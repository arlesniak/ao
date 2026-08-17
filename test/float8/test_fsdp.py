# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of bf16 versus float8 with FSDP on. At a high level:
1. start with a reference model, with FSDP on
2. run forward + backward + optim for 2 iterations
3. repeat 2 with float8 enabled (2 iterations needed for delayed scaling)
4. compare outputs and state dict between (2) and (3), should be close
"""

import copy
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear_utils import (
    convert_to_float8_training,
)
from torchao.float8.float8_utils import compute_error

torch.manual_seed(0)

B, M, K, N = 8, 8, 32, 32
LR = 0.01
N_ITER = 2
SQNR_THRESHOLD = 15.0


def get_device_type():
    """Return the current accelerator device type, falling back to CPU."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"


DEVICE_TYPE = get_device_type()


def get_backend():
    """Return the appropriate distributed backend for the current device."""
    return "nccl" if DEVICE_TYPE == "cuda" else "xccl"


@dataclass(frozen=True)
class RunArgs:
    """Configuration passed to each spawned FSDP worker."""

    emulate: bool
    base_dtype: torch.dtype
    compile: bool


@contextmanager
def distributed_context(rank, world_size):
    """Set up and tear down the NCCL process group for a single worker."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(get_backend(), rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()


def get_model(K, N, base_dtype=torch.float32):
    return nn.Sequential(
        nn.Linear(K, N, dtype=base_dtype),
        nn.ReLU(),
        nn.Linear(N, N, dtype=base_dtype),
        nn.ReLU(),
    )


def build_models(rank, base_dtype):
    """Return (bf16_reference, float8) models, both wrapped in FSDP."""
    reference = get_model(K, N, base_dtype=base_dtype).to(rank)
    float8 = copy.deepcopy(reference)

    # Note: we only iterate over `scaling_type_weight` because FSDP only interacts
    # with weights.
    convert_to_float8_training(float8, config=Float8LinearConfig())

    # To compile FSDP, we need use_orig_params to True
    # TODO: FSDP(torch.compile(model), use_orig_params=True) doesn't work yet.
    reference = FSDP(reference, use_orig_params=True)
    float8 = FSDP(float8, use_orig_params=True)
    return reference, float8


def make_local_batches(rank, world_size, base_dtype):
    """Slice global input/grad tensors into this rank's local shard."""
    # Note: we need two different inputs to properly measure the impact of
    # delayed scaling, before the first input uses dynamic scaling to
    # populate the buffers
    # TODO(future PR): delete ^, since we deleted delayed scaling
    inputs_global = [
        torch.randn(B, M, K, device=DEVICE_TYPE).to(base_dtype) for _ in range(N_ITER)
    ]
    grads_global = [
        torch.randn(B, M, N, device=DEVICE_TYPE).to(base_dtype) for _ in range(N_ITER)
    ]

    # basic distributed data sampling
    assert B % world_size == 0
    start = int(rank / world_size * B)
    end = int((rank + 1) / world_size * B)
    inputs_local = [t[start:end].to(rank) for t in inputs_global]
    grads_local = [g[start:end].to(rank) for g in grads_global]
    return inputs_local, grads_local


def forward_backward(model, optim, x, grad):
    optim.zero_grad()
    y = model(x)
    y.backward(grad)
    optim.step()
    return y


def all_gather_cat(local, world_size, base_dtype, rank):
    """All-gather a local tensor across ranks and concatenate along dim 0."""
    gathered = [
        torch.zeros(*local.shape, dtype=base_dtype).to(rank) for _ in range(world_size)
    ]
    dist.all_gather(gathered, local)
    return torch.cat(gathered, dim=0)


def full_state_dict(model):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


def assert_sqnr(v1, v2, msg=""):
    sqnr = compute_error(v1, v2)
    assert sqnr > SQNR_THRESHOLD, f"SQNR of {sqnr} is too low{msg}"


# taken from https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# and modified
def fsdp_main(rank, world_size, args: RunArgs):
    torch.accelerator.set_device_index(rank)
    print("args", args)

    with distributed_context(rank, world_size):
        reference, float8 = build_models(rank, args.base_dtype)
        opt_ref = torch.optim.SGD(reference.parameters(), lr=LR)
        opt_fp8 = torch.optim.SGD(float8.parameters(), lr=LR)

        inputs_local, grads_local = make_local_batches(
            rank, world_size, args.base_dtype
        )

        y_local = y_local_fp8 = None
        for i in range(N_ITER):
            # We first run one iteration without compile, as a workaround to compile
            # the float8 layer. In the first iter, float8 layers take the
            # "is_amax_initialized == False" branch; afterwards the True branch.
            # TODO: Need to fix compile to run without this workaround.
            if i == 1 and args.compile:
                reference = torch.compile(reference)
                float8 = torch.compile(float8)
            y_local = forward_backward(
                reference, opt_ref, inputs_local[i], grads_local[i]
            )
            y_local_fp8 = forward_backward(
                float8, opt_fp8, inputs_local[i], grads_local[i]
            )
            _ = compute_error(y_local, y_local_fp8)  # noqa: F841

        # compare gathered outputs
        y_global = all_gather_cat(y_local, world_size, args.base_dtype, rank)
        y_global_fp8 = all_gather_cat(y_local_fp8, world_size, args.base_dtype, rank)
        if rank == 0:
            assert_sqnr(y_global, y_global_fp8)

        # compare global state dicts
        # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
        dist.barrier()
        cpu_state = full_state_dict(reference)
        cpu_state_fp8 = full_state_dict(float8)
        if rank == 0:
            for k, v1 in cpu_state.items():
                v2 = cpu_state_fp8[k]
                v1, v2 = v1.cpu(), v2.cpu()
                assert_sqnr(v1, v2, msg=f", k: {k}, v1: {v1}, v2: {v2}")


def run(compile_fsdp: bool = False):
    base_dtype = torch.bfloat16

    emulate = False
    if not torch.accelerator.is_available():
        warnings.warn("Accelerator not available, running in emulation_mode")
        emulate = True
    elif DEVICE_TYPE == "cuda" and torch.cuda.get_device_capability() < (8, 9):
        warnings.warn(
            f"CUDA capability {torch.cuda.get_device_capability()} < (8.9), running in emulation mode"
        )
        emulate = True

    world_size = torch.accelerator.device_count() if not emulate else 1
    args = RunArgs(emulate=emulate, base_dtype=base_dtype, compile=compile_fsdp)
    mp.spawn(fsdp_main, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    fire.Fire(run)
