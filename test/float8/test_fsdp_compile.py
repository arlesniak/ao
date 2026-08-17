# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Test autocast + torch.compile + FSDP + Float8Linear
"""

import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torchao.float8 import Float8LinearConfig
from torchao.float8.float8_linear_utils import convert_to_float8_training

torch.manual_seed(0)


@dataclass(frozen=True)
class Config:
    """Static configuration for the FSDP + float8 compile test."""

    batch: int = 8
    m: int = 8
    k: int = 32
    n: int = 32
    lr: float = 0.01
    n_iter: int = 1

    master_addr: str = "localhost"
    master_port: str = "12355"

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (self.batch, self.m, self.k, self.n)


CFG = Config()


@contextmanager
def process_group(rank: int, world_size: int):
    """Set up and tear down the distributed process group."""
    os.environ["MASTER_ADDR"] = CFG.master_addr
    os.environ["MASTER_PORT"] = CFG.master_port

    device_type = torch.accelerator.current_accelerator().type
    backend = dist.get_default_backend_for_device(device_type)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        yield
    finally:
        dist.destroy_process_group()


def build_model(emulate: bool, base_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Build a small MLP and convert its linear layers to float8 training."""
    model = nn.Sequential(
        nn.Linear(CFG.k, CFG.n, dtype=base_dtype),
        nn.ReLU(),
    )
    convert_to_float8_training(model, config=Float8LinearConfig(emulate=emulate))
    return model


def fsdp_worker(rank: int, world_size: int, emulate: bool) -> None:
    """Per-rank entry point run under ``mp.spawn``."""
    with process_group(rank, world_size):
        device_type = torch.accelerator.current_accelerator().type
        torch.accelerator.set_device_index(rank)
        device = f"{device_type}:{rank}"

        model = build_model(emulate).to(device)
        # To compile FSDP, we need use_orig_params=True.
        model = FSDP(model, use_orig_params=True)
        model = torch.compile(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr * world_size)
        input_local = torch.randn(*CFG.input_shape, device=device)

        for _ in range(CFG.n_iter):
            optimizer.zero_grad()
            with torch.autocast(device_type):
                y_local = model(input_local)
            y_local.sum().backward()
            optimizer.step()

    print("done!")


def should_emulate() -> bool:
    """Decide whether to run in emulation mode based on available hardware."""
    if not torch.accelerator.is_available():
        warnings.warn("GPU not available, running in emulation mode", stacklevel=2)
        return True
    if torch.cuda.is_available() and torch.cuda.get_device_capability() < (9, 0):
        warnings.warn(
            f"CUDA capability {torch.cuda.get_device_capability()} < (9.0), "
            "running in emulation mode",
            stacklevel=2,
        )
        return True
    return False


def run() -> None:
    world_size = torch.accelerator.device_count()
    mp.spawn(
        fsdp_worker,
        args=(world_size, should_emulate()),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    fire.Fire(run)
