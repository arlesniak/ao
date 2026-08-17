# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Test numerics of manually defined float16 TP vs float8 TP of toy models

Note: for now, this does not run in CI.
TODO(future): make this run in CI
"""

import os

import torch
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)
from tqdm import tqdm

from torchao.float8 import Float8LinearConfig
from torchao.float8.config import (
    CastConfig,
    Float8LinearRecipeName,
    ScalingType,
    e4m3_dtype,
)
from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.float8.float8_scaling_utils import NoopFwToFloat8BwDynamic
from torchao.float8.float8_training_tensor import (
    Float8TrainingTensor,
    GemmInputRole,
    LinearMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor
from torchao.testing.training.dtensor_utils import (
    _test_lowp_mlp_tensor_parallelism_base,
)
from torchao.utils import torch_version_at_least

torch.set_float32_matmul_precision("high")


def get_device_type() -> str:
    """Current accelerator type, e.g. "cuda" / "xpu"."""
    return str(torch.accelerator.current_accelerator())


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", -1))


def bind_local_device() -> str:
    """Pin this process to its local device and return the device type."""
    device_type = get_device_type()
    torch.get_device_module(device_type).set_device(torch.distributed.get_rank())
    return device_type


def build_device_mesh() -> DeviceMesh:
    return init_device_mesh(get_device_type(), (get_world_size(),))


def setup_distributed():
    device_mesh = build_device_mesh()
    # seed must be the same in all processes
    torch.manual_seed(1)
    bind_local_device()
    return device_mesh


def _to_fp8_tensor(tensor, fp8_dtype=e4m3_dtype, role=None):
    scale = tensor_to_scale(tensor, fp8_dtype).float()
    return hp_tensor_and_scale_to_float8(tensor, scale, fp8_dtype, None, role)


def _distributed_fp8_tensor(tensor, mesh, placement, role=None):
    return DTensor.from_local(
        _to_fp8_tensor(tensor, role=role), mesh, [placement], run_check=False
    )


def _test_scaled_mm(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)
    y_fp32 = torch.eye(size, device=device).t()

    placement_combs = (
        (Shard(0), Replicate()),
        (Replicate(), Shard(1)),
        (Shard(1), Shard(0)),
    )
    expected_dt_out_shape = (
        (size * world_size, size),
        (size, size * world_size),
        (size, size),
    )
    for idx, (lhs_placement, rhs_placement) in enumerate(placement_combs):
        dist_x_fp8 = _distributed_fp8_tensor(
            x_fp32, mesh, lhs_placement, GemmInputRole.INPUT
        )
        dist_y_fp8 = _distributed_fp8_tensor(
            y_fp32, mesh, rhs_placement, GemmInputRole.WEIGHT
        )

        assert isinstance(dist_x_fp8.to_local(), Float8TrainingTensor)
        assert isinstance(dist_y_fp8.to_local(), Float8TrainingTensor)
        assert dist_x_fp8.to_local()._orig_dtype == torch.float32
        out_fp8 = torch.mm(dist_x_fp8, dist_y_fp8)
        local_fp8_out = out_fp8.to_local()
        assert out_fp8.shape == expected_dt_out_shape[idx], (idx, local_fp8_out.shape)

        # after mm the out dtype should be fp32
        assert local_fp8_out.dtype == torch.float32


def _test_fp8_redistribute(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype
    world_size = mesh.size()

    x_fp32 = torch.rand(size, size, device=device)

    dist_x_fp8 = _distributed_fp8_tensor(x_fp32, mesh, Shard(0))
    out_dist = dist_x_fp8.redistribute(placements=[Replicate()])
    assert out_dist.shape == (size * world_size, size)
    assert out_dist.placements == (Replicate(),)
    out_local = out_dist.to_local()
    # after allgather the out shape should be replicate
    assert out_local.shape == (size * world_size, size)
    from torch.distributed._functional_collectives import AsyncCollectiveTensor

    if isinstance(out_local, AsyncCollectiveTensor):
        out_local = out_local.wait()

    assert isinstance(out_local, Float8TrainingTensor)
    assert out_local._data.dtype == fp8_dtype


def _test_dtensor_cast_to_fp8(mesh: DeviceMesh, size=16):
    device = mesh.device_type
    fp8_dtype = e4m3_dtype

    x_fp32 = torch.rand(size, size, device=device)
    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])

    dist_x_scale = tensor_to_scale(dist_x_fp32, fp8_dtype).float()
    assert isinstance(dist_x_scale, DTensor)

    dist_x_fp8 = hp_tensor_and_scale_to_float8(dist_x_fp32, dist_x_scale, fp8_dtype)
    assert isinstance(dist_x_fp8, DTensor)


def _test_dtensor_fp8_autograd(mesh: DeviceMesh, size=16):
    # TODO: re-enable after fixing https://github.com/pytorch/ao/issues/4106
    if torch_version_at_least("2.11.0.dev"):
        print("skipping _test_dtensor_fp8_autograd for torch >= 2.11.0.dev")
        return
    device = mesh.device_type
    fp8_dtype = e4m3_dtype

    x_fp32 = torch.rand(size, size, device=device, requires_grad=True)
    local_weight = torch.rand(2 * size, size, device=device, requires_grad=True)
    target = torch.rand(size, 2 * size, device=device)

    dist_x_fp32 = distribute_tensor(x_fp32, mesh, [Shard(0)])
    dist_wight_fp32 = distribute_tensor(local_weight, mesh, [Shard(0)])
    dist_target = distribute_tensor(target, mesh, [Shard(0)])

    dist_x_fp8 = _to_fp8_tensor(dist_x_fp32, fp8_dtype, GemmInputRole.INPUT)
    dist_weight_fp8 = _to_fp8_tensor(
        dist_wight_fp32, fp8_dtype, GemmInputRole.WEIGHT
    )

    out = torch.nn.functional.linear(dist_x_fp8, dist_weight_fp8)
    out = NoopFwToFloat8BwDynamic.apply(out, LinearMMConfig(), fp8_dtype)
    assert isinstance(out, DTensor), f"Expected DTensor, got {type(out)}"
    loss = torch.sum(torch.abs(out - dist_target))
    loss.backward()


def _test_fp8_mlp_tensor_parallelism_eager(mesh: DeviceMesh, size=32):
    _test_fp8_mlp_tensor_parallelism(mesh, size, compile=False)


def _test_fp8_mlp_tensor_parallelism(mesh, size, compile):
    configs = (
        (Float8LinearConfig(emulate=True), True),
        (Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE), False),
    )
    for config, allgather_in_lowp in configs:
        object.__setattr__(config, "emulate", True)
        _test_lowp_mlp_tensor_parallelism_base(
            mesh, config, size, compile=compile, allgather_in_lowp=allgather_in_lowp
        )


def _test_fp8_mlp_tensor_parallelism_compile(mesh: DeviceMesh, size=32):
    _test_fp8_mlp_tensor_parallelism(mesh, size, compile=True)


def _test_distribute_fsdp_tensor_subclass(tp_mesh: DeviceMesh):
    torch.manual_seed(42)
    device = tp_mesh.device_type
    model = Transformer(ModelArgs(dropout_p=0.0, weight_tying=False)).to(device)
    convert_to_float8_training(
        model,
        config=Float8LinearConfig(
            enable_fsdp_float8_all_gather=True,
            cast_config_weight=CastConfig(scaling_type=ScalingType.DYNAMIC),
        ),
    )
    # test Float8ColwiseParallel
    colwise_param = distribute_tensor(
        model.layers[0].attention.wq.weight, tp_mesh, [Shard(0)]
    )
    assert isinstance(colwise_param, DTensor) and isinstance(
        colwise_param._local_tensor, WeightWithDynamicFloat8CastTensor
    ), (
        f"expect DTensor(local_tensor={WeightWithDynamicFloat8CastTensor}) but got {colwise_param}"
    )
    # test Float8RowwiseParallel
    rowwise_param = distribute_tensor(
        model.layers[0].attention.wo.weight, tp_mesh, [Shard(1)]
    )
    assert isinstance(rowwise_param, DTensor) and isinstance(
        rowwise_param._local_tensor, WeightWithDynamicFloat8CastTensor
    ), (
        f"expect DTensor(local_tensor={WeightWithDynamicFloat8CastTensor}) but got {colwise_param}"
    )


if __name__ == "__main__":
    # float8 only works on CUDA H100 and XPU so we only test cuda/XPU we follow
    # other test files to not use TestCase but instead just add the test
    # cases in the main func.
    device_mesh = setup_distributed()
    tests = [
        _test_scaled_mm,
        _test_fp8_redistribute,
        _test_dtensor_cast_to_fp8,
        _test_dtensor_fp8_autograd,
        _test_fp8_mlp_tensor_parallelism_eager,
        _test_fp8_mlp_tensor_parallelism_compile,
        _test_distribute_fsdp_tensor_subclass,
    ]

    for test in tqdm(tests, desc="Running tests"):
        try:
            test(device_mesh)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            raise e

    torch.distributed.destroy_process_group()
