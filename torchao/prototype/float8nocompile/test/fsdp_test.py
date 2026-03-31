# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
######################################################################
#
# To run these unit tests, use the following command:
#
# torchrun --nproc_per_node=${NUM_GPUS} -m pytest test/fsdp_test.py
#
#######################################################################
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard

from torchao.float8.float8_linear_utils import convert_to_float8_training
from torchao.prototype.float8nocompile.float8nocompile_linear_utils import (
    convert_to_float8_nocompile_training,
)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2048, 4096, bias=False),
            nn.Linear(4096, 16, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _get_device_type() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return ""


def _get_backend(device_type: str) -> str:
    return "xccl" if device_type == "xpu" else "nccl"


def setup_distributed(device_type: str):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(_get_backend(device_type), rank=rank, world_size=world_size)
    if device_type == "xpu":
        torch.xpu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)


@pytest.fixture
def model1():
    torch.manual_seed(0)
    return TestModel()


@pytest.fixture
def model2():
    torch.manual_seed(0)
    return TestModel()


def test_model_weights_and_gradients(model1, model2):
    device_type = _get_device_type()
    assert device_type, "No XPU/CUDA accelerator available"
    device = torch.device(device_type)

    setup_distributed(device_type)

    model1 = model1.to(torch.bfloat16).to(device)
    model2 = model2.to(torch.bfloat16).to(device)

    # compare production float8 linear conversion with no-compile version
    convert_to_float8_training(model2)
    convert_to_float8_nocompile_training(model1)

    # distributed training with FSDP2
    fully_shard(model1)
    fully_shard(model2)

    input_tensor = torch.randn(
        16, 2048, requires_grad=True, dtype=torch.bfloat16, device=device
    )
    input_copy1 = input_tensor.clone().detach().requires_grad_(True)
    input_copy2 = input_tensor.clone().detach().requires_grad_(True)

    loss_fn = nn.MSELoss()

    output1 = model1(input_copy1)
    output2 = model2(input_copy2)

    loss1 = loss_fn(output1, torch.zeros_like(output1))
    loss2 = loss_fn(output2, torch.zeros_like(output2))

    loss1.backward()
    loss2.backward()

    # compare the outputs, weight gradients, and input gradients
    assert torch.allclose(output1, output2, atol=0, rtol=0)
    assert torch.allclose(input_copy1.grad, input_copy2.grad, atol=0, rtol=0)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(param1.grad, param2.grad)

    dist.destroy_process_group()
