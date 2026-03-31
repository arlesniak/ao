# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e
IS_ROCM=$(rocm-smi --version || true)
HAS_ACCEL=$(python - <<'PY'
import torch

has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
has_cuda = torch.cuda.is_available()
print("1" if (has_xpu or has_cuda) else "0")
PY
)

# These tests do not work on ROCm yet and require accelerator devices.
if [ -z "$IS_ROCM" ] && [ "$HAS_ACCEL" = "1" ]
then
./test/float8/test_fsdp.sh
./test/float8/test_fsdp_compile.sh
./test/float8/test_dtensor.sh
python test/float8/test_fsdp2/test_fsdp2.py
fi

echo "all multi gpu tests successful"
