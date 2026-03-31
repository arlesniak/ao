# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
#!/bin/bash

# terminate script on first error
set -e

launch() {
    echo "launching compile_fsdp $COMPILE"

    # Use oneCCL-friendly defaults for XPU while preserving CUDA fallback.
    CCL_LOG_LEVEL=info NCCL_DEBUG=WARN python test/float8/test_fsdp.py \
        --compile_fsdp $COMPILE

    echo "✅ All Tests Passed ✅"
}

if ! python - <<'PY'
import sys
import torch

has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
has_cuda = torch.cuda.is_available()
sys.exit(0 if (has_xpu or has_cuda) else 1)
PY
then
    echo "Skipping test_fsdp.sh because no XPU/CUDA devices are available."
    exit
fi

COMPILE=False launch
COMPILE=True launch
