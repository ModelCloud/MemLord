# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import pytest
import torch

from memlord import MemLord

def cuda_or_skip(min_gpus: int = 1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < min_gpus:
        pytest.skip(f"Requires CUDA with at least {min_gpus} GPU(s)")

def alloc_tensor_on(device, shape=(128, 128), dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=device)

def test_normal_out_does_not_allocate_cpu():
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()
    buf = torch.empty(1000, dtype=torch.float32)

    before, _ = tr.allocated(torch.device("cpu"))
    # NOTE: Some PyTorch builds require size= when using out= with float mean/std.
    torch.normal(0.0, 1.0, size=buf.shape, out=buf)
    after, _ = tr.allocated(torch.device("cpu"))

    assert after == before  # no new allocation

@pytest.mark.cuda
def test_same_device_dtype_cast_counts_once_with_snakehooks_cuda():
    cuda_or_skip(1)
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_torch()
    tr.hook_into_python()  # counts creation/clone-like allocs

    d0 = torch.device("cuda:0")
    t = alloc_tensor_on(d0, (256, 128), dtype=torch.float32)
    # bytes expected for the fp16 destination allocation
    approx_added = t.numel() * 2  # 2 bytes per element for float16

    before, _ = tr.allocated(d0)
    u = t.to(dtype=torch.float16)  # same device; should be counted once
    mid, _ = tr.allocated(d0)

    assert mid >= before + int(approx_added * 0.8)

    # cleanup to exercise frees
    del t, u
    for _ in range(3):
        torch.cuda.synchronize()
        gc.collect()

    freed, _ = tr.freed(d0)
    assert freed >= int(approx_added * 0.8)
