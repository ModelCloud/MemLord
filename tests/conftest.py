# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import os
import pytest
import torch
import torch.nn as nn

from memlord import MemLord

# ---------- Helpers (shared) ----------

DTYPES_TENSOR = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.int32,
    torch.complex64,
]

DTYPES_MODULE = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.complex64,
]

def device_count(kind="cuda"):
    if kind == "cuda" and torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def cuda_or_skip(min_gpus: int = 1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < min_gpus:
        pytest.skip(f"Requires CUDA with at least {min_gpus} GPU(s)")

def alloc_tensor_on(device, shape=(128, 128), dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=device)

def module_linear(in_f=64, out_f=64, device="cpu", dtype=torch.float32):
    m = nn.Linear(in_f, out_f, bias=True)
    m = m.to(dtype=dtype)
    return m.to(device)

def all_cuda_devices():
    if not torch.cuda.is_available():
        return []
    return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def ensure_debug_off_and_restore_env():
    """Keep DEBUG off unless a test explicitly turns it on; restore env afterwards."""
    old = os.environ.get("DEBUG")
    if "DEBUG" in os.environ:
        del os.environ["DEBUG"]
    try:
        yield
    finally:
        if old is None:
            if "DEBUG" in os.environ:
                del os.environ["DEBUG"]
        else:
            os.environ["DEBUG"] = old

@pytest.fixture
def tracker():
    # Disable auto-GC by default (empty strategy = no bands matched)
    return MemLord(auto_gc_strategy={})
