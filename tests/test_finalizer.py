# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import gc
import os

import pytest
import torch
import torch.nn as nn

from memlord import MemLord


# -----------------------
# Helpers (local to file)
# -----------------------

def alloc_tensor_on(device, shape=(128, 128), dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=device)

def module_linear(in_f=64, out_f=64, device="cpu", dtype=torch.float32):
    m = nn.Linear(in_f, out_f, bias=True)
    m = m.to(dtype=dtype)
    return m.to(device)

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def module_nbytes(m: nn.Module) -> int:
    total = 0
    for p in m.parameters():
        total += p.numel() * p.element_size()
    for b in m.buffers():
        total += b.numel() * b.element_size()
    return total

def cuda_or_skip(min_gpus: int = 1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < min_gpus:
        pytest.skip(f"Requires CUDA with at least {min_gpus} GPU(s)")

# Ensure DEBUG is off (less noisy logs during tests)
@pytest.fixture(autouse=True)
def ensure_debug_off_and_restore_env():
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


# --------------------------------------------
# Finalizer basics: CPU tensor, explicit alloc
# --------------------------------------------

def test_finalizer_tensor_cpu_del_triggers_freed():
    tracker = MemLord(auto_gc_strategy={})  # disable auto-GC resets
    tracker.hook_into_python(enable_factory_wrappers=False)

    t = alloc_tensor_on("cpu", (256, 256))
    expected = tensor_nbytes(t)

    # allocate() records bytes and (since python hooks are active) registers finalizer
    tracker.allocate(t)

    # Drop the only reference and force GC — finalizer should credit freed bytes
    del t
    gc.collect()

    freed, _ = tracker.freed(torch.device("cpu"))
    assert freed >= expected


# ---------------------------------------------------------
# Aliasing: finalizer must NOT fire until last ref is gone
# ---------------------------------------------------------

def test_finalizer_tensor_cpu_aliasing_only_last_del_triggers():
    tracker = MemLord(auto_gc_strategy={})
    tracker.hook_into_python(enable_factory_wrappers=False)

    t = alloc_tensor_on("cpu", (128, 128))
    alias = t  # second reference
    expected = tensor_nbytes(t)

    tracker.allocate(t)  # registers finalizer for the underlying object

    # Delete only one reference; object still alive via 'alias'
    del t
    gc.collect()

    freed_before, _ = tracker.freed(torch.device("cpu"))
    assert freed_before == 0  # finalizer must NOT have run yet

    # Now delete the last reference; finalizer should fire
    del alias
    gc.collect()

    freed_after, _ = tracker.freed(torch.device("cpu"))
    assert freed_after >= expected


# -------------------------------------------------
# Module finalizer: CPU nn.Module lifetime end free
# -------------------------------------------------

def test_finalizer_module_cpu_del_triggers_freed():
    tracker = MemLord(auto_gc_strategy={})
    tracker.hook_into_python(enable_factory_wrappers=False)

    m = module_linear(128, 256, device="cpu", dtype=torch.float32)
    expected = module_nbytes(m)

    tracker.allocate(m)

    del m
    gc.collect()

    freed, _ = tracker.freed(torch.device("cpu"))
    assert freed >= expected


# ----------------------------------------------------------
# Factory wrappers path: auto-register without tracker.allocate
# ----------------------------------------------------------

def test_finalizer_factory_wrappers_auto_register_tensor_cpu():
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)

    try:
        t = torch.empty((64, 64), dtype=torch.float32, device="cpu")
        expected = tensor_nbytes(t)
        # NOTE: We did NOT call tracker.allocate(t); finalizer alone should credit freed.

        del t
        gc.collect()

        freed, _ = tracker.freed(torch.device("cpu"))
        assert freed >= expected
    finally:
        # restore original torch factories for isolation
        hooks.disable()


# --------------------------
# CUDA variants (if present)
# --------------------------

@pytest.mark.cuda
def test_finalizer_tensor_cuda_del_triggers_freed():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    tracker.hook_into_python(enable_factory_wrappers=False)

    d0 = torch.device("cuda:0")
    t = alloc_tensor_on(d0, (128, 128))
    expected = tensor_nbytes(t)

    tracker.allocate(t)

    del t
    gc.collect()

    freed_cuda0, _ = tracker.freed(d0)
    assert freed_cuda0 >= expected


@pytest.mark.cuda
def test_finalizer_tensor_cuda_aliasing_only_last_del_triggers():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    tracker.hook_into_python(enable_factory_wrappers=False)

    d0 = torch.device("cuda:0")
    t = alloc_tensor_on(d0, (64, 64))
    alias = t
    expected = tensor_nbytes(t)

    tracker.allocate(t)

    del t
    gc.collect()
    freed_before, _ = tracker.freed(d0)
    assert freed_before == 0  # still referenced by 'alias'

    del alias
    gc.collect()
    freed_after, _ = tracker.freed(d0)
    assert freed_after >= expected


@pytest.mark.cuda
def test_finalizer_factory_wrappers_auto_register_tensor_cuda():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)

    try:
        d0 = torch.device("cuda:0")
        t = torch.empty((32, 32), dtype=torch.float32, device=d0)
        expected = tensor_nbytes(t)
        # Not calling tracker.allocate(t)

        del t
        gc.collect()

        freed_cuda0, _ = tracker.freed(d0)
        assert freed_cuda0 >= expected
    finally:
        hooks.disable()


# -------------------------------------------------------------
# Idempotence: finalizer runs only once, even with extra GC runs
# -------------------------------------------------------------

def test_finalizer_idempotent_on_multiple_gc_runs_cpu():
    tracker = MemLord(auto_gc_strategy={})
    tracker.hook_into_python(enable_factory_wrappers=False)

    t = alloc_tensor_on("cpu", (100, 100))
    expected = tensor_nbytes(t)
    tracker.allocate(t)

    del t
    gc.collect()
    freed1, _ = tracker.freed(torch.device("cpu"))

    # Additional GC passes should not change the freed counter for this object
    gc.collect()
    gc.collect()
    freed2, _ = tracker.freed(torch.device("cpu"))

    assert freed1 >= expected
    assert freed2 == freed1
