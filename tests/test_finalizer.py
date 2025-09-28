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
# Finalizer basics: CPU tensor (auto-hooked)
# --------------------------------------------

def test_finalizer_tensor_cpu_del_triggers_freed():
    tracker = MemLord(auto_gc_strategy={})  # disable auto-GC resets
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        t = alloc_tensor_on("cpu", (256, 256))
        expected = tensor_nbytes(t)

        # Drop the only reference and force GC — finalizer should credit freed bytes
        del t
        gc.collect()

        freed, _ = tracker.freed(torch.device("cpu"))
        assert freed >= expected
    finally:
        hooks.disable()


# ---------------------------------------------------------
# Aliasing: finalizer must NOT fire until last ref is gone
# ---------------------------------------------------------

def test_finalizer_tensor_cpu_aliasing_only_last_del_triggers():
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        t = alloc_tensor_on("cpu", (128, 128))
        alias = t  # second reference
        expected = tensor_nbytes(t)

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
    finally:
        hooks.disable()


# -------------------------------------------------
# Module finalizer: CPU nn.Module lifetime end free
# (parameters are created via torch factories -> auto-registered)
# -------------------------------------------------

def test_finalizer_module_cpu_del_triggers_freed():
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        # Enable hooks BEFORE creating the module so its parameters get registered
        m = module_linear(128, 256, device="cpu", dtype=torch.float32)
        expected = module_nbytes(m)

        del m
        gc.collect()

        freed, _ = tracker.freed(torch.device("cpu"))
        assert freed >= expected
    finally:
        hooks.disable()


# ----------------------------------------------------------
# Factory wrappers path: auto-register without manual tracking
# ----------------------------------------------------------

def test_finalizer_factory_wrappers_auto_register_tensor_cpu():
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        t = torch.empty((64, 64), dtype=torch.float32, device="cpu")
        expected = tensor_nbytes(t)

        del t
        gc.collect()

        freed, _ = tracker.freed(torch.device("cpu"))
        assert freed >= expected
    finally:
        hooks.disable()


# --------------------------
# CUDA variants (if present)
# --------------------------

@pytest.mark.cuda
def test_finalizer_tensor_cuda_del_triggers_freed():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        d0 = torch.device("cuda:0")
        t = alloc_tensor_on(d0, (128, 128))
        expected = tensor_nbytes(t)

        del t
        gc.collect()

        freed_cuda0, _ = tracker.freed(d0)
        assert freed_cuda0 >= expected
    finally:
        hooks.disable()


@pytest.mark.cuda
def test_finalizer_tensor_cuda_aliasing_only_last_del_triggers():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        d0 = torch.device("cuda:0")
        t = alloc_tensor_on(d0, (64, 64))
        alias = t
        expected = tensor_nbytes(t)

        del t
        gc.collect()
        freed_before, _ = tracker.freed(d0)
        assert freed_before == 0  # still referenced by 'alias'

        del alias
        gc.collect()
        freed_after, _ = tracker.freed(d0)
        assert freed_after >= expected
    finally:
        hooks.disable()


@pytest.mark.cuda
def test_finalizer_factory_wrappers_auto_register_tensor_cuda():
    cuda_or_skip(1)
    tracker = MemLord(auto_gc_strategy={})
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        d0 = torch.device("cuda:0")
        t = torch.empty((32, 32), dtype=torch.float32, device=d0)
        expected = tensor_nbytes(t)

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
    hooks = tracker.hook_into_python(enable_factory_wrappers=True)
    try:
        t = alloc_tensor_on("cpu", (100, 100))
        expected = tensor_nbytes(t)

        del t
        gc.collect()
        freed1, _ = tracker.freed(torch.device("cpu"))

        # Additional GC passes should not change the freed counter for this object
        gc.collect()
        gc.collect()
        freed2, _ = tracker.freed(torch.device("cpu"))

        assert freed1 >= expected
        assert freed2 == freed1
    finally:
        hooks.disable()


# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import threading
import time
import torch

from memlord import MemLord

def test_finalizer_does_not_trigger_autogc(monkeypatch):
    import threading
    import gc
    import torch
    from memlord import MemLord

    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    # Create tensor in a worker and HOLD a reference until we patch.
    ready = threading.Event()
    release = threading.Event()
    holder = {}

    def worker():
        t = torch.empty(1024)
        holder["t"] = t  # keep alive until we say release
        ready.set()
        release.wait()
        # drop last ref -> triggers finalizer
        holder.pop("t", None)
        gc.collect()

    th = threading.Thread(target=worker)
    th.start()
    assert ready.wait(timeout=2.0)

    # Patch AFTER allocation but BEFORE finalizer can run.
    calls = {"n": 0}
    def spy(*a, **k):
        calls["n"] += 1
    monkeypatch.setattr(tr, "_maybe_auto_gc", spy)

    # Now let the worker drop the ref; finalizers may run.
    release.set()
    th.join(timeout=2.0)

    # Force finalizers to run (a few passes for good measure).
    for _ in range(5):
        gc.collect()

    # Finalizers must NOT call auto-GC.
    assert calls["n"] == 0


def test_no_deadlock_with_concurrent_finalizers():
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    stop = threading.Event()
    exc = []

    def worker():
        try:
            while not stop.is_set():
                x = torch.empty(4096)
                y = torch.empty(8192)
                del x, y
                gc.collect()
        except Exception as e:
            exc.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()

    time.sleep(0.5)
    stop.set()
    for th in threads:
        th.join(timeout=2.0)

    assert not exc  # no exceptions or deadlocks observed
