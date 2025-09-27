# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
from typing import Dict

import pytest
import torch
import torch.nn as nn

from memlord import MemLord

# ---------- Helpers ----------

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
    torch.complex64,   # NOTE: int dtypes are invalid for nn.Module.to(dtype=…)
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

# ---------- Core API: CPU-only ----------

def test_allocate_and_free_tensor_cpu(tracker: MemLord):
    t = alloc_tensor_on("cpu", (256, 256))
    nbytes = t.numel() * t.element_size()

    tracker.allocate(t)
    raw, human = tracker.allocated(torch.device("cpu"))
    assert raw >= nbytes
    assert isinstance(human, str) and "B" in human

    tracker.free(t)
    raw_freed, _ = tracker.freed(torch.device("cpu"))
    assert raw_freed >= nbytes

def test_allocate_and_free_module_cpu(tracker: MemLord):
    m = module_linear(128, 256, device="cpu", dtype=torch.float32)
    total = 0
    for p in m.parameters():
        total += p.numel() * p.element_size()
    for b in m.buffers():
        total += b.numel() * b.element_size()

    tracker.allocate(m)
    got, _ = tracker.allocated(torch.device("cpu"))
    assert got >= total

    tracker.free(m)
    freed, _ = tracker.freed(torch.device("cpu"))
    assert freed >= total

def test_tuple_inputs_single_call(tracker: MemLord, capsys):
    os.environ["DEBUG"] = "1"  # enable logs to check summary count (runtime-active)
    t1 = alloc_tensor_on("cpu", (128, 128))
    t2 = alloc_tensor_on("cpu", (64, 64))
    total = t1.numel() * t1.element_size() + t2.numel() * t2.element_size()

    tracker.allocate((t1, t2))
    out = capsys.readouterr().out
    assert out.count("[allocate-summary]") == 1  # one summary per call

    raw, _ = tracker.allocated(torch.device("cpu"))
    assert raw >= total

    tracker.free((t1, t2))
    out = capsys.readouterr().out
    assert out.count("[free-summary]") == 1
    freed, _ = tracker.freed(torch.device("cpu"))
    assert freed >= total

# ---------- Aggregation queries ----------

def test_type_aggregation_without_index(tracker: MemLord):
    cpu = torch.device("cpu")
    t1 = alloc_tensor_on(cpu, (100, 100))
    t2 = alloc_tensor_on(cpu, (200, 50))
    tracker.allocate((t1, t2))

    total_by_cpu, _ = tracker.allocated(torch.device("cpu"))  # type aggregator
    direct, _ = tracker.allocated(cpu)  # exact device (same on CPU)
    assert total_by_cpu == direct

# ---------- Hotspot tracking ----------

def test_hotspot_tracks_file_line(tracker: MemLord):
    t = alloc_tensor_on("cpu", (64, 64))
    tracker.allocate(t)
    top = tracker.top_alloc_site()
    assert top is not None
    dev, site, bytes_ = top
    assert isinstance(site, str) and ":" in site
    assert bytes_ > 0

    tracker.free(t)
    top_free = tracker.top_free_site()
    assert top_free is not None
    dev2, site2, bytes2 = top_free
    assert ":" in site2 and bytes2 > 0

# ---------- Auto-GC (per-device) ----------

@pytest.mark.cuda
def test_auto_gc_runs_and_resets_freed_per_device(monkeypatch):
    cuda_or_skip(1)

    # Strategy: single band over all usage, trigger on any freed >= 1 byte
    strategy = {
        (0, 101): {"metric": "freed", "threshold": {"bytes": 1}},
    }
    tr = MemLord(auto_gc_strategy=strategy)
    d0 = torch.device("cuda:0")

    # Make GC deterministic
    import memlord.core as core_mod
    calls = []
    def fake_gc(dev):
        calls.append(dev)
        return True
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: fake_gc(d))

    t = alloc_tensor_on(d0, (256, 256))
    tr.allocate(t)

    tr.free(t)
    # GC should have run and reset freed counter
    assert calls == [d0]
    fr1, _ = tr.freed(d0)
    assert fr1 == 0  # reset after GC

# ---------- Hooks: basic behaviors ----------

@pytest.mark.cuda
def test_hooks_install_and_restore(tracker: MemLord):
    cuda_or_skip(1)
    hooks = tracker.hook_into_torch()
    assert callable(torch.Tensor.to)
    assert callable(nn.Module.to)
    if hasattr(torch.cuda, "synchronize"):
        assert callable(torch.cuda.synchronize)
    hooks.disable()
    assert callable(torch.Tensor.to)
    assert callable(nn.Module.to)

@pytest.mark.cuda
def test_tensor_to_alloc_and_free_cuda_to_cpu(tracker: MemLord):
    cuda_or_skip(1)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        t = alloc_tensor_on(d0, (512, 512))
        size = t.numel() * t.element_size()

        out = t.to("cpu")
        assert out.device.type == "cpu"

        cpu_total, _ = tracker.allocated(torch.device("cpu"))
        assert cpu_total >= size

        freed_cuda0, _ = tracker.freed(torch.device("cuda:0"))
        assert freed_cuda0 >= size
    finally:
        hooks.disable()

@pytest.mark.cuda
def test_tensor_to_cuda_to_cuda_deferred_free_nb(tracker: MemLord):
    cuda_or_skip(2)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        t = alloc_tensor_on(d0, (256, 256))
        nbytes = t.numel() * t.element_size()

        out = t.to(d1, non_blocking=True)
        assert out.device == d1

        alloc_d1, _ = tracker.allocated(d1)
        assert alloc_d1 >= nbytes

        freed_d0_before, _ = tracker.freed(d0)
        assert freed_d0_before < nbytes  # now truly deferred

        torch.cuda.synchronize()
        freed_d0_after, _ = tracker.freed(d0)
        assert freed_d0_after >= nbytes
    finally:
        hooks.disable()

@pytest.mark.cuda
def test_module_to_alloc_and_free_cuda_to_cuda_nb(tracker: MemLord):
    cuda_or_skip(2)
    hooks = tracker.hook_into_torch()
    try:
        m = module_linear(512, 512, device="cuda:0", dtype=torch.float32)

        mod_bytes = 0
        for p in m.parameters():
            mod_bytes += p.numel() * p.element_size()
        for b in m.buffers():
            mod_bytes += b.numel() * b.element_size()

        m.to("cuda:1", non_blocking=True)

        alloc_d1, _ = tracker.allocated(torch.device("cuda:1"))
        assert alloc_d1 >= mod_bytes

        freed_d0_before, _ = tracker.freed(torch.device("cuda:0"))
        assert freed_d0_before < mod_bytes

        torch.cuda.synchronize()
        freed_d0_after, _ = tracker.freed(torch.device("cuda:0"))
        assert freed_d0_after >= mod_bytes
    finally:
        hooks.disable()

@pytest.mark.cuda
def test_stream_and_event_sync_polling(tracker: MemLord):
    cuda_or_skip(2)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        t = alloc_tensor_on(d0, (128, 128))
        nbytes = t.numel() * t.element_size()

        out = t.to(d1, non_blocking=True)

        s = torch.cuda.Stream(device=d1.index)
        e = torch.cuda.Event(blocking=False, enable_timing=False)
        with torch.cuda.stream(s):
            _ = out + 1
            e.record(s)

        freed_before, _ = tracker.freed(d0)
        assert freed_before < nbytes

        e.synchronize()
        freed_after, _ = tracker.freed(d0)
        assert freed_after >= nbytes
    finally:
        hooks.disable()

# ---------- Aggregation across multiple CUDA devices ----------

@pytest.mark.cuda
def test_allocated_freed_aggregate_cuda_type(tracker: MemLord):
    cuda_or_skip(1)
    devs = all_cuda_devices()
    tensors = []
    sizes: Dict[torch.device, int] = {}
    for d in devs:
        t = alloc_tensor_on(d, (64, 64))
        tensors.append(t)
        sizes[d] = t.numel() * t.element_size()
    tracker.allocate(tuple(tensors))

    total_by_type, _ = tracker.allocated(torch.device("cuda"))
    assert total_by_type >= sum(sizes.values())

    tracker.free(tuple(tensors))
    freed_by_type, _ = tracker.freed(torch.device("cuda"))
    assert freed_by_type >= sum(sizes.values())

# ---------- Percent threshold resolution (strategy) ----------

def test_percent_threshold_band_selection_and_trigger(monkeypatch):
    # Build a MemLord with a two-band strategy:
    #  (0,50):   metric=max,   threshold=1024 bytes
    #  (50,101): metric=freed, threshold=percent=0.1 of capacity
    strategy = {
        (0, 50):   {"metric": "max",   "threshold": {"bytes": 1024}},
        (50, 101): {"metric": "freed", "threshold": {"percent": 0.1}},
    }
    tr = MemLord(auto_gc_strategy=strategy)
    dev = torch.device("cpu")

    # Monkeypatch backend GC to be deterministic
    import memlord.core as core_mod
    calls = []
    def fake_gc(d):
        calls.append(d)
        return True
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: fake_gc(d))

    # Band A: used% = 10 -> (0,50), metric=max, threshold=1024 bytes
    def stats_low(_dev):
        return (10 * 1024, 1 * 1024, 10.0)  # total=10KiB, used=1KiB, used%=10
    monkeypatch.setattr(tr, "_device_memory_stats", stats_low)

    tr._allocated_by_dev[dev] = 2048  # 2KiB
    tr._freed_by_dev[dev] = 0
    tr._maybe_auto_gc(dev)
    assert calls == [dev]  # triggered by band (0,50), max metric

    # Band B: used% = 60 -> (50,101), metric=freed, threshold=0.1% of total
    # For total=10KiB, 0.1% = 10 bytes; pick a small but >10 value to trigger
    def stats_high(_dev):
        return (10 * 1024, 6 * 1024, 60.0)
    monkeypatch.setattr(tr, "_device_memory_stats", stats_high)

    calls.clear()
    tr._allocated_by_dev[dev] = 0
    tr._freed_by_dev[dev] = 64
    tr._maybe_auto_gc(dev)
    assert calls == [dev]
    # Freed counter should reset
    assert tr._freed_by_dev[dev] == 0

# ---------- Hook logs present when DEBUG=1 ----------

@pytest.mark.cuda
def test_logs_include_hook_labels_and_free_and_allocate(capsys):
    cuda_or_skip(1)
    os.environ["DEBUG"] = "1"
    tr = MemLord(auto_gc_strategy={})
    hooks = tr.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        t = alloc_tensor_on(d0, (64, 64))
        _ = t.to("cpu")  # CPU alloc (hook) + cuda:0 free (hook)
        out = capsys.readouterr().out
        assert "[allocate]" in out and "(hook)" in out
        assert "[free]" in out and "(hook)" in out
    finally:
        hooks.disable()
        if "DEBUG" in os.environ:
            del os.environ["DEBUG"]

# ==============================
# Parameterized dtype test cases
# ==============================

@pytest.mark.parametrize("dtype", DTYPES_TENSOR)
def test_cpu_dtype_allocation_and_free(tracker: MemLord, dtype):
    t = alloc_tensor_on("cpu", (257, 129), dtype=dtype)
    expected = t.numel() * t.element_size()

    tracker.allocate(t)
    got, _ = tracker.allocated(torch.device("cpu"))
    assert got >= expected

    tracker.free(t)
    freed, _ = tracker.freed(torch.device("cpu"))
    assert freed >= expected

@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES_TENSOR)
def test_cuda_to_cpu_dtype_move_alloc_free(tracker: MemLord, dtype):
    cuda_or_skip(1)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        t = alloc_tensor_on(d0, (123, 321), dtype=dtype)
        expected = t.numel() * t.element_size()

        out = t.to("cpu")
        assert out.device.type == "cpu" and out.dtype == dtype

        alloc_cpu, _ = tracker.allocated(torch.device("cpu"))
        assert alloc_cpu >= expected

        freed_cuda0, _ = tracker.freed(torch.device("cuda:0"))
        assert freed_cuda0 >= expected
    finally:
        hooks.disable()

@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES_TENSOR)
def test_cuda_to_cuda_dtype_move_deferred_nb(tracker: MemLord, dtype):
    cuda_or_skip(2)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        t = alloc_tensor_on(d0, (111, 222), dtype=dtype)
        expected = t.numel() * t.element_size()

        out = t.to(d1, non_blocking=True)
        assert out.device == d1 and out.dtype == dtype

        alloc_d1, _ = tracker.allocated(d1)
        assert alloc_d1 >= expected

        freed_d0_before, _ = tracker.freed(d0)
        assert freed_d0_before < expected  # deferred until sync

        torch.cuda.synchronize()
        freed_d0_after, _ = tracker.freed(d0)
        assert freed_d0_after >= expected
    finally:
        hooks.disable()

@pytest.mark.cuda
@pytest.mark.parametrize("dtype", DTYPES_MODULE)
def test_module_param_dtype_tracking_cuda(tracker: MemLord, dtype):
    cuda_or_skip(1)
    hooks = tracker.hook_into_torch()
    try:
        d0 = torch.device("cuda:0")
        m = module_linear(129, 257, device=d0, dtype=dtype)

        total = 0
        for p in m.parameters():
            total += p.numel() * p.element_size()
        for b in m.buffers():
            total += b.numel() * b.element_size()

        m.to("cpu")
        alloc_cpu, _ = tracker.allocated(torch.device("cpu"))
        freed_cuda0, _ = tracker.freed(d0)
        assert alloc_cpu >= total
        assert freed_cuda0 >= total
    finally:
        hooks.disable()
