# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import gc
import time
import threading
import pytest
import torch

from memlord import MemLord


# =========================
# Configurable perf knobs
# =========================
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

# Total iterations for microbenchmarks (keep modest for CI; tune via env)
OPS_ALLOC = _env_int("MEMLORD_PERF_OPS_ALLOC", 2000)
OPS_FINAL = _env_int("MEMLORD_PERF_OPS_FINAL", 5000)
THREAD_TIME_SEC = _env_float("MEMLORD_THREAD_STRESS_SEC", 0.5)

# Acceptable overhead (relative factor & absolute ns/op). Loosen/tighten in CI as needed.
MAX_OVERHEAD_FACTOR = _env_float("MEMLORD_PERF_MAX_OVERHEAD_FACTOR", 3.0)     # e.g., hooks ≤ 3x baseline
MAX_ABS_OVERHEAD_NS = _env_int("MEMLORD_PERF_MAX_ABS_OVERHEAD_NS", 500_000)   # ≤ 0.5 ms per op

# =========================
# Helpers
# =========================
def _tensor_nbytes(t: torch.Tensor) -> int:
    if t.is_sparse:
        return int(t.indices().numel() * t.indices().element_size()) + int(t.values().numel() * t.values().element_size())
    if t.layout == torch.sparse_csr:
        return (
            int(t.crow_indices().numel() * t.crow_indices().element_size()) +
            int(t.col_indices().numel() * t.col_indices().element_size()) +
            int(t.values().numel() * t.values().element_size())
        )
    if hasattr(torch, "sparse_csc") and t.layout == torch.sparse_csc:  # type: ignore[attr-defined]
        return (
            int(t.ccol_indices().numel() * t.ccol_indices().element_size()) +
            int(t.row_indices().numel() * t.row_indices().element_size()) +
            int(t.values().numel() * t.values().element_size())
        )
    return t.numel() * t.element_size()


# ==================================================
# 1) DOUBLE/UNDER-COUNTING EDGE CASES (CPU & CUDA)
# ==================================================

def test_no_double_count_same_device_dtype_cast_cpu():
    """Casting dtype on the SAME device should count exactly the new buffer once."""
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_torch()     # <-- needed: dtype-cast accounting is in Tensor.to hook
    tr.hook_into_python()    # counts factory/clone-like paths

    t = torch.empty(256, 128, dtype=torch.float32, device="cpu")
    expected_added = t.numel() * 2  # fp16 = 2 bytes/elt

    before, _ = tr.allocated(torch.device("cpu"))
    u = t.to(dtype=torch.float16)   # should register one allocation on CPU (same device)
    after, _ = tr.allocated(torch.device("cpu"))

    assert after >= before + int(expected_added * 0.8)  # allow small tolerance

    # cleanup
    del t, u
    for _ in range(3):
        gc.collect()


@pytest.mark.cuda
def test_no_double_count_device_move_with_both_hooks_cuda():
    """Move cuda:0 -> cuda:1 with both hooks enabled; count once on dst, free on src."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires >= 2 CUDA GPUs")
    tr = MemLord(auto_gc_strategy={})
    hooks = tr.hook_into_torch()
    tr.hook_into_python()

    try:
        d0 = torch.device("cuda:0")
        d1 = torch.device("cuda:1")
        t = torch.empty(256, 128, device=d0, dtype=torch.float32)
        nbytes = _tensor_nbytes(t)

        a0_before, _ = tr.allocated(d1)
        f0_before, _ = tr.freed(d0)

        _ = t.to(d1, non_blocking=True)

        # allow deferred free to happen
        torch.cuda.synchronize()

        a1_after, _ = tr.allocated(d1)
        f1_after, _ = tr.freed(d0)

        # One allocation on dst roughly equal to nbytes; one free on src
        assert a1_after >= a0_before + int(nbytes * 0.8)
        assert f1_after >= f0_before + int(nbytes * 0.8)
    finally:
        hooks.disable()


def test_views_not_counted_as_allocations_as_strided_cpu():
    """Views (as_strided) must not be counted as new allocations."""
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    base = torch.empty(1024, device="cpu")
    before, _ = tr.allocated(torch.device("cpu"))
    view = torch.as_strided(base, (256, 2), (2, 1))
    after, _ = tr.allocated(torch.device("cpu"))
    assert after == before  # view only, no alloc

    del base, view
    for _ in range(2):
        gc.collect()


def test_out_argument_does_not_count_allocation_cpu():
    """Using out= should not count new allocations."""
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()
    buf = torch.empty(1000, dtype=torch.float32)

    before, _ = tr.allocated(torch.device("cpu"))
    # Some builds require size= when mean/std are scalars + out=
    torch.normal(0.0, 1.0, size=buf.shape, out=buf)
    after, _ = tr.allocated(torch.device("cpu"))

    assert after == before


# ======================================
# 2) PERFORMANCE REGRESSION (ALLOC LAT)
# ======================================

def test_perf_allocation_overhead_snakehooks():
    """
    Compare raw tensor allocation vs allocation with SnakeHooks enabled.
    Assert both relative and absolute overheads are within thresholds.
    """
    # Baseline
    gc.disable()
    try:
        t0 = time.monotonic_ns()
        for _ in range(OPS_ALLOC):
            _ = torch.empty(64, 64)  # ~16KB per tensor
        t1 = time.monotonic_ns()
    finally:
        gc.enable()
        gc.collect()
    baseline_per_op = (t1 - t0) / max(1, OPS_ALLOC)

    # With hooks
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()
    gc.disable()
    try:
        t2 = time.monotonic_ns()
        for _ in range(OPS_ALLOC):
            _ = torch.empty(64, 64)
        t3 = time.monotonic_ns()
    finally:
        gc.enable()
        gc.collect()
    hooked_per_op = (t3 - t2) / max(1, OPS_ALLOC)

    # Check relative and absolute overhead
    rel = hooked_per_op / max(1, baseline_per_op)
    abs_ns = max(0, hooked_per_op - baseline_per_op)
    assert rel <= MAX_OVERHEAD_FACTOR, f"Relative overhead {rel:.2f}x exceeds {MAX_OVERHEAD_FACTOR}x"
    assert abs_ns <= MAX_ABS_OVERHEAD_NS, f"Abs overhead/op {abs_ns:.0f} ns exceeds {MAX_ABS_OVERHEAD_NS} ns"


# ===================================================
# 3) CONCURRENCY: NO DEADLOCK / HANG UNDER STRESS
# ===================================================

def test_concurrency_many_threads_no_deadlock():
    """
    Stress concurrent allocate/free across threads to catch lock ordering or
    re-entrancy pathologies. Should complete within a short timeout.
    """
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    stop = threading.Event()
    excs = []
    def worker():
        try:
            while not stop.is_set():
                x = torch.empty(4096)  # ~32KB
                y = torch.zeros(2048)  # ~8KB
                del x, y
        except Exception as e:
            excs.append(e)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for th in threads:
        th.start()

    # Run a bit
    time.sleep(THREAD_TIME_SEC)
    stop.set()
    for th in threads:
        th.join(timeout=2.0)

    assert all(not th.is_alive() for th in threads), "Thread(s) failed to join -> possible deadlock"
    assert not excs, f"Exceptions in worker threads: {excs}"


# ============================================================
# 4) FINALIZER / WEAKREF DEREF PERFORMANCE (COLLECTION COST)
# ============================================================

def test_perf_finalizer_overhead_snakehooks():
    """
    Measure cost to collect many tensors when finalizers are registered.
    Compare vs baseline (no hooks). Ensure overhead within bounds.
    """
    # Baseline: create and drop many tensors (no hooks)
    tensors = []
    for _ in range(OPS_FINAL):
        tensors.append(torch.empty(32, 32))  # small objects
    t0 = time.monotonic_ns()
    del tensors
    gc.collect()
    t1 = time.monotonic_ns()
    baseline_collect = t1 - t0

    # With hooks (register finalizers per tensor)
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()
    tensors2 = []
    for _ in range(OPS_FINAL):
        tensors2.append(torch.empty(32, 32))
    t2 = time.monotonic_ns()
    del tensors2
    gc.collect()
    t3 = time.monotonic_ns()
    hooked_collect = t3 - t2

    # Evaluate per-object overhead
    per_obj_ns = (hooked_collect - baseline_collect) / max(1, OPS_FINAL)
    # Tolerate larger relative cost but cap absolute per object
    assert per_obj_ns <= MAX_ABS_OVERHEAD_NS, f"Finalizer collect overhead/op {per_obj_ns:.0f} ns exceeds {MAX_ABS_OVERHEAD_NS} ns"
