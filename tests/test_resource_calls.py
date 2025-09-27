# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import time
import pytest
import torch

from memlord.util import get_device_mem_stats


def _avg_call_time_seconds(fn, iters: int = 5000) -> float:
    # Warmup
    for _ in range(50):
        fn()
    # Small sleep to let per-device pollers initialize and populate
    time.sleep(0.25)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - t0
    return elapsed / iters


def test_cpu_memstats_cached_speed():
    cpu = torch.device("cpu")

    # Function under test: cached lookup (lock + dict read)
    def _read():
        _ = get_device_mem_stats(cpu)

    avg = _avg_call_time_seconds(_read, iters=5000)
    # Require < 10 microseconds per call
    assert avg < 1e-5, f"CPU cached read too slow: {avg:.6f}s per call"


@pytest.mark.cuda
def test_cuda_memstats_cached_speed():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("Requires at least one CUDA device")

    d0 = torch.device("cuda", 0)

    def _read():
        _ = get_device_mem_stats(d0)

    avg = _avg_call_time_seconds(_read, iters=5000)
    # Require < 10 microseconds per call
    assert avg < 1e-5, f"CUDA cached read too slow: {avg:.6f}s per call"
