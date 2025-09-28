# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from memlord import MemLord

@pytest.mark.cuda
def test_autogc_on_cuda_resets_all_cuda_counters(monkeypatch):
    """Spec: when backend GC for CUDA runs (empty_cache), all CUDA device counters are reset."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires >=2 CUDA GPUs for cross-device reset test")

    tr = MemLord(auto_gc_strategy={(0, 101): {"metric": "allocated", "threshold": {"bytes": 1}}})
    d0 = torch.device("cuda:0")
    d1 = torch.device("cuda:1")

    # Seed counters to non-zero for both devices
    tr._allocated_by_dev[d0] = 10
    tr._freed_by_dev[d0] = 7
    tr._allocated_by_dev[d1] = 11
    tr._freed_by_dev[d1] = 3

    import memlord.core as core_mod
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: True)

    def stats(_dev):
        return (100, 90, 90.0)  # select some band
    monkeypatch.setattr(tr, "_device_memory_stats", stats)

    tr._maybe_auto_gc(d0)

    # EXPECTATION: all CUDA devices reset
    assert tr._allocated_by_dev.get(d0, -1) == 0
    assert tr._freed_by_dev.get(d0, -1) == 0
    assert tr._allocated_by_dev.get(d1, -1) == 0
    assert tr._freed_by_dev.get(d1, -1) == 0
