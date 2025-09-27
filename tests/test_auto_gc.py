# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from memlord import MemLord


def test_auto_gc_uses_larger_of_alloc_or_freed(monkeypatch):
    calls = []

    def fake_gc(dev):
        calls.append(dev)
        return True  # make sure the GC branch executes

    # Fixed threshold: 1 KiB
    lord = MemLord(auto_gc_bytes=1024)

    # Patch the module where MemLord resolves _run_backend_gc
    import memlord.core as core_mod
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: fake_gc(d))

    dev = torch.device("cpu")  # counter logic is backend-agnostic

    # --- Case 1: allocations alone exceed threshold (allocated > freed) ---
    lord._allocated_by_dev[dev] = 2048  # 2 KiB allocated
    lord._freed_by_dev[dev] = 0
    lord._maybe_auto_gc(dev)
    assert calls == [dev], "GC should trigger when allocated >= threshold even if freed is 0"

    # --- Case 2: frees exceed threshold (freed > allocated) ---
    calls.clear()
    lord._allocated_by_dev[dev] = 512    # below threshold
    lord._freed_by_dev[dev] = 4096       # 4 KiB freed since last GC
    lord._maybe_auto_gc(dev)
    assert calls == [dev], "GC should trigger when freed >= threshold even if allocated is below threshold"
