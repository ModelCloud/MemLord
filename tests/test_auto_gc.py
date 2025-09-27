# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from memlord import MemLord


def test_auto_gc_strategy_with_percent(monkeypatch):
    calls = []

    def fake_gc(dev):
        calls.append(dev)
        return True

    import memlord.core as core_mod
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: fake_gc(d))

    # Strategy: low band uses "max" with bytes threshold
    #           high band uses "freed" with percent threshold
    strategy = {
        (0, 50):   {"metric": "max",   "threshold": {"bytes": 1024}},
        (50, 101): {"metric": "freed", "threshold": {"percent": 0.1}},  # 0.1% of total
    }
    lord = MemLord(auto_gc_strategy=strategy)

    dev = torch.device("cpu")

    # Monkeypatch memory stats to simulate device total/used
    def stats_low(_dev):
        return (10 * 1024, 1 * 1024, 10.0)  # total=10KiB, used=1KiB, used%=10%

    monkeypatch.setattr(lord, "_device_memory_stats", stats_low)

    # allocated=2KiB, freed=0, max=2KiB >= 1KiB threshold
    lord._allocated_by_dev[dev] = 2048
    lord._freed_by_dev[dev] = 0
    lord._maybe_auto_gc(dev)
    assert calls == [dev]

    # High band: used=60%
    def stats_high(_dev):
        return (10 * 1024, 6 * 1024, 60.0)

    monkeypatch.setattr(lord, "_device_memory_stats", stats_high)

    calls.clear()
    lord._allocated_by_dev[dev] = 512
    lord._freed_by_dev[dev] = 20 * 1024  # big freed
    lord._maybe_auto_gc(dev)
    assert calls == [dev]
