# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch
from memlord import MemLord
import pytest
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



# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from memlord import MemLord

def test_auto_gc_runs_and_resets_freed_per_device(monkeypatch):
    # Strategy: single band over all usage, trigger on any freed >= 1 byte
    strategy = {(0, 101): {"metric": "freed", "threshold": {"bytes": 1}}}
    tr = MemLord(auto_gc_strategy=strategy)
    dev = torch.device("cpu")

    # Deterministic backend GC
    import memlord.core as core_mod
    calls = []
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: calls.append(d) or True)

    # Force band + threshold
    def stats_low(_dev):
        return (10 * 1024, 1 * 1024, 10.0)  # total=10KiB, used=1KiB, 10%
    monkeypatch.setattr(tr, "_device_memory_stats", stats_low)

    tr._freed_by_dev[dev] = 64
    tr._maybe_auto_gc(dev)

    assert calls == [dev]
    # After GC, counters may be reset to 0 OR removed. Treat missing as 0.
    assert tr._freed_by_dev.get(dev, 0) == 0
    assert tr._allocated_by_dev.get(dev, 0) == 0

def test_auto_gc_logs_even_when_debug_off(monkeypatch, capsys):
    strategy = {(0, 101): {"metric": "freed", "threshold": {"bytes": 1}}}
    tr = MemLord(auto_gc_strategy=strategy)
    dev = torch.device("cpu")

    import memlord.core as core_mod
    monkeypatch.setattr(core_mod, "_run_backend_gc", lambda d: True)

    def stats(_dev):
        return (1024, 900, 87.9)
    monkeypatch.setattr(tr, "_device_memory_stats", stats)

    # Trigger
    tr._allocated_by_dev[dev] = 2048
    tr._freed_by_dev[dev] = 2048
    tr._maybe_auto_gc(dev)

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "[auto_gc]" in out
