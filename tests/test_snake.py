# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import torch

from memlord import MemLord

def _size_bytes(t: torch.Tensor) -> int:
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

def test_factories_allocation_accounting_cpu():
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    cpu = torch.device("cpu")
    before, _ = tr.allocated(cpu)  # measure BEFORE creating tensors

    # Cover a bunch of factories
    a = torch.empty(123, 321)            # empty
    b = torch.zeros_like(a)              # *_like
    c = torch.rand(128, 17)              # rand
    d = torch.eye(99)                    # eye
    e = torch.linspace(0, 1, steps=257)  # linspace
    f = torch.randperm(513)              # randperm
    g = torch.normal(0.0, 1.0, size=(64, 33))  # normal
    h = torch.clone(c)                   # clone()

    total = sum(_size_bytes(x) for x in [a, b, c, d, e, f, g, h])

    after, _ = tr.allocated(cpu)
    # Heuristic tolerance: at least ~80% of the created bytes should be counted.
    assert after >= before + int(total * 0.8)

    # Cleanup to exercise finalizers (not strictly needed for this assertion)
    del a, b, c, d, e, f, g, h
    for _ in range(3):
        gc.collect()

def test_as_strided_should_not_count_as_allocation():
    tr = MemLord(auto_gc_strategy={})
    tr.hook_into_python()

    base = torch.empty(1024)
    before, _ = tr.allocated(torch.device("cpu"))
    view = torch.as_strided(base, (256, 2), (2, 1))  # view into base storage, no new storage
    after, _ = tr.allocated(torch.device("cpu"))

    # as_strided must not be counted as a new allocation (view only)
    assert after == before

    # keep references so finalizers don't muddy counters (not required here)
    del view, base
