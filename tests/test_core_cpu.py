# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import torch

from memlord import MemLord
from conftest import alloc_tensor_on, module_linear, DTYPES_TENSOR

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
    import os
    os.environ["DEBUG"] = "1"
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

def test_type_aggregation_without_index(tracker: MemLord):
    cpu = torch.device("cpu")
    t1 = alloc_tensor_on(cpu, (100, 100))
    t2 = alloc_tensor_on(cpu, (200, 50))
    tracker.allocate((t1, t2))

    total_by_cpu, _ = tracker.allocated(torch.device("cpu"))  # type aggregator
    direct, _ = tracker.allocated(cpu)  # exact device (same on CPU)
    assert total_by_cpu == direct

def test_cpu_dtype_allocation_and_free(tracker: MemLord):
    for dtype in DTYPES_TENSOR:
        t = alloc_tensor_on("cpu", (257, 129), dtype=dtype)
        expected = t.numel() * t.element_size()

        tracker.allocate(t)
        got, _ = tracker.allocated(torch.device("cpu"))
        assert got >= expected

        tracker.free(t)
        freed, _ = tracker.freed(torch.device("cpu"))
        assert freed >= expected
