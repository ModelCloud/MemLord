# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


import os
import pytest
import torch
import torch.nn as nn

from memlord import MemLord
from conftest import (
    cuda_or_skip, alloc_tensor_on, module_linear,
    all_cuda_devices, DTYPES_TENSOR, DTYPES_MODULE
)

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
        assert freed_d0_before < nbytes  # truly deferred

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

@pytest.mark.cuda
def test_allocated_freed_aggregate_cuda_type(tracker: MemLord):
    cuda_or_skip(1)
    devs = all_cuda_devices()
    tensors = []
    sizes = {}
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
        assert "[allocate]" in out and "[hook]" in out and ", now " in out
        assert "[free]" in out and "[hook]" in out and ", now " in out
    finally:
        hooks.disable()
        if "DEBUG" in os.environ:
            del os.environ["DEBUG"]

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
