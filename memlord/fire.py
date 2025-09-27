# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import threading

import torch
import torch.nn as nn

from .util import (
    RESET, RED, GREEN, YELLOW,
    log as _log, format_bytes, best_user_frame,
)

# ---------- Public helpers ----------

def run_backend_gc(dev: torch.device) -> bool:
    try:
        if dev.type == "cuda":
            if dev.index is not None:
                torch.cuda.set_device(dev.index)
            torch.cuda.empty_cache()
            return True
        if dev.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()  # type: ignore[attr-defined]
            return True
        if dev.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
            return True
        if dev.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()  # type: ignore[attr-defined]
            return True
        return False
    except Exception:
        return False

def sum_for_device(table: Dict[torch.device, int], query: torch.device) -> int:
    if query.index is None:
        return sum(v for d, v in table.items() if d.type == query.type)
    else:
        return table.get(torch.device(query.type, query.index), 0)

# =========================
# Torch .to() Hook Manager
# =========================

@dataclass
class _PendingFree:
    src_dev: torch.device
    bytes_: int
    event: Optional["torch.cuda.Event"]
    dst_dev: Optional[torch.device] = None


# thread-local to shuttle module->tensor non_blocking intent
_TLS = threading.local()


class TorchMoveHooks:
    """
    Monkey-patches:
      - torch.Tensor.to       (does all accounting & deferral)
      - nn.Module.to          (thin; forwards non_blocking via thread-local)
      - torch.cuda.synchronize
      - torch.cuda.Stream.synchronize
      - torch.cuda.Event.synchronize
    """
    def __init__(self, tracker: "MemLord") -> None:
        self.tracker = tracker
        self._orig_tensor_to = None
        self._orig_module_to = None
        self._orig_cuda_synchronize = None
        self._orig_stream_synchronize = None
        self._orig_event_synchronize = None
        self._enabled = False
        self._pending: List[_PendingFree] = []
        self._pending_lock = threading.Lock()  # <-- thread safety for queue

    # ------- pending frees -------
    def _poll_pending(self) -> None:
        with self._pending_lock:
            if not self._pending:
                return
            still_waiting: List[_PendingFree] = []
            batch: Dict[torch.device, int] = {}
            for item in self._pending:
                ev = item.event
                if ev is None:
                    batch[item.src_dev] = batch.get(item.src_dev, 0) + item.bytes_
                    continue
                if ev.query():
                    batch[item.src_dev] = batch.get(item.src_dev, 0) + item.bytes_
                else:
                    still_waiting.append(item)
            self._pending = still_waiting
        if batch:
            self.tracker._apply_sizes_free(batch)

    # ------- enable/disable -------
    def enable(self) -> None:
        if self._enabled:
            return

        # Save originals
        self._orig_tensor_to = torch.Tensor.to
        self._orig_module_to = nn.Module.to
        self._orig_cuda_synchronize = getattr(torch.cuda, "synchronize", None)

        StreamCls = getattr(torch.cuda, "Stream", None)
        EventCls  = getattr(torch.cuda, "Event", None)
        if StreamCls is not None:
            self._orig_stream_synchronize = getattr(StreamCls, "synchronize", None)
        if EventCls is not None:
            self._orig_event_synchronize = getattr(EventCls, "synchronize", None)

        # Patch Tensor.to (does all accounting & deferral)
        def tensor_to_wrapper(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            src_dev = t.device
            try:
                sizes_src = self.tracker._sizes_by_device_instance(t)
            except Exception:
                sizes_src = {}

            # Prefer explicit kwarg; else inherit from module.to via thread-local
            if "non_blocking" in kwargs:
                non_blocking = bool(kwargs["non_blocking"])
            else:
                non_blocking = bool(getattr(_TLS, "module_non_blocking", False))

            out = self._orig_tensor_to(t, *args, **kwargs)  # type: ignore[misc]
            dst_dev = out.device

            try:
                sizes_dst = self.tracker._sizes_by_device_instance(out)
            except Exception:
                sizes_dst = {}
            if sizes_dst:
                self.tracker._apply_sizes_allocate(sizes_dst)

            added_deferred = False
            if dst_dev != src_dev and sizes_src:
                defer = (
                    non_blocking and
                    torch.cuda.is_available() and
                    src_dev.type == "cuda" and
                    dst_dev.type == "cuda"
                )
                if defer:
                    if dst_dev.index is not None:
                        torch.cuda.set_device(dst_dev.index)
                    stream = torch.cuda.current_stream()
                    ev = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)
                    stream.record_event(ev)
                    with self._pending_lock:
                        for sdev, b in sizes_src.items():
                            if b <= 0:
                                continue
                            if sdev.type == "cuda":
                                self._pending.append(_PendingFree(src_dev=sdev, bytes_=b, event=ev, dst_dev=dst_dev))
                                added_deferred = True
                            else:
                                # non-CUDA sources can free immediately
                                pass
                    # free non-CUDA sources immediately (outside lock)
                    non_cuda_batch = {sdev: b for sdev, b in sizes_src.items() if sdev.type != "cuda" and b > 0}
                    if non_cuda_batch:
                        self.tracker._apply_sizes_free(non_cuda_batch)
                else:
                    self.tracker._apply_sizes_free(sizes_src)

            # Only poll when we did NOT queue deferred frees (keep deferred until sync)
            if not added_deferred:
                self._poll_pending()
            return out

        # Patch Module.to (NO accounting; just forward non_blocking intent via thread-local)
        def module_to_wrapper(m: nn.Module, *args: Any, **kwargs: Any):
            nb = bool(kwargs.get("non_blocking", False))
            old = getattr(_TLS, "module_non_blocking", None)
            _TLS.module_non_blocking = nb
            try:
                return self._orig_module_to(m, *args, **kwargs)  # type: ignore[misc]
            finally:
                if old is None and hasattr(_TLS, "module_non_blocking"):
                    delattr(_TLS, "module_non_blocking")
                else:
                    _TLS.module_non_blocking = old

        # Patch torch.cuda.synchronize (module function)
        def cuda_synchronize_wrapper(*args: Any, **kwargs: Any):
            result = self._orig_cuda_synchronize(*args, **kwargs) if self._orig_cuda_synchronize else None
            self._poll_pending()
            return result

        # Patch Stream.synchronize (instance method)
        def stream_synchronize_wrapper(self_stream, *args: Any, **kwargs: Any):
            res = self._orig_stream_synchronize(self_stream, *args, **kwargs) if self._orig_stream_synchronize else None
            self._poll_pending()
            return res

        # Patch Event.synchronize (instance method)
        def event_synchronize_wrapper(self_event, *args: Any, **kwargs: Any):
            res = self._orig_event_synchronize(self_event, *args, **kwargs) if self._orig_event_synchronize else None
            self._poll_pending()
            return res

        # Install patches
        torch.Tensor.to = tensor_to_wrapper  # type: ignore[assignment]
        nn.Module.to = module_to_wrapper     # type: ignore[assignment]
        if self._orig_cuda_synchronize is not None:
            torch.cuda.synchronize = cuda_synchronize_wrapper  # type: ignore[assignment]
        if StreamCls is not None and self._orig_stream_synchronize is not None:
            StreamCls.synchronize = stream_synchronize_wrapper  # type: ignore[assignment]
        if EventCls is not None and self._orig_event_synchronize is not None:
            EventCls.synchronize = event_synchronize_wrapper  # type: ignore[assignment]

        self._enabled = True
        hooked = ["Tensor.to", "Module.to"]
        if self._orig_cuda_synchronize is not None:
            hooked.append("cuda.synchronize")
        if self._orig_stream_synchronize is not None:
            hooked.append("Stream.synchronize")
        if self._orig_event_synchronize is not None:
            hooked.append("Event.synchronize")
        _log(f"{YELLOW}[hooks]{RESET} Patched: {', '.join(hooked)}")

    def disable(self) -> None:
        if not self._enabled:
            return
        if self._orig_tensor_to is not None:
            torch.Tensor.to = self._orig_tensor_to  # type: ignore[assignment]
        if self._orig_module_to is not None:
            nn.Module.to = self._orig_module_to     # type: ignore[assignment]
        if self._orig_cuda_synchronize is not None:
            torch.cuda.synchronize = self._orig_cuda_synchronize  # type: ignore[assignment]

        StreamCls = getattr(torch.cuda, "Stream", None)
        EventCls  = getattr(torch.cuda, "Event", None)
        if StreamCls is not None and self._orig_stream_synchronize is not None:
            StreamCls.synchronize = self._orig_stream_synchronize  # type: ignore[assignment]
        if EventCls is not None and self._orig_event_synchronize is not None:
            EventCls.synchronize = self._orig_event_synchronize  # type: ignore[assignment]

        self._orig_tensor_to = None
        self._orig_module_to = None
        self._orig_cuda_synchronize = None
        self._orig_stream_synchronize = None
        self._orig_event_synchronize = None
        self._enabled = False
        self._poll_pending()
        _log(f"{YELLOW}[hooks]{RESET} Restored patches")

    # Optional: public poll, e.g., after torch.cuda.synchronize()
    def poll(self) -> None:
        self._poll_pending()
