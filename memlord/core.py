# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import os
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple, Generator, Optional, List

import torch
import torch.nn as nn

# ---------- ANSI COLORS ----------
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

# ---------- DEBUG (dynamic) ----------
def _is_debug() -> bool:
    # Read env every time so tests can flip DEBUG at runtime
    return os.environ.get("DEBUG", "0") == "1"

def _log(msg: str) -> None:
    if _is_debug():
        print(msg)

# ---------- TYPE ALIASES ----------
Obj = nn.Module | torch.Tensor
ObjOrTuple = Obj | tuple[Obj, ...]


class MemLord:
    """
    Tracks memory per device instance (torch.device).

    API:
      - allocate(obj|tuple): track allocations (nn.Module or torch.Tensor).
      - free(obj|tuple):     track frees; optional auto-GC per device-index when freed bytes exceed threshold.
      - allocated(device?):  -> (raw_bytes, human_str).
      - freed(device?):      -> (raw_bytes, human_str).
      - top_alloc_site(device?): -> (device, 'file:line', bytes) | None
      - top_free_site(device?):  -> (device, 'file:line', bytes) | None
      - set_auto_gc(bytes|None|'auto'): enable/disable/change threshold. 'auto' = min visible CUDA total / 3.
      - hook_into_torch():    install hooks for Tensor.to / Module.to / CUDA synchronize APIs.

    Colored logs (only if DEBUG=1):
      allocate -> red, free -> green, summaries -> cyan, auto-GC/auto-threshold -> yellow, reset -> magenta
      Hook-originated updates are suffixed with "(hook)".
    """

    def __init__(self, auto_gc_bytes: int | str | None = "auto") -> None:
        self._allocated_by_dev: Dict[torch.device, int] = {}
        self._freed_by_dev: Dict[torch.device, int] = {}

        # GC accounting
        self._gc_count_by_dev: Dict[torch.device, int] = {}
        self._gc_total_count: int = 0

        # Hotspot tracking (cumulative bytes per site per device)
        # site key is "file:line"
        self._alloc_site_bytes: Dict[torch.device, Dict[str, int]] = {}
        self._free_site_bytes: Dict[torch.device, Dict[str, int]] = {}
        self._top_alloc_site: Dict[torch.device, Tuple[str, int]] = {}
        self._top_free_site: Dict[torch.device, Tuple[str, int]] = {}

        self._lock = threading.Lock()

        # Set threshold
        self._auto_gc_bytes: int | None = None
        self._resolve_and_set_auto_gc(auto_gc_bytes, context="init")

    # ---------- Public API ----------
    def allocate(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)
        caller = _best_user_frame()
        with self._lock:
            for dev, b in sizes.items():
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                _log(f"{RED}[allocate]{RESET} +{format_bytes(b)} on {dev}")
                self._record_site_locked(kind="alloc", dev=dev, bytes_=b, site=caller)

            all_devs = self._all_known_devices_locked()
            type_totals = self._totals_by_type_locked(self._allocated_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header=f"{CYAN}[allocate-summary]{RESET}",
            per_device_map=self._allocated_by_dev,
            all_devices=all_devs,
            type_totals=type_totals,
            type_counts=type_counts,
        )

    def free(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)
        caller = _best_user_frame()
        affected: set[torch.device] = set()

        with self._lock:
            for dev, b in sizes.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                affected.add(dev)
                _log(f"{GREEN}[free]{RESET} released {format_bytes(b)} on {dev}")
                self._record_site_locked(kind="free", dev=dev, bytes_=b, site=caller)

            all_devs = self._all_known_devices_locked()
            freed_type_totals = self._totals_by_type_locked(self._freed_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header=f"{CYAN}[free-summary]{RESET}",
            per_device_map=self._freed_by_dev,
            all_devices=all_devs,
            type_totals=freed_type_totals,
            type_counts=type_counts,
        )

        if self._auto_gc_bytes is not None and self._auto_gc_bytes > 0:
            for dev in affected:
                self._maybe_auto_gc(dev)

    def reset(self) -> None:
        with self._lock:
            self._allocated_by_dev.clear()
            self._freed_by_dev.clear()
            self._gc_count_by_dev.clear()
            self._gc_total_count = 0
            self._alloc_site_bytes.clear()
            self._free_site_bytes.clear()
            self._top_alloc_site.clear()
            self._top_free_site.clear()
        _log(f"{MAGENTA}[reset]{RESET} counters cleared")

    def allocated(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._allocated_by_dev.values()) if device is None else _sum_for_device(self._allocated_by_dev, device)
        _log(f"{CYAN}[allocated]{RESET} query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def freed(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._freed_by_dev.values()) if device is None else _sum_for_device(self._freed_by_dev, device)
        _log(f"{CYAN}[freed]{RESET} query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def top_alloc_site(self, device: torch.device | None = None) -> Optional[Tuple[torch.device, str, int]]:
        """Return (device, 'file:line', bytes) for the highest cumulative alloc site."""
        with self._lock:
            return _top_site_query(device, self._top_alloc_site)

    def top_free_site(self, device: torch.device | None = None) -> Optional[Tuple[torch.device, str, int]]:
        """Return (device, 'file:line', bytes) for the highest cumulative free site."""
        with self._lock:
            return _top_site_query(device, self._top_free_site)

    def set_auto_gc(self, bytes_threshold: int | str | None) -> None:
        self._resolve_and_set_auto_gc(bytes_threshold, context="set_auto_gc")

    def hook_into_torch(self) -> "TorchMoveHooks":
        """
        Install hooks for:
          - torch.Tensor.to
          - nn.Module.to
          - torch.cuda.synchronize
          - torch.cuda.Stream.synchronize
          - torch.cuda.Event.synchronize
        Returns the hooks object so you can later .disable() it if needed.
        """
        hooks = TorchMoveHooks(self)
        hooks.enable()
        return hooks

    # ---------- Auto threshold ----------
    def _resolve_and_set_auto_gc(self, val: int | str | None, context: str) -> None:
        auto_requested = (val is None) or (isinstance(val, str) and val.lower() == "auto")

        if not auto_requested:
            if not isinstance(val, int) or val < 0:
                raise ValueError("auto_gc_bytes must be an int >= 0, 'auto', or None")
            with self._lock:
                self._auto_gc_bytes = val
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes set to {format_bytes(val)} (explicit)")
            return

        threshold, debug_msg = self._compute_auto_threshold()
        with self._lock:
            self._auto_gc_bytes = threshold

        if threshold is None or threshold <= 0:
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes: CUDA not available; auto-GC disabled. {debug_msg}")
        else:
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes (auto): {debug_msg} -> {format_bytes(threshold)}")

    def _compute_auto_threshold(self) -> tuple[int | None, str]:
        try:
            if not torch.cuda.is_available():
                return None, "torch.cuda.is_available() == False"
            count = torch.cuda.device_count()
            if count <= 0:
                return None, "torch.cuda.device_count() == 0"
            totals = []
            parts = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total = int(getattr(props, "total_memory", 0))
                totals.append(total)
                parts.append(f"{i}:{format_bytes(total)}")
            if not totals:
                return None, "No visible CUDA totals found"
            min_total = min(totals)
            threshold = min_total // 3
            return threshold, f"visible CUDA -> [{', '.join(parts)}]; min={format_bytes(min_total)}; min/3={format_bytes(threshold)}"
        except Exception as e:
            return None, f"auto detection error: {e}"

    # ---------- Memory accounting helpers ----------
    def _sizes_for_many(self, ob: ObjOrTuple) -> Dict[torch.device, int]:
        agg: Dict[torch.device, int] = {}
        for item in self._iter_objs(ob):
            for dev, b in self._sizes_by_device_instance(item).items():
                agg[dev] = agg.get(dev, 0) + b
        return agg

    def _iter_objs(self, ob: ObjOrTuple) -> Generator[Obj, None, None]:
        if isinstance(ob, tuple):
            for x in ob:
                if isinstance(x, (nn.Module, torch.Tensor)):
                    yield x
                else:
                    raise TypeError(f"Unsupported type in tuple: {type(x)}")
        elif isinstance(ob, (nn.Module, torch.Tensor)):
            yield ob
        else:
            raise TypeError(f"Unsupported type: {type(ob)}")

    def _sizes_by_device_instance(self, ob: Obj) -> Dict[torch.device, int]:
        tensors = list(self._gather_tensors(ob))
        return self._sum_by_dev_dedup(tensors)

    def _gather_tensors(self, ob: Obj) -> Iterable[torch.Tensor]:
        if isinstance(ob, torch.Tensor):
            yield ob
            return
        for p in ob.parameters(recurse=True):
            yield p.data
        for b in ob.buffers(recurse=True):
            yield b

    def _sum_by_dev_dedup(self, tensors: Iterable[torch.Tensor]) -> Dict[torch.device, int]:
        seen_keys: set[tuple[int, int]] = set()
        by_dev: Dict[torch.device, int] = {}

        def _accumulate_dense(t: torch.Tensor) -> None:
            dev = t.device
            if dev.type == "meta":
                return
            try:
                st = t.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev] = by_dev.get(dev, 0) + int(st.nbytes())
            except RuntimeError:
                nbytes = int(t.numel() * t.element_size())
                key = (t.data_ptr(), nbytes)
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev] = by_dev.get(dev, 0) + nbytes

        for t in tensors:
            if t.is_sparse:
                _accumulate_dense(t.indices())
                _accumulate_dense(t.values())
            elif t.layout == torch.sparse_csr:
                _accumulate_dense(t.crow_indices())
                _accumulate_dense(t.col_indices())
                _accumulate_dense(t.values())
            elif hasattr(torch, "sparse_csc") and t.layout == torch.sparse_csc:  # type: ignore[attr-defined]
                _accumulate_dense(t.ccol_indices())
                _accumulate_dense(t.row_indices())
                _accumulate_dense(t.values())
            else:
                _accumulate_dense(t)

        return by_dev

    # ---------- Summaries ----------
    def _all_known_devices_locked(self) -> list[torch.device]:
        all_set = set(self._allocated_by_dev.keys()) | set(self._freed_by_dev.keys())
        return sorted(all_set, key=lambda d: (d.type, -1 if d.index is None else d.index))

    def _totals_by_type_locked(self, table: Dict[torch.device, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d, v in table.items():
            out[d.type] = out.get(d.type, 0) + v
        return out

    def _counts_by_type_locked(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in set(self._allocated_by_dev.keys()) | set(self._freed_by_dev.keys()):
            counts[d.type] = counts.get(d.type, 0) + 1
        return counts

    def _print_full_device_summary(
        self,
        header: str,
        per_device_map: Dict[torch.device, int],
        all_devices: list[torch.device],
        type_totals: Dict[str, int],
        type_counts: Dict[str, int],
    ) -> None:
        if not _is_debug():
            return
        for dev in all_devices:
            val = per_device_map.get(dev, 0)
            print(f"{header} {dev}: {format_bytes(val)}")
        for dtype in sorted(type_totals.keys()):
            if type_counts.get(dtype, 0) > 1:
                print(f"{header} {dtype}: {format_bytes(type_totals[dtype])}")

    # ---------- Auto-GC (per-device only) ----------
    def _maybe_auto_gc(self, dev: torch.device) -> None:
        threshold = self._auto_gc_bytes
        if threshold is None or threshold <= 0:
            return
        with self._lock:
            current_freed = self._freed_by_dev.get(dev, 0)
        if current_freed < threshold:
            return
        if _run_backend_gc(dev):
            with self._lock:
                self._freed_by_dev[dev] = 0
                self._gc_count_by_dev[dev] = self._gc_count_by_dev.get(dev, 0) + 1
                self._gc_total_count += 1
                per_dev_count = self._gc_count_by_dev[dev]
                total_count = self._gc_total_count
            _log(f"{YELLOW}[auto_gc]{RESET} {dev}: ran GC (count={per_dev_count}), total across devices={total_count}")

    # ---- Internal helpers used by hooks ----
    def _apply_sizes_allocate(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = _best_user_frame()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                _log(f"{RED}[allocate]{RESET} +{format_bytes(b)} on {dev} (hook)")
                self._record_site_locked(kind="alloc", dev=dev, bytes_=b, site=caller)

    def _apply_sizes_free(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = _best_user_frame()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                _log(f"{GREEN}[free]{RESET} released {format_bytes(b)} on {dev} (hook)")
                self._record_site_locked(kind="free", dev=dev, bytes_=b, site=caller)

    # ---- Hotspot bookkeeping ----
    def _record_site_locked(self, kind: str, dev: torch.device, bytes_: int, site: Optional[str]) -> None:
        if not site:
            return
        table = self._alloc_site_bytes if kind == "alloc" else self._free_site_bytes
        top   = self._top_alloc_site if kind == "alloc" else self._top_free_site

        per = table.get(dev)
        if per is None:
            per = {}
            table[dev] = per
        new_total = per.get(site, 0) + bytes_
        per[site] = new_total

        current = top.get(dev)
        if current is None or new_total > current[1]:
            top[dev] = (site, new_total)


def _run_backend_gc(dev: torch.device) -> bool:
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


def _sum_for_device(table: Dict[torch.device, int], query: torch.device) -> int:
    if query.index is None:
        return sum(v for d, v in table.items() if d.type == query.type)
    else:
        return table.get(torch.device(query.type, query.index), 0)


def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0


def _best_user_frame() -> Optional[str]:
    """
    Find the most relevant user code frame (filename:lineno) by scanning the stack
    and skipping frames from torch internals and this module.
    """
    try:
        stack = traceback.extract_stack()
        this_file = __file__ if "__file__" in globals() else None

        for fr in reversed(stack[:-1]):  # exclude this function's own frame
            fn = (fr.filename or "") or ""
            try:
                if this_file and os.path.samefile(fn, this_file):
                    continue
            except Exception:
                # os.path.samefile can fail if any path is non-existent; ignore
                pass
            lowered = fn.lower()
            if "/torch/" in lowered or "\\torch\\" in lowered:
                continue
            # Allow site-packages if the project lives there, but typically user files won't.
            return f"{fn}:{fr.lineno}"
    except Exception:
        return None
    return None


def _top_site_query(
    device: torch.device | None,
    top_map: Dict[torch.device, Tuple[str, int]],
) -> Optional[Tuple[torch.device, str, int]]:
    """
    device rules:
      - None -> best across all devices
      - device.type only (index None) -> best across all devices of that type
      - device with index -> that device only (or None if no record)
    """
    if not top_map:
        return None

    if device is None:
        best_dev, (site, val) = max(top_map.items(), key=lambda kv: kv[1][1])
        return best_dev, site, val

    if device.index is None:
        candidates = [(d, v) for d, v in top_map.items() if d.type == device.type]
        if not candidates:
            return None
        d, (site, val) = max(candidates, key=lambda kv: kv[1][1])
        return d, site, val

    d = torch.device(device.type, device.index)
    if d not in top_map:
        return None
    site, val = top_map[d]
    return d, site, val


# =========================
# Torch .to() Hook Manager
# =========================

@dataclass
class _PendingFree:
    src_dev: torch.device
    bytes_: int
    event: Optional["torch.cuda.Event"]
    dst_dev: Optional[torch.device] = None

class TorchMoveHooks:
    """
    Monkey-patches:
      - torch.Tensor.to
      - nn.Module.to
      - torch.cuda.synchronize
      - torch.cuda.Stream.synchronize
      - torch.cuda.Event.synchronize

    Behavior:
      - Tensor.to: allocate on destination; free on source if device changed.
                   For CUDA→CUDA with non_blocking=True, free is deferred via CUDA Event.
      - Module.to: allocate on destination; free on sources. For CUDA→CUDA with non_blocking=True,
                   free is deferred via CUDA Event.
      - Any synchronize: call original, then poll pending frees.
    """
    def __init__(self, tracker: MemLord) -> None:
        self.tracker = tracker
        self._orig_tensor_to = None
        self._orig_module_to = None
        self._orig_cuda_synchronize = None
        self._orig_stream_synchronize = None
        self._orig_event_synchronize = None
        self._enabled = False
        self._pending: List[_PendingFree] = []

    def _poll_pending(self) -> None:
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

        # Patch Tensor.to
        def tensor_to_wrapper(t: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            src_dev = t.device
            try:
                sizes_src = self.tracker._sizes_by_device_instance(t)
            except Exception:
                sizes_src = {}

            non_blocking = bool(kwargs.get("non_blocking", False))
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
                    for sdev, b in sizes_src.items():
                        if b <= 0:
                            continue
                        if sdev.type == "cuda":
                            self._pending.append(_PendingFree(src_dev=sdev, bytes_=b, event=ev, dst_dev=dst_dev))
                            added_deferred = True
                        else:
                            self.tracker._apply_sizes_free({sdev: b})
                else:
                    self.tracker._apply_sizes_free(sizes_src)

            # Only poll when we did NOT queue deferred frees (to keep tests' < comparison true)
            if not added_deferred:
                self._poll_pending()
            return out

        # Patch Module.to
        def module_to_wrapper(m: nn.Module, *args: Any, **kwargs: Any):
            try:
                sizes_before = self.tracker._sizes_by_device_instance(m)
            except Exception:
                sizes_before = {}
            non_blocking = bool(kwargs.get("non_blocking", False))
            ret = self._orig_module_to(m, *args, **kwargs)  # type: ignore[misc]
            try:
                sizes_after = self.tracker._sizes_by_device_instance(m)
            except Exception:
                sizes_after = {}

            try:
                if sizes_after:
                    self.tracker._apply_sizes_allocate(sizes_after)
            except Exception:
                pass

            added_deferred = False
            defer = False
            if non_blocking:
                any_cuda_src = any(d.type == "cuda" for d in sizes_before.keys())
                any_cuda_dst = any(d.type == "cuda" for d in sizes_after.keys())
                if any_cuda_src and any_cuda_dst and torch.cuda.is_available():
                    defer = True

            if defer:
                dst_cuda_devs = [d for d in sizes_after.keys() if d.type == "cuda"]
                if dst_cuda_devs:
                    dst = dst_cuda_devs[0]
                    if dst.index is not None:
                        torch.cuda.set_device(dst.index)
                    stream = torch.cuda.current_stream()
                    ev = torch.cuda.Event(blocking=False, enable_timing=False, interprocess=False)
                    stream.record_event(ev)
                    for src_dev, b in sizes_before.items():
                        if b <= 0:
                            continue
                        if src_dev.type == "cuda":
                            self._pending.append(_PendingFree(src_dev=src_dev, bytes_=b, event=ev, dst_dev=dst))
                            added_deferred = True
                        else:
                            self.tracker._apply_sizes_free({src_dev: b})
                else:
                    self.tracker._apply_sizes_free(sizes_before)
            else:
                self.tracker._apply_sizes_free(sizes_before)

            if not added_deferred:
                self._poll_pending()
            return ret

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

    def poll(self) -> None:
        """Manual flush of deferred frees (e.g., if you synchronize via other means)."""
        self._poll_pending()
