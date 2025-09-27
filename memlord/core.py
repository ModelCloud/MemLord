# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import threading
from typing import Any, Dict, Iterable, Tuple, Generator, Optional

import torch
import torch.nn as nn

from .util import (
    RESET, RED, GREEN, YELLOW, CYAN, MAGENTA,
    is_debug, log as _log, format_bytes, best_user_frame,
)
from .fire import TorchMoveHooks, run_backend_gc as _run_backend_gc, sum_for_device as _sum_for_device


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
        caller = best_user_frame()
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
        caller = best_user_frame()
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
        if not is_debug():
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
        caller = best_user_frame()
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
        caller = best_user_frame()
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
