# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import threading
from typing import Dict, Iterable, Tuple, Generator, Optional

import torch
import torch.nn as nn

# Hard dependency (PyPI/GitHub: modelcloud/device-smi)
import device_smi

from .util import (
    RESET, RED, GREEN, YELLOW, CYAN, MAGENTA,
    is_debug, log as _log, format_bytes, best_user_frame,
)
from .fire import TorchMoveHooks, run_backend_gc as _run_backend_gc, sum_for_device as _sum_for_device


# ---------- TYPE ALIASES ----------
Obj = nn.Module | torch.Tensor
ObjOrTuple = Obj | tuple[Obj, ...]


"""
Auto-GC Strategy (stacked / banded)
-----------------------------------

MemLord uses a configurable `auto_gc_strategy` instead of a single byte threshold.

Strategy shape:
    {
        (lo_pct: number, hi_pct: number): {
            "metric": "allocated" | "freed" | "max",
            "threshold": {
                "bytes":   int,     # absolute threshold in bytes
                "percent": float,   # percent of device capacity (VRAM on GPU, RAM on CPU); e.g. 0.5 means 0.5%
            }
        },
        ...
    }

Semantics:
  • At each allocation/free (and hook variants), MemLord:
      1) Computes device usage (used/total) as a percent (used_pct).
      2) Selects the FIRST band (lo <= used_pct < hi) in insertion order.
      3) Resolves the effective threshold in bytes by taking the MAX of:
           - "bytes" (if provided)
           - "percent" * device_total (if provided and total is known)
      4) Computes the metric value:
           - "allocated": tracked allocated bytes for device
           - "freed":     bytes freed since last successful GC (resets post-GC)
           - "max":       max(allocated, freed)
      5) If metric_value >= threshold → run backend GC.
         On success: reset freed counter for device; increment GC counters; log details.

Device capacity sources:
  • CUDA: total from torch.cuda.get_device_properties(i).total_memory;
          used from max(torch.cuda.memory_reserved(i), torch.cuda.memory_allocated(i)).
  • CPU:  via device_smi.System().memory() -> (total, used)
  • Others: if unknown, "percent" thresholds are ignored; only "bytes" thresholds apply.
"""


class MemLord:
    """
    Tracks memory per device instance (torch.device) with an auto-GC strategy.

    API:
      - allocate(obj|tuple): track allocations (nn.Module or torch.Tensor).
      - free(obj|tuple):     track frees; auto-GC driven by `auto_gc_strategy`.
      - allocated(device?):  -> (raw_bytes, human_str).
      - freed(device?):      -> (raw_bytes, human_str).
      - top_alloc_site(device?): -> (device, 'file:line', bytes) | None
      - top_free_site(device?):  -> (device, 'file:line', bytes) | None
      - set_auto_gc_strategy(dict|None): replace strategy. If None, uses a default.
      - hook_into_torch():   install hooks for Tensor/Module move + CUDA sync APIs.

    Colored logs (only if DEBUG=1):
      allocate -> red, free -> green, summaries -> cyan, auto-GC/strategy -> yellow, reset -> magenta
      Hook-originated updates are suffixed with "(hook)".
    """

    def __init__(self, auto_gc_strategy: Optional[dict] = None) -> None:
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

        # Auto-GC strategy
        self._auto_gc_strategy: dict = {}
        self.set_auto_gc_strategy(auto_gc_strategy)

    # ---------- Public API ----------
    def allocate(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)
        caller = best_user_frame()
        affected: set[torch.device] = set()

        with self._lock:
            for dev, b in sizes.items():
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                affected.add(dev)
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

        # Strategy check post-allocation
        for dev in affected:
            self._maybe_auto_gc(dev)

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

        # Strategy check post-free
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

    def set_auto_gc_strategy(self, strategy: Optional[dict]) -> None:
        """
        Replace the current auto-GC strategy. If None, install a reasonable default.

        Strategy shape:
            {
                (lo_pct:int|float, hi_pct:int|float): {
                    "metric": "allocated" | "freed" | "max",
                    "threshold": {
                        "bytes":   int,        # absolute
                        "percent": float,      # percent of device capacity (VRAM on GPU, RAM on CPU)
                    }
                },
                ...
            }
        """
        if strategy is None:
            # Default: thresholds scale with capacity; get stricter as usage climbs.
            strategy = {
                (0, 50):   {"metric": "max", "threshold": {"percent": 0.50}},  # 0.50% of capacity
                (50, 75):  {"metric": "max", "threshold": {"percent": 0.33}},  # 0.33% of capacity
                (75, 101): {"metric": "max", "threshold": {"percent": 0.25}},  # 0.25% of capacity
            }

        # Light validation
        for k, v in strategy.items():
            if not (isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, (int, float)) for x in k)):
                raise ValueError(f"Strategy key must be (lo_pct, hi_pct), got {k}")
            metric = v.get("metric")
            if metric not in ("allocated", "freed", "max"):
                raise ValueError(f"Invalid metric {metric}; choose 'allocated'|'freed'|'max'")
            thr = v.get("threshold", {})
            if not isinstance(thr, dict) or not thr:
                raise ValueError("Each rule must provide a non-empty 'threshold' dict")
            if not any(k in thr for k in ("bytes", "percent")):
                raise ValueError("Threshold must include 'bytes' and/or 'percent'")

        with self._lock:
            self._auto_gc_strategy = dict(strategy)

        _log(f"{YELLOW}[set_auto_gc_strategy]{RESET} installed {len(strategy)} band(s)")

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

    # ---------- Device memory stats ----------
    def _device_memory_stats(self, dev: torch.device) -> Tuple[Optional[int], Optional[int], float]:
        """
        Returns (total_bytes, used_bytes, used_pct) for the device.

        - CUDA: total from get_device_properties.total_memory.
                used = max(memory_reserved, memory_allocated).
        - CPU:  via device_smi.System().memory() (total, used).
        - Others: (None, None, 0.0)
        """
        if dev.type == "cuda" and torch.cuda.is_available():
            idx = 0 if dev.index is None else int(dev.index)
            try:
                props = torch.cuda.get_device_properties(idx)
                total = int(getattr(props, "total_memory", 0))
            except Exception:
                total = 0

            try:
                alloc = int(torch.cuda.memory_allocated(idx))
                reserv = int(torch.cuda.memory_reserved(idx))
                used = max(alloc, reserv)
            except Exception:
                used = 0

            used_pct = (used / total * 100.0) if (total > 0) else 0.0
            return (total if total > 0 else None,
                    used if used > 0 else 0,
                    used_pct)

        if dev.type == "cpu":
            # device-smi hard dependency
            sys = device_smi.System()
            mem = sys.memory()            # expected attributes: total, used (bytes)
            total = int(mem.total)
            used = int(mem.used)
            used_pct = (used / total * 100.0) if total > 0 else 0.0
            return total, used, used_pct

        # Non-CUDA/CPU or unknown types
        return (None, None, 0.0)

    # ---------- Strategy selection & evaluation ----------
    def _select_rule_for_used_pct(self, used_pct: float) -> Optional[dict]:
        """
        Find the first rule whose band (lo, hi) satisfies lo <= used_pct < hi.
        Strategy iteration order follows dict insertion order (Python 3.7+),
        so you can control precedence by ordering the bands.
        """
        strat = self._auto_gc_strategy
        for (lo, hi), rule in strat.items():
            if used_pct >= float(lo) and used_pct < float(hi):
                return rule
        return None

    def _resolve_threshold_bytes(self, rule: dict, dev: torch.device) -> int:
        """
        Resolve the rule's threshold into bytes. If multiple components are present,
        returns the maximum. Unknown capacity => 'percent' component is ignored.
        """
        thr = rule.get("threshold", {})
        out = 0

        if "bytes" in thr:
            try:
                out = max(out, int(thr["bytes"]))
            except Exception:
                pass

        if "percent" in thr:
            total, _, _ = self._device_memory_stats(dev)
            if total:
                try:
                    pct = float(thr["percent"])
                    out = max(out, int(total * (pct / 100.0)))
                except Exception:
                    pass

        return out

    def _metric_value(self, metric: str, dev: torch.device) -> int:
        if metric == "allocated":
            with self._lock:
                return self._allocated_by_dev.get(dev, 0)
        elif metric == "freed":
            with self._lock:
                return self._freed_by_dev.get(dev, 0)
        elif metric == "max":
            with self._lock:
                return max(self._allocated_by_dev.get(dev, 0), self._freed_by_dev.get(dev, 0))
        else:
            return 0

    # ---------- Auto-GC (per-device, strategy-driven) ----------
    def _maybe_auto_gc(self, dev: torch.device) -> None:
        """
        Apply the configured auto-GC strategy for this device.
        """
        # 1) Compute used % (and total if available)
        total_bytes, used_bytes, used_pct = self._device_memory_stats(dev)

        # 2) Find matching rule
        rule = self._select_rule_for_used_pct(used_pct)
        if not rule:
            return

        metric = rule.get("metric", "max")
        threshold_bytes = self._resolve_threshold_bytes(rule, dev)

        if threshold_bytes <= 0:
            # No meaningful threshold for this device/band
            return

        # 3) Evaluate metric
        value = self._metric_value(metric, dev)

        # 4) Compare and GC
        if value < threshold_bytes:
            return

        if _run_backend_gc(dev):
            with self._lock:
                # Keep allocated accounting; reset freed-after-last-GC counter.
                self._freed_by_dev[dev] = 0
                self._gc_count_by_dev[dev] = self._gc_count_by_dev.get(dev, 0) + 1
                self._gc_total_count += 1
                per_dev_count = self._gc_count_by_dev[dev]
                total_count = self._gc_total_count

            _log(
                f"{YELLOW}[auto_gc]{RESET} {dev}: ran GC "
                f"(count={per_dev_count}), total across devices={total_count}; "
                f"band_used≈{used_pct:.1f}% metric={metric} "
                f"{format_bytes(value)} >= {format_bytes(threshold_bytes)}"
            )

    # ---- Internal helpers used by hooks ----
    def _apply_sizes_allocate(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = best_user_frame()
        affected: set[torch.device] = set()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                affected.add(dev)
                _log(f"{RED}[allocate]{RESET} +{format_bytes(b)} on {dev} (hook)")
                self._record_site_locked(kind="alloc", dev=dev, bytes_=b, site=caller)
        for dev in affected:
            self._maybe_auto_gc(dev)

    def _apply_sizes_free(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = best_user_frame()
        affected: set[torch.device] = set()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                affected.add(dev)
                _log(f"{GREEN}[free]{RESET} released {format_bytes(b)} on {dev} (hook)")
                self._record_site_locked(kind="free", dev=dev, bytes_=b, site=caller)
        for dev in affected:
            self._maybe_auto_gc(dev)

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
