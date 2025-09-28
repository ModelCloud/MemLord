# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import threading
from typing import Dict, Iterable, Tuple, Generator, Optional

import torch
import torch.nn as nn

from .util import (
    is_debug, debug, info, warn, error, format_bytes, best_user_frame,
    get_device_mem_stats,  # device-smi based pollers
)
from .fire import TorchMoveHooks, run_backend_gc as _run_backend_gc, sum_for_device as _sum_for_device
from .snake import SnakeHooks  # Python-level GC/finalizer hooks


# ---------- TYPE ALIASES ----------
Obj = nn.Module | torch.Tensor
ObjOrTuple = Obj | tuple[Obj, ...]


"""
Auto-GC Strategy (stacked / banded)
-----------------------------------

Strategy shape:
    {
        (lo_pct: number, hi_pct: number): {
            "metric": "allocated" | "freed" | "max",
            "threshold": {
                "bytes":   int,     # absolute threshold in bytes
                "percent": float,   # percent of device capacity (RAM/VRAM)
            },
        },
        ...
    }

Band selection is half-open: lo <= used_pct < hi.
"""


class MemLord:
    """
    Tracks memory per device instance (torch.device) with an auto-GC strategy,
    Torch move hooks, and optional Python GC hooks.

    Debug-only logs (DEBUG=1):
      - allocate/free events, summaries
    Always-on logs:
      - auto-GC event summaries
    """

    def __init__(self, auto_gc_strategy: Optional[dict] = None, *, track_callsite: bool = False) -> None:
        self._allocated_by_dev: Dict[torch.device, int] = {}
        self._freed_by_dev: Dict[torch.device, int] = {}

        # GC accounting
        self._gc_count_by_dev: Dict[torch.device, int] = {}
        self._gc_total_count: int = 0

        # Hotspot tracking (cumulative bytes per site per device)
        self._alloc_site_bytes: Dict[torch.device, Dict[str, int]] = {}
        self._free_site_bytes: Dict[torch.device, Dict[str, int]] = {}
        self._top_alloc_site: Dict[torch.device, Tuple[str, int]] = {}
        self._top_free_site: Dict[torch.device, Tuple[str, int]] = {}

        # Re-entrant lock to avoid deadlocks from nested acquire
        self._lock = threading.RLock()

        # Auto-GC strategy
        self._auto_gc_strategy: dict = {}
        self.set_auto_gc_strategy(auto_gc_strategy)

        # Python-layer GC hooks
        self._snake_hooks: Optional[SnakeHooks] = None

        # Call-site tracking flag (disabled by default)
        self._track_callsite: bool = bool(track_callsite)

        self._in_finalizer_local = threading.local()

    # ---------- Public API ----------
    def set_callsite_tracking(self, enabled: bool) -> None:
        self._track_callsite = bool(enabled)

    def allocate(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)  # storage-dedup path (accurate)
        caller = best_user_frame() if self._track_callsite else None
        affected: set[torch.device] = set()

        # If Python hooks are active, register finalizers for tracked objects
        if self._snake_hooks is not None:
            if isinstance(ob, tuple):
                for x in ob:
                    if isinstance(x, (nn.Module, torch.Tensor)):
                        self._snake_hooks.register(x)
            elif isinstance(ob, (nn.Module, torch.Tensor)):
                self._snake_hooks.register(ob)

        with self._lock:
            for dev, b in sizes.items():
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                total_now = self._allocated_by_dev[dev]
                affected.add(dev)
                debug(f"[allocate] +{format_bytes(b)} on {dev}, now {format_bytes(total_now)}")
                self._record_site_locked(kind="alloc", dev=dev, bytes_=b, site=caller)

            all_devs = self._all_known_devices_locked()
            type_totals = self._totals_by_type_locked(self._allocated_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header="[allocate-summary]",
            per_device_map=self._allocated_by_dev,
            all_devices=all_devs,
            type_totals=type_totals,
            type_counts=type_counts,
        )

        for dev in affected:
            self._maybe_auto_gc(dev)

    def free(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)  # storage-dedup path (accurate)
        caller = best_user_frame() if self._track_callsite else None
        affected: set[torch.device] = set()

        with self._lock:
            for dev, b in sizes.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                total_now = self._allocated_by_dev[dev]
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                affected.add(dev)
                debug(f"[free] -{format_bytes(b)} on {dev}, now {format_bytes(total_now)}")
                self._record_site_locked(kind="free", dev=dev, bytes_=b, site=caller)

            all_devs = self._all_known_devices_locked()
            freed_type_totals = self._totals_by_type_locked(self._freed_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header="[free-summary]",
            per_device_map=self._freed_by_dev,
            all_devices=all_devs,
            type_totals=freed_type_totals,
            type_counts=type_counts,
        )

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
        debug("[reset] counters cleared")

    def allocated(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._allocated_by_dev.values()) if device is None else _sum_for_device(self._allocated_by_dev, device)
        debug(f"[allocated] query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def freed(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._freed_by_dev.values()) if device is None else _sum_for_device(self._freed_by_dev, device)
        debug(f"[freed] query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def top_alloc_site(self, device: torch.device | None = None) -> Optional[Tuple[torch.device, str, int]]:
        with self._lock:
            return _top_site_query(device, self._top_alloc_site)

    def top_free_site(self, device: torch.device | None = None) -> Optional[Tuple[torch.device, str, int]]:
        with self._lock:
            return _top_site_query(device, self._top_free_site)

    def set_auto_gc_strategy(self, strategy: Optional[dict] = None) -> None:
        if strategy is None:
            strategy = {
                (0, 50):   {"metric": "max", "threshold": {"percent": 60.0}},
                (50, 75):  {"metric": "max", "threshold": {"percent": 50.0}},
                (75, 101): {"metric": "max", "threshold": {"percent": 40.0,}},
            }

        for k, v in strategy.items():
            if not (isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, (int, float)) for x in k)):
                raise ValueError(f"Strategy key must be (lo_pct, hi_pct), got {k}")
            metric = v.get("metric")
            if metric not in ("allocated", "freed", "max"):
                raise ValueError(f"Invalid metric {metric}; choose 'allocated'|'freed'|'max'")
            thr = v.get("threshold", {})
            if not isinstance(thr, dict) or not thr:
                raise ValueError("Each rule must provide a non-empty 'threshold' dict")
            if not any(k2 in thr for k2 in ("bytes", "percent")):
                raise ValueError("Threshold must include 'bytes' and/or 'percent'")

        with self._lock:
            self._auto_gc_strategy = dict(strategy)

        debug(f"[set_auto_gc_strategy] installed {len(strategy)} band(s)")

    def hook_into_torch(self) -> "TorchMoveHooks":
        hooks = TorchMoveHooks(self)
        hooks.enable()
        return hooks

    def hook_into_python(self, *, enable_factory_wrappers: bool = True, enable_deep_autowrap: bool = True) -> "SnakeHooks":
        if self._snake_hooks is not None:
            return self._snake_hooks
        sh = SnakeHooks(self, enable_factory_wrappers=enable_factory_wrappers, enable_deep_autowrap=enable_deep_autowrap)
        sh.enable()
        self._snake_hooks = sh
        return sh

    # ---------- Memory accounting helpers ----------
    def _sizes_for_many(self, ob: ObjOrTuple) -> Dict[torch.device, int]:
        """Accurate (storage-dedup) path — used by explicit allocate()/free()."""
        agg: Dict[torch.device, int] = {}
        for item in self._iter_objs(ob):
            for dev, b in self._sizes_by_device_instance(item).items():
                agg[dev] = agg.get(dev, 0) + b
        return agg

    def _sizes_for_many_fast(self, ob: ObjOrTuple) -> Dict[torch.device, int]:
        """
        Safe (no storage access) path — used by TorchMoveHooks to avoid segfaults.
        Dedup is not attempted; we count by tensor nbytes only.
        """
        agg: Dict[torch.device, int] = {}
        it = ob if isinstance(ob, tuple) else (ob,)
        for x in it:
            if isinstance(x, torch.Tensor):
                for dev, b in self._sizes_by_device_instance_fast_tensor(x).items():
                    agg[dev] = agg.get(dev, 0) + b
            elif isinstance(x, nn.Module):
                for t in self._gather_tensors(x):
                    for dev, b in self._sizes_by_device_instance_fast_tensor(t).items():
                        agg[dev] = agg.get(dev, 0) + b
            else:
                raise TypeError(f"Unsupported type: {type(x)}")
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
        """Accurate (storage-dedup) path — may call untyped_storage()."""
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

    # ---- FAST / SAFE size for a single tensor (no storage access) ----
    def _sizes_by_device_instance_fast_tensor(self, t: torch.Tensor) -> Dict[torch.device, int]:
        dev = t.device
        if dev.type == "meta":
            return {}
        total = 0
        # Handle sparse / CSR / CSC by summing components via numel*element_size (safe)
        if t.is_sparse:
            total += int(t.indices().numel() * t.indices().element_size())
            total += int(t.values().numel() * t.values().element_size())
        elif t.layout == torch.sparse_csr:
            total += int(t.crow_indices().numel() * t.crow_indices().element_size())
            total += int(t.col_indices().numel() * t.col_indices().element_size())
            total += int(t.values().numel() * t.values().element_size())
        elif hasattr(torch, "sparse_csc") and t.layout == torch.sparse_csc:  # type: ignore[attr-defined]
            total += int(t.ccol_indices().numel() * t.ccol_indices().element_size())
            total += int(t.row_indices().numel() * t.row_indices().element_size())
            total += int(t.values().numel() * t.values().element_size())
        else:
            total += int(t.numel() * t.element_size())
        if total <= 0:
            return {}
        return {dev: total}

    # ---- Accurate / storage-dedup sum across tensors ----
    def _sum_by_dev_dedup(self, tensors: Iterable[torch.Tensor]) -> Dict[torch.device, int]:
        seen_keys: set[tuple[int, int]] = set()
        by_dev: Dict[torch.device, int] = {}

        def _accumulate_dense(t: torch.Tensor) -> None:
            dev = t.device
            if dev.type == "meta":
                return
            # NOTE: storage path can segfault in some edge cases; only used in explicit allocate()/free()
            try:
                st = t.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev] = by_dev.get(dev, 0) + int(st.nbytes())
            except Exception:
                # Fallback: safe size
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
            debug(f"{header} {dev}: {format_bytes(val)}")
        for dtype in sorted(type_totals.keys()):
            if type_counts.get(dtype, 0) > 1:
                debug(f"{header} {dtype}: {format_bytes(type_totals[dtype])}")

    # ---------- Device memory stats ----------
    def _device_memory_stats(self, dev: torch.device) -> Tuple[Optional[int], Optional[int], float]:
        return get_device_mem_stats(dev)

    # ---------- Family-wide counter reset ----------
    def _reset_counters_family_locked(self, dev_type: str) -> None:
        """Caller must hold self._lock."""
        for d in list(self._allocated_by_dev.keys()):
            if d.type == dev_type:
                self._allocated_by_dev[d] = 0
        for d in list(self._freed_by_dev.keys()):
            if d.type == dev_type:
                self._freed_by_dev[d] = 0

    def _reset_counters_family(self, dev_type: str) -> None:
        """Public wrapper that acquires the lock."""
        with self._lock:
            self._reset_counters_family_locked(dev_type)

    # ---------- Strategy selection & evaluation ----------
    def _select_rule_for_used_pct(self, used_pct: float) -> Optional[tuple[tuple[float, float], dict]]:
        strat = self._auto_gc_strategy
        for (lo, hi), rule in strat.items():
            if used_pct >= float(lo) and used_pct < float(hi):
                return (float(lo), float(hi)), rule
        return None

    def _resolve_threshold_bytes(self, rule: dict, dev: torch.device) -> int:
        """
        Build threshold bytes. If both 'bytes' and 'percent' are provided,
        use the MIN of the two (more conservative).
        """
        thr = rule.get("threshold", {})
        candidates = []

        if "bytes" in thr:
            try:
                b = int(thr["bytes"])
                if b > 0:
                    candidates.append(b)
            except Exception:
                pass

        if "percent" in thr:
            total, _, _ = self._device_memory_stats(dev)
            if total:
                try:
                    pct = float(thr["percent"])
                    if pct > 0:
                        candidates.append(int(total * (pct / 100.0)))
                except Exception:
                    pass

        if not candidates:
            return 0
        return min(candidates)

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
        total_bytes, used_bytes, used_pct = self._device_memory_stats(dev)
        sel = self._select_rule_for_used_pct(used_pct)
        if not sel:
            return
        (band_lo, band_hi), rule = sel

        metric = rule.get("metric", "max")
        threshold_bytes = self._resolve_threshold_bytes(rule, dev)
        if threshold_bytes <= 0:
            return

        value = self._metric_value(metric, dev)
        if value < threshold_bytes:
            return

        # Run backend GC
        if _run_backend_gc(dev):
            with self._lock:
                # Snapshot BEFORE zeroing to include prior counters
                snap_alloc = dict(self._allocated_by_dev)
                snap_freed = dict(self._freed_by_dev)
                snap_gc_counts = dict(self._gc_count_by_dev)
                all_devs = self._all_known_devices_locked()

                # Reset allocated/freed for the entire backend family (cuda/xpu/mps/npu)
                self._reset_counters_family_locked(dev.type)

                # Bump GC counters for triggering device + global
                self._gc_count_by_dev[dev] = self._gc_count_by_dev.get(dev, 0) + 1
                per_dev_count = self._gc_count_by_dev[dev]
                self._gc_total_count += 1
                total_count = self._gc_total_count

            # Build a point-in-time summary across all known devices
            lines = []
            lines.append(
                f"[auto_gc] {dev}: ran GC "
                f"(dev_gc_runs={per_dev_count}, global_gc_runs={total_count}); "
                f"band={band_lo:.0f}–{band_hi:.0f}% used≈{used_pct:.1f}%; "
                f"metric={metric} value={format_bytes(value)} threshold={format_bytes(threshold_bytes)}; "
                f"counters_reset_family={dev.type}"
            )

            # Per-device snapshot lines (allocated/freed from snapshot + live device-smi stats)
            for d in all_devs:
                t_bytes, u_bytes, u_pct = self._device_memory_stats(d)
                a = snap_alloc.get(d, 0)
                f = snap_freed.get(d, 0)
                gc_runs = snap_gc_counts.get(d, 0) + (1 if d == dev else 0)
                tb = "unknown" if t_bytes is None else format_bytes(t_bytes)
                ub = "unknown" if u_bytes is None else format_bytes(u_bytes)
                lines.append(
                    f"  - {d}: allocated={format_bytes(a)}, freed={format_bytes(f)}, "
                    f"dev_gc_runs={gc_runs}; device_smi(total={tb}, used={ub}, used%={u_pct:.1f}%)"
                )

            # Always emit the auto-GC event (ignores DEBUG flag)
            info("\n".join(lines))

    # ---------- Finalizer-safe free path ----------
    def _apply_sizes_free_finalizer(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b

    # ---- Internal helpers used by hooks ----
    def _apply_sizes_allocate(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = best_user_frame() if self._track_callsite else None
        affected: set[torch.device] = set()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                total_now = self._allocated_by_dev[dev]
                affected.add(dev)
                debug(f"[allocate] [hook] +{format_bytes(b)} on {dev}, now {format_bytes(total_now)}")
                self._record_site_locked(kind="alloc", dev=dev, bytes_=b, site=caller)
        for dev in affected:
            self._maybe_auto_gc(dev)

    def _apply_sizes_free(self, sizes_by_dev: Dict[torch.device, int]) -> None:
        if not sizes_by_dev:
            return
        caller = best_user_frame() if self._track_callsite else None
        affected: set[torch.device] = set()
        with self._lock:
            for dev, b in sizes_by_dev.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                total_now = self._allocated_by_dev[dev]
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                affected.add(dev)
                debug(f"[free] [hook] -{format_bytes(b)} on {dev}, now {format_bytes(total_now)}")
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
