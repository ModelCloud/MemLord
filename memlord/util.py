# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import os
import threading
import time
import traceback
from typing import Dict, Optional, Tuple

import torch
# Hard dependency (PyPI/GitHub: modelcloud/device-smi)
from device_smi import Device

# ---------- ANSI COLORS ----------
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

# ---------- DEBUG (dynamic) ----------
def is_debug() -> bool:
    return os.environ.get("DEBUG", "0") == "1"

def log(msg: str) -> None:
    if is_debug():
        print(msg)

# ---------- Formatters ----------
def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0

# ---------- Stack helpers ----------
def best_user_frame() -> Optional[str]:
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
                import os as _os
                if this_file and _os.path.exists(fn) and _os.path.exists(this_file):
                    if _os.path.samefile(fn, this_file):
                        continue
            except Exception:
                pass
            lowered = fn.lower()
            if "/torch/" in lowered or "\\torch\\" in lowered:
                continue
            return f"{fn}:{fr.lineno}"
    except Exception:
        return None
    return None


# =================================================================
#                Per-Device Memory Pollers (device-smi)
# =================================================================

# Public polling cadence (seconds). Can be overridden with MEMLORD_MEM_POLL_MS.
_POLL_INTERVAL_SEC_DEFAULT = 0.200  # 200 ms
try:
    _POLL_INTERVAL_SEC = float(int(os.environ.get("MEMLORD_MEM_POLL_MS", "200"))) / 1000.0
except Exception:
    _POLL_INTERVAL_SEC = _POLL_INTERVAL_SEC_DEFAULT

# Thread-safe cache: { torch.device -> (total_bytes, used_bytes, used_pct) }
_mem_cache: Dict[torch.device, Tuple[Optional[int], Optional[int], float]] = {}
_cache_lock = threading.Lock()

# Device handles and workers
_device_handles: Dict[torch.device, Device] = {}
_workers_started = False
_workers_lock = threading.Lock()


def _mk_cuda_key(idx: int) -> str:
    # device-smi naming convention for CUDA adapters
    return f"cuda:{idx}"


def _register_cpu_if_needed() -> None:
    global _device_handles
    td = torch.device("cpu")
    if td in _device_handles:
        return
    # Initialize CPU handle & cache with total capacity
    cpu = Device("cpu")
    total = int(cpu.memory_total)
    with _cache_lock:
        _device_handles[td] = cpu
        _mem_cache[td] = (total, 0, 0.0)


def _register_cuda_if_needed() -> None:
    global _device_handles
    if not torch.cuda.is_available():
        return
    try:
        n = torch.cuda.device_count()
    except Exception:
        n = 0
    for idx in range(n):
        td = torch.device("cuda", idx)
        if td in _device_handles:
            continue
        key = _mk_cuda_key(idx)
        try:
            h = Device(key)
            total = int(h.memory_total)
        except Exception:
            # Skip this CUDA adapter if not visible to device-smi
            continue
        with _cache_lock:
            _device_handles[td] = h
            _mem_cache[td] = (total, 0, 0.0)


def _poller_loop(td: torch.device, handle: Device, interval_sec: float) -> None:
    """
    Dedicated thread per device. Refreshes live metrics at the given cadence.
    """
    name = f"memlord-poller-{td.type}{'' if td.index is None else ':' + str(td.index)}"
    try:
        threading.current_thread().name = name  # cosmetic
    except Exception:
        pass

    # Ensure cache has total capacity on first tick
    with _cache_lock:
        current = _mem_cache.get(td)
        if not current or current[0] is None:
            try:
                total = int(handle.memory_total)
            except Exception:
                total = None
            _mem_cache[td] = (total, 0, 0.0)

    while True:
        try:
            # Refresh live stats
            handle.metrics()
            used = int(handle.memory_used())

            with _cache_lock:
                total = _mem_cache.get(td, (None, None, 0.0))[0]
                if total is None:
                    try:
                        total = int(handle.memory_total)
                    except Exception:
                        total = None
                pct = (used / total * 100.0) if (total and total > 0) else 0.0
                _mem_cache[td] = (total, used, pct)
        except Exception:
            # Keep the last good values; try again next tick
            pass

        time.sleep(interval_sec)


def _start_workers_if_needed() -> None:
    global _workers_started
    if _workers_started:
        return
    with _workers_lock:
        if _workers_started:
            return

        # Register CPU and CUDA devices
        _register_cpu_if_needed()
        _register_cuda_if_needed()

        # Spawn one daemon thread per device
        with _cache_lock:
            items = list(_device_handles.items())
        for td, handle in items:
            t = threading.Thread(
                target=_poller_loop,
                args=(td, handle, _POLL_INTERVAL_SEC),
                daemon=True,
            )
            t.start()

        _workers_started = True
        if is_debug():
            log(f"{YELLOW}[memlord]{RESET} started {len(items)} per-device poller thread(s) @ {int(_POLL_INTERVAL_SEC*1000)} ms")


def get_device_mem_stats(dev: torch.device) -> Tuple[Optional[int], Optional[int], float]:
    """
    Return (total_bytes, used_bytes, used_pct) for the given torch.device from the
    background per-device poller cache. Starts workers lazily on first use and
    auto-registers devices discovered later (e.g., after CUDA init).
    """
    # Ensure pollers are running
    _start_workers_if_needed()

    # If a new device appears later (rare), register its worker lazily
    if dev not in _device_handles:
        try:
            if dev.type == "cpu":
                _register_cpu_if_needed()
            elif dev.type == "cuda":
                # Register all visible CUDA devices; cheap if repeated
                _register_cuda_if_needed()
                # If still not present, try to bind this index specifically
                if dev not in _device_handles and dev.index is not None:
                    key = _mk_cuda_key(int(dev.index))
                    h = Device(key)  # may raise; let it bubble to the except
                    total = int(h.memory_total)
                    with _cache_lock:
                        _device_handles[dev] = h
                        _mem_cache[dev] = (total, 0, 0.0)
                    # Spawn its worker
                    t = threading.Thread(
                        target=_poller_loop, args=(dev, h, _POLL_INTERVAL_SEC), daemon=True
                    )
                    t.start()
        except Exception:
            # Leave missing; return empty stats for now
            pass

    with _cache_lock:
        return _mem_cache.get(dev, (None, None, 0.0))
