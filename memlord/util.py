# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import atexit
import os
import threading
import traceback
import weakref
from typing import Dict, Optional, Tuple

import torch
from device_smi import Device  # hard dependency as requested
from logbar import LogBar

# ---------- Logger ----------
logger = LogBar("MemLord")

def is_debug() -> bool:
    return os.environ.get("DEBUG", "0") == "1"

def _prefix(msg: str) -> str:
    return f"MemLord: {msg}"

def debug(msg: str) -> None:
    """Debug log, only if DEBUG=1."""
    if is_debug():
        logger.debug(_prefix(msg))

def info(msg: str) -> None:
    logger.info(_prefix(msg))

def warn(msg: str) -> None:
    logger.warn(_prefix(msg))

def error(msg: str) -> None:
    logger.error(_prefix(msg))

# Back-compat aliases
log = debug
log_always = info

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
            if this_file and os.path.exists(fn) and os.path.exists(this_file):
                try:
                    if os.path.samefile(fn, this_file):
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


# ============================================================================
# Background device memory polling (CPU and CUDA) using device-smi
# ============================================================================

# Poll cadence (seconds). Can override with MEMLORD_POLL_INTERVAL_MS=integer
_POLL_INTERVAL = max(0.01, float(os.environ.get("MEMLORD_POLL_INTERVAL_MS", "200")) / 1000.0)

# Global shutdown coordination
_SHUTTING_DOWN = False
_shutdown_lock = threading.Lock()

# Track active SnakeHooks (weakly) so atexit can disable them before weakref.finalize pass
_active_snake_hooks: "weakref.WeakSet" = weakref.WeakSet()

# Per-device pollers and stats
_pollers_lock = threading.Lock()
_pollers: Dict[str, "DevicePoller"] = {}

_stats_lock = threading.Lock()
# key -> (total_bytes, used_bytes, used_pct)
_stats: Dict[str, Tuple[int, int, float]] = {}


def is_shutting_down() -> bool:
    return _SHUTTING_DOWN


def register_snake_hooks_instance(hook_obj) -> None:
    """
    Keep a weak reference to active SnakeHooks so we can disable/unwrap at exit.
    Does not create cycles.
    """
    try:
        _active_snake_hooks.add(hook_obj)
    except Exception:
        pass


def _devkey_from_torch_device(dev: torch.device) -> str:
    if dev.type == "cpu":
        return "cpu"
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else 0
        return f"cuda:{idx}"
    # Fallback: stringify
    if dev.index is None:
        return dev.type
    return f"{dev.type}:{dev.index}"


def _ensure_poller(key: str) -> None:
    with _pollers_lock:
        if key in _pollers:
            return
        poller = DevicePoller(key)
        _pollers[key] = poller
        poller.start()


def _stop_all_device_pollers() -> None:
    with _pollers_lock:
        ps = list(_pollers.values())
    for p in ps:
        p.stop()
    for p in ps:
        p.join(timeout=1.0)
    with _pollers_lock:
        _pollers.clear()


def get_device_mem_stats(dev: torch.device) -> Tuple[Optional[int], Optional[int], float]:
    """
    Returns (total_bytes, used_bytes, used_pct) for the given device, based on
    the latest snapshot from the background poller. If the poller hasn't run
    yet, best-effort values (possibly zeros) are returned.

    For CUDA devices, pass a device with a concrete index (e.g., cuda:0).
    """
    key = _devkey_from_torch_device(dev)

    if is_shutting_down():
        # Do NOT create new pollers during teardown/finalizers.
        with _stats_lock:
            tup = _stats.get(key)
        return (tup[0], tup[1], tup[2]) if tup else (None, None, 0.0)

    _ensure_poller(key)

    with _stats_lock:
        tup = _stats.get(key)

    if tup is None:
        return (None, None, 0.0)

    total, used, pct = tup
    return (total, used, pct)


class DevicePoller:
    """
    One poller per device key (e.g., "cpu", "cuda:0"). Reads memory_total once,
    then samples memory_used() roughly every _POLL_INTERVAL seconds.

    device-smi notes:
      - CPU:   Device("cpu")        -> .memory_total, .memory_used()
      - CUDA:  Device("cuda:IDX")   -> .memory_total, .memory_used()
    """
    def __init__(self, key: str) -> None:
        self.key = key
        self._stop_evt = threading.Event()
        self._thr = threading.Thread(target=self._run, name=f"memlord-poller-{key}", daemon=True)

        # Resolve a device-smi Device handle
        self._dev = Device(key)
        self._total = int(getattr(self._dev, "memory_total"))

        # Initialize snapshot so callers see something immediately
        with _stats_lock:
            _stats[self.key] = (self._total, 0, 0.0)

    def start(self) -> None:
        self._thr.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thr.join(timeout=timeout)

    def _run(self) -> None:
        # Tight loop sampling memory_used() until asked to stop
        iv = _POLL_INTERVAL
        while not self._stop_evt.is_set():
            try:
                used = int(self._dev.memory_used())  # JIT/instant used value
            except Exception:
                used = 0
            pct = (float(used) / float(self._total) * 100.0) if self._total > 0 else 0.0
            with _stats_lock:
                _stats[self.key] = (self._total, used, pct)
            # Wait with interruptible sleep
            self._stop_evt.wait(iv)


# ============================================================================
# atexit cleanup: stop pollers and disable SnakeHooks before weakref.finalize
# ============================================================================

def _memlord_atexit_cleanup():
    global _SHUTTING_DOWN
    with _shutdown_lock:
        if _SHUTTING_DOWN:
            return
        _SHUTTING_DOWN = True

    # 1) Stop background pollers first (prevents late memory_used calls)
    try:
        _stop_all_device_pollers()
    except Exception:
        pass

    # 2) Disable active SnakeHooks (unwrap torch so no more finalizers are registered)
    try:
        for h in list(_active_snake_hooks):
            try:
                h.disable()
            except Exception:
                pass
    except Exception:
        pass

# Register now; atexit runs handlers in LIFO order.
atexit.register(_memlord_atexit_cleanup)
