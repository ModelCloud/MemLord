# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import os
import traceback
from typing import Optional

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
                # os.path.samefile may fail for non-existent paths; ignore exceptions
                import os
                if this_file and os.path.exists(fn) and os.path.exists(this_file):
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
