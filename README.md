# MemLord

**MemLord** is a tiny, fast PyTorch memory accountant that tracks allocated/freed bytes **per `torch.device` _and_ device index** (e.g., `cuda:0`, `cuda:1`, `cpu`). It now features a **banded auto-GC strategy**, **Python finalizers for `del`**, **optional call-site tracking**, and **background device usage polling** via [`device-smi`](https://github.com/ModelCloud/Device-SMI).

---

## Highlights

- ✅ `allocate()` / `free()` for `nn.Module` **or** `torch.Tensor` (or a tuple of them)
- ✅ Dedup by storage (avoids double-counting shared views/views of the same storage)
- ✅ `allocated()` / `freed()` → `(raw_bytes, "human string")` for **all devices**, a **device type** (e.g. `torch.device("cuda")`), or a **specific index** (e.g. `torch.device("cuda", 1)`)
- ✅ **Auto-GC strategy** (banded). Pick metric: **`allocated`**, **`freed`**, or **`max`**; thresholds in **bytes** or **percent of device capacity**
- ✅ **Per-device index** auto-GC (e.g. only `cuda:1` triggers when it crosses its band threshold)
- ✅ **Python GC hooks**: finalizers detect `del`/GC of tensors & modules → auto-credited to `freed`
- ✅ **Torch move hooks**: intercept `Tensor.to` / `Module.to` / CUDA sync to count moves & deferred frees
- ✅ **Background memory polling** (every ~200ms) for **CPU & CUDA** via `device-smi`
- ✅ **Optional call-site tracking** (file:line) and **hotspot queries** (disabled by default)
- ✅ DEBUG-guarded, colored logs (`DEBUG=1`)

> ℹ️ **Performance defaults**:  
> • Call-site stack traces are **off by default** (enable when you need hotspots).  
> • `device-smi` polling runs in **one thread per device** (CPU + each CUDA index) to keep calls off your hot path.

---

## Install

```bash
# MemLord (editable for dev)
pip install -e .
````

# Quick Start
```py
import torch
import torch.nn as nn
from memlord import MemLord

# Create tracker (no auto-GC by default here for determinism)
lord = MemLord(auto_gc_strategy={})

# (Optional) Hook into Torch moves (Tensor.to / Module.to / CUDA sync)
torch_hooks = lord.hook_into_torch()

# (Optional) Hook Python GC: register finalizers on new tensors/modules
# Deep autowrap wraps many torch factories + tensor methods
py_hooks = lord.hook_into_python(
    enable_factory_wrappers=True,
    enable_deep_autowrap=True,
)

# Track explicit objects (you can still call allocate/free manually if you want)
x = torch.empty((1024, 1024), device="cuda:0")
lord.allocate(x)

# Move: counted via Torch hooks
y = x.to("cuda:1", non_blocking=True)
torch.cuda.synchronize()

# Deletion: counted via Python finalizer; you don't need to call free()
del x, y

# Or explicitly free (if you're not relying on finalizers)
# lord.free((x, y))

# Query totals
print(lord.allocated())                           # all devices
print(lord.allocated(torch.device("cuda")))       # all cuda:* combined
print(lord.allocated(torch.device("cuda", 1)))    # only cuda:1

# Optional: enable call-site tracking then query hotspots
lord.set_callsite_tracking(True)
# ... do some work that allocates/frees ...
print(lord.top_alloc_site())                      # (device, "file:line", bytes)
print(lord.top_free_site(torch.device("cuda")))   # best among all cuda:* devices

# Cleanup
torch_hooks.disable()
py_hooks.disable()

```

# Auto-GC Strategy
Instead of a single byte threshold, MemLord uses a stacked/banded strategy keyed by “used %” of the device. Each band specifies:

metric: one of "allocated", "freed", "max" (max of allocated vs freed counters)

threshold: any combination of bytes and/or percent (percent of the device capacity)

Format:
```py
{
  (lo_pct, hi_pct): {
    "metric": "allocated" | "freed" | "max",
    "threshold": {
      "bytes":   int,      # absolute threshold
      "percent": float,    # % of device capacity (VRAM or system RAM)
    }
  },
  ...
}
```

Example: scale up aggressiveness as memory fills

```py
strategy = {
    (0, 50):   {"metric": "max",   "threshold": {"percent": 50.0}},  # <=50% used
    (50, 75):  {"metric": "max",   "threshold": {"percent": 33.0}},  # 50-75%
    (75, 101): {"metric": "freed", "threshold": {"percent": 10.0}},  # >75%
}
lord = MemLord(auto_gc_strategy=strategy)

```

Trigger semantics:

MemLord polls total/used bytes via device-smi (CPU & CUDA), derives used %, selects the matching band, resolves threshold to bytes (combining bytes/percent as max), then compares the chosen metric value to that threshold. If exceeded, it calls the backend GC for that device and resets the per-device freed counter.

🧪 For deterministic tests, you can monkeypatch MemLord._device_memory_stats or pass {} to disable auto-GC (no bands matched).

Per-device percent threshold of min vram size / 3 or ~33%:

```py
strategy = {
    (0, 101): {"metric": "max", "threshold": {"percent": 33.0}}
}
lord.set_auto_gc_strategy(legacy_like)

```
# Python GC Hooks (Finalizers)

MemLord can auto-credit frees when the last reference to a tensor/module is gone (e.g., del t or variable going out of scope).

Enable once:

```py
lord.hook_into_python(
  enable_factory_wrappers=True,   # wrap common torch factories
  enable_deep_autowrap=True,      # also wrap many tensor methods + torch.* funcs
)

```

How it works:

On creation (factories/methods we wrap) or when you call allocate(obj), MemLord registers a weakref.finalize callback with a precomputed size snapshot and a weakref to MemLord (no strong refs ⇒ no cycles).

When the last reference dies, the finalizer runs and credits the object’s bytes to the device’s freed counter.

You can still call free(obj) explicitly, but don’t double-count: either rely on finalizers or call free()—avoid doing both for the same lifetime.

Aliasing safe: the finalizer only fires when the last reference is gone, so a = t; b = t; del a won’t trigger yet; del b will.

## Torch Move Hooks

```pyc

hooks = lord.hook_into_torch()
# Tracks:
#   - torch.Tensor.to(...)
#   - nn.Module.to(...)
#   - torch.cuda.synchronize / Stream.synchronize / Event.synchronize
# Useful for counting non-blocking moves and deferred frees reliably.
hooks.disable()

```

## Optional Call-Site Tracking (Hotspots)

Stack-walking is expensive; disabled by default. Enable only when you need it:

```py
lord.set_callsite_tracking(True)
# OR: MemLord(..., track_callsite=True)

```

Then:

```pycon
lord.top_alloc_site()                 # (device, "file:line", cumulative bytes)
lord.top_free_site(torch.device("cuda"))

```

If tracking is off, hotspot tables remain empty.

## Background Device Usage (device-smi)

MemLord fetches total/used bytes & used % for CPU and CUDA devices via device-smi
.
To avoid hot-path latency, MemLord uses one background poller thread per device (CPU + each cuda:i) that updates usage roughly every 200ms. Strategy checks read the latest snapshot.

# API Reference
`MemLord(auto_gc_strategy: Optional[dict] = None, *, track_callsite: bool = False)`


Create a tracker. If auto_gc_strategy is None, a reasonable banded default is installed.
If you want no auto-GC, pass {}.

`set_auto_gc_strategy(strategy: dict) -> None`

Replace strategy at runtime (shape as shown above).

`set_callsite_tracking(enabled: bool) -> None`

Toggle call-site hotspot tracking (off by default).

`allocate(obj | tuple[obj, ...]) -> None`

Record allocations for a tensor/module or a tuple of them. Also registers finalizers if Python hooks are active.

`free(obj | tuple[obj, ...]) -> None`

Record frees (and may trigger auto-GC per strategy). If you rely on finalizers, avoid double-counting by not calling free() for the same object lifetime.

`allocated(device: Optional[torch.device]) -> tuple[int, str]`

Bytes allocated (raw, human).

None → all devices

torch.device("cuda") → all CUDA devices

torch.device("cuda", i) → only that index

`freed(device: Optional[torch.device]) -> tuple[int, str]`

Bytes freed (raw, human). Same device selection rules.

`top_alloc_site(device: Optional[torch.device]) -> Optional[tuple[torch.device, str, int]]`
`top_free_site(device: Optional[torch.device]) -> Optional[tuple[torch.device, str, int]]`

Return (device, "file:line", cumulative bytes). Requires call-site tracking ON.

`hook_into_torch() -> TorchMoveHooks`

Enable move/sync hooks. Call .disable() to restore originals.

`hook_into_python(*, enable_factory_wrappers=True, enable_deep_autowrap=True) -> SnakeHooks`

Enable finalizers and auto-registration by wrapping many creation paths. Call .disable() to restore originals.

`reset() -> None`

Clear all counters & hotspot tables (does not change strategies or hooks).

# Usage Patterns

A. Pure finalizers (no manual free/allocate)

```py
lord = MemLord(auto_gc_strategy={})
lord.hook_into_python(enable_factory_wrappers=True, enable_deep_autowrap=True)

t = torch.zeros((1_000, 1_000), device="cuda:0")
del t
```

Finalizer credits 'freed' after GC; optionally gc.collect() in tests


B. Manual accounting (no Python wrappers)

```py
lord = MemLord(auto_gc_strategy={})
x = torch.empty((10_000, 10_000), device="cpu")
lord.allocate(x)
# ...
lord.free(x)
```

C. Strategy tuned for pressure

```py
strategy = {
    (0, 60):   {"metric": "max",   "threshold": {"percent": 40}},
    (60, 85):  {"metric": "max",   "threshold": {"percent": 25}},
    (85, 101): {"metric": "freed", "threshold": {"bytes": 256 * 1024**2}},  # 256 MiB
}
lord = MemLord(auto_gc_strategy=strategy)
```

# Logging

Set DEBUG=1 to see colored logs:

```pycon
DEBUG=1 python your_script.py
```


[allocate] (red), [free] (green), per-call summaries (cyan)

[auto_gc] decisions and counts (yellow)

[reset] events (magenta)

# Notes & Gotchas

Double counting: If Python finalizers are enabled, avoid calling free() for the same object lifetime (or you will credit twice). It’s fine to mix approaches across different objects—just be consistent per object.

CUDA caching allocator: Freed bytes reflect MemLord’s logical accounting. Actual VRAM release is subject to PyTorch’s allocator; your auto-GC strategy can trigger a backend GC to purge caches.

Call-site hotspots: They’re empty unless set_callsite_tracking(True) has been called before allocations/frees occur (stack walking is disabled otherwise).

Background pollers: Start lazily on first usage; one thread per device; ~200ms cadence.