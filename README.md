# MemLord

**MemLord** is a tiny PyTorch memory accountant that tracks allocated/freed bytes **per `torch.device` _and_ device index** (e.g., `cuda:0`, `cuda:1`). It supports:

- ✅ `allocate()` / `free()` for `nn.Module` **or** `torch.Tensor` (or a tuple of them)
- ✅ Dedup by storage (avoids double-counting shared views)
- ✅ `allocated()` / `freed()` — return `(raw_bytes, "human string")` for **all** devices, a device type (e.g. `torch.device("cuda")`), or a specific device index (e.g. `torch.device("cuda", 1)`)
- ✅ Per-device-index **auto-GC** (e.g., only `cuda:1` triggers GC when its freed-bytes cross the threshold)
- ✅ `"auto"` threshold: **min visible GPU total / 3** (recomputed on demand)
- ✅ DEBUG-guarded, colored logs (`DEBUG=1`)
- ✅ Hotspot tracking: `top_alloc_site()` / `top_free_site()` → show biggest cumulative site per device
- ✅ Optional hooks to intercept **`.to()`** moves and **defer frees** for CUDA async transfers

> ⚠️ Logs are printed only when `DEBUG=1` is set in the environment.  
> Example: `DEBUG=1 python train.py`

---

## Install

```bash
pip install -e .
```

# Usage
```py
from memlord import MemLord

# auto threshold = min(visible GPU total) / 3
lord = MemLord(auto_gc_bytes="auto")

# optional: hook into Tensor.to / Module.to / CUDA synchronize calls
hooks = lord.hook_into_torch()

# track an object (module or tensor, or tuple of them)
m = nn.Linear(1024, 1024, bias=False).to("cuda:0")
lord.allocate(m)

# later, when you’re done with it:
lord.free(m)

# query totals
print(lord.allocated())                           # all devices
print(lord.allocated(torch.device("cuda")))       # all cuda:* combined
print(lord.allocated(torch.device("cuda", 1)))    # only cuda:1

# See top sites
print(lord.top_alloc_site())                      # across all devices
print(lord.top_alloc_site(torch.device("cuda")))  # best among all cuda:* devices

# change the auto-GC threshold at runtime
lord.set_auto_gc("auto")                          # recompute min visible GPU / 3
lord.set_auto_gc(8 * 1024**3)                     # fixed 8 GiB

# unhook if needed
hooks.disable()
```