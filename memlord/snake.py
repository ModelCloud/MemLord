# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import weakref
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# ---- Utilities to compute sizes (dedup by storage) -----------------

def _gather_tensors_from_obj(ob: nn.Module | torch.Tensor) -> Iterable[torch.Tensor]:
    if isinstance(ob, torch.Tensor):
        yield ob
        return
    # nn.Module
    for p in ob.parameters(recurse=True):
        yield p.data
    for b in ob.buffers(recurse=True):
        yield b


def _sizes_by_device_instance(ob: nn.Module | torch.Tensor) -> Dict[torch.device, int]:
    seen_keys: set[Tuple[int, int]] = set()  # (data_ptr, nbytes)
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

    for t in _gather_tensors_from_obj(ob):
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


# ---- Finalizer callback (module-level to avoid capturing 'self') -------------

def _finalize_cb(lord_ref: "weakref.ReferenceType[MemLord]", sizes_by_dev: Dict[torch.device, int]) -> None:  # type: ignore[name-defined]  # MemLord is defined in core.py at runtime
    try:
        lord = lord_ref()
        if lord is not None and sizes_by_dev:
            lord._apply_sizes_free(sizes_by_dev)  # noqa: SLF001 (internal hook by design)
    except Exception:
        # Swallow errors during interpreter shutdown / late GC
        pass


# ---- Python-level hooks ------------------------------------------------------

class SnakeHooks:
    """
    Python-level GC hooks for MemLord:
      • Registers weakref.finalize for tensors/modules so that `del obj` events
        (object GC) credit freed bytes back to MemLord.
      • Optionally wraps selected torch factory functions to auto-track
        newly created tensors.

    Design for no cycles:
      • The finalizer function is module-level (_finalize_cb) and receives only:
          - a weakref to MemLord (no strong ref to MemLord, no ref to SnakeHooks)
          - a precomputed sizes_by_dev dict (ints + torch.device keys)
      • We do NOT store the Finalize object anywhere; registration is sufficient.

    Enable with:
        hooks = SnakeHooks(memlord_instance)
        hooks.enable()   # optional: enable_factory_wrappers=True to auto-track factories
        ...
        hooks.disable()
    """

    _FACTORY_NAMES = (
        "empty", "zeros", "ones", "full",
        "rand", "randn", "randint",
        "tensor", "from_numpy",
        "empty_like", "zeros_like", "ones_like",
        "rand_like", "randn_like",
        "clone",
    )

    def __init__(self, lord: "MemLord", enable_factory_wrappers: bool = True) -> None:  # type: ignore[name-defined]
        # Store only a weakref to MemLord to avoid cycles
        self._lord_ref: "weakref.ReferenceType[MemLord]" = weakref.ref(lord)  # type: ignore[name-defined]
        self._enabled = False
        self._wrap_factories = enable_factory_wrappers
        self._orig_fns: Dict[str, Any] = {}

    # ---- Public API ----
    def enable(self) -> None:
        if self._enabled:
            return
        if self._wrap_factories:
            self._patch_factories()
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        self._restore_factories()
        # We did not keep references to finalizers; they remain registered.
        self._enabled = False

    def register(self, ob: nn.Module | torch.Tensor) -> None:
        """
        Manually register an object for GC finalization.
        Used by MemLord.allocate(...) to ensure tracked objects are covered.

        This does NOT retain 'ob' nor create reference cycles:
          - We precompute sizes now.
          - We pass a weakref to MemLord into a module-level callback.
          - We do NOT store the Finalize object; registration is enough.
        """
        if not isinstance(ob, (torch.Tensor, nn.Module)):
            return
        sizes = _sizes_by_device_instance(ob)
        if not sizes:
            return

        # Register finalizer with no strong references to `ob` or to SnakeHooks.
        weakref.finalize(ob, _finalize_cb, self._lord_ref, sizes)

    # ---- Internals: factory patching ----

    def _patch_factories(self) -> None:
        for name in self._FACTORY_NAMES:
            orig = getattr(torch, name, None)
            if not callable(orig):
                continue

            def _make_wrapper(fn: Any):
                lord_ref = self._lord_ref  # capture weakref only
                def _wrapper(*args, **kwargs):
                    out = fn(*args, **kwargs)
                    # Register only tensors (or sequences of tensors)
                    if isinstance(out, torch.Tensor):
                        sizes = _sizes_by_device_instance(out)
                        if sizes:
                            weakref.finalize(out, _finalize_cb, lord_ref, sizes)
                    elif isinstance(out, (tuple, list)):
                        for x in out:
                            if isinstance(x, torch.Tensor):
                                sizes = _sizes_by_device_instance(x)
                                if sizes:
                                    weakref.finalize(x, _finalize_cb, lord_ref, sizes)
                    return out
                _wrapper.__name__ = f"memlord_wrapped_{fn.__name__}"
                _wrapper.__doc__ = (fn.__doc__ or "") + "\n\n[wrapped by MemLord SnakeHooks]"
                _wrapper.__module__ = fn.__module__
                return _wrapper

            self._orig_fns[name] = orig
            setattr(torch, name, _make_wrapper(orig))

    def _restore_factories(self) -> None:
        if not self._orig_fns:
            return
        for name, fn in self._orig_fns.items():
            setattr(torch, name, fn)
        self._orig_fns.clear()
