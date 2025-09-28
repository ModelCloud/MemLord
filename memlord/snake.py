# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import types
import weakref
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .util import register_snake_hooks_instance, is_shutting_down


# ============== helpers for size snapshot (duplicate core logic safely) ==============

def _gather_tensors(ob: nn.Module | torch.Tensor) -> Iterable[torch.Tensor]:
    if isinstance(ob, torch.Tensor):
        yield ob
        return
    # nn.Module
    for p in ob.parameters(recurse=True):
        yield p.data
    for b in ob.buffers(recurse=True):
        yield b


def _sum_by_dev_dedup(tensors: Iterable[torch.Tensor]) -> Dict[torch.device, int]:
    seen_keys: set[Tuple[int, int]] = set()
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


def _sizes_for_object(ob: nn.Module | torch.Tensor) -> Dict[torch.device, int]:
    tensors = list(_gather_tensors(ob))
    return _sum_by_dev_dedup(tensors)


# ========================== finalizer callback (no cycles) ===========================

def _finalize_cb(lord_ref: "weakref.ReferenceType", sizes_by_dev: Dict[torch.device, int]) -> None:
    """
    Runs when the last strong reference to the tracked object is gone.
    Uses a weakref to MemLord to avoid cycles. Uses precomputed size snapshot.
    """
    try:
        lord = lord_ref()
        if lord is not None and sizes_by_dev:
            # Call the lightweight internal method that updates counters.
            lord._apply_sizes_free(sizes_by_dev)  # type: ignore[attr-defined]
    except Exception:
        # Never raise from finalizers
        pass


# ================================ SnakeHooks class ==================================

class SnakeHooks:
    """
    Python-level hooks to auto-register finalizers for tensors/modules.

    Features:
      - Wrap common torch factory functions (e.g., torch.empty/zeros/ones/full/tensor/rand/randn)
      - Optional deeper wrapping for additional producers (left minimal by default)
      - Register a weakref.finalize per object with a precomputed size snapshot
      - Avoids atexit participation (fin.atexit = False)
      - Avoids cycles by holding only a weakref to MemLord in finalizers
      - Idempotent enable/disable

    NOTE: We keep a per-instance WeakKeyDictionary to avoid double-registering the same object.
    """

    _FACTORY_FUNCS = [
        "empty", "zeros", "ones", "full",
        "rand", "randn", "tensor",
        # You can extend this list if desired.
    ]

    def __init__(self, lord, enable_factory_wrappers: bool = True, enable_deep_autowrap: bool = True) -> None:
        self._lord_ref: "weakref.ReferenceType" = weakref.ref(lord)
        self._wrap_factories = bool(enable_factory_wrappers)
        self._deep_autowrap = bool(enable_deep_autowrap)

        self._enabled: bool = False
        # Keep (module, name) -> original callable so we can restore
        self._orig: List[Tuple[object, str, object]] = []
        # Track objects already registered to avoid duplicate finalizers
        self._seen = weakref.WeakKeyDictionary()  # type: ignore[type-arg]

    # ----------------------------- public API -----------------------------

    def enable(self) -> None:
        if self._enabled:
            return
        if self._wrap_factories:
            self._patch_factories()
        if self._deep_autowrap:
            # Minimal deep autowrap (safe no-ops by default). Extend if needed.
            self._patch_tensor_methods()
            self._patch_additional()
        self._enabled = True
        register_snake_hooks_instance(self)

    def disable(self) -> None:
        if not self._enabled:
            return
        # Restore all patched callables
        for owner, name, orig in reversed(self._orig):
            try:
                setattr(owner, name, orig)
            except Exception:
                pass
        self._orig.clear()
        self._enabled = False

    # Allow MemLord.allocate(...) to register explicitly
    def register(self, ob: nn.Module | torch.Tensor) -> None:
        if is_shutting_down():
            return
        if not isinstance(ob, (torch.Tensor, nn.Module)):
            return
        # Avoid double-registering same object
        try:
            if self._seen.get(ob, False):
                return
            self._seen[ob] = True
        except Exception:
            # If WeakKeyDictionary doesn't accept (rare types), just continue; duplicate finalizers are unlikely here
            pass

        sizes = _sizes_for_object(ob)
        if not sizes:
            return
        fin = weakref.finalize(ob, _finalize_cb, self._lord_ref, sizes)
        # Prevent atexit registry participation to avoid interpreter-shutdown races
        try:
            fin.atexit = False  # type: ignore[attr-defined]
        except Exception:
            pass

    # ----------------------------- patchers ------------------------------

    def _patch_factories(self) -> None:
        for name in self._FACTORY_FUNCS:
            if not hasattr(torch, name):
                continue
            orig = getattr(torch, name)
            if not callable(orig):
                continue
            wrapper = self._wrap_factory(orig)
            self._orig.append((torch, name, orig))
            setattr(torch, name, wrapper)

    def _patch_tensor_methods(self) -> None:
        """
        Optionally wrap a couple of tensor instance methods that are known to produce
        new tensors. We keep this minimal to reduce maintenance risk.
        """
        # Example: clone()
        if hasattr(torch.Tensor, "clone"):
            orig = torch.Tensor.clone
            wrapper = self._wrap_tensor_method(orig)
            self._orig.append((torch.Tensor, "clone", orig))
            setattr(torch.Tensor, "clone", wrapper)

    def _patch_additional(self) -> None:
        """
        Hook a couple of top-level APIs that often create tensors.
        Keep conservative; extend if you need wider coverage.
        """
        # Example: torch.arange
        if hasattr(torch, "arange") and callable(getattr(torch, "arange")):
            orig = torch.arange
            wrapper = self._wrap_factory(orig)
            self._orig.append((torch, "arange", orig))
            setattr(torch, "arange", wrapper)

    # --------------------------- wrappers --------------------------------

    def _wrap_factory(self, fn):
        lord_ref = self._lord_ref
        seen = self._seen

        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if is_shutting_down():
                return out

            def _reg(t: torch.Tensor):
                try:
                    if seen.get(t, False):
                        return
                    seen[t] = True
                except Exception:
                    pass
                sizes = _sizes_for_object(t)
                if not sizes:
                    return
                fin = weakref.finalize(t, _finalize_cb, lord_ref, sizes)
                try:
                    fin.atexit = False  # type: ignore[attr-defined]
                except Exception:
                    pass

            if isinstance(out, torch.Tensor):
                _reg(out)
            elif isinstance(out, (tuple, list)):
                for x in out:
                    if isinstance(x, torch.Tensor):
                        _reg(x)
            return out

        wrapper.__name__ = getattr(fn, "__name__", "memlord_wrapped_factory")
        wrapper.__doc__ = getattr(fn, "__doc__", "")
        wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
        wrapper.__module__ = getattr(fn, "__module__", "torch")
        return wrapper

    def _wrap_tensor_method(self, meth):
        lord_ref = self._lord_ref
        seen = self._seen

        def _method(self, *args, **kwargs):
            out = meth(self, *args, **kwargs)
            if is_shutting_down():
                return out

            def _reg(t: torch.Tensor):
                try:
                    if seen.get(t, False):
                        return
                    seen[t] = True
                except Exception:
                    pass
                sizes = _sizes_for_object(t)
                if not sizes:
                    return
                fin = weakref.finalize(t, _finalize_cb, lord_ref, sizes)
                try:
                    fin.atexit = False  # type: ignore[attr-defined]
                except Exception:
                    pass

            if isinstance(out, torch.Tensor):
                _reg(out)
            elif isinstance(out, (tuple, list)):
                for x in out:
                    if isinstance(x, torch.Tensor):
                        _reg(x)
            return out

        _method.__name__ = getattr(meth, "__name__", "memlord_wrapped_tensor_method")
        _method.__doc__ = getattr(meth, "__doc__", "")
        _method.__qualname__ = getattr(meth, "__qualname__", _method.__name__)
        _method.__module__ = getattr(meth, "__module__", "torch")
        return _method
