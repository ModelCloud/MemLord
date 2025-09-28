# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import weakref
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .util import register_snake_hooks_instance, is_shutting_down


# ===================== safe size helpers (no storage access) ======================

def _sizes_for_tensor_fast(t: torch.Tensor) -> Dict[torch.device, int]:
    dev = t.device
    if dev.type == "meta":
        return {}
    total = 0
    # Sum components for sparse layouts safely
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


def _sizes_for_object_safe(ob: nn.Module | torch.Tensor) -> Dict[torch.device, int]:
    sizes: Dict[torch.device, int] = {}
    if isinstance(ob, torch.Tensor):
        for d, b in _sizes_for_tensor_fast(ob).items():
            sizes[d] = sizes.get(d, 0) + b
        return sizes

    # nn.Module: accumulate over params and buffers
    for p in ob.parameters(recurse=True):
        for d, b in _sizes_for_tensor_fast(p.data).items():
            sizes[d] = sizes.get(d, 0) + b
    for bbuf in ob.buffers(recurse=True):
        for d, b in _sizes_for_tensor_fast(bbuf).items():
            sizes[d] = sizes.get(d, 0) + b
    return sizes


# ========================== finalizer callback (no cycles / no GC) ===================

def _finalize_cb(lord_ref: "weakref.ReferenceType", sizes_by_dev: Dict[torch.device, int]) -> None:
    """
    Runs when the last strong reference to the tracked object is gone.
    Uses a weakref to MemLord to avoid cycles. Uses precomputed size snapshot.
    IMPORTANT: Do NOT trigger auto-GC or poller work here; only bump counters.
    """
    try:
        lord = lord_ref()
        if lord is not None and sizes_by_dev:
            lord._apply_sizes_free_finalizer(sizes_by_dev)  # finalizer-safe path (no GC)
    except Exception:
        # Never raise from finalizers
        pass


# ================================ SnakeHooks class ==================================

class SnakeHooks:
    """
    Python-level hooks to auto-register finalizers for tensors/modules and
    attribute allocate/free accounting to creation/destruction events.

    Features:
      - Wrap common torch factory functions (broad coverage, see list below)
      - Minimal deep autowrap for additional producers (clone())
      - On creation: compute SAFE sizes, bump allocation via MemLord, then register a finalizer
        with the same sizes to guarantee alloc/free symmetry.
      - Skips allocation counting when an 'out=' buffer is provided.
      - Avoids atexit participation (fin.atexit = False)
      - Avoids cycles by holding only a weakref to MemLord in finalizers
      - Idempotent enable/disable
      - Per-instance WeakKeyDictionary ensures no double-registration

    NOTE: We intentionally DO NOT call any method that touches untyped_storage() here.
    """

    # Broad factory coverage. Many models rely on these APIs for new allocations.
    _FACTORY_FUNCS = [
        # core factories
        "empty", "zeros", "ones", "full",
        "rand", "randn", "tensor", "arange",

        # *_like variants
        "empty_like", "zeros_like", "ones_like", "full_like",
        "rand_like", "randn_like",

        # sampling / indexing / sequences
        "randint", "randperm", "linspace", "logspace", "eye", "tri",

        # strided / low-level alloc
        "empty_strided", "as_strided",

        # distributions (common init)
        "normal",
        # (add more as needed)
    ]

    def __init__(self, lord, enable_factory_wrappers: bool = True, enable_deep_autowrap: bool = True) -> None:
        self._lord_ref: "weakref.ReferenceType" = weakref.ref(lord)
        self._wrap_factories = bool(enable_factory_wrappers)
        self._deep_autowrap = bool(enable_deep_autowrap)

        self._enabled: bool = False
        # Keep (module, name) -> original callable so we can restore
        self._orig: List[Tuple[object, str, object]] = []
        # Track objects already registered to avoid duplicate finalizers
        self._seen: "weakref.WeakKeyDictionary[object, bool]" = weakref.WeakKeyDictionary()  # type: ignore[type-arg]

    # ----------------------------- public API -----------------------------

    def enable(self) -> None:
        if self._enabled:
            return
        if self._wrap_factories:
            self._patch_factories()
        if self._deep_autowrap:
            # Minimal deep autowrap — clone() returns a new tensor
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

    # Allow MemLord.allocate(...) to register explicitly (fallback path)
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
            pass

        # SAFE sizes, allocate, then register finalizer with the SAME sizes
        sizes = _sizes_for_object_safe(ob)
        if not sizes:
            return

        lord = self._lord_ref()
        if lord is not None:
            lord._apply_sizes_allocate(sizes)

        fin = weakref.finalize(ob, _finalize_cb, self._lord_ref, sizes)
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
        # Example: clone() -> returns a new tensor on same device
        if hasattr(torch.Tensor, "clone"):
            orig = torch.Tensor.clone
            wrapper = self._wrap_tensor_method(orig)
            self._orig.append((torch.Tensor, "clone", orig))
            setattr(torch.Tensor, "clone", wrapper)

    def _patch_additional(self) -> None:
        # Placeholder for additional producers
        pass

    # --------------------------- wrappers --------------------------------

    def _wrap_factory(self, fn):
        lord_ref = self._lord_ref
        seen = self._seen

        def _reg(t: torch.Tensor, pre_sizes: Optional[Dict[torch.device, int]] = None):
            # Register a finalizer with precomputed sizes to keep symmetry
            try:
                if seen.get(t, False):
                    return
                seen[t] = True
            except Exception:
                pass
            sizes = pre_sizes if pre_sizes is not None else _sizes_for_object_safe(t)
            if not sizes:
                return
            fin = weakref.finalize(t, _finalize_cb, lord_ref, sizes)
            try:
                fin.atexit = False  # type: ignore[attr-defined]
            except Exception:
                pass

        def _maybe_is_out_arg(kwargs: dict) -> bool:
            # If an explicit out= tensor is provided, the call writes into it (no new alloc).
            if not isinstance(kwargs, dict):
                return False
            out = kwargs.get("out", None)
            if out is None:
                return False
            # torch APIs often accept out as Tensor or (Tensor,) or (Tensor, Tensor)
            if isinstance(out, torch.Tensor):
                return True
            if isinstance(out, (tuple, list)) and any(isinstance(x, torch.Tensor) for x in out):
                return True
            return False

        def wrapper(*args, **kwargs):
            # If 'out=' is provided, we skip counting allocation (it's reusing memory),
            # but we still need to register finalizers for any new tensor(s) returned.
            has_out = _maybe_is_out_arg(kwargs)
            out = fn(*args, **kwargs)
            if is_shutting_down():
                return out

            lord = lord_ref()

            # Handle a single tensor
            if isinstance(out, torch.Tensor):
                sizes = _sizes_for_object_safe(out)
                if sizes and lord is not None and not has_out:
                    lord._apply_sizes_allocate(sizes)
                _reg(out, sizes)
                return out

            # Handle tuple/list of tensors
            if isinstance(out, (tuple, list)):
                tensors: List[torch.Tensor] = [x for x in out if isinstance(x, torch.Tensor)]
                if tensors:
                    agg: Dict[torch.device, int] = {}
                    per: List[Tuple[torch.Tensor, Dict[torch.device, int]]] = []
                    for x in tensors:
                        sx = _sizes_for_object_safe(x)
                        per.append((x, sx))
                        for d, b in sx.items():
                            agg[d] = agg.get(d, 0) + b
                    if lord is not None and agg and not has_out:
                        lord._apply_sizes_allocate(agg)
                    for x, sx in per:
                        _reg(x, sx)
                return out

            # Non-tensor outputs pass through
            return out

        # Metadata preservation
        wrapper.__name__ = getattr(fn, "__name__", "memlord_wrapped_factory")
        wrapper.__doc__ = getattr(fn, "__doc__", "")
        wrapper.__qualname__ = getattr(fn, "__qualname__", wrapper.__name__)
        wrapper.__module__ = getattr(fn, "__module__", "torch")
        return wrapper

    def _wrap_tensor_method(self, meth):
        lord_ref = self._lord_ref
        seen = self._seen

        def _reg(t: torch.Tensor, pre_sizes: Optional[Dict[torch.device, int]] = None):
            try:
                if seen.get(t, False):
                    return
                seen[t] = True
            except Exception:
                pass
            sizes = pre_sizes if pre_sizes is not None else _sizes_for_object_safe(t)
            if not sizes:
                return
            fin = weakref.finalize(t, _finalize_cb, lord_ref, sizes)
            try:
                fin.atexit = False  # type: ignore[attr-defined]
            except Exception:
                pass

        def _method(self, *args, **kwargs):
            out = meth(self, *args, **kwargs)
            if is_shutting_down():
                return out

            lord = lord_ref()
            # clone and similar return tensors (or collections)
            if isinstance(out, torch.Tensor):
                sizes = _sizes_for_object_safe(out)
                if sizes and lord is not None:
                    lord._apply_sizes_allocate(sizes)
                _reg(out, sizes)
                return out

            if isinstance(out, (tuple, list)):
                tensors: List[torch.Tensor] = [x for x in out if isinstance(x, torch.Tensor)]
                if tensors:
                    agg: Dict[torch.device, int] = {}
                    per: List[Tuple[torch.Tensor, Dict[torch.device, int]]] = []
                    for x in tensors:
                        sx = _sizes_for_object_safe(x)
                        per.append((x, sx))
                        for d, b in sx.items():
                            agg[d] = agg.get(d, 0) + b
                    if lord is not None and agg:
                        lord._apply_sizes_allocate(agg)
                    for x, sx in per:
                        _reg(x, sx)
                return out

            return out

        _method.__name__ = getattr(meth, "__name__", "memlord_wrapped_tensor_method")
        _method.__doc__ = getattr(meth, "__doc__", "")
        _method.__qualname__ = getattr(meth, "__qualname__", _method.__name__)
        _method.__module__ = getattr(meth, "__module__", "torch")
        return _method
