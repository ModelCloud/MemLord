# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import types
import weakref
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


# =========================
# Size accounting utilities
# =========================

def _gather_tensors_from_obj(ob: nn.Module | torch.Tensor) -> Iterable[torch.Tensor]:
    if isinstance(ob, torch.Tensor):
        yield ob
        return
    for p in ob.parameters(recurse=True):
        yield p.data
    for b in ob.buffers(recurse=True):
        yield b


def _sizes_by_device_instance(ob: nn.Module | torch.Tensor) -> Dict[torch.device, int]:
    seen: set[Tuple[int, int]] = set()  # (data_ptr, nbytes)
    by_dev: Dict[torch.device, int] = {}

    def _acc_dense(t: torch.Tensor) -> None:
        dev = t.device
        if dev.type == "meta":
            return
        try:
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key in seen:
                return
            seen.add(key)
            by_dev[dev] = by_dev.get(dev, 0) + int(st.nbytes())
        except RuntimeError:
            nbytes = int(t.numel() * t.element_size())
            key = (t.data_ptr(), nbytes)
            if key in seen:
                return
            seen.add(key)
            by_dev[dev] = by_dev.get(dev, 0) + nbytes

    for t in _gather_tensors_from_obj(ob):
        if t.is_sparse:
            _acc_dense(t.indices())
            _acc_dense(t.values())
        elif t.layout == torch.sparse_csr:
            _acc_dense(t.crow_indices())
            _acc_dense(t.col_indices())
            _acc_dense(t.values())
        elif hasattr(torch, "sparse_csc") and t.layout == torch.sparse_csc:  # type: ignore[attr-defined]
            _acc_dense(t.ccol_indices())
            _acc_dense(t.row_indices())
            _acc_dense(t.values())
        else:
            _acc_dense(t)
    return by_dev


# ======================================
# Finalizer (module-level; no strong refs)
# ======================================

def _finalize_cb(lord_ref: "weakref.ReferenceType[MemLord]", sizes_by_dev: Dict[torch.device, int]) -> None:  # type: ignore[name-defined]
    try:
        lord = lord_ref()
        if lord is not None and sizes_by_dev:
            lord._apply_sizes_free(sizes_by_dev)  # internal SLF by design
    except Exception:
        pass


# ==================
# Python GC hook API
# ==================

class SnakeHooks:
    """
    Python-level GC hooks for MemLord.

    What it does:
      • Registers weakref.finalize on tensors/modules so that when the **last**
        reference drops (e.g., `del t`), MemLord credits their bytes to "freed".
      • Optionally wraps many torch factory functions, tensor methods, and a wide
        set of top-level `torch.*` functions to auto-register new tensors without
        the user needing to call `MemLord.allocate(...)`.

    Safety:
      • The finalizer is module-level and holds only a **weakref to MemLord** plus
        a precomputed sizes snapshot. We do **not** store finalizer objects.
      • Wrappers capture the weakref to MemLord (not `self`), avoiding cycles.

    Configuration:
      SnakeHooks(lord,
                 enable_factory_wrappers=True,
                 enable_deep_autowrap=True)
    """

    # Common top-level factories (explicit list for reliability)
    _FACTORY_NAMES = (
        "empty", "zeros", "ones", "full",
        "rand", "randn", "randint", "randperm",
        "tensor", "as_tensor", "from_numpy", "frombuffer",
        "arange", "linspace", "logspace", "eye",
        "empty_like", "zeros_like", "ones_like", "full_like",
        "range",  # deprecated alias sometimes present
        "stack", "hstack", "vstack", "dstack", "column_stack",
        "cat", "concat", "tile", "repeat_interleave",
        "where", "nonzero", "index_select", "gather",
        "tril", "triu", "diag", "diagonal",
        "nn.functional.pad",  # in case it's re-exported; best-effort
    )

    # Tensor instance methods that often produce new tensors
    _TENSOR_METHODS = (
        "clone", "detach",
        "to", "cpu", "cuda",
        "contiguous",
        "view", "reshape", "flatten",
        "transpose", "permute",
        "unsqueeze", "squeeze",
        "expand", "repeat",
        "narrow",
        "new_empty", "new_zeros", "new_ones", "new_full", "new_tensor",
        "type", "type_as",
        "float", "half", "bfloat16", "int", "long", "bool",
        "pin_memory",
    )

    # Names in the torch namespace we should not wrap (modules, types, etc.)
    _NS_EXCLUDE_PREFIXES = (
        "_", "nn", "optim", "distributed", "fx", "onnx", "utils", "hub",
        "amp", "backends", "library", "jax", "mps", "xpu",
        "cuda", "backends", "testing",
        "autograd",  # lots of wrappers inside; avoid
        "overrides",
    )
    _NS_EXCLUDE_NAMES = {
        "Tensor", "dtype", "device", "layout", "memory_format", "Generator",
        "Size", "Storage", "stream", "manual_seed", "seed", "initial_seed",
        "set_default_dtype", "get_default_dtype", "set_printoptions",
    }

    def __init__(
        self,
        lord: "MemLord",  # type: ignore[name-defined]
        enable_factory_wrappers: bool = True,
        enable_deep_autowrap: bool = True,
    ) -> None:
        self._lord_ref: "weakref.ReferenceType[MemLord]" = weakref.ref(lord)  # type: ignore[name-defined]
        self._enabled = False
        self._wrap_factories = enable_factory_wrappers
        self._deep_autowrap = enable_deep_autowrap

        self._orig_factories: Dict[str, Any] = {}
        self._orig_tensor_methods: Dict[str, Any] = {}
        self._orig_ns_funcs: Dict[str, Any] = {}

    # ---- Public API ----

    def enable(self) -> None:
        if self._enabled:
            return
        if self._wrap_factories:
            self._patch_factories()
        if self._deep_autowrap:
            self._patch_tensor_methods()
            self._patch_torch_namespace()
        self._enabled = True

    def disable(self) -> None:
        if not self._enabled:
            return
        self._restore_factories()
        self._restore_tensor_methods()
        self._restore_torch_namespace()
        self._enabled = False

    def register(self, ob: nn.Module | torch.Tensor) -> None:
        """
        Manually register an object for GC finalization. Safe to skip if
        wrappers are enabled; kept for explicit control and modules.
        """
        if not isinstance(ob, (torch.Tensor, nn.Module)):
            return
        sizes = _sizes_by_device_instance(ob)
        if not sizes:
            return
        weakref.finalize(ob, _finalize_cb, self._lord_ref, sizes)

    # ---- Internal: wrap helpers ----

    def _wrap_callable(self, fn: Any) -> Any:
        lord_ref = self._lord_ref

        def _wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
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

        try:
            _wrapper.__name__ = f"memlord_wrapped_{getattr(fn, '__name__', 'callable')}"
        except Exception:
            pass
        _wrapper.__doc__ = (getattr(fn, "__doc__", "") or "") + "\n\n[wrapped by MemLord SnakeHooks]"
        _wrapper.__module__ = getattr(fn, "__module__", "torch")
        setattr(_wrapper, "__memlord_wrapped__", True)
        return _wrapper

    def _wrap_method(self, meth: Any) -> Any:
        lord_ref = self._lord_ref

        def _meth(self, *args, **kwargs):
            out = meth(self, *args, **kwargs)
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

        try:
            _meth.__name__ = f"memlord_wrapped_{getattr(meth, '__name__', 'method')}"
        except Exception:
            pass
        _meth.__doc__ = (getattr(meth, "__doc__", "") or "") + "\n\n[wrapped by MemLord SnakeHooks]"
        _meth.__module__ = getattr(meth, "__module__", "torch")
        setattr(_meth, "__memlord_wrapped__", True)
        return _meth

    # ---- Patching sets ----

    def _patch_factories(self) -> None:
        for name in self._FACTORY_NAMES:
            # Some names like "nn.functional.pad" may not exist at torch.*
            if "." in name:
                # best-effort dotted lookup
                root = torch
                parts = name.split(".")
                ok = True
                for p in parts[:-1]:
                    root = getattr(root, p, None)
                    if root is None:
                        ok = False
                        break
                if not ok:
                    continue
                leaf = parts[-1]
                fn = getattr(root, leaf, None)
                if not callable(fn):
                    continue
                if getattr(fn, "__memlord_wrapped__", False):
                    continue
                self._orig_factories[name] = fn
                setattr(root, leaf, self._wrap_callable(fn))
            else:
                fn = getattr(torch, name, None)
                if not callable(fn):
                    continue
                if getattr(fn, "__memlord_wrapped__", False):
                    continue
                self._orig_factories[name] = fn
                setattr(torch, name, self._wrap_callable(fn))

        # Also cover torch.from_file if present (rare)
        fn = getattr(torch, "from_file", None)
        if callable(fn) and not getattr(fn, "__memlord_wrapped__", False):
            self._orig_factories["from_file"] = fn
            setattr(torch, "from_file", self._wrap_callable(fn))

    def _restore_factories(self) -> None:
        for name, fn in self._orig_factories.items():
            if "." in name:
                root = torch
                parts = name.split(".")
                ok = True
                for p in parts[:-1]:
                    root = getattr(root, p, None)
                    if root is None:
                        ok = False
                        break
                if ok:
                    setattr(root, parts[-1], fn)
            else:
                setattr(torch, name, fn)
        self._orig_factories.clear()

    def _patch_tensor_methods(self) -> None:
        T = torch.Tensor
        for name in self._TENSOR_METHODS:
            meth = getattr(T, name, None)
            if not callable(meth):
                continue
            if getattr(meth, "__memlord_wrapped__", False):
                continue
            self._orig_tensor_methods[name] = meth
            setattr(T, name, self._wrap_method(meth))

    def _restore_tensor_methods(self) -> None:
        T = torch.Tensor
        for name, meth in self._orig_tensor_methods.items():
            setattr(T, name, meth)
        self._orig_tensor_methods.clear()

    def _patch_torch_namespace(self) -> None:
        """
        Broad best-effort pass: wrap torch.* callables that aren't explicitly
        excluded. This catches many functions not listed in _FACTORY_NAMES.
        """
        for name in dir(torch):
            if name in self._NS_EXCLUDE_NAMES:
                continue
            if any(name.startswith(pref) for pref in self._NS_EXCLUDE_PREFIXES):
                continue
            if name in self._orig_factories:  # already wrapped as a factory
                continue
            obj = getattr(torch, name, None)
            # Only wrap plain functions / builtins; skip classes, modules, descriptors
            if not callable(obj):
                continue
            if isinstance(obj, (type, types.ModuleType)):
                continue
            if getattr(obj, "__memlord_wrapped__", False):
                continue
            try:
                # Try a cheap probe: call signature might be large; just wrap blindly.
                self._orig_ns_funcs[name] = obj
                setattr(torch, name, self._wrap_callable(obj))
            except Exception:
                # Be conservative: skip if setting fails
                self._orig_ns_funcs.pop(name, None)
                continue

    def _restore_torch_namespace(self) -> None:
        for name, fn in self._orig_ns_funcs.items():
            try:
                setattr(torch, name, fn)
            except Exception:
                pass
        self._orig_ns_funcs.clear()
