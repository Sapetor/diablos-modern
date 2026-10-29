"""Block-kernel registry for the compiled (fast-solver) path.

Each compilable block family contributes builder functions
``build_<block>(ctx) -> executor`` registered under their canonical fn-name(s)
(see ``lib.engine.block_names.canonical_fn``). ``SystemCompiler``'s
``_create_block_executor`` computes one ``BuildContext`` per block and dispatches
to the registered builder, replacing the historical ~1700-line if/elif ladder of
closure factories. The same registry is intended to back the post-solve replay
loop in ``simulation_engine.run_compiled_simulation`` so the two paths cannot
diverge.

Migration is incremental: families move here one at a time, each gated behind the
compiled-path golden harness (``tests/regression/test_compiled_golden.py``).
Blocks not yet migrated fall through to the legacy if/elif in the compiler.
"""

import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from blocks.base_block import BaseBlock
from lib.engine.zero_crossing import EventSpec, signal_scalar  # noqa: F401 (re-export)
from lib.user_blocks import USER_BLOCK_FLAG, USER_BLOCK_NAMESPACE, USER_MODULE_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class BuildContext:
    """Per-block inputs shared by every kernel builder.

    Computed once by the compiler before dispatch. Builders read only the fields
    they need (e.g. a source block uses ``b_name`` and ``params`` only).
    """

    block: Any
    b_name: str
    fn: str
    params: Dict
    input_sources: List[Optional[str]]  # signal key per input port (None -> 0.0)
    deps: Dict
    state_map: Dict
    block_matrices: Dict
    # Per-compile store of latch/mode holders, shared between a block's kernel
    # builder and its event builder (they run as two separate dispatches but
    # must close over the *same* mutable holder). Optional so the many direct
    # BuildContext(...) constructions in tests keep working.
    mode_registry: Optional[Dict[str, Dict[str, Any]]] = None

    def latch(self, **initial) -> Dict[str, Any]:
        """Return this block's mode holder, creating it on first request.

        The holder carries ``mode`` (the live discrete state), ``init`` (what a
        run reset restores), ``event_driven`` (set by the block's event builder
        to claim the latch) and ``frozen``.

        It starts *unfrozen*, i.e. free-running: the kernel updates the mode
        from whatever it sees, the historical probe-order-dependent behaviour.
        ``lib.engine.zero_crossing`` freezes the claimed holders for the length
        of a run, so that only an event's discrete update at a located
        switching instant may change ``mode`` -- which is what makes the ODE
        right-hand side a pure function of ``(t, y)`` within a segment.  It
        unfreezes them again if the chattering guard has to give up.
        """
        registry = self.mode_registry
        if registry is None:
            registry = {}
            self.mode_registry = registry
        holder = registry.get(self.b_name)
        if holder is None:
            holder = dict(initial)
            holder.setdefault("frozen", False)
            holder.setdefault("event_driven", False)
            registry[self.b_name] = holder
        return holder


# canonical fn-name -> builder(ctx) -> executor closure, for the built-in
# kernels (and any registered from an ordinary, non-user module). Kernels from
# user block modules are *not* stored here: see "User block kernels" below.
KERNEL_BUILDERS: Dict[str, Callable[[BuildContext], Callable]] = {}

_KERNEL_KIND = "kernel"
_EVENTS_KIND = "event builder"

# --------------------------------------------------------------------------- #
# User block kernels.
#
# A user block module (``lib.user_blocks``, imported as
# ``diablos_user_blocks.<stem>``) registers through the same decorators, but
# its builders are bound to *that module*, not to the bare canonical name: two
# trusted projects may each define a ``ReviewTripler`` with its own kernel, and
# a process-global name lookup would hand both diagrams whichever was imported
# last. The builders are stored in the defining module's namespace (the
# builder's ``__globals__``) and found again from the block's class, which the
# loader stamps with the same namespace. Module names are no help here: both
# projects' files are ``diablos_user_blocks.review_tripler``.
# --------------------------------------------------------------------------- #

#: Key, in a user module's globals, of its ``{kind: {name: builder}}`` registry.
_SCOPED_REGISTRY = "__diablos_compiled_builders__"


def _is_user_module(module_name) -> bool:
    return str(module_name or "").startswith(USER_MODULE_PREFIX + ".")


def _builder_namespace(builder) -> Optional[Dict[str, Any]]:
    """The globals of the module that defined ``builder`` (None if unknown)."""
    namespace = getattr(builder, "__globals__", None)
    if isinstance(namespace, dict):
        return namespace
    module = sys.modules.get(getattr(builder, "__module__", "") or "")
    return getattr(module, "__dict__", None)


def _block_class(block) -> Optional[type]:
    """Implementation class of a DBlock, block instance, or block class."""
    if block is None:
        return None
    instance = getattr(block, "block_instance", None)
    if instance is not None:
        return type(instance)
    if isinstance(block, type):
        return block
    if isinstance(block, BaseBlock):
        return type(block)
    return None


def _is_user_class(cls) -> bool:
    return bool(getattr(cls, USER_BLOCK_FLAG, False)) or _is_user_module(
        getattr(cls, "__module__", "")
    )


def _class_namespace(cls) -> Optional[Dict[str, Any]]:
    """Namespace of the module that defined block class ``cls``.

    Prefers the loader's stamp; otherwise the globals of a function defined in
    the class body (immune to ``sys.modules`` being rebound to a same-named
    module from another project), and only last ``sys.modules``.
    """
    namespace = cls.__dict__.get(USER_BLOCK_NAMESPACE)
    if isinstance(namespace, dict):
        return namespace
    for member in vars(cls).values():
        func = getattr(member, "fget", None) or getattr(member, "__func__", None) or member
        found = getattr(func, "__globals__", None)
        if isinstance(found, dict) and found.get("__name__") == cls.__module__:
            return found
    module = sys.modules.get(getattr(cls, "__module__", "") or "")
    return getattr(module, "__dict__", None)


def _user_builder(cls, kind, name):
    namespace = _class_namespace(cls)
    if not namespace:
        return None
    return namespace.get(_SCOPED_REGISTRY, {}).get(kind, {}).get(name)


def _lookup(registry, kind, name, block):
    """Builder for ``name``: the block's own if it is a user block, else global.

    A user block only ever gets a builder registered by its own module -- never
    a built-in one that happens to share its canonical name, and never another
    project's.
    """
    cls = _block_class(block)
    if cls is not None and _is_user_class(cls):
        return _user_builder(cls, kind, name)
    return registry.get(name)


def kernel(*names):
    """Decorator: register a builder under one or more canonical fn-names."""

    def deco(builder):
        for name in names:
            _register(KERNEL_BUILDERS, name, builder, _KERNEL_KIND)
        return builder

    return deco


def _is_builtin(builder) -> bool:
    """True for a builder defined in one of this package's family modules."""
    module = getattr(builder, "__module__", "") or ""
    return module == __name__ or module.startswith(__name__ + ".")


def _register(registry, name, builder, kind):
    """Add ``builder`` under ``name`` unless that would shadow a built-in.

    User block modules register through the same decorators, so a user file
    declaring ``@kernel("Gain")`` must not silently replace the engine's own
    Gain kernel for every diagram. A user module's builder goes into that
    module's own registry (see "User block kernels" above); re-registering it
    (a palette reload re-imports the module) simply replaces it there.
    """
    existing = registry.get(name)
    if existing is not None and _is_builtin(existing) and not _is_builtin(builder):
        logger.warning(
            "Ignoring %s %r from %s: it would replace the built-in one.",
            kind,
            name,
            getattr(builder, "__module__", "?"),
        )
        return
    if _is_user_module(getattr(builder, "__module__", "")):
        namespace = _builder_namespace(builder)
        if namespace is not None:
            namespace.setdefault(_SCOPED_REGISTRY, {}).setdefault(kind, {})[name] = builder
            return
    registry[name] = builder


def get_kernel_builder(fn, block=None):
    """Return the builder for canonical name ``fn`` (or None).

    ``block`` (a DBlock, block instance or block class) selects the registry: a
    user block resolves only kernels registered by its own module; anything
    else -- or no ``block`` -- uses the global (built-in) registry.
    """
    return _lookup(KERNEL_BUILDERS, _KERNEL_KIND, fn, block)


# --------------------------------------------------------------------------- #
# Zero-crossing (event) registry.
#
# A discontinuous block declares its switching surfaces next to the kernel that
# implements the discontinuity, so the two cannot drift.  A builder returns a
# list of ``EventSpec`` (possibly empty, e.g. when the limits are infinite);
# ``SystemCompiler.compile_system`` collects them and attaches them to the
# compiled RHS, and ``lib.engine.zero_crossing.solve_with_events`` drives them.
# Adding events to a new block means one ``@events("Fn")`` builder here -- the
# runner never learns about individual block types.
# --------------------------------------------------------------------------- #

# canonical fn-name -> builder(ctx) -> List[EventSpec]
EVENT_BUILDERS: Dict[str, Callable[[BuildContext], List[Any]]] = {}


def events(*names):
    """Decorator: register an event builder under one or more canonical names."""

    def deco(builder):
        for name in names:
            _register(EVENT_BUILDERS, name, builder, _EVENTS_KIND)
        return builder

    return deco


def get_event_builder(fn, block=None):
    """Return the event builder for canonical name ``fn`` (or None).

    ``block`` selects the registry exactly as in :func:`get_kernel_builder`.
    """
    return _lookup(EVENT_BUILDERS, _EVENTS_KIND, fn, block)


def build_events(ctx):
    """Event specs contributed by one block, or ``[]`` when it has none."""
    builder = get_event_builder(ctx.fn, ctx.block)
    if builder is None:
        return []
    return list(builder(ctx) or [])


# Import family modules for their registration side effects. Keep at the bottom
# to avoid import cycles (family modules import `kernel` from this module).
from lib.engine.compiler_kernels import sources  # noqa: E402,F401
from lib.engine.compiler_kernels import math  # noqa: E402,F401
from lib.engine.compiler_kernels import nonlinear  # noqa: E402,F401
from lib.engine.compiler_kernels import routing  # noqa: E402,F401
from lib.engine.compiler_kernels import state  # noqa: E402,F401
from lib.engine.compiler_kernels import pde  # noqa: E402,F401
from lib.engine.compiler_kernels import field  # noqa: E402,F401
from lib.engine.compiler_kernels import discrete  # noqa: E402,F401
from lib.engine.compiler_kernels import function  # noqa: E402,F401
