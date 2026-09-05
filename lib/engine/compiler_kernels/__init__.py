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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from lib.engine.zero_crossing import EventSpec, signal_scalar  # noqa: F401 (re-export)


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


# canonical fn-name -> builder(ctx) -> executor closure
KERNEL_BUILDERS: Dict[str, Callable[[BuildContext], Callable]] = {}


def kernel(*names):
    """Decorator: register a builder under one or more canonical fn-names."""

    def deco(builder):
        for name in names:
            KERNEL_BUILDERS[name] = builder
        return builder

    return deco


def get_kernel_builder(fn):
    """Return the registered builder for canonical name ``fn`` (or None)."""
    return KERNEL_BUILDERS.get(fn)


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
            EVENT_BUILDERS[name] = builder
        return builder

    return deco


def get_event_builder(fn):
    """Return the registered event builder for canonical name ``fn`` (or None)."""
    return EVENT_BUILDERS.get(fn)


def build_events(ctx):
    """Event specs contributed by one block, or ``[]`` when it has none."""
    builder = EVENT_BUILDERS.get(ctx.fn)
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
