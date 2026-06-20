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
    input_sources: List[Optional[str]]   # signal key per input port (None -> 0.0)
    deps: Dict
    state_map: Dict
    block_matrices: Dict


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


# Import family modules for their registration side effects. Keep at the bottom
# to avoid import cycles (family modules import `kernel` from this module).
from lib.engine.compiler_kernels import sources  # noqa: E402,F401
