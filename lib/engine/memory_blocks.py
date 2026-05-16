"""
Single source of truth for memory-block classification.

Memory blocks are diagrams elements whose output at time t depends only on
*past* inputs (or no inputs at all), not on the current input.  They serve
two related but distinct purposes in DiaBloS:

1. **Initialization safety** (used by SimulationEngine.identify_memory_blocks
   in lib/engine/simulation_engine.py): the engine's Loop 1 calls every
   memory block once with `output_only=True` to emit y[0] = h(x[0]) before
   any source has driven its inputs.  A block can only land in this set
   if it's safe to execute with an empty inputs dict — i.e. it either
   uses `inputs.get(k, default)` or doesn't read inputs at all.

2. **Algebraic-loop detection** (used by
   ValidationHelper.detect_algebraic_loops in lib/improvements.py): when
   doing a topological sort to decide whether a feedback cycle is purely
   algebraic, edges into memory blocks are removed because the
   block's output for the current step does not depend on its input
   for the current step — the cycle is broken in time.

For nearly every block these two notions coincide: if a block is safe to
call output_only, it also breaks algebraic loops, and vice versa.  The
constants below are therefore shared between both call sites.

Conditional memory blocks (TranFn/DiscreteTranFn strictly proper,
StateSpace/DiscreteStateSpace with D=0) live in dedicated helpers because
the per-block decision depends on resolved parameters, not just on the
function name.
"""

from __future__ import annotations
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Unconditional memory blocks
# ---------------------------------------------------------------------------

#: Block functions that are always memory blocks regardless of parameters.
#: They all satisfy: execute() is safe with an empty inputs dict OR the
#: block declares `requires_inputs=False`.
OUTPUT_ONLY_SAFE_BLOCK_FNS: frozenset = frozenset({
    # Core stateful blocks
    'Integrator',
    'StateVariable',
    'TransportDelay',
    'Delay',

    # Optimization primitives (hold previous update / state)
    'Adam',
    'Momentum',

    # Sample-rate adapters (hold previous sample)
    'FirstOrderHold',
    'ZeroOrderHold',
    'RateLimiter',
    'RateTransition',

    # Control blocks hardened to use inputs.get() — see audit priority #7
    'Deriv',
    'Hysteresis',
    'PID',

    # Stochastic sources (no inputs at all)
    'PRBS',
})


# ---------------------------------------------------------------------------
# Conditional memory blocks
# ---------------------------------------------------------------------------


def _params_source(block: Any) -> dict:
    """Prefer resolved exec_params; fall back to raw params for callers
    that haven't run resolve_params yet (e.g. ValidationHelper which is
    invoked during canvas validation)."""
    return getattr(block, 'exec_params', None) or getattr(block, 'params', {}) or {}


def is_strictly_proper_tf(block: Any) -> bool:
    """True iff `block` is a TranFn or DiscreteTranFn with deg(den) > deg(num).

    Strictly-proper TFs have no direct feedthrough — their y[t] depends
    only on past inputs and on internal state, so they are memory blocks.
    Proper TFs (equal degrees) DO have feedthrough and are NOT memory.
    """
    block_fn = getattr(block, 'block_fn', '')
    if block_fn not in ('TranFn', 'DiscreteTranFn'):
        return False
    p = _params_source(block)
    num = p.get('numerator', [])
    den = p.get('denominator', [])
    return len(den) > len(num)


def is_zero_D_statespace(block: Any) -> bool:
    """True iff `block` is a StateSpace/DiscreteStateSpace with D = 0.

    A zero-D state-space realisation has no direct feedthrough u → y,
    so y[t] depends only on x[t] (which is set from past inputs).
    """
    block_fn = getattr(block, 'block_fn', '')
    if block_fn not in ('StateSpace', 'DiscreteStateSpace'):
        return False
    p = _params_source(block)
    D = np.array(p.get('D', [[0.0]]))
    try:
        return bool(np.all(D == 0))
    except (TypeError, ValueError):
        # D might still be a workspace-variable string if no resolution
        # has happened yet — be conservative and treat as non-memory.
        return False


def is_memory_block(block: Any) -> bool:
    """Combined check: block is a memory block by name OR by conditional rule."""
    block_fn = getattr(block, 'block_fn', '')
    if block_fn in OUTPUT_ONLY_SAFE_BLOCK_FNS:
        return True
    if is_strictly_proper_tf(block):
        return True
    if is_zero_D_statespace(block):
        return True
    return False
