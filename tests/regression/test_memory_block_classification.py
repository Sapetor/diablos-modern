"""
Regression tests for algebraic-loop detector memory block classification.

BUG DESCRIPTION
---------------
identify_memory_blocks() in SimulationEngine missed several stateful blocks
that are safe to classify as memory (i.e. can be called with output_only=True
during Loop 1 initialisation without crashing):
  ZeroOrderHold, FirstOrderHold, RateTransition, Adam, Momentum, RateLimiter.

These blocks use inputs.get(port, default) so they return a sensible held
value when the engine calls them with an empty inputs dict during Loop 1.
When they appeared in a feedback loop the algebraic-loop detector therefore
raised a spurious "algebraic loop detected" error.

OUT-OF-SCOPE (follow-up task)
------------------------------
PID, Hysteresis, and Deriv also hold cross-step state and break algebraic
loops, but their execute() accesses inputs[key] directly and would raise
KeyError if called with output_only=True.  They must not be added to
memory_blocks until execute() is hardened.  They are covered by the
separate MEMORY_BLOCK_TYPES set in lib/improvements.py (topological-sort
loop detector) which does not call execute().

NOTE on block_fn strings
------------------------
The block_fn used by the engine is the *block_name* property (the string
returned by the block class), not the Python file name:
  - blocks/derivative.py                → block_name == "Deriv"
  - blocks/discrete_transfer_function.py → block_name == "DiscreteTranFn"

NOTE on conditional classifications
-------------------------------------
DiscreteTranFn: classified as memory only when strictly-proper (len(den) >
len(num)), i.e. D=0.  Proper TFs need their current input for correct t=0
output; classifying them as memory would give wrong initialisation.

StateSpace / DiscreteStateSpace: classified as memory only when D=0 (same
reason as above).
"""

import pytest
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def engine(qapp, simulation_model):
    """SimulationEngine with empty model — uses shared conftest fixtures."""
    from lib.engine.simulation_engine import SimulationEngine
    return SimulationEngine(simulation_model)


# ---------------------------------------------------------------------------
# Unconditionally-stateful blocks safe for output_only=True
# ---------------------------------------------------------------------------

UNCONDITIONAL_STATEFUL_BLOCK_FNS = [
    # Core integrating blocks (sanity checks — were always classified)
    "Integrator",
    "StateVariable",
    "TransportDelay",
    # Newly fixed: stateful blocks that hold a previous output (safe with
    # output_only=True because they use inputs.get() with a default)
    "ZeroOrderHold",
    "FirstOrderHold",
    "RateTransition",
    "RateLimiter",
    "Adam",
    "Momentum",
]


def _make_stub(block_fn: str, extra_params: dict = None) -> SimpleNamespace:
    """Minimal stub that identify_memory_blocks() can inspect."""
    params = {}
    if extra_params:
        params.update(extra_params)
    return SimpleNamespace(
        name=f"{block_fn.lower()}__test",
        block_fn=block_fn,
        params=params,
        exec_params={},
        b_type=2,           # regular block
        block_class=None,   # disable the block_class instantiation path
    )


@pytest.mark.regression
class TestMemoryBlockClassification:

    @pytest.mark.parametrize("block_fn", UNCONDITIONAL_STATEFUL_BLOCK_FNS)
    def test_stateful_block_classified_as_memory(self, engine, block_fn):
        """Every unconditionally-stateful block must land in engine.memory_blocks."""
        stub = _make_stub(block_fn)
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name in engine.memory_blocks, (
            f"'{block_fn}' holds cross-step state and is safe for "
            f"output_only=True, but was NOT classified as a memory block. "
            f"The algebraic-loop detector will fire spuriously on feedback "
            f"loops through this block."
        )

    # ------------------------------------------------------------------
    # TranFn — conditional on strict properness
    # ------------------------------------------------------------------

    def test_tranfn_strictly_proper_is_memory(self, engine):
        """TranFn with len(den) > len(num) must be classified as memory."""
        stub = _make_stub("TranFn", extra_params={
            "numerator": [1.0],
            "denominator": [1.0, 1.0],
        })
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name in engine.memory_blocks, (
            "TranFn strictly-proper (len(den) > len(num)) was not classified "
            "as a memory block."
        )

    def test_tranfn_non_strictly_proper_not_memory(self, engine):
        """TranFn with len(den) == len(num) must NOT be classified as memory."""
        stub = _make_stub("TranFn", extra_params={
            "numerator": [1.0, 2.0],
            "denominator": [1.0, 1.0],
        })
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name not in engine.memory_blocks, (
            "TranFn with equal-degree num/den was wrongly classified as memory."
        )

    # ------------------------------------------------------------------
    # DiscreteTranFn — conditional: strictly-proper only
    # ------------------------------------------------------------------

    def test_discrete_tranfn_strictly_proper_is_memory(self, engine):
        """Strictly-proper DiscreteTranFn (len(den) > len(num)) must be memory."""
        stub = _make_stub("DiscreteTranFn", extra_params={
            "numerator": [1.0],
            "denominator": [1.0, 1.0],
        })
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name in engine.memory_blocks, (
            "Strictly-proper DiscreteTranFn was not classified as memory."
        )

    def test_discrete_tranfn_proper_not_memory(self, engine):
        """Proper DiscreteTranFn (len(den) == len(num)) must NOT be memory.

        A proper discrete TF has D!=0; classifying it as memory causes the
        engine to call execute with output_only=True (u forced to 0), which
        gives the wrong initial output when the real input is non-zero.
        """
        stub = _make_stub("DiscreteTranFn", extra_params={
            "numerator": [1.0, 0.5],
            "denominator": [1.0, 0.2],
        })
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name not in engine.memory_blocks, (
            "Proper DiscreteTranFn (D!=0) was wrongly classified as memory; "
            "this would cause a wrong initial output when input is non-zero."
        )

    # ------------------------------------------------------------------
    # StateSpace / DiscreteStateSpace — conditional on D==0
    # ------------------------------------------------------------------

    def test_statespace_zero_D_is_memory(self, engine):
        """StateSpace with D=0 must be classified as memory."""
        stub = _make_stub("StateSpace", extra_params={"D": [[0.0]]})
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name in engine.memory_blocks, (
            "StateSpace with D=0 was not classified as a memory block."
        )

    def test_discrete_statespace_zero_D_is_memory(self, engine):
        """DiscreteStateSpace with D=0 must be classified as memory."""
        stub = _make_stub("DiscreteStateSpace", extra_params={"D": [[0.0]]})
        engine.active_blocks_list = [stub]
        engine.identify_memory_blocks()
        assert stub.name in engine.memory_blocks, (
            "DiscreteStateSpace with D=0 was not classified as a memory block."
        )

    # ------------------------------------------------------------------
    # Combined: all unconditional blocks classified at once
    # ------------------------------------------------------------------

    def test_multiple_blocks_all_classified(self, engine):
        """All unconditionally-stateful blocks in one list are classified as memory."""
        stubs = [_make_stub(fn) for fn in UNCONDITIONAL_STATEFUL_BLOCK_FNS]
        engine.active_blocks_list = stubs
        engine.identify_memory_blocks()
        missed = [s.block_fn for s in stubs if s.name not in engine.memory_blocks]
        assert not missed, (
            f"The following stateful blocks were NOT classified as memory: "
            f"{missed}"
        )
