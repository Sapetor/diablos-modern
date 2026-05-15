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

FOLLOW-UP FIX (audit priority #7)
----------------------------------
PID, Hysteresis, and Deriv were hardened to use inputs.get() so they no
longer raise KeyError when called with an empty inputs dict.  They are now
also registered in identify_memory_blocks().

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
    # Audit priority #7: Control blocks hardened to use inputs.get()
    "Deriv",
    "Hysteresis",
    "PID",
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


# ---------------------------------------------------------------------------
# Audit priority #7: empty-inputs safety for hardened Control blocks
# ---------------------------------------------------------------------------

@pytest.mark.regression
class TestHardenedBlockEmptyInputs:
    """
    Verify PID, Hysteresis, and Deriv do not raise when called with an empty
    inputs dict (simulating the output_only=True path used by Loop 1).
    """

    def test_pid_safe_on_empty_inputs(self):
        """PID with empty inputs returns current integral (should be ~0 on first call)."""
        import numpy as np
        from blocks.pid import PIDBlock
        block = PIDBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        result = block.execute(time=0.1, inputs={}, params=params, dtime=0.01)
        assert isinstance(result, dict), "PID must return a dict"
        assert 0 in result, "PID must return output port 0"
        assert float(result[0]) == pytest.approx(0.0, abs=1e-9), (
            "PID with no accumulated error should output ~0 on first call"
        )

    def test_pid_output_only_returns_held_integral(self):
        """PID output_only after one normal step returns the held integral, not a crash."""
        import numpy as np
        from blocks.pid import PIDBlock
        block = PIDBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        params["Ki"] = 1.0
        # Normal step: setpoint=1, measurement=0 → error=1, integral accumulates
        block.execute(time=0.0, inputs={0: np.array([1.0]), 1: np.array([0.0])},
                      params=params, dtime=0.01)
        # output_only step: empty inputs — must not crash
        result = block.execute(time=0.01, inputs={}, params=params, dtime=0.01)
        assert isinstance(result, dict)
        assert 0 in result

    def test_hysteresis_safe_on_empty_inputs(self):
        """Hysteresis with empty inputs returns the held _state (low by default)."""
        import numpy as np
        from blocks.hysteresis import HysteresisBlock
        block = HysteresisBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        result = block.execute(time=0.0, inputs={}, params=params)
        assert isinstance(result, dict), "Hysteresis must return a dict"
        assert 0 in result, "Hysteresis must return output port 0"
        # input=0.0 is between lower=-0.5 and upper=0.5, so state = low = 0.0
        assert float(result[0]) == pytest.approx(float(params["low"]), abs=1e-9)

    def test_hysteresis_output_only_returns_held_state(self):
        """Hysteresis output_only after a switch returns the switched state."""
        import numpy as np
        from blocks.hysteresis import HysteresisBlock
        block = HysteresisBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        # Drive input above upper threshold → state switches to high
        block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params)
        # output_only path: empty inputs — must return held high state
        result = block.execute(time=0.01, inputs={}, params=params)
        assert isinstance(result, dict)
        assert float(result[0]) == pytest.approx(float(params["high"]), abs=1e-9)

    def test_derivative_safe_on_empty_inputs(self):
        """Deriv with empty inputs on first call returns zeros (no crash)."""
        import numpy as np
        from blocks.derivative import DerivativeBlock
        block = DerivativeBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        params["_init_start_"] = True
        result = block.execute(time=0.0, inputs={}, params=params)
        assert isinstance(result, dict), "Deriv must return a dict"
        assert 0 in result, "Deriv must return output port 0"
        assert np.allclose(result[0], 0.0), "Deriv first-call output must be zero"

    def test_derivative_output_only_returns_last_derivative(self):
        """Deriv output_only after a normal step returns the last computed didt."""
        import numpy as np
        from blocks.derivative import DerivativeBlock
        block = DerivativeBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        params["_init_start_"] = True
        # Step 1 (init)
        block.execute(time=0.0, inputs={0: np.array([0.0])}, params=params)
        # Step 2: input rises to 1.0 over dt=0.1 → didt = 10.0
        r2 = block.execute(time=0.1, inputs={0: np.array([1.0])}, params=params)
        expected_didt = r2[0].copy()
        # Step 3: output_only (empty inputs) — must return held didt, not recompute
        r3 = block.execute(time=0.2, inputs={}, params=params)
        assert isinstance(r3, dict)
        assert np.allclose(r3[0], expected_didt), (
            "Deriv output_only must return last computed derivative, not recompute with zero input"
        )

    def test_pid_output_only_does_not_corrupt_state(self):
        """Calling PID with empty inputs (Loop 1 output_only) must NOT mutate
        _prev_e or _d_state. Otherwise the next real step's derivative term
        is computed from corrupted state."""
        import numpy as np
        from blocks.pid import PIDBlock
        block = PIDBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        params['Kp'] = 1.0
        params['Ki'] = 0.5
        params['Kd'] = 1.0  # nonzero — exposes the bug

        # Run a real step with setpoint=1, measurement=0 → error=1
        real_inputs = {0: np.array([1.0]), 1: np.array([0.0])}
        block.execute(time=0.0, inputs=real_inputs, params=params, dtime=0.01)

        # Capture state after real step
        prev_e_after_real = params.get('_prev_e')
        d_state_after_real = params.get('_d_state')

        # Now simulate Loop 1 output_only call (empty inputs)
        block.execute(time=0.01, inputs={}, params=params, dtime=0.01)

        # State must be unchanged
        assert params.get('_prev_e') == prev_e_after_real, (
            f"_prev_e was corrupted by output_only call: "
            f"{prev_e_after_real} -> {params.get('_prev_e')}"
        )
        assert params.get('_d_state') == d_state_after_real, (
            f"_d_state was corrupted: {d_state_after_real} -> {params.get('_d_state')}"
        )

    def test_hysteresis_output_only_does_not_flip_state(self):
        """With non-symmetric thresholds (both above 0), the default 0.0 input
        must not spuriously trigger a transition. Empty inputs → return current state."""
        import numpy as np
        from blocks.hysteresis import HysteresisBlock
        block = HysteresisBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        # Configure so 0.0 input would naively cross the upper threshold
        params['upper'] = -0.05
        params['lower'] = -0.5
        params['low'] = 0.0
        params['high'] = 1.0
        params['_state'] = 0.0   # currently in low state
        params['_init_start_'] = False  # already initialized

        # Output-only call with empty inputs — must NOT flip to high
        result = block.execute(time=0.0, inputs={}, params=params, dtime=0.01)
        assert params['_state'] == 0.0, "Hysteresis state was spuriously flipped"
        assert float(result[0]) == pytest.approx(0.0), (
            "Hysteresis output_only must return current state, not flip"
        )

    def test_adam_output_only_does_not_corrupt_moments(self):
        """Adam with empty inputs must not increment _t_ or mutate _m_/_v_."""
        import numpy as np
        from blocks.optimization_primitives.adam import AdamBlock
        block = AdamBlock()
        params = {k: v["default"] if isinstance(v, dict) else v
                  for k, v in block.params.items()}

        # Run a real step
        grad = np.array([1.0, 0.5])
        block.execute(time=0.0, inputs={0: grad}, params=params)

        m_after_real = params['_m_'].copy()
        v_after_real = params['_v_'].copy()
        t_after_real = params['_t_']

        # Output-only call
        block.execute(time=0.01, inputs={}, params=params)

        assert np.allclose(params['_m_'], m_after_real), "_m_ was corrupted by output_only call"
        assert np.allclose(params['_v_'], v_after_real), "_v_ was corrupted by output_only call"
        assert params['_t_'] == t_after_real, "_t_ was incremented by output_only call"

    def test_momentum_output_only_does_not_corrupt_velocity(self):
        """Momentum with empty inputs must not mutate _velocity_."""
        import numpy as np
        from blocks.optimization_primitives.momentum import MomentumBlock
        block = MomentumBlock()
        params = {k: v["default"] if isinstance(v, dict) else v
                  for k, v in block.params.items()}

        # Run a real step
        grad = np.array([1.0, 0.5])
        block.execute(time=0.0, inputs={0: grad}, params=params)

        velocity_after_real = params['_velocity_'].copy()

        # Output-only call
        block.execute(time=0.01, inputs={}, params=params)

        assert np.allclose(params['_velocity_'], velocity_after_real), (
            "_velocity_ was corrupted by output_only call"
        )

    def test_rate_limiter_output_only_does_not_corrupt_prev(self):
        """RateLimiter with empty inputs must not overwrite _prev."""
        import numpy as np
        from blocks.rate_limiter import RateLimiterBlock
        block = RateLimiterBlock()
        params = {k: (v["default"] if isinstance(v, dict) else v)
                  for k, v in block.params.items()}

        # Run a real step to initialize
        block.execute(time=0.0, inputs={0: np.array([2.0])}, params=params, dtime=0.01)

        prev_after_real = np.array(params['_prev']).copy()

        # Output-only call with empty inputs
        block.execute(time=0.01, inputs={}, params=params, dtime=0.01)

        assert np.allclose(params['_prev'], prev_after_real), (
            "_prev was corrupted by output_only call"
        )
