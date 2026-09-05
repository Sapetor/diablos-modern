"""
Unit tests for Integrator block.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestIntegratorBlock:
    """Tests for Integrator block."""

    def test_integrator_initial_condition(self):
        """Test integrator outputs initial condition at t=0."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        params = {"init_conds": 5.0, "method": "FWD_EULER", "_init_start_": True}

        # First call with output_only to get initial value
        result = block.execute(
            time=0.0, inputs={0: np.array([0.0])}, params=params, output_only=True
        )
        assert np.isclose(result[0][0], 5.0), "Initial output should equal init_conds"

    def test_integrator_fwd_euler(self):
        """Test forward Euler integration accumulates correctly."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 0.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # First call initializes and integrates: 0 + 0.1*1 = 0.1
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        assert np.isclose(result[0][0], 0.1, atol=0.01), (
            f"FWD_EULER first step: expected ~0.1, got {result[0][0]}"
        )

    def test_integrator_fwd_euler_accumulation(self):
        """Test forward Euler accumulates over multiple steps."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 0.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # Run 10 steps with constant input of 1.0
        for i in range(10):
            result = block.execute(
                time=i * dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime
            )

        # After 10 steps: 10 * 0.1 * 1.0 = 1.0
        assert np.isclose(result[0][0], 1.0, atol=0.1), (
            f"Accumulated integral should be ~1.0, got {result[0][0]}"
        )

    def test_integrator_solve_ivp(self):
        """Test SOLVE_IVP integration method."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 0.0,
            "method": "SOLVE_IVP",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # First call initializes and integrates
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        # After one step integrating constant 1.0 over dtime=0.1, should be ~0.1
        assert np.isclose(result[0][0], 0.1, atol=0.02), (
            f"SOLVE_IVP: expected ~0.1, got {result[0][0]}"
        )

    def test_integrator_solve_ivp_accumulation(self):
        """Test SOLVE_IVP accumulates over multiple steps."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 0.0,
            "method": "SOLVE_IVP",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # Run 10 steps with constant input of 1.0
        for i in range(10):
            result = block.execute(
                time=i * dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime
            )

        # After 10 steps: integral of 1.0 over 1.0s should be ~1.0
        assert np.isclose(result[0][0], 1.0, atol=0.1), (
            f"SOLVE_IVP accumulated: expected ~1.0, got {result[0][0]}"
        )

    def test_integrator_vector_input(self):
        """Test integrator handles vector inputs."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 0.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        vec_input = np.array([1.0, 2.0, 3.0])

        # Execute - will expand init_conds to match input dimensions
        result = block.execute(time=0.0, inputs={0: vec_input}, params=params, dtime=dtime)

        assert result[0].shape == vec_input.shape, "Output shape should match input"
        # After one step: dtime * [1, 2, 3] = [0.1, 0.2, 0.3]
        expected = dtime * vec_input
        assert np.allclose(result[0], expected, atol=0.01), f"Expected {expected}, got {result[0]}"

    def test_integrator_tustin(self):
        """Test Tustin (trapezoidal) integration."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {"init_conds": 0.0, "method": "TUSTIN", "_init_start_": True, "_name_": "TestInt"}

        # First call initializes with mem_list containing zeros
        # Tustin: x_new = x_old + 0.5 * dtime * (u_old + u_new)
        # First step: u_old from mem_list = 0, u_new = 1.0
        # x = 0 + 0.5 * 0.1 * (0 + 1) = 0.05
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)

        # Second step: u_old = 1.0, u_new = 1.0
        # x = 0.05 + 0.5 * 0.1 * (1 + 1) = 0.05 + 0.1 = 0.15
        result = block.execute(time=dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        assert np.isclose(result[0][0], 0.15, atol=0.02), (
            f"TUSTIN: expected ~0.15, got {result[0][0]}"
        )

    def test_integrator_negative_input(self):
        """Test integrator with negative input values."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 5.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # Run 10 steps with constant negative input
        for i in range(10):
            result = block.execute(
                time=i * dtime, inputs={0: np.array([-1.0])}, params=params, dtime=dtime
            )

        # Should decrease from 5.0 by 10 * 0.1 * 1.0 = 1.0, so result is 4.0
        expected = 5.0 - 10 * dtime * 1.0
        assert np.isclose(result[0][0], expected, atol=0.1), (
            f"Expected {expected}, got {result[0][0]}"
        )

    def test_integrator_zero_input(self):
        """Test integrator with zero input stays at initial condition."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 3.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # Run 5 steps with zero input
        for i in range(5):
            result = block.execute(
                time=i * dtime, inputs={0: np.array([0.0])}, params=params, dtime=dtime
            )

        # Should stay at initial condition
        assert np.isclose(result[0][0], 3.0, atol=0.01), f"Expected 3.0, got {result[0][0]}"

    def test_integrator_output_only_mode(self):
        """Test output_only mode returns current value without integrating."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        dtime = 0.1
        params = {
            "init_conds": 2.0,
            "method": "FWD_EULER",
            "_init_start_": True,
            "_name_": "TestInt",
        }

        # Initialize
        block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=dtime)
        val_after_init = params["mem"][0]

        # Call output_only - should not change mem
        result = block.execute(
            time=dtime, inputs={0: np.array([1.0])}, params=params, dtime=dtime, output_only=True
        )
        val_after_output_only = params["mem"][0]

        # mem should be unchanged after output_only call
        assert val_after_output_only == val_after_init, (
            "output_only should not change internal state"
        )


@pytest.mark.unit
class TestIntegratorMethodNames:
    """The fixed-step 4-stage strategy is called "RK4", not "RK45".

    It is classical Runge-Kutta, not scipy's adaptive RK4(5), and it shipped
    mislabelled for years -- so "RK45" survives as a *legacy alias* that saved
    .diablos files still hold. Both spellings must select the same path, in the
    block and in ``SimulationEngine.count_rk45_integrators`` (which enables the
    interpreter's four sub-steps by string-matching the resolved name).
    """

    def test_option_list_offers_the_corrected_spelling(self):
        from blocks.integrator import INTEGRATOR_METHODS

        assert "RK4" in INTEGRATOR_METHODS
        assert "RK45" not in INTEGRATOR_METHODS

    def test_legacy_rk45_resolves_to_the_rk4_strategy(self):
        from blocks.integrator import resolve_method

        assert resolve_method("RK45") == "RK4"
        assert resolve_method("RK4") == "RK4"
        assert resolve_method("FWD_EULER") == "FWD_EULER"

    def test_every_offered_option_is_its_own_canonical_name(self):
        """An option the dropdown offers must not itself be an alias, or the
        saved value and the strategy actually run would drift apart."""
        from blocks.integrator import INTEGRATOR_METHODS, METHOD_ALIASES, resolve_method

        for option in INTEGRATOR_METHODS:
            assert resolve_method(option) == option

        # Every alias must land on something execute() actually implements.
        for legacy, canonical in METHOD_ALIASES.items():
            assert legacy not in INTEGRATOR_METHODS
            assert canonical in INTEGRATOR_METHODS

    def test_rk4_takes_the_four_stage_path(self):
        """ "RK4" must set up the sub-step state for the four-stage schedule."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        params = {"init_conds": 0.0, "method": "RK4", "_init_start_": True, "_name_": "I"}

        result = block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.1)

        assert params["nb_loop"] == 1  # stage 1 of 4, not a completed step
        assert 0 not in result  # stages 0-2 publish nothing
        assert np.isclose(params["aux"][0], 0.05)  # x_n + K1/2

    def test_rk4_and_legacy_rk45_produce_identical_state_over_a_full_cycle(self):
        from blocks.integrator import IntegratorBlock

        def run(method):
            block = IntegratorBlock()
            params = {"init_conds": 0.0, "method": method, "_init_start_": True, "_name_": "I"}
            for _ in range(4):
                block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.1)
            return float(params["mem"][0])

        assert np.isclose(run("RK4"), run("RK45"))

    def test_legacy_rk45_completes_a_step_after_four_stages(self):
        """A diagram saved before the rename still runs the full RK4 cycle."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        params = {"init_conds": 0.0, "method": "RK45", "_init_start_": True, "_name_": "I"}

        for _ in range(4):
            result = block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.1)

        # RK4 on dy/dt = 1 over h = 0.1: (K1 + 2K2 + 2K3 + K4)/6 = h.
        assert np.isclose(result[0][0], 0.1)
        assert params["nb_loop"] == 0

    def test_unknown_method_returns_an_error_dict(self):
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        params = {"init_conds": 0.0, "method": "MAGIC", "_init_start_": True, "_name_": "I"}

        result = block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.1)

        assert result.get("E") is True
        assert "MAGIC" in result["error"]


@pytest.mark.unit
class TestIntegratorEulerVariants:
    """Pin what BWD_EULER actually computes.

    It is NOT implicit (backward) Euler: a plain integrator cannot form
    ``y[k+1] = y[k] + h*u[k+1]`` on the interpreted path, because ``u`` at the new
    time is produced by the rest of the diagram from the new state -- only the
    compiled solver solves the whole system simultaneously. What the block does
    is explicit Euler on the *previous* input sample, i.e. one extra step of lag.
    The block doc says so; these tests keep the doc honest.
    """

    @staticmethod
    def _run(method, inputs, dtime=0.1):
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        params = {"init_conds": 0.0, "method": method, "_init_start_": True, "_name_": "I"}
        out = []
        for k, u in enumerate(inputs):
            out.append(
                float(block.execute(k * dtime, {0: np.array([u])}, params, dtime=dtime)[0][0])
            )
        return out

    def test_fwd_euler_uses_the_current_input_sample(self):
        u = [1.0, 2.0, 3.0]
        assert self._run("FWD_EULER", u) == pytest.approx([0.1, 0.3, 0.6])

    def test_bwd_euler_uses_the_previous_input_sample(self):
        u = [1.0, 2.0, 3.0]
        # y[k+1] = y[k] + h*u[k-1], with u[-1] taken as 0 by the seeded mem_list.
        assert self._run("BWD_EULER", u) == pytest.approx([0.0, 0.1, 0.3])

    def test_bwd_euler_is_not_implicit_euler(self):
        """Implicit Euler on dy/dt = u would give h*(u[0]+...+u[k])."""
        u = [1.0, 2.0, 3.0]
        implicit = [0.1, 0.3, 0.6]
        assert self._run("BWD_EULER", u) != pytest.approx(implicit)

    def test_tustin_is_the_trapezoidal_rule(self):
        u = [1.0, 2.0, 3.0]
        # 0.5*h*(u[k] + u[k-1]) accumulated, with u[-1] = 0.
        assert self._run("TUSTIN", u) == pytest.approx([0.05, 0.2, 0.45])

    def test_the_documented_behaviour_is_stated_in_the_block_doc(self):
        from blocks.integrator import IntegratorBlock

        doc = IntegratorBlock().doc
        assert "NOT" in doc and "implicit Euler" in doc
        assert "u[k-1]" in doc
