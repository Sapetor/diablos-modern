"""
Regression tests for the interpreter's RK45 sub-stepping and for the
``mem``/``output`` aliasing that shifted the in-place integration methods.

Four independent defects, all on the interpreted path (the compiled solver
ignores the Integrator's ``method`` entirely and was never affected):

1. A four-call RK4 cycle advanced the clock by ``2 * sim_dt`` -- sub-step 0
   added a whole step on top of the two halves -- while the integrator advanced
   one ``h``.  Every RK45 trace came out stretched 2x in time: the integral of a
   unit step reached only 0.5 at t=1.0.

2. The ``_skip_`` flag that tells sinks "this is a stage evaluation, not a
   sample" was written to ``block.params``, but ``execute()`` receives
   ``block.exec_params``, which is normally served from cache.  Scope and Export
   never saw it and recorded all four sub-steps: 21 scope samples against a
   6-entry timeline.

3. Memory blocks publish ``params["output"]`` (the step's start state) once per
   call, so all four stages evaluated the derivative at the same state.  With
   K1 = K2 = K3 = K4 the weighted average collapses to K1, making "RK45"
   forward Euler at four times the cost.

4. ``exec_params["output"] = exec_params["mem"]`` bound both names to one array.
   FWD_EULER / BWD_EULER / TUSTIN / RK45 all do ``mem += ...`` in place, so the
   reported output tracked the state instead of lagging it, shifting every
   sample one step early (the integral of a unit step read t + h).  SOLVE_IVP
   rebinds ``mem`` to a fresh array and so never showed it.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _new_dsim():
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    dsim.execution_init_time = lambda: dsim.sim_time
    dsim.pyqtPlotScope = lambda: None
    return dsim, {b.fn_name: b for b in dsim.menu_blocks}


def _drive(dsim, max_calls=4000):
    calls = 0
    while dsim.execution_initialized and calls < max_calls:
        dsim.execution_loop_headless()
        calls += 1
    assert calls < max_calls, "run did not terminate"
    return calls


def _run_ramp(method, sim_time=1.0, sim_dt=0.1):
    """Step -> Integrator -> Scope, i.e. y = integral of a unit step."""
    dsim, menu = _new_dsim()
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    integ = dsim.add_block(menu["integrator"], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    integ.params["init_conds"] = 0.0
    integ.params["method"] = method
    dsim.add_line((step.name, 0, step.out_coords[0]), (integ.name, 0, integ.in_coords[0]))
    dsim.add_line((integ.name, 0, integ.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time, dsim.sim_dt, dsim.plot_trange = sim_time, sim_dt, sim_time
    assert dsim.execution_init() is True
    calls = _drive(dsim)
    return (
        np.ravel(dsim._timeline_list),
        np.ravel(scope.exec_params["vector"]),
        calls,
    )


def _run_decay(method, sim_time=0.5, sim_dt=0.1):
    """Integrator with unit negative feedback: x' = -x, x(0) = 1.

    The feedback is what exposes defect 3 -- with a source-driven integrator
    the stage states never feed back into the derivative, so all four stages
    coincide whether or not they are published.
    """
    dsim, menu = _new_dsim()
    integ = dsim.add_block(menu["integrator"], QPoint(300, 100))
    gain = dsim.add_block(menu["gain"], QPoint(300, 300))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    integ.params["init_conds"] = 1.0
    integ.params["method"] = method
    gain.params["gain"] = -1.0
    dsim.add_line((integ.name, 0, integ.out_coords[0]), (gain.name, 0, gain.in_coords[0]))
    dsim.add_line((gain.name, 0, gain.out_coords[0]), (integ.name, 0, integ.in_coords[0]))
    dsim.add_line((integ.name, 0, integ.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time, dsim.sim_dt, dsim.plot_trange = sim_time, sim_dt, sim_time
    assert dsim.execution_init() is True
    _drive(dsim)
    return np.ravel(dsim._timeline_list), np.ravel(scope.exec_params["vector"])


@pytest.mark.regression
class TestRk45Clock:
    def test_cycle_advances_exactly_one_step(self, qapp):
        """Four sub-steps == one sim_dt of simulated time (was 2 * sim_dt)."""
        timeline, _, calls = _run_ramp("RK45", sim_time=1.0, sim_dt=0.1)
        assert len(timeline) == 11
        assert timeline[-1] == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(np.diff(timeline), 0.1, atol=1e-9)
        # Four calls per grid point, less the stage the init sequence performs.
        assert calls == 40

    def test_rk45_grid_matches_the_other_methods(self, qapp):
        """RK45 must land on the same time grid as a single-call method."""
        t_rk, y_rk, _ = _run_ramp("RK45")
        t_ivp, y_ivp, _ = _run_ramp("SOLVE_IVP")
        assert len(t_rk) == len(t_ivp)
        assert np.allclose(t_rk, t_ivp, atol=1e-9)
        # y = t for the integral of a unit step; the 2x stretch made this 0.5.
        assert y_rk[-1] == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(y_rk, t_rk, atol=1e-9)

    def test_horizon_not_a_multiple_of_the_step(self, qapp):
        timeline, _, _ = _run_ramp("RK45", sim_time=0.5, sim_dt=0.2)
        assert np.allclose(timeline, [0.0, 0.2, 0.4], atol=1e-9)


@pytest.mark.regression
class TestRk45SubStepsAreNotSampled:
    def test_scope_records_one_sample_per_cycle(self, qapp):
        """The _skip_ flag must reach exec_params, the dict execute() gets."""
        timeline, vector, _ = _run_ramp("RK45")
        assert len(vector) == len(timeline), "sinks recorded RK45 stage evaluations"

    def test_skip_flag_reaches_exec_params(self, qapp):
        dsim, menu = _new_dsim()
        step = dsim.add_block(menu["step"], QPoint(100, 100))
        integ = dsim.add_block(menu["integrator"], QPoint(300, 100))
        scope = dsim.add_block(menu["scope"], QPoint(500, 100))
        integ.params["method"] = "RK45"
        integ.params["init_conds"] = 0.0
        dsim.add_line((step.name, 0, step.out_coords[0]), (integ.name, 0, integ.in_coords[0]))
        dsim.add_line((integ.name, 0, integ.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
        dsim.sim_time, dsim.sim_dt, dsim.plot_trange = 1.0, 0.1, 1.0
        assert dsim.execution_init() is True

        seen = set()
        for _ in range(8):
            dsim.execution_loop_headless()
            # Scope clears the flag when it honours it, so a False here means
            # either "not a sub-step" or "sub-step, and the sink consumed it".
            seen.add(dsim.rk_counter % 4)
            assert "_skip_" in scope.exec_params
        assert seen, "loop did not run"


@pytest.mark.regression
class TestRk45IsActuallyRk4:
    def test_matches_textbook_rk4_on_a_feedback_system(self, qapp):
        """x' = -x with the stage states published: true RK4, not Euler."""
        h = 0.1
        expected = [1.0]
        for _ in range(5):
            x = expected[-1]
            k1 = h * (-x)
            k2 = h * (-(x + k1 / 2))
            k3 = h * (-(x + k2 / 2))
            k4 = h * (-(x + k3))
            expected.append(x + (k1 + 2 * k2 + 2 * k3 + k4) / 6)

        _, vector = _run_decay("RK45", sim_time=0.5, sim_dt=h)
        assert len(vector) == len(expected)
        assert np.allclose(vector, expected, atol=1e-12)

    def test_not_degenerate_to_forward_euler(self, qapp):
        """The precise symptom of unpublished stage states."""
        h = 0.1
        euler = [1.0]
        for _ in range(5):
            euler.append(euler[-1] + h * (-euler[-1]))

        _, vector = _run_decay("RK45", sim_time=0.5, sim_dt=h)
        assert not np.allclose(vector, euler, atol=1e-9), (
            "RK45 collapsed to forward Euler -- stage states are not being published"
        )

    def test_fourth_order_convergence(self, qapp):
        """Halving the step must cut the error by roughly 16."""
        errors = []
        for dt in (0.1, 0.05):
            timeline, vector = _run_decay("RK45", sim_time=0.5, sim_dt=dt)
            errors.append(abs(vector[-1] - np.exp(-timeline[-1])))
        ratio = errors[0] / errors[1]
        assert 8.0 < ratio < 32.0, f"expected ~16x error reduction, got {ratio:.1f}x"


@pytest.mark.regression
class TestIntegratorOutputIsNotAliased:
    @pytest.mark.parametrize("method", ["FWD_EULER", "RK45", "SOLVE_IVP"])
    def test_integral_of_a_unit_step_is_t(self, qapp, method):
        """In-place ``mem += ...`` must not mutate the reported output.

        Aliasing made these methods report t + h at every sample.  BWD_EULER
        (t - h, it integrates the *previous* input) and TUSTIN (t - h/2,
        trapezoid) are excluded: those offsets are exact for their schemes,
        not aliasing.
        """
        timeline, vector, _ = _run_ramp(method, sim_time=1.0, sim_dt=0.1)
        assert np.allclose(vector, timeline, atol=1e-9)
        assert vector[-1] == pytest.approx(1.0, abs=1e-9)

    def test_sync_integrator_output_copies(self, qapp):
        """The engine helper must break the reference, not rebind it."""
        from lib.engine.simulation_engine import SimulationEngine

        block = type(
            "B", (), {"block_fn": "Integrator", "exec_params": {"mem": np.array([1.0, 2.0])}}
        )()
        SimulationEngine.sync_integrator_output(block)
        out = block.exec_params["output"]
        assert out is not block.exec_params["mem"]
        block.exec_params["mem"] += 5.0
        assert np.allclose(out, [1.0, 2.0]), "output followed an in-place update to mem"

    def test_sync_integrator_output_ignores_other_blocks(self, qapp):
        from lib.engine.simulation_engine import SimulationEngine

        block = type("B", (), {"block_fn": "Gain", "exec_params": {"mem": np.array([1.0])}})()
        SimulationEngine.sync_integrator_output(block)
        assert "output" not in block.exec_params
