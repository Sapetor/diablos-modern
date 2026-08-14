"""
Regression tests for sampling a *continuous* block (sampling_time > 0).

BUG DESCRIPTION
---------------
``Integrator``, ``TranFn``, ``StateSpace`` and ``PID`` all expose a
``sampling_time`` parameter, and the engine gates them correctly: with
``Ts > 0`` the block is executed only every Ts seconds and holds its output
in between (``DBlock.should_execute``).  But the step each block *integrated
with* came from ``exec_params['dtime']``, which was unconditionally stamped
with the global ``sim_dt``.  A block therefore advanced by ``dt`` once every
``Ts`` seconds and ran ``Ts/dt`` times too slowly:

    Step -> TranFn 1/(s+1) -> Scope, sim_dt=0.01, Ts=0.5
        y(3) = 0.058   (expected 0.950)
    Step -> Integrator -> Scope, sim_dt=0.01, Ts=0.5
        y(3) = 0.070   (expected 3.0)

Worse, the compiled fast solver (the default) ignored ``sampling_time``
entirely, so the same diagram produced a *continuous* answer on the fast path
and the wrong sampled answer on the interpreter path.

A third, related defect surfaced while pinning the PID numbers: ``PID`` is a
memory block, so the simulation loop's first pass calls it with
``output_only=True``, but ``PIDBlock.execute`` only checked for *missing*
inputs.  With ``input_queue`` still holding the previous step's values it
integrated twice per timestep, doubling the Ki contribution.

THE FIX
-------
- ``DBlock.execution_step(sim_dt)`` returns Ts for discrete-rate blocks and
  ``sim_dt`` otherwise; it is stamped into ``exec_params['dtime']`` (in
  ``_resolve_block_params`` and re-stamped at the end of
  ``propagate_sample_times`` once inherited rates are resolved).
- ``SystemCompiler.check_compilability`` rejects any block with a declared
  sample time > 0, so sampled diagrams fall back to the interpreter, which
  honours the rate.
- ``PIDBlock.execute`` honours ``output_only``.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _new_dsim(sim_time, sim_dt):
    from lib.lib import DSim

    dsim = DSim()
    # The headless harness pokes ``buttons_list[6].active`` after init.
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    dsim.sim_time = sim_time
    dsim.sim_dt = sim_dt
    dsim.plot_trange = sim_time
    dsim.execution_init_time = lambda: dsim.sim_time
    return dsim


def _build_chain(fn_name, params, sim_time=3.0, sim_dt=0.01):
    """Step -> <block> -> Scope."""
    dsim = _new_dsim(sim_time, sim_dt)
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    blk = dsim.add_block(menu[fn_name], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    blk.params.update(params)
    dsim.add_line((step.name, 0, step.out_coords[0]), (blk.name, 0, blk.in_coords[0]))
    dsim.add_line((blk.name, 0, blk.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    return dsim, blk, scope


def _run_interpreted(dsim, scope):
    # Drive on the completion flag, not on a time comparison: the loop ends
    # at the horizon (see tests/regression/test_simulation_horizon.py), and
    # stepping past that point appends samples on top of blocks that
    # reset_memblocks() has already re-armed.
    assert dsim.execution_init() is True
    while dsim.execution_initialized:
        dsim.execution_loop_headless()
    return np.asarray(scope.exec_params["vector"]).flatten()


@pytest.mark.regression
class TestSampledContinuousBlocks:
    def test_sampled_transfer_function_matches_zoh_response(self, qapp):
        """1/(s+1) sampled at Ts=0.5 must still reach the true step response.

        The bug produced 0.058 at t=3 (50x too slow) because the block
        advanced by sim_dt=0.01 once per 0.5s sample.
        """
        dsim, _, scope = _build_chain(
            "transfer_function",
            {"numerator": [1.0], "denominator": [1.0, 1.0], "sampling_time": 0.5},
        )
        vec = _run_interpreted(dsim, scope)
        exact = 1.0 - np.exp(-3.0)
        assert vec[-1] == pytest.approx(exact, abs=1e-3), (
            f"Sampled TranFn step response should reach {exact:.4f}, got {vec[-1]:.4f}"
        )

    def test_sampled_integrator_uses_sample_period_as_step(self, qapp):
        """Integrating a unit step at Ts=0.5 must accumulate 0.5 per sample.

        The bug accumulated sim_dt=0.01 per sample (y(3)=0.07).
        """
        dsim, _, scope = _build_chain("integrator", {"init_conds": 0.0, "sampling_time": 0.5})
        vec = _run_interpreted(dsim, scope)
        assert vec[-1] == pytest.approx(3.0, abs=1e-9), (
            f"Sampled integrator should accumulate 0.5/sample, got {vec[-1]:.4f}"
        )

    def test_dtime_is_stamped_with_the_sample_period(self, qapp):
        """exec_params['dtime'] must carry the block's own execution step."""
        dsim, blk, _ = _build_chain("integrator", {"init_conds": 0.0, "sampling_time": 0.5})
        assert dsim.execution_init() is True
        active = {b.name: b for b in dsim.engine.active_blocks_list}
        integrator = active[blk.name]
        assert integrator.effective_sample_time == 0.5
        assert integrator.exec_params["dtime"] == 0.5
        # Continuous blocks keep the base step.
        step_block = next(b for b in dsim.engine.active_blocks_list if b.block_fn == "Step")
        assert step_block.exec_params["dtime"] == dsim.sim_dt

    def test_execution_step_helper(self, qapp, sample_block):
        """DBlock.execution_step: Ts when gated, sim_dt when continuous."""
        sample_block.effective_sample_time = -1.0
        assert sample_block.execution_step(0.01) == 0.01
        sample_block.effective_sample_time = 0.0  # unresolved 'inherit'
        assert sample_block.execution_step(0.01) == 0.01
        sample_block.effective_sample_time = 0.25
        assert sample_block.execution_step(0.01) == 0.25


@pytest.mark.regression
class TestSampledStaircaseAlignment:
    """A gated block must emit y[k] (computed from x[k]) across the whole
    interval [kTs, (k+1)Ts).

    The Integrator returns its *post-update* state, so holding the value its
    state-updating execute returned made the staircase step a full sample
    period early: the integral of a unit step read 0.5 at t=0.49, before the
    first sample instant had even been reached.
    """

    def _trace(self, params, sim_time=3.0, sim_dt=0.01):
        dsim, _, scope = _build_chain("integrator", params, sim_time=sim_time, sim_dt=sim_dt)
        vec = _run_interpreted(dsim, scope)
        t = np.asarray(dsim._timeline_list).flatten()
        n = min(len(t), len(vec))
        return t[:n], vec[:n]

    def _at(self, t, y, when):
        return y[int(np.argmin(np.abs(t - when)))]

    def test_staircase_steps_at_sample_instants(self, qapp):
        t, y = self._trace({"init_conds": 0.0, "sampling_time": 0.5})
        # Held across the first interval, *not* stepped early.
        assert self._at(t, y, 0.0) == pytest.approx(0.0, abs=1e-12)
        assert self._at(t, y, 0.25) == pytest.approx(0.0, abs=1e-12)
        assert self._at(t, y, 0.49) == pytest.approx(0.0, abs=1e-12)
        # Steps exactly at the sample instants.
        assert self._at(t, y, 0.5) == pytest.approx(0.5, abs=1e-12)
        assert self._at(t, y, 0.99) == pytest.approx(0.5, abs=1e-12)
        assert self._at(t, y, 1.0) == pytest.approx(1.0, abs=1e-12)
        assert self._at(t, y, 1.5) == pytest.approx(1.5, abs=1e-12)

    def test_initial_condition_is_held_over_the_first_interval(self, qapp):
        t, y = self._trace({"init_conds": 2.0, "sampling_time": 0.5})
        assert self._at(t, y, 0.0) == pytest.approx(2.0, abs=1e-12)
        assert self._at(t, y, 0.49) == pytest.approx(2.0, abs=1e-12)
        assert self._at(t, y, 0.5) == pytest.approx(2.5, abs=1e-12)

    def test_continuous_integrator_trace_is_unchanged(self, qapp):
        """Guard: the continuous integrator must still track y(t) = t exactly."""
        t, y = self._trace({"init_conds": 0.0, "sampling_time": -1.0})
        for when in (0.0, 0.5, 1.0, 2.0, 3.0):
            assert self._at(t, y, when) == pytest.approx(when, abs=1e-9)

    def test_zero_D_statespace_alignment_is_unchanged(self, qapp):
        """Guard: StateSpace computes y from the pre-update state and was
        already aligned; it must stay that way."""
        dsim, _, scope = _build_chain(
            "statespace",
            {
                "A": [[-1.0]],
                "B": [[1.0]],
                "C": [[1.0]],
                "D": [[0.0]],
                "init_conds": [0.0],
                "sampling_time": 0.5,
            },
        )
        vec = _run_interpreted(dsim, scope)
        t = np.asarray(dsim._timeline_list).flatten()
        n = min(len(t), len(vec))
        t, vec = t[:n], vec[:n]
        assert self._at(t, vec, 0.49) == pytest.approx(0.0, abs=1e-12)
        assert self._at(t, vec, 0.5) == pytest.approx(1 - np.exp(-0.5), abs=1e-9)
        assert self._at(t, vec, 0.99) == pytest.approx(1 - np.exp(-0.5), abs=1e-9)
        assert self._at(t, vec, 1.0) == pytest.approx(1 - np.exp(-1.0), abs=1e-9)

    def test_output_is_post_update_flag(self):
        """Only blocks returning the advanced state declare the flag."""
        from blocks.integrator import IntegratorBlock
        from blocks.statespace import StateSpaceBlock
        from blocks.zero_order_hold import ZeroOrderHoldBlock

        assert IntegratorBlock().output_is_post_update is True
        assert StateSpaceBlock().output_is_post_update is False
        assert ZeroOrderHoldBlock().output_is_post_update is False


@pytest.mark.regression
class TestSampledBlocksAreNotCompiled:
    """The compiled path has no notion of sample instants, so a diagram with a
    discrete rate must fall back to the interpreter instead of silently
    simulating it as continuous."""

    def _tf_diagram(self, sampling_time):
        return _build_chain(
            "transfer_function",
            {"numerator": [1.0], "denominator": [1.0, 1.0], "sampling_time": sampling_time},
        )

    def test_continuous_diagram_is_still_compilable(self, qapp):
        dsim, _, _ = self._tf_diagram(-1.0)
        assert dsim.engine.check_compilability(dsim.blocks_list) is True

    def test_sampled_diagram_is_not_compilable(self, qapp):
        dsim, _, _ = self._tf_diagram(0.5)
        assert dsim.engine.check_compilability(dsim.blocks_list) is False

    def test_batch_run_falls_back_to_interpreter(self, qapp):
        """With the fast solver enabled, a sampled diagram must still be run by
        the interpreter — and give the correct answer."""
        dsim, _, scope = self._tf_diagram(0.5)
        dsim.use_fast_solver = True
        # execution_batch's interpreter fallback runs the *interactive* loop,
        # which would pop a SignalPlot window; suppress it so the test stays
        # headless.
        dsim.pyqtPlotScope = lambda: None
        assert dsim.execution_init() is True
        dsim.execution_batch()
        assert dsim.last_solver_type == "Standard (Interpreter)"
        vec = np.asarray(scope.exec_params["vector"]).flatten()
        assert vec[-1] == pytest.approx(1.0 - np.exp(-3.0), abs=1e-3)


@pytest.mark.regression
class TestPidOutputOnly:
    def _pid_diagram(self, sampling_time, sim_time=3.0, sim_dt=0.01):
        dsim = _new_dsim(sim_time, sim_dt)
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        step = dsim.add_block(menu["step"], QPoint(100, 100))
        const = dsim.add_block(menu["constant"], QPoint(100, 300))
        pid = dsim.add_block(menu["pid"], QPoint(300, 100))
        scope = dsim.add_block(menu["scope"], QPoint(500, 100))
        const.params["value"] = 0.0
        pid.params.update(
            {"Kp": 0.0, "Ki": 1.0, "Kd": 0.0, "sampling_time": sampling_time}
        )
        dsim.add_line((step.name, 0, step.out_coords[0]), (pid.name, 0, pid.in_coords[0]))
        dsim.add_line((const.name, 0, const.out_coords[0]), (pid.name, 1, pid.in_coords[1]))
        dsim.add_line((pid.name, 0, pid.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
        return dsim, scope

    def test_integral_is_not_doubled(self, qapp):
        """Pure-integral PID on a unit error must integrate once per timestep.

        Before the fix the memory-block first pass integrated a second time,
        giving u(3)=6.03 instead of 3.0.
        """
        dsim, scope = self._pid_diagram(-1.0)
        vec = _run_interpreted(dsim, scope)
        assert vec[-1] == pytest.approx(3.0, abs=1e-9), (
            f"Ki=1 on unit error should integrate to 3, got {vec[-1]:.4f}"
        )

    def test_sampled_pid_uses_sample_period(self, qapp):
        """Same PID at Ts=0.5 must accumulate 0.5 per sample (0.01 before)."""
        dsim, scope = self._pid_diagram(0.5)
        vec = _run_interpreted(dsim, scope)
        assert vec[-1] == pytest.approx(3.0, abs=1e-9)
