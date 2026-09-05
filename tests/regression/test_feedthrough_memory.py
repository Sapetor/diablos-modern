"""
Regression tests for feedthrough memory blocks delivering a stale output.

lib/engine/memory_blocks.py defines a memory block as one whose output "depends
only on *past* inputs ... not on the current input".  ZeroOrderHold, RateLimiter
and PID are all in OUTPUT_ONLY_SAFE_BLOCK_FNS, but none of them satisfies that:
each has direct feedthrough (b_type == 2), and a ZOH at a sample instant outputs
precisely the value it has just sampled.

The interpreter's first loop propagates every memory block's ``output_only``
value (the previously held one) and the hierarchy loop deliberately skips
propagating memory blocks, on the assumption that the first loop's value was
already correct.  For a feedthrough block it is not: the freshly sampled value
never reached consumers until the following step, so a ZOH staircase's edges
lagged by exactly one solver step, and the lag tracked sim_dt:

    Ts=0.2, sim_dt=0.100  -> edges at 0.3,   0.5,   0.7
    Ts=0.2, sim_dt=0.025  -> edges at 0.225, 0.425, 0.625

Feedthrough memory blocks now refresh their consumers' already-counted input
queues in place after executing (propagate_outputs(count=False), which must not
re-count the delivery or a multi-input consumer could fire early).

Removing that delay exposed a second defect it had been masking: PID added
``e * dt`` on its very first call, at t0, before any time had elapsed, so its
integral led the true one by a step for the whole run.  The one-step-late
delivery had been cancelling it exactly.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _run(src_fn, src_params, mid_fn, mid_params, sim_dt, sim_time=0.8):
    """source -> block under test -> scope."""
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    src = dsim.add_block(menu[src_fn], QPoint(100, 100))
    mid = dsim.add_block(menu[mid_fn], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    src.params.update(src_params)
    mid.params.update(mid_params)
    dsim.add_line((src.name, 0, src.out_coords[0]), (mid.name, 0, mid.in_coords[0]))
    dsim.add_line((mid.name, 0, mid.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time, dsim.sim_dt, dsim.plot_trange = sim_time, sim_dt, sim_time
    dsim.execution_init_time = lambda: dsim.sim_time
    dsim.pyqtPlotScope = lambda: None
    dsim.use_fast_solver = False
    assert dsim.execution_init() is True
    calls = 0
    while dsim.execution_initialized and calls < 5000:
        dsim.execution_loop_headless()
        calls += 1
    assert calls < 5000, "run did not terminate"
    return np.ravel(dsim._timeline_list), np.ravel(scope.exec_params["vector"])


def _edges(t, y):
    """Times at which the output changes value."""
    return [round(float(t[i]), 6) for i in range(1, len(y)) if y[i] != y[i - 1]]


@pytest.mark.regression
class TestZohEdgesLandOnSampleInstants:
    @pytest.mark.parametrize("sim_dt", [0.1, 0.05, 0.025])
    def test_edges_do_not_lag_the_solver_step(self, qapp, sim_dt):
        t, y = _run("ramp", {}, "zero_order_hold", {"sampling_time": 0.2}, sim_dt)
        assert _edges(t, y) == [0.2, 0.4, 0.6, 0.8]

    def test_staircase_is_independent_of_the_solver_step(self, qapp):
        levels = []
        for sim_dt in (0.1, 0.05, 0.025):
            t, y = _run("ramp", {}, "zero_order_hold", {"sampling_time": 0.2}, sim_dt)
            levels.append(sorted({round(float(v), 6) for v in y}))
        assert levels[0] == levels[1] == levels[2] == [0.0, 0.2, 0.4, 0.6, 0.8]

    def test_first_sample_is_taken_at_t0(self, qapp):
        """A held value of 0 at t=0 would be the same lag, one sample earlier."""
        _, y = _run(
            "constant", {"value": 5.0}, "zero_order_hold", {"sampling_time": 0.2}, 0.1, sim_time=0.4
        )
        assert np.allclose(y, 5.0)


@pytest.mark.regression
class TestFeedthroughTracksItsInput:
    def test_rate_limiter_does_not_lag(self, qapp):
        """Unconstrained, a rate limiter must reproduce its input exactly."""
        t, y = _run("ramp", {}, "ratelimiter", {}, 0.1, sim_time=0.4)
        assert np.allclose(y, t, atol=1e-9)

    def test_strictly_proper_blocks_are_untouched(self, qapp):
        """b_type 1 keeps the old path: its output really is past-input only."""
        t, y = _run("step", {}, "integrator", {"init_conds": 0.0}, 0.1, sim_time=0.5)
        assert np.allclose(y, t, atol=1e-9)


@pytest.mark.regression
class TestPidIntegralStartsAtZero:
    # The end-to-end values (Ki=1 on a unit error over 3 s == 3.0, continuous
    # and sampled) are asserted by TestPidOutputOnly in
    # tests/regression/test_continuous_block_sampling.py, which wires both of
    # the PID's ports. Those two assertions are what caught this defect when
    # the one-step-late delivery that had been masking it was removed.

    def test_no_integral_accrues_before_time_passes(self, qapp):
        """Directly: the first execute must not advance the integral."""
        from blocks.pid import PIDBlock

        block = PIDBlock()
        params = {"Kp": 0.0, "Ki": 1.0, "Kd": 0.0, "dtime": 0.1, "_init_start_": True}
        first = block.execute(
            time=0.0, inputs={0: np.array([1.0]), 1: np.array([0.0])}, params=params
        )
        assert float(np.ravel(first[0])[0]) == pytest.approx(0.0)
        second = block.execute(
            time=0.1, inputs={0: np.array([1.0]), 1: np.array([0.0])}, params=params
        )
        assert float(np.ravel(second[0])[0]) == pytest.approx(0.1)


@pytest.mark.regression
class TestRefreshDoesNotRecount:
    def test_refresh_leaves_data_received_alone(self, qapp):
        """Re-counting would let a multi-input consumer fire before it is ready."""
        from lib.engine.simulation_engine import SimulationEngine

        dsim_engine = SimulationEngine.__new__(SimulationEngine)

        class FakeChild:
            name = "child"
            data_received = 0
            input_queue = {}

        child = FakeChild()
        src = type("S", (), {"name": "src", "data_sent": 0})()
        line = type("L", (), {"srcblock": "src", "srcport": 0, "dstblock": "child", "dstport": 0})()
        # propagate_outputs resolves its targets from the active block/line
        # lists (cached adjacency), so populate both.
        dsim_engine.active_blocks_list = [child]
        dsim_engine.active_line_list = [line]

        SimulationEngine.propagate_outputs(dsim_engine, src, {0: 1.0})
        assert (child.data_received, child.input_queue[0]) == (1, 1.0)

        SimulationEngine.propagate_outputs(dsim_engine, src, {0: 2.0}, count=False)
        assert child.input_queue[0] == 2.0, "refresh must overwrite the value"
        assert child.data_received == 1, "refresh must not count as a new arrival"
