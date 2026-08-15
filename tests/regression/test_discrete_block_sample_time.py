"""
Regression tests for the sample-time default on the z-domain blocks.

DiscreteTransferFunction and DiscreteStateSpace used to default to
``sampling_time = -1``, the marker meaning "continuous".  A pure recursion in
the sample index k has no continuous-time reading, and the blocks gate on
``sampling_time > 0``, so the default made them advance one k per *solver
step*: the same diagram produced a different physical response when sim_dt
changed.

    sim_dt=0.10   y = 0, 0, 0.1,  0.25,  0.425, 0.6125
    sim_dt=0.05   y = 0, 0, 0.05, 0.125, 0.2125, 0.3062, ...

The default is now 0 (inherit), so a rate arrives from the upstream discrete
source and the response becomes sim_dt-invariant.  When nothing can be
inherited the old behaviour still applies -- existing diagrams keep their
numbers -- but the engine now warns instead of doing it silently, via the
optional ``requires_sample_time`` block property.
"""

import logging

import numpy as np
import pytest
from PyQt5.QtCore import QPoint

# y[k] = 0.5*y[k-1] + u[k-1]
NUM = [1.0]
DEN = [1.0, -0.5]


def _run(upstream_ts, block_ts, sim_dt, sim_time=1.0):
    """ramp -> [ZOH @ upstream_ts] -> DiscreteTF -> scope."""
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    src = dsim.add_block(menu["ramp"], QPoint(100, 100))
    dtf = dsim.add_block(menu["discrete_transfer_function"], QPoint(400, 100))
    scope = dsim.add_block(menu["scope"], QPoint(600, 100))
    dtf.params.update({"numerator": NUM, "denominator": DEN})
    if block_ts is not None:
        dtf.params["sampling_time"] = block_ts

    if upstream_ts is None:
        dsim.add_line((src.name, 0, src.out_coords[0]), (dtf.name, 0, dtf.in_coords[0]))
    else:
        zoh = dsim.add_block(menu["zero_order_hold"], QPoint(250, 100))
        zoh.params["sampling_time"] = upstream_ts
        dsim.add_line((src.name, 0, src.out_coords[0]), (zoh.name, 0, zoh.in_coords[0]))
        dsim.add_line((zoh.name, 0, zoh.out_coords[0]), (dtf.name, 0, dtf.in_coords[0]))
    dsim.add_line((dtf.name, 0, dtf.out_coords[0]), (scope.name, 0, scope.in_coords[0]))

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
    return dtf.effective_sample_time, np.ravel(scope.exec_params["vector"])


def _distinct(values):
    """The staircase's successive levels, i.e. one entry per actual sample."""
    out = []
    for v in values:
        if not out or v != out[-1]:
            out.append(round(float(v), 9))
    return out


@pytest.mark.regression
class TestDiscreteBlockDefaults:
    def test_default_is_inherit_not_continuous(self, qapp):
        from blocks.discrete_statespace import DiscreteStateSpaceBlock
        from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock

        for cls in (DiscreteTransferFunctionBlock, DiscreteStateSpaceBlock):
            block = cls()
            assert block.params["sampling_time"]["default"] == 0.0, (
                f"{cls.__name__} must default to inherit, not the continuous marker"
            )
            assert block.requires_sample_time is True

    def test_marker_is_opt_in(self, qapp):
        """Continuous blocks legitimately default to -1 and must not warn."""
        from blocks.integrator import IntegratorBlock
        from blocks.transfer_function import TransferFunctionBlock

        for cls in (IntegratorBlock, TransferFunctionBlock):
            block = cls()
            assert block.requires_sample_time is False
            assert block.params["sampling_time"]["default"] == -1.0


@pytest.mark.regression
class TestRateInheritance:
    def test_inherits_the_upstream_discrete_rate(self, qapp):
        ts, _ = _run(upstream_ts=0.2, block_ts=None, sim_dt=0.05)
        assert ts == pytest.approx(0.2)

    def test_inherited_response_is_independent_of_the_solver_step(self, qapp):
        """The point of the change: same diagram, same samples, any sim_dt."""
        ts_a, y_a = _run(upstream_ts=0.2, block_ts=None, sim_dt=0.1)
        ts_b, y_b = _run(upstream_ts=0.2, block_ts=None, sim_dt=0.05)
        assert ts_a == ts_b == pytest.approx(0.2)
        assert _distinct(y_a) == _distinct(y_b)
        # Sampled ramp through y[k] = 0.5 y[k-1] + u[k-1].
        assert _distinct(y_a) == [0.0, 0.2, 0.5, 0.85]

    def test_explicit_period_overrides_inheritance(self, qapp):
        ts, _ = _run(upstream_ts=0.2, block_ts=0.1, sim_dt=0.05)
        assert ts == pytest.approx(0.1)


@pytest.mark.regression
class TestUnresolvedRate:
    def test_behaviour_is_unchanged_when_nothing_can_be_inherited(self, qapp):
        """Existing diagrams keep their numbers -- one k per solver step."""
        ts, y = _run(upstream_ts=None, block_ts=None, sim_dt=0.1, sim_time=0.5)
        assert ts == -1.0
        assert np.allclose(y, [0.0, 0.0, 0.1, 0.25, 0.425, 0.6125], atol=1e-9)

    def test_that_fallback_is_still_solver_step_dependent(self, qapp):
        """Pinned deliberately: this is exactly what the warning is about."""
        _, y_a = _run(upstream_ts=None, block_ts=None, sim_dt=0.1, sim_time=0.5)
        _, y_b = _run(upstream_ts=None, block_ts=None, sim_dt=0.05, sim_time=0.5)
        assert _distinct(y_a) != _distinct(y_b)

    def test_warns_when_no_rate_resolves(self, qapp, caplog):
        with caplog.at_level(logging.WARNING, logger="lib.engine.simulation_engine"):
            _run(upstream_ts=None, block_ts=None, sim_dt=0.1, sim_time=0.5)
        hits = [r for r in caplog.records if "no resolved sample time" in r.getMessage()]
        assert len(hits) == 1
        msg = hits[0].getMessage()
        assert "depends on the simulation step size" in msg
        assert "sampling_time" in msg

    def test_does_not_warn_once_a_rate_resolves(self, qapp, caplog):
        with caplog.at_level(logging.WARNING, logger="lib.engine.simulation_engine"):
            _run(upstream_ts=0.2, block_ts=None, sim_dt=0.05)
        assert not [r for r in caplog.records if "no resolved sample time" in r.getMessage()]
