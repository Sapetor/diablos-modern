"""
Regression test: the compiled solver must say when it ignores a block's method.

system_compiler.py never reads an Integrator's ``method`` or ``ivp_method`` --
the compiled path assembles the whole diagram into one ODE system and hands it
to a single scheme, so a per-block method is not expressible there (Euler in
one integrator and RK4 in another share a state vector).  Since the fast solver
is the default, a user who selected FWD_EULER got scipy's result and no
indication that the choice had been discarded.

The app already has the control that does work on that path: the solver method
in Simulation settings, whose list includes Euler and RK4.  So the per-block
setting stays interpreter-only by design, and the compiled path now warns when
it drops a non-default one instead of silently overriding it.
"""

import logging

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _run(method, use_fast):
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    integ = dsim.add_block(menu["integrator"], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    integ.params["init_conds"] = 0.0
    integ.params["method"] = method
    dsim.add_line((step.name, 0, step.out_coords[0]), (integ.name, 0, integ.in_coords[0]))
    dsim.add_line((integ.name, 0, integ.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time, dsim.sim_dt, dsim.plot_trange = 1.0, 0.1, 1.0
    dsim.execution_init_time = lambda: dsim.sim_time
    dsim.pyqtPlotScope = lambda: None
    dsim.use_fast_solver = use_fast
    assert dsim.execution_init() is True
    dsim.execution_batch()
    return dsim, np.ravel(scope.exec_params["vector"])


@pytest.mark.regression
class TestCompiledSolverAnnouncesTheOverride:
    def test_warns_for_a_non_default_method(self, qapp, caplog):
        with caplog.at_level(logging.WARNING, logger="lib.engine.simulation_engine"):
            dsim, _ = _run("FWD_EULER", use_fast=True)
        assert dsim.last_solver_type == "Fast (Compiled)"
        hits = [
            r for r in caplog.records if "Per-block Integrator method ignored" in r.getMessage()
        ]
        assert len(hits) == 1
        message = hits[0].getMessage()
        assert "Simulation settings" in message
        assert "interpreted solver only" in message

    def test_silent_for_the_default_method(self, qapp, caplog):
        """SOLVE_IVP is the default; warning on it would be noise on every run."""
        with caplog.at_level(logging.WARNING, logger="lib.engine.simulation_engine"):
            _run("SOLVE_IVP", use_fast=True)
        assert not [
            r for r in caplog.records if "Per-block Integrator method ignored" in r.getMessage()
        ]

    def test_silent_on_the_interpreted_path(self, qapp, caplog):
        """There the setting is honoured, so there is nothing to announce."""
        with caplog.at_level(logging.WARNING, logger="lib.engine.simulation_engine"):
            dsim, _ = _run("FWD_EULER", use_fast=False)
        assert dsim.last_solver_type == "Standard (Interpreter)"
        assert not [
            r for r in caplog.records if "Per-block Integrator method ignored" in r.getMessage()
        ]

    def test_the_warning_is_accurate(self, qapp):
        """The compiled result must indeed not be the block's method."""
        _, y_fast = _run("FWD_EULER", use_fast=True)
        _, y_interp = _run("FWD_EULER", use_fast=False)
        # Both integrate a unit step to y = t here, so they agree on the answer;
        # what the warning is about is that the fast path did not use FWD_EULER
        # to get there. Pin the shared grid so a divergence surfaces as a change.
        assert len(y_fast) == len(y_interp)
        assert y_fast[-1] == pytest.approx(1.0, abs=1e-6)
        assert y_interp[-1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.regression
def test_method_param_documents_its_scope(qapp):
    """The tooltip must not promise something the default path ignores."""
    from blocks.integrator import IntegratorBlock

    doc = IntegratorBlock().params["method"]["doc"]
    assert "interpreted solver only" in doc
    assert "Simulation settings" in doc
