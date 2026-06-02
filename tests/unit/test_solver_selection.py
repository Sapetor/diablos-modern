"""
Unit tests for compiled-solver selection.

Covers:
  - the in-house fixed-step integrators (Euler, RK4) for accuracy/ordering,
  - SimulationEngine.update_sim_params plumbing (method/rtol/atol, backward compat),
  - .diablos persistence of solver settings through FileService serialize/load.
"""

import numpy as np
import pytest

from lib.engine.simulation_engine import (
    SimulationEngine,
    integrate_fixed_step,
    SCIPY_SOLVER_METHODS,
    FIXED_STEP_METHODS,
)


@pytest.mark.unit
class TestFixedStepIntegrators:
    def _solve(self, scheme, n=200):
        # dy/dt = -y, y(0) = 1  ->  y(t) = exp(-t)
        t_eval = np.linspace(0.0, 1.0, n)
        y0 = np.array([1.0])
        model_func = lambda t, y: -y
        y_hist = integrate_fixed_step(model_func, t_eval, y0, scheme)
        return t_eval, y_hist

    def test_euler_tracks_decay(self):
        t_eval, y = self._solve("euler")
        assert np.isclose(y[0, -1], np.exp(-1.0), atol=1e-2)

    def test_rk4_tracks_decay_precisely(self):
        t_eval, y = self._solve("rk4")
        assert np.isclose(y[0, -1], np.exp(-1.0), atol=1e-6)

    def test_rk4_more_accurate_than_euler(self):
        analytic = np.exp(-1.0)
        _, y_euler = self._solve("euler", n=50)
        _, y_rk4 = self._solve("rk4", n=50)
        err_euler = abs(y_euler[0, -1] - analytic)
        err_rk4 = abs(y_rk4[0, -1] - analytic)
        assert err_rk4 < err_euler

    def test_initial_condition_preserved(self):
        _, y = self._solve("rk4")
        assert np.isclose(y[0, 0], 1.0)

    def test_rk4_exact_on_linear_growth(self):
        # dy/dt = 2  ->  y(t) = 2t (RK4 is exact for polynomials up to degree 4)
        t_eval = np.linspace(0.0, 5.0, 11)
        y = integrate_fixed_step(lambda t, yy: np.array([2.0]), t_eval, np.array([0.0]), "rk4")
        assert np.allclose(y[0], 2.0 * t_eval, atol=1e-9)


@pytest.mark.unit
class TestUpdateSimParams:
    def test_defaults(self):
        eng = SimulationEngine(model=None)
        assert eng.solver_method == "RK45"
        assert eng.rtol == 1e-9
        assert eng.atol == 1e-12

    def test_backward_compatible_two_arg_call(self):
        eng = SimulationEngine(model=None)
        eng.update_sim_params(5.0, 0.001)
        assert eng.sim_time == 5.0 and eng.sim_dt == 0.001
        # Solver settings untouched by the legacy 2-arg signature.
        assert eng.solver_method == "RK45"
        assert eng.rtol == 1e-9

    def test_sets_solver_fields(self):
        eng = SimulationEngine(model=None)
        eng.update_sim_params(1.0, 0.01, solver_method="LSODA", rtol=1e-6, atol=1e-9)
        assert eng.solver_method == "LSODA"
        assert eng.rtol == 1e-6
        assert eng.atol == 1e-9

    def test_partial_update_keeps_others(self):
        eng = SimulationEngine(model=None)
        eng.update_sim_params(1.0, 0.01, solver_method="BDF")
        assert eng.solver_method == "BDF"
        assert eng.rtol == 1e-9  # unchanged

    def test_method_sets_are_disjoint(self):
        assert set(SCIPY_SOLVER_METHODS).isdisjoint(FIXED_STEP_METHODS)


@pytest.mark.unit
class TestSolverPersistence:
    # `file_service` fixture (conftest.py) provides a FileService backed by a
    # SimulationModel and a session QApplication.
    def test_serialize_then_load_roundtrip(self, file_service):
        sim_params = {
            "sim_time": 3.0,
            "sim_dt": 0.005,
            "plot_trange": 100,
            "solver_method": "RK4",
            "rtol": 1e-6,
            "atol": 1e-8,
        }
        data = file_service.serialize(modern_ui_data=None, sim_params=sim_params)
        assert data["sim_data"]["solver_method"] == "RK4"
        assert data["sim_data"]["rtol"] == 1e-6
        assert data["sim_data"]["atol"] == 1e-8

        # Round-trip back through the loader.
        loaded = file_service.apply_loaded_data(data)
        assert loaded["solver_method"] == "RK4"
        assert loaded["rtol"] == 1e-6
        assert loaded["atol"] == 1e-8

    def test_load_legacy_file_defaults_to_rk45(self, file_service):
        # Legacy file without solver keys.
        legacy = {"sim_data": {"sim_time": 2.0, "sim_dt": 0.01, "sim_trange": 50}}
        loaded = file_service.apply_loaded_data(legacy)
        assert loaded["solver_method"] == "RK45"
        assert loaded["rtol"] == 1e-9


class _MockModel:
    """Minimal model sufficient for the compiled-solver path."""

    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}

    def link_goto_from(self):
        pass


@pytest.mark.unit
class TestCompiledSolverEndToEnd:
    """
    Run a tiny Constant(2.0) -> Integrator diagram through the compiled solver
    with each selectable method and check the integrated result.

    The exact integral over [0, 5] is 2 * 5 = 10 regardless of method.
    """

    def _run(self, qapp, method, t_end=5.0, dt=0.01):
        from PyQt5.QtCore import QRect, QPoint
        from lib.engine.simulation_engine import SimulationEngine
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine

        model = _MockModel()
        const = DBlock("Constant", "C1", coords=QRect(0, 0, 50, 50), color="blue")
        const.block_fn = "Constant"
        const.params["value"] = 2.0
        const.hierarchy = 0

        integ = DBlock("Integrator", "I1", coords=QRect(100, 0, 50, 50), color="green")
        integ.block_fn = "Integrator"
        integ.params["init_conds"] = 0.0
        integ.hierarchy = 1

        line = DLine(sid=0, srcblock=const.name, srcport=0,
                     dstblock=integ.name, dstport=0,
                     points=[QPoint(0, 0), QPoint(1, 1)])
        model.blocks_list = [const, integ]
        model.line_list = [line]

        engine = SimulationEngine(model)
        engine.solver_method = method
        engine.initialize_execution([const, integ], [line])
        ok = engine.run_compiled_simulation([const, integ], [line], (0.0, t_end), dt)
        assert ok, f"compiled run failed for method {method}"
        return float(engine.outs[:, -1].ravel()[0])

    @pytest.mark.parametrize("method", ["RK45", "RK23", "DOP853", "LSODA", "RK4", "Euler"])
    def test_integrates_constant(self, qapp, method):
        final = self._run(qapp, method)
        assert np.isclose(final, 10.0, atol=1e-2), f"{method}: expected ~10.0, got {final}"

    def test_unknown_method_falls_back_to_rk45(self, qapp):
        # Unknown solver names degrade gracefully to RK45 rather than crashing.
        final = self._run(qapp, "NOPE")
        assert np.isclose(final, 10.0, atol=1e-2)
