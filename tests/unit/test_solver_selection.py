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


def _constant_integrator_scope(method="RK45", value=2.0):
    """Build a valid Constant -> Integrator -> Scope compiled test diagram."""
    from PyQt5.QtCore import QRect, QPoint
    from blocks.constant import ConstantBlock
    from blocks.integrator import IntegratorBlock
    from blocks.scope import ScopeBlock
    from lib.engine.simulation_engine import SimulationEngine
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine

    def _defaults(block):
        return {
            k: v["default"] if isinstance(v, dict) and "default" in v else v
            for k, v in block.params.items()
        }

    model = _MockModel()
    const_block = ConstantBlock()
    const = DBlock(
        "Constant",
        "C1",
        coords=QRect(0, 0, 50, 50),
        color="blue",
        in_ports=0,
        out_ports=1,
        b_type=const_block.b_type,
        params=_defaults(const_block),
        block_class=ConstantBlock,
        category=const_block.category,
    )
    const.params["value"] = value
    const.hierarchy = 0

    integrator_block = IntegratorBlock()
    integ = DBlock(
        "Integrator",
        "I1",
        coords=QRect(100, 0, 50, 50),
        color="green",
        in_ports=1,
        out_ports=1,
        b_type=integrator_block.b_type,
        params=_defaults(integrator_block),
        block_class=IntegratorBlock,
        category=integrator_block.category,
    )
    integ.params["init_conds"] = 0.0
    integ.hierarchy = 1

    scope_block = ScopeBlock()
    scope = DBlock(
        "Scope",
        "S1",
        coords=QRect(200, 0, 50, 50),
        color="red",
        in_ports=1,
        out_ports=0,
        b_type=scope_block.b_type,
        params=_defaults(scope_block),
        block_class=ScopeBlock,
        category=scope_block.category,
    )
    scope.params["labels"] = "output"
    scope.hierarchy = 2

    lines = [
        DLine(
            sid=0,
            srcblock=const.name,
            srcport=0,
            dstblock=integ.name,
            dstport=0,
            points=[QPoint(0, 0), QPoint(1, 1)],
        ),
        DLine(
            sid=1,
            srcblock=integ.name,
            srcport=0,
            dstblock=scope.name,
            dstport=0,
            points=[QPoint(1, 1), QPoint(2, 2)],
        ),
    ]
    model.blocks_list = [const, integ, scope]
    model.line_list = lines

    engine = SimulationEngine(model)
    engine.solver_method = method
    return engine, const, integ, scope, lines


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

        line = DLine(
            sid=0,
            srcblock=const.name,
            srcport=0,
            dstblock=integ.name,
            dstport=0,
            points=[QPoint(0, 0), QPoint(1, 1)],
        )
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


@pytest.mark.unit
class TestCompiledSolverDiagnosticsAndCache:
    def _run(self, engine, lines, t_end=1.0, dt=0.1):
        ok = engine.run_compiled_simulation(engine.model.blocks_list, lines, (0.0, t_end), dt)
        assert ok, engine.error_msg
        return engine.get_solver_diagnostics()

    def test_scipy_solver_diagnostics_are_recorded(self, qapp):
        engine, _const, _integ, _scope, lines = _constant_integrator_scope("RK45")

        diag = self._run(engine, lines, t_end=1.0, dt=0.1)

        assert diag["success"] is True
        assert diag["backend"] == "scipy"
        assert diag["method_requested"] == "RK45"
        assert diag["method_used"] == "RK45"
        assert diag["n_states"] == 1
        assert diag["n_blocks"] == 3
        assert diag["n_lines"] == 2
        assert diag["n_time_points"] == 11
        assert diag["n_output_steps"] == 10
        assert diag["nfev"] > 0
        assert diag["compile_cache_hit"] is False
        assert diag["compile_cache_misses_total"] == 1
        assert diag["compile_wall_time"] >= 0.0
        assert diag["solve_wall_time"] >= 0.0
        assert diag["replay_wall_time"] >= 0.0
        assert diag["total_wall_time"] >= diag["solve_wall_time"]
        assert diag["output_range"]["max"] > 0.0

    def test_fixed_step_diagnostics_include_estimated_rhs_evals(self, qapp):
        engine, _const, _integ, _scope, lines = _constant_integrator_scope("RK4")

        diag = self._run(engine, lines, t_end=1.0, dt=0.1)

        assert diag["backend"] == "fixed_step"
        assert diag["method_used"] == "RK4"
        assert diag["nfev"] == 40  # 10 output steps * 4 RK stages
        assert diag["status"] == 0

    def test_unknown_solver_fallback_is_reported(self, qapp):
        engine, _const, _integ, _scope, lines = _constant_integrator_scope("NOPE")

        diag = self._run(engine, lines, t_end=1.0, dt=0.1)

        assert diag["backend"] == "scipy"
        assert diag["method_requested"] == "NOPE"
        assert diag["method_used"] == "RK45"
        assert "unknown solver" in diag["fallback_reason"]

    def test_repeat_run_reuses_compiled_system_cache(self, qapp, monkeypatch):
        engine, _const, _integ, _scope, lines = _constant_integrator_scope("RK45")
        calls = {"n": 0}
        original = engine.compiler.compile_system

        def counting_compile(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(engine.compiler, "compile_system", counting_compile)

        first = self._run(engine, lines, t_end=1.0, dt=0.1)
        second = self._run(engine, lines, t_end=1.0, dt=0.1)

        assert calls["n"] == 1
        assert first["compile_cache_hit"] is False
        assert second["compile_cache_hit"] is True
        assert second["compile_cache_hits_total"] == 1
        assert second["compile_cache_misses_total"] == 1

    def test_parameter_change_invalidates_compiled_system_cache(self, qapp, monkeypatch):
        engine, const, _integ, _scope, lines = _constant_integrator_scope("RK45", value=2.0)
        calls = {"n": 0}
        original = engine.compiler.compile_system

        def counting_compile(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(engine.compiler, "compile_system", counting_compile)

        first = self._run(engine, lines, t_end=1.0, dt=0.1)
        const.params["value"] = 3.0
        second = self._run(engine, lines, t_end=1.0, dt=0.1)

        assert calls["n"] == 2
        assert first["compile_cache_hit"] is False
        assert second["compile_cache_hit"] is False
        assert second["compile_cache_misses_total"] == 2
        assert np.isclose(float(engine.outs[:, -1].ravel()[0]), 3.0, atol=1e-2)
