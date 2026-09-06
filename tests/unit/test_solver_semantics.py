"""Solver semantics: the stiffness heuristic, the "auto" method, and fallback.

Three subjects, all documented in ``docs/SOLVER_SEMANTICS.md``:

* the stiffness heuristic (``lib/engine/solver_diagnostics.estimate_stiffness``),
  in two layers:

* the pure function, driven with a hand-built solver result and an analytic
  right-hand side, so each of its two gates can be exercised on its own;
* real compiled runs of diagrams assembled from blocks — a stiff Van der Pol
  oscillator (Integrator + Sum + Gain + Product) must be flagged under RK45 and
  not under Radau, and a smooth second-order system must never be flagged, at
  any output resolution;
* the ``"auto"`` solver setting, which must resolve to LSODA at run time while
  leaving RK45 as the default;
* the recorded reason a diagram was declined by the compiler, which is what the
  property panel shows instead of a silent fall-back to the interpreter.
"""

import types

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, QRect

from lib.engine.compiled_runner import (
    AUTO_RESOLVED_METHOD,
    AUTO_SOLVER_METHOD,
    resolve_solver_method,
)
from lib.engine.simulation_engine import SimulationEngine
from lib.engine.solver_diagnostics import (
    STIFFNESS_INDEX,
    STIFFNESS_WORK_RATIO,
    estimate_stiffness,
    format_stiffness_for_log,
)
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


# --------------------------------------------------------------------------- #
# Pure-function layer
# --------------------------------------------------------------------------- #


def _sol(nfev, n_points, n_states=2, y=None):
    """A minimal stand-in for the attributes estimate_stiffness reads."""
    t = np.linspace(0.0, 1.0, n_points)
    if y is None:
        y = np.zeros((n_states, n_points))
    return types.SimpleNamespace(t=t, y=np.asarray(y, dtype=float), nfev=nfev)


def _linear_rhs(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return lambda t, y: matrix @ np.asarray(y, dtype=float)


@pytest.mark.unit
class TestEstimateStiffnessApplicability:
    def test_implicit_method_is_not_probed(self):
        # Someone already on Radau/BDF/LSODA has nothing to be told.
        for method in ("Radau", "BDF", "LSODA"):
            assert (
                estimate_stiffness(sol=_sol(10000, 11), method=method, dt=0.1, n_states=2) is None
            )

    def test_fixed_step_method_is_not_probed(self):
        for method in ("RK4", "Euler"):
            assert estimate_stiffness(sol=_sol(400, 11), method=method, dt=0.1, n_states=2) is None

    def test_stateless_system_is_not_probed(self):
        # A purely algebraic diagram never calls solve_ivp at all.
        assert estimate_stiffness(sol=_sol(0, 11), method="RK45", dt=0.1, n_states=0) is None

    def test_missing_nfev_is_not_probed(self):
        sol = types.SimpleNamespace(t=np.linspace(0, 1, 11), y=np.zeros((1, 11)))
        assert estimate_stiffness(sol=sol, method="RK45", dt=0.1, n_states=1) is None

    @pytest.mark.parametrize("method", ["RK45", "RK23", "DOP853"])
    def test_every_explicit_method_is_probed(self, method):
        result = estimate_stiffness(sol=_sol(600, 11), method=method, dt=0.1, n_states=2)
        assert result is not None and "work_ratio" in result


@pytest.mark.unit
class TestEstimateStiffnessGates:
    def test_cheap_run_short_circuits_before_the_jacobian(self):
        """A solver that kept up with the output grid is never probed further."""
        calls = {"n": 0}

        def counting_rhs(t, y):
            calls["n"] += 1
            return -1e6 * np.asarray(y, dtype=float)

        result = estimate_stiffness(
            sol=_sol(60, 11),  # 10 steps over 10 intervals -> work_ratio 1.0
            method="RK45",
            dt=0.1,
            n_states=1,
            model_func=counting_rhs,
        )
        assert result["work_ratio"] == pytest.approx(1.0)
        assert result["suspected"] is False
        assert result["stiffness_index"] is None
        assert calls["n"] == 0, "the Jacobian probe must not run below the work-ratio gate"

    def test_hard_working_but_smooth_system_is_not_flagged(self):
        """The work-ratio gate alone is not enough.

        A perfectly benign oscillator sampled coarsely makes the solver take
        many steps per output sample, but its eigenvalues are slow: an implicit
        method would buy nothing, so nothing is suggested.
        """
        smooth = _linear_rhs([[0.0, 1.0], [-4.0, -1.2]])
        result = estimate_stiffness(
            sol=_sol(6000, 11, y=np.ones((2, 11))),  # work_ratio = 100
            method="RK45",
            dt=0.1,
            n_states=2,
            model_func=smooth,
        )
        assert result["work_ratio"] > STIFFNESS_WORK_RATIO
        assert result["stiffness_index"] < STIFFNESS_INDEX
        assert result["suspected"] is False

    def test_stiff_system_that_worked_hard_is_flagged(self):
        stiff = _linear_rhs([[-1e4, 0.0], [0.0, -1.0]])
        result = estimate_stiffness(
            sol=_sol(6000, 11, y=np.ones((2, 11))),
            method="RK45",
            dt=0.01,
            n_states=2,
            model_func=stiff,
        )
        assert result["suspected"] is True
        assert result["stiffness_index"] == pytest.approx(100.0, rel=1e-3)
        assert result["eig_ratio"] == pytest.approx(1e4, rel=1e-3)
        assert result["suggested_method"] == "LSODA"

    def test_fast_oscillation_is_not_stiffness(self):
        """Purely imaginary eigenvalues: the answer really does move that fast."""
        oscillator = _linear_rhs([[0.0, 1e4], [-1e4, 0.0]])
        result = estimate_stiffness(
            sol=_sol(60000, 11, y=np.ones((2, 11))),
            method="RK45",
            dt=0.01,
            n_states=2,
            model_func=oscillator,
        )
        assert result["work_ratio"] > STIFFNESS_WORK_RATIO
        assert result["stiffness_index"] == pytest.approx(0.0, abs=1e-6)
        assert result["suspected"] is False

    def test_single_fast_mode_is_flagged_without_an_eigenvalue_spread(self):
        """``dy/dt = -1e4 (y - u(t))`` is the textbook stiff scalar problem.

        Its Jacobian has one eigenvalue, so the classic max/min stiffness *ratio*
        is 1 and says nothing. What makes it stiff is the fast mode against the
        output step, which is exactly what ``stiffness_index`` measures.
        """
        result = estimate_stiffness(
            sol=_sol(6000, 11, n_states=1, y=np.ones((1, 11))),
            method="RK45",
            dt=0.01,
            n_states=1,
            model_func=_linear_rhs([[-1e4]]),
        )
        assert result["eig_ratio"] == pytest.approx(1.0)
        assert result["stiffness_index"] == pytest.approx(100.0, rel=1e-3)
        assert result["suspected"] is True

    def test_no_model_func_reports_work_only(self):
        result = estimate_stiffness(sol=_sol(60000, 11), method="RK45", dt=0.01, n_states=2)
        assert result["work_ratio"] > STIFFNESS_WORK_RATIO
        assert result["stiffness_index"] is None
        assert result["suspected"] is False

    def test_large_state_vector_skips_the_jacobian(self):
        """A 2-D PDE has thousands of states; n+1 RHS calls is not 'cheap'."""
        calls = {"n": 0}

        def counting_rhs(t, y):
            calls["n"] += 1
            return -np.asarray(y, dtype=float)

        result = estimate_stiffness(
            sol=_sol(60000, 11, n_states=500, y=np.ones((500, 11))),
            method="RK45",
            dt=0.01,
            n_states=500,
            model_func=counting_rhs,
        )
        assert result["stiffness_index"] is None
        assert result["suspected"] is False
        assert calls["n"] == 0

    def test_a_raising_rhs_degrades_to_work_only(self):
        def broken(t, y):
            raise ValueError("boom")

        result = estimate_stiffness(
            sol=_sol(60000, 11, y=np.ones((2, 11))),
            method="RK45",
            dt=0.01,
            n_states=2,
            model_func=broken,
        )
        assert result["suspected"] is False
        assert result["stiffness_index"] is None


@pytest.mark.unit
class TestFormatStiffnessForLog:
    def test_silent_when_not_suspected(self):
        assert format_stiffness_for_log(None, "RK45") == ""
        assert format_stiffness_for_log({"suspected": False}, "RK45") == ""

    def test_names_the_method_and_the_suggestion(self):
        text = format_stiffness_for_log(
            {
                "suspected": True,
                "work_ratio": 35.6,
                "stiffness_index": 100.0,
                "suggested_method": "LSODA",
            },
            "RK45",
        )
        assert "RK45" in text and "LSODA" in text and "36" in text


# --------------------------------------------------------------------------- #
# End-to-end layer: real diagrams through the compiled engine
# --------------------------------------------------------------------------- #


class _MockModel:
    def __init__(self, blocks=None, lines=None):
        self.blocks_list = list(blocks or [])
        self.line_list = list(lines or [])
        self.variables = {}

    def link_goto_from(self):
        pass


def _defaults(instance):
    return {
        k: v["default"] if isinstance(v, dict) and "default" in v else v
        for k, v in instance.params.items()
    }


def _block(cls, block_fn, sid, in_ports, out_ports, **params):
    instance = cls()
    block = DBlock(
        block_fn,
        sid,
        coords=QRect(60 * sid, 0, 50, 50),
        color="blue",
        in_ports=in_ports,
        out_ports=out_ports,
        # b_type/category are optional on a block class; 0 ("instantaneous") is
        # the right default for every primitive used here — the source branch of
        # initialize_execution keys off the port count, not off b_type alone.
        b_type=getattr(instance, "b_type", 0),
        params=_defaults(instance),
        block_class=cls,
        category=getattr(instance, "category", "Other"),
    )
    block.params.update(params)
    return block


def _wire(sid, src, srcport, dst, dstport):
    return DLine(
        sid=sid,
        srcblock=src.name,
        srcport=srcport,
        dstblock=dst.name,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(1, 1)],
    )


def _van_der_pol(mu):
    """x1'' - mu (1 - x1^2) x1' + x1 = 0, wired out of primitive blocks.

    x1' = x2                       (integrator i1)
    x2' = mu (1 - x1^2) x2 - x1    (integrator i2)

    The canonical stiff test problem: for large mu the relaxation oscillator
    spends most of its period on a slow manifold whose transverse mode decays
    ~mu times faster than the motion along it.
    """
    from blocks.constant import ConstantBlock
    from blocks.gain import GainBlock
    from blocks.integrator import IntegratorBlock
    from blocks.sigproduct import SigProductBlock
    from blocks.sum import SumBlock

    one = _block(ConstantBlock, "Constant", 0, 0, 1, value=1.0)
    i1 = _block(IntegratorBlock, "Integrator", 1, 1, 1, init_conds=2.0)  # x1
    i2 = _block(IntegratorBlock, "Integrator", 2, 1, 1, init_conds=0.0)  # x2
    sq = _block(SigProductBlock, "SgProd", 3, 2, 1)  # x1 * x1
    one_minus = _block(SumBlock, "Sum", 4, 2, 1, sign="+-")  # 1 - x1^2
    damp = _block(SigProductBlock, "SgProd", 5, 2, 1)  # (1-x1^2) * x2
    gain = _block(GainBlock, "Gain", 6, 1, 1, gain=mu)
    acc = _block(SumBlock, "Sum", 7, 2, 1, sign="+-")  # mu(...)x2 - x1

    blocks = [one, i1, i2, sq, one_minus, damp, gain, acc]
    lines = [
        _wire(0, i2, 0, i1, 0),  # x2 -> integrator -> x1
        _wire(1, i1, 0, sq, 0),
        _wire(2, i1, 0, sq, 1),
        _wire(3, one, 0, one_minus, 0),
        _wire(4, sq, 0, one_minus, 1),
        _wire(5, one_minus, 0, damp, 0),
        _wire(6, i2, 0, damp, 1),
        _wire(7, damp, 0, gain, 0),
        _wire(8, gain, 0, acc, 0),
        _wire(9, i1, 0, acc, 1),
        _wire(10, acc, 0, i2, 0),
    ]
    return blocks, lines


def _smooth_second_order():
    """x'' + 1.2 x' + 4 x = 0 — a well-damped, entirely non-stiff oscillator."""
    from blocks.gain import GainBlock
    from blocks.integrator import IntegratorBlock
    from blocks.sum import SumBlock

    i1 = _block(IntegratorBlock, "Integrator", 0, 1, 1, init_conds=1.0)  # x
    i2 = _block(IntegratorBlock, "Integrator", 1, 1, 1, init_conds=0.0)  # x'
    k = _block(GainBlock, "Gain", 2, 1, 1, gain=4.0)
    c = _block(GainBlock, "Gain", 3, 1, 1, gain=1.2)
    acc = _block(SumBlock, "Sum", 4, 2, 1, sign="--")

    blocks = [i1, i2, k, c, acc]
    lines = [
        _wire(0, i2, 0, i1, 0),
        _wire(1, i1, 0, k, 0),
        _wire(2, i2, 0, c, 0),
        _wire(3, k, 0, acc, 0),
        _wire(4, c, 0, acc, 1),
        _wire(5, acc, 0, i2, 0),
    ]
    return blocks, lines


def _run_compiled(blocks, lines, method, t_end, dt, rtol=1e-6, atol=1e-9):
    model = _MockModel(blocks, lines)
    engine = SimulationEngine(model)
    engine.update_sim_params(t_end, dt, solver_method=method, rtol=rtol, atol=atol)
    engine.initialize_execution(blocks, lines)
    ok = engine.run_compiled_simulation(blocks, lines, (0.0, t_end), dt)
    assert ok, engine.error_msg
    return engine.get_solver_diagnostics()


@pytest.mark.unit
class TestStiffnessOnRealDiagrams:
    def test_stiff_van_der_pol_is_flagged_under_rk45(self, qapp):
        blocks, lines = _van_der_pol(mu=1000.0)
        diag = _run_compiled(blocks, lines, "RK45", t_end=1.0, dt=0.05)

        assert diag["success"] is True
        assert diag["stiffness_suspected"] is True
        stiffness = diag["stiffness"]
        assert stiffness["work_ratio"] > STIFFNESS_WORK_RATIO
        assert stiffness["stiffness_index"] > STIFFNESS_INDEX
        assert stiffness["suggested_method"] == "LSODA"

    def test_the_same_diagram_is_not_flagged_under_radau(self, qapp):
        # Radau is already the right tool; there is nothing to suggest, so the
        # heuristic does not run at all and the flat flag stays False.
        blocks, lines = _van_der_pol(mu=1000.0)
        diag = _run_compiled(blocks, lines, "Radau", t_end=1.0, dt=0.05)

        assert diag["success"] is True
        assert diag["stiffness"] is None
        assert diag["stiffness_suspected"] is False

    @pytest.mark.parametrize("dt", [0.01, 0.5])
    def test_smooth_second_order_is_never_flagged(self, qapp, dt):
        # Coarse output (dt=0.5) makes even this benign system look expensive
        # per sample; the eigenvalue gate is what keeps it quiet.
        blocks, lines = _smooth_second_order()
        diag = _run_compiled(blocks, lines, "RK45", t_end=20.0, dt=dt, rtol=1e-9, atol=1e-12)

        assert diag["success"] is True
        assert diag["stiffness_suspected"] is False

    def test_fixed_step_run_records_no_stiffness(self, qapp):
        blocks, lines = _smooth_second_order()
        diag = _run_compiled(blocks, lines, "RK4", t_end=5.0, dt=0.01)

        assert diag["backend"] == "fixed_step"
        assert diag["stiffness"] is None
        assert diag["stiffness_suspected"] is False


# --------------------------------------------------------------------------- #
# The "auto" solver setting
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestAutoSolverMethod:
    def test_auto_resolves_to_lsoda(self):
        assert resolve_solver_method(AUTO_SOLVER_METHOD) == AUTO_RESOLVED_METHOD
        assert resolve_solver_method("AUTO") == AUTO_RESOLVED_METHOD
        assert resolve_solver_method(" Auto ") == AUTO_RESOLVED_METHOD

    def test_every_other_name_passes_through(self):
        for name in ("RK45", "Radau", "BDF", "LSODA", "RK4", "Euler", "NOPE"):
            assert resolve_solver_method(name) == name

    def test_empty_setting_defaults_to_rk45(self):
        assert resolve_solver_method(None) == "RK45"
        assert resolve_solver_method("") == "RK45"

    def test_default_is_still_rk45(self):
        # The whole point of "auto" being opt-in: no existing diagram's numbers
        # move because the default changed.
        assert SimulationEngine(model=None).solver_method == "RK45"

    def test_auto_run_reports_the_scheme_it_used(self, qapp):
        blocks, lines = _smooth_second_order()
        diag = _run_compiled(blocks, lines, AUTO_SOLVER_METHOD, t_end=2.0, dt=0.05)

        assert diag["method_requested"] == AUTO_SOLVER_METHOD
        assert diag["method_used"] == AUTO_RESOLVED_METHOD
        assert diag["fallback_reason"] is None  # not a degraded fallback

    def test_auto_is_offered_in_the_simulation_dialog(self, qapp):
        from lib.dialogs import SimulationDialog

        assert AUTO_SOLVER_METHOD in SimulationDialog.SOLVER_METHODS
        # Index 0 is also the fallback for an unrecognised stored method.
        assert SimulationDialog.SOLVER_METHODS[0] == "RK45"

    def test_auto_round_trips_through_save_and_load(self, file_service):
        data = file_service.serialize(
            modern_ui_data=None,
            sim_params={
                "sim_time": 3.0,
                "sim_dt": 0.005,
                "plot_trange": 100,
                "solver_method": AUTO_SOLVER_METHOD,
                "rtol": 1e-6,
                "atol": 1e-8,
                "zero_crossing": True,
            },
        )
        assert data["sim_data"]["solver_method"] == AUTO_SOLVER_METHOD
        assert file_service.apply_loaded_data(data)["solver_method"] == AUTO_SOLVER_METHOD

    def test_auto_round_trips_through_the_dialog(self, qapp):
        from lib.dialogs import SimulationDialog

        dialog = SimulationDialog(
            sim_time=1.0,
            sim_dt=0.01,
            plot_trange=100,
            solver_method=AUTO_SOLVER_METHOD,
        )
        try:
            assert dialog.solver_method_combo.currentText() == AUTO_SOLVER_METHOD
            assert dialog.get_values()["solver_method"] == AUTO_SOLVER_METHOD
        finally:
            dialog.deleteLater()


# --------------------------------------------------------------------------- #
# Why a diagram was declined by the compiler
# --------------------------------------------------------------------------- #


@pytest.mark.unit
class TestCompileFallbackReason:
    def _engine(self, blocks):
        engine = SimulationEngine(_MockModel(blocks, []))
        return engine

    def test_a_compilable_diagram_records_no_reason(self, qapp):
        blocks, _lines = _smooth_second_order()
        engine = self._engine(blocks)
        assert engine.check_compilability(blocks) is True
        assert engine.get_compile_fallback_reason() is None

    def test_a_discrete_rate_names_the_block_and_the_reason(self, qapp):
        from blocks.zero_order_hold import ZeroOrderHoldBlock

        zoh = _block(ZeroOrderHoldBlock, "ZeroOrderHold", 0, 1, 1, sampling_time=0.1)
        engine = self._engine([zoh])
        assert engine.check_compilability([zoh]) is False
        reason = engine.get_compile_fallback_reason()
        assert zoh.name in reason and "discrete sample time" in reason

    def test_hysteresis_reason_points_at_zero_crossing(self, qapp):
        from blocks.hysteresis import HysteresisBlock

        relay = _block(HysteresisBlock, "Hysteresis", 0, 1, 1)
        engine = self._engine([relay])
        engine.zero_crossing = False
        assert engine.check_compilability([relay]) is False
        assert "zero-crossing" in engine.get_compile_fallback_reason()

        # With detection on it compiles, and the stale reason is cleared.
        engine.zero_crossing = True
        assert engine.check_compilability([relay]) is True
        assert engine.get_compile_fallback_reason() is None

    def test_an_uncompilable_block_says_so(self, qapp):
        from blocks.noise import NoiseBlock

        noise = _block(NoiseBlock, "Noise", 0, 0, 1)
        engine = self._engine([noise])
        assert engine.check_compilability([noise]) is False
        assert "no compiled kernel" in engine.get_compile_fallback_reason()
