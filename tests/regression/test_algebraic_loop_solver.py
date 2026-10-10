"""
Algebraic loops are solved iteratively instead of being rejected.

Before 2026-10-10 any feedthrough cycle without a memory block failed init with
"Algebraic loop detected", and the usual workaround (a unit Delay in the
feedback path) only converges when the loop is a contraction:
scripts/benchmarks/pathsim/s2_algebraic_loop.py measured it diverging for
y = u - 3y and oscillating for y = u - tanh(3y). Each loop is now a strongly
connected component solved as a unit every step (lib/engine/algebraic_loop.py).
"""

import numpy as np
import pytest
from scipy.optimize import brentq


def _run(builder, tmp_path, sim_time, sim_dt):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram

    path = tmp_path / "loop.diablos"
    builder.save(str(path))
    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = True
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    data = harvest_scope_signals(dsim) if ok else None
    return ok, err, dsim, data


def _scalar_loop(expr, signs="+-", u=1.0, sim_time=0.5, sim_dt=0.1):
    """u -> Sum -> y, y -> Function(expr) -> Sum; Scope on y."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    c = b.add_block("Constant", 0, 0, name="u", params={"value": u})
    s = b.add_block("Sum", 0, 0, name="s", params={"sign": signs}, in_ports=2)
    f = b.add_block("Function", 0, 0, name="f", params={"expression": expr})
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    b.connect(c, 0, s, 0)
    b.connect(s, 0, f, 0)
    b.connect(f, 0, s, 1)
    b.connect(s, 0, sc, 0)
    return b


@pytest.mark.regression
class TestScalarLoops:
    @pytest.mark.parametrize(
        "expr, fn",
        [
            ("0.5*u[0]", lambda y: 0.5 * y),
            ("3.0*u[0]", lambda y: 3.0 * y),  # the delay workaround diverges here
            ("tanh(3.0*u[0])", lambda y: np.tanh(3.0 * y)),  # ... and oscillates here
        ],
    )
    def test_matches_the_root(self, qapp, tmp_path, expr, fn):
        ok, err, _, data = _run(_scalar_loop(expr), tmp_path, 0.5, 0.1)
        assert ok, err
        y_ref = brentq(lambda y: y - (1.0 - fn(y)), -10, 10, xtol=1e-15)
        assert np.allclose(data["signals"]["y"], y_ref, atol=1e-12)

    def test_downstream_scope_records_one_sample_per_step(self, qapp, tmp_path):
        """The Scope is downstream of the loop, not in it, so it must not be
        re-executed by the solver's trial evaluations."""
        ok, err, _, data = _run(_scalar_loop("0.5*u[0]"), tmp_path, 0.5, 0.1)
        assert ok, err
        assert len(data["signals"]["y"]) == len(data["timeline"]) == 6

    def test_runs_on_the_interpreter_with_a_reason(self, qapp, tmp_path):
        ok, err, dsim, _ = _run(_scalar_loop("0.5*u[0]"), tmp_path, 0.5, 0.1)
        assert ok, err
        reason = dsim.engine.get_compile_fallback_reason()
        assert "algebraic loop" in reason and "s" in reason

    def test_unsolvable_loop_reports_non_convergence(self, qapp, tmp_path):
        """y = u + (y + 1) has no solution."""
        ok, err, dsim, _ = _run(_scalar_loop("u[0] + 1.0", signs="++"), tmp_path, 0.5, 0.1)
        assert not ok
        assert "did not converge" in err
        assert dsim.engine.error_block


@pytest.mark.regression
class TestStructuredLoops:
    def test_two_feedback_paths_into_one_sum(self, qapp, tmp_path):
        """One SCC with two cycles: y = u - 0.2 y - 0.3 y."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=0.3, sim_dt=0.1)
        c = b.add_block("Constant", 0, 0, name="u", params={"value": 1.0})
        s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+--"}, in_ports=3)
        g1 = b.add_block("Gain", 0, 0, name="g1", params={"gain": 0.2})
        g2 = b.add_block("Gain", 0, 0, name="g2", params={"gain": 0.3})
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(c, 0, s, 0)
        b.connect(s, 0, g1, 0)
        b.connect(s, 0, g2, 0)
        b.connect(g1, 0, s, 1)
        b.connect(g2, 0, s, 2)
        b.connect(s, 0, sc, 0)
        ok, err, _, data = _run(b, tmp_path, 0.3, 0.1)
        assert ok, err
        assert np.allclose(data["signals"]["y"], 1.0 / 1.5, atol=1e-12)

    def test_loop_through_a_proper_transfer_function(self, qapp, tmp_path):
        """Unity feedback around G = (s+2)/(s+1), which has D = 1.

        G carries state, so the solver's repeated trial evaluations must not
        advance it. Closed loop (s+2)/(2s+3): y(t) = 2/3 - exp(-1.5 t)/6.
        """
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=3.0, sim_dt=0.001)
        c = b.add_block("Constant", 0, 0, name="u", params={"value": 1.0})
        s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+-"}, in_ports=2)
        g = b.add_block(
            "TranFn", 0, 0, name="g", params={"numerator": [1.0, 2.0], "denominator": [1.0, 1.0]}
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(c, 0, s, 0)
        b.connect(s, 0, g, 0)
        b.connect(g, 0, s, 1)
        b.connect(g, 0, sc, 0)
        ok, err, _, data = _run(b, tmp_path, 3.0, 0.001)
        assert ok, err
        t, y = data["timeline"], data["signals"]["y"]
        exact = 2.0 / 3.0 - np.exp(-1.5 * t) / 6.0
        assert y[0] == pytest.approx(0.5, abs=1e-12)
        # The interpreter holds the TF input over each step: first order in dt.
        assert np.max(np.abs(y - exact)) < 1e-3

    def test_loop_through_a_discrete_block_is_rejected_by_name(self, qapp, tmp_path):
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=0.3, sim_dt=0.1)
        c = b.add_block("Constant", 0, 0, name="u", params={"value": 1.0})
        s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+-"}, in_ports=2)
        d = b.add_block(
            "DiscreteTranFn",
            0,
            0,
            name="d",
            params={"numerator": [0.5], "denominator": [1.0], "sampling_time": 0.1},
            in_ports=1,
            out_ports=1,
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(c, 0, s, 0)
        b.connect(s, 0, d, 0)
        b.connect(d, 0, s, 1)
        b.connect(s, 0, sc, 0)
        ok, err, _, _ = _run(b, tmp_path, 0.3, 0.1)
        assert not ok
        assert "discrete block" in err
