"""
Algebraic loops are solved iteratively instead of being rejected.

Before 2026-10-10 any feedthrough cycle without a memory block failed init with
"Algebraic loop detected", and the usual workaround (a unit Delay in the
feedback path) only converges when the loop is a contraction:
scripts/benchmarks/pathsim/s2_algebraic_loop.py measured it diverging for
y = u - 3y and oscillating for y = u - tanh(3y). Each loop is now a strongly
connected component solved as a unit every step (lib/engine/algebraic_loop.py).
Since phase 3 of the hybrid compiled path the compiler solves them too, inside
the RHS (lib/engine/compiled_loops.py), so a diagram with a loop compiles and
linearizes; the interpreter's solver remains for interpreter runs.
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

    def test_runs_on_the_compiled_path(self, qapp, tmp_path):
        ok, err, dsim, data = _run(_scalar_loop("0.5*u[0]"), tmp_path, 0.5, 0.1)
        assert ok, err
        assert dsim.engine.get_compile_fallback_reason() is None
        assert dsim.engine.last_solver_diagnostics.get("backend")
        assert np.allclose(data["signals"]["y"], 1.0 / 1.5, atol=1e-12)

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


def _zoh_loop(sim_time=1.0):
    """ramp -> ZOH(0.2) -> Sum(+-) -> y, y -> Gain(0.5) -> Sum. y = r(kT) / 1.5."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=0.1)
    r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
    z = b.add_block(
        "ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2}, in_ports=1, out_ports=1
    )
    s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+-"}, in_ports=2)
    g = b.add_block("Gain", 0, 0, name="g", params={"gain": 0.5})
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    b.connect(r, 0, z, 0)
    b.connect(z, 0, s, 0)
    b.connect(s, 0, g, 0)
    b.connect(g, 0, s, 1)
    b.connect(s, 0, sc, 0)
    return b


@pytest.mark.regression
class TestLoopBehindAZoh:
    """The loop must wait for the ZOH's fresh sample, like any other consumer
    of a feedthrough memory block (tests/regression/test_sampled_loop_delay.py);
    solving on the stale held value would lag every edge by one solver step."""

    @pytest.mark.parametrize("sim_dt", [0.1, 0.05])
    def test_staircase_edges_land_on_sample_instants(self, qapp, tmp_path, sim_dt):
        """Driven the way the GUI runs: execution_init + the step loop."""
        from lib.cli import load_diagram

        path = tmp_path / "zoh_loop.diablos"
        _zoh_loop().save(str(path))
        dsim, _ = load_diagram(str(path))
        dsim.sim_time, dsim.sim_dt, dsim.plot_trange = 1.0, sim_dt, 1.0
        dsim.execution_init_time = lambda: dsim.sim_time
        dsim.pyqtPlotScope = lambda: None
        dsim.use_fast_solver = True
        assert dsim.execution_init() is True
        calls = 0
        while dsim.execution_initialized and calls < 5000:
            dsim.execution_loop_headless()
            calls += 1
        t = np.ravel(dsim._timeline_list)
        scope = next(b for b in dsim.engine.active_blocks_list if b.block_fn == "Scope")
        y = np.ravel(scope.exec_params["vector"])
        expected = np.floor(t / 0.2 + 1e-9) * 0.2 / 1.5
        assert np.allclose(y, expected, atol=1e-12)

    def test_loop_fed_by_a_zoh_behind_another_loop(self, qapp, tmp_path):
        """ramp -> loop A -> ZOH(0.2) -> loop B. The ZOH's input exists only
        once loop A is solved, so loop B must not solve on the ZOH's stale
        value in the same sweep. yB = r(kT) / 2.25."""
        from lib.analysis.resim import harvest_scope_signals
        from lib.cli import load_diagram
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=1.0, sim_dt=0.1)
        r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
        sa = b.add_block("Sum", 0, 0, name="sa", params={"sign": "+-"}, in_ports=2)
        ga = b.add_block("Gain", 0, 0, name="ga", params={"gain": 0.5})
        z = b.add_block(
            "ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2}, in_ports=1, out_ports=1
        )
        sb = b.add_block("Sum", 0, 0, name="sb", params={"sign": "+-"}, in_ports=2)
        gb = b.add_block("Gain", 0, 0, name="gb", params={"gain": 0.5})
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        for src, dst, port in [
            (r, sa, 0), (sa, ga, 0), (ga, sa, 1), (sa, z, 0),
            (z, sb, 0), (sb, gb, 0), (gb, sb, 1), (sb, sc, 0),
        ]:  # fmt: skip
            b.connect(src, 0, dst, port)
        path = tmp_path / "chain.diablos"
        b.save(str(path))
        dsim, _ = load_diagram(str(path))
        ok, err = dsim.run_tuning_simulation(1.0, 0.1)
        assert ok, err
        data = harvest_scope_signals(dsim)
        t, y = data["timeline"], data["signals"]["y"]
        expected = np.floor(t / 0.2 + 1e-9) * 0.2 / 2.25
        assert np.allclose(y, expected, atol=1e-12)


@pytest.mark.regression
class TestLoopsOnTheCompiledPath:
    def _proper_tf_loop(self):
        """r -> Sum(+-) -> (s+2)/(s+1) -> feedback: closed loop (s+2)/(2s+3)."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=3.0, sim_dt=0.01)
        r = b.add_block("Step", 0, 0, name="r", params={"value": 1.0, "delay": 0.0, "type": "up"})
        e = b.add_block("Sum", 0, 0, name="e", params={"sign": "+-"}, in_ports=2)
        g = b.add_block(
            "TranFn", 0, 0, name="g", params={"numerator": [1.0, 2.0], "denominator": [1.0, 1.0]}
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(r, 0, e, 0)
        b.connect(e, 0, g, 0)
        b.connect(g, 0, e, 1)
        b.connect(g, 0, sc, 0)
        return b

    def test_loop_through_a_proper_tf_is_exact(self, qapp, tmp_path):
        """The interpreter's fixed step gives 1.5e-4 here; compiled, the
        error is the adaptive solver's tolerance."""
        from scipy import signal

        ok, err, dsim, data = _run(self._proper_tf_loop(), tmp_path, 3.0, 0.01)
        assert ok, err
        assert dsim.engine.get_compile_fallback_reason() is None
        t = np.ravel(data["timeline"])
        _, y_ref, _ = signal.lsim(([1.0, 2.0], [2.0, 3.0]), np.ones_like(t), t)
        assert np.max(np.abs(np.ravel(data["signals"]["y"]) - y_ref)) < 1e-8

    def test_linearizer_solves_through_the_loop(self, qapp, tmp_path):
        from lib.analysis.linearizer import Linearizer
        from lib.cli import load_diagram

        path = tmp_path / "loop.diablos"
        self._proper_tf_loop().save(str(path))
        dsim, _ = load_diagram(str(path))
        names = {b.block_fn: b.name for b in dsim.blocks_list}
        res = Linearizer(dsim).linearize_at_point(
            input_blocks=[names["Step"]], output_blocks=[names["TranFn"]]
        )
        # (s+2)/(2s+3) = 0.5 + 0.25/(s+1.5)
        assert np.allclose(res["A"], [[-1.5]], atol=1e-5)
        assert np.allclose(res["D"], [[0.5]], atol=1e-6)
        assert np.allclose(np.ravel(res["B"]) * np.ravel(res["C"]), [0.25], atol=1e-5)

    @pytest.mark.parametrize("which", ["gain", "proper_tf"])
    def test_python_export_matches_the_compiled_run(self, qapp, tmp_path, which):
        import subprocess
        import sys

        from lib.cli import export_python
        from lib.diagram_builder import DiagramBuilder

        if which == "gain":
            b = DiagramBuilder(sim_time=0.3, sim_dt=0.1)
            c = b.add_block("Constant", 0, 0, name="u", params={"value": 1.0})
            s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+-"}, in_ports=2)
            g = b.add_block("Gain", 0, 0, name="g", params={"gain": 0.5})
            sc = b.add_block(
                "Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0
            )
            b.connect(c, 0, s, 0)
            b.connect(s, 0, g, 0)
            b.connect(g, 0, s, 1)
            b.connect(s, 0, sc, 0)
            sim = (0.3, 0.1)
        else:
            b = self._proper_tf_loop()
            sim = (3.0, 0.01)
        path = tmp_path / "loop.diablos"
        b.save(str(path))
        script = tmp_path / "loop.py"
        export_python(str(path), out_path=str(script))
        out = tmp_path / "run.npz"
        proc = subprocess.run(
            [sys.executable, str(script), "--no-plot", "--out", str(out)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        exported = np.load(out)
        ok, err, _, data = _run(b, tmp_path, *sim)
        assert ok, err
        y_ref = np.ravel(data["signals"]["y"])
        assert np.max(np.abs(np.ravel(exported["y"]) - y_ref)) < 1e-8
