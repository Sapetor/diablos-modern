"""Importable headless simulation API (``lib/api.py``)."""

import os
import subprocess
import sys

import numpy as np
import pytest

from lib.api import Diagram, SimulationError, from_builder, load
from lib.diagram_builder import DiagramBuilder

pytestmark = pytest.mark.integration

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIRST_ORDER = os.path.join(REPO, "examples", "first_order_step_response.diablos")
MASKED_SUBSYSTEM = os.path.join(REPO, "examples", "library_block_demo.diablos")


def _ramp(gain=3.0, sim_time=2.0):
    """Step(1 at t=0) -> Gain -> Integrator -> Scope: the trace is gain * t."""
    b = DiagramBuilder(sim_time=sim_time, sim_dt=0.01)
    u = b.add_block("Step", 0, 0, name="u", params={"value": 1.0, "delay": 0.0})
    k = b.add_block("Gain", 100, 0, name="k", params={"gain": gain})
    x = b.add_block("Integrator", 200, 0, name="x")
    y = b.add_block("Scope", 300, 0, name="y", params={"labels": "x"})
    b.connect(u, 0, k, 0)
    b.connect(k, 0, x, 0)
    b.connect(x, 0, y, 0)
    return b


def _noisy(sim_time=1.0):
    """Constant + Noise -> Scope. Noise has no compiled kernel."""
    b = DiagramBuilder(sim_time=sim_time, sim_dt=0.01)
    c = b.add_block("Constant", 0, 0, name="c", params={"value": 1.0})
    n = b.add_block("Noise", 0, 100, name="n", params={"seed": 7})
    s = b.add_block("Sum", 100, 0, name="s", params={"sign": "++"})
    y = b.add_block("Scope", 200, 0, name="y", params={"labels": "y"})
    b.connect(c, 0, s, 0)
    b.connect(n, 0, s, 1)
    b.connect(s, 0, y, 0)
    return b


class TestLoadAndRun:
    def test_load_runs_a_diagram_file_with_its_own_settings(self):
        sim = load(FIRST_ORDER)
        assert isinstance(sim, Diagram)
        assert sim.blocks == ["step0", "tranfn1", "scope2"]
        assert sim.block_type("tranfn1") == "TranFn"
        assert sim.params("tranfn1")["denominator"] == [2.0, 1.0]

        res = sim.run()
        assert res.engine == "compiled"
        assert res.fallback_reason is None
        assert res.names == ["reference", "output"]
        # Defaults come from the file: 10 s at 0.01 s.
        assert res.time[-1] == pytest.approx(sim.sim_time)
        assert len(res.time) == round(sim.sim_time / sim.sim_dt) + 1
        assert all(len(v) == len(res.time) for v in res.signals.values())
        assert res.diagnostics["success"] is True
        assert res.diagnostics["method_used"] == "RK45"
        assert res.diagnostics["nfev"] > 0
        # 1/(2s+1) to a unit step at t=1: 1 - exp(-(t-1)/2).
        t = res.time
        expected = np.where(t >= 1.0, 1.0 - np.exp(-(t - 1.0) / 2.0), 0.0)
        assert np.allclose(res["output"], expected, atol=1e-4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(str(tmp_path / "nope.diablos"))

    def test_unknown_signal_lists_the_names(self):
        res = load(FIRST_ORDER).run(t=1.0)
        with pytest.raises(KeyError, match="output"):
            res["nope"]

    def test_bad_arguments(self):
        sim = load(FIRST_ORDER)
        with pytest.raises(ValueError, match="solver"):
            sim.run(solver="fast")
        with pytest.raises(ValueError, match="positive"):
            sim.run(t=0.0)
        with pytest.raises(KeyError, match="nope"):
            sim.set_param("nope", "gain", 1.0)
        with pytest.raises(KeyError, match="gain"):
            sim.set_param("tranfn1", "gain", 1.0)


class TestParameters:
    def test_set_param_changes_later_runs(self):
        sim = from_builder(_ramp(gain=3.0))
        assert sim.run()["x"][-1] == pytest.approx(6.0)
        sim.set_param("k", "gain", 5.0)
        assert sim.get_param("k", "gain") == 5.0
        assert sim.run()["x"][-1] == pytest.approx(10.0)

    def test_run_overrides_are_restored(self):
        sim = from_builder(_ramp(gain=3.0))
        res = sim.run(params={("k", "gain"): 0.5})
        assert res["x"][-1] == pytest.approx(1.0)
        assert sim.get_param("k", "gain") == 3.0
        assert sim.run()["x"][-1] == pytest.approx(6.0)

    def test_results_do_not_alias_between_runs(self):
        sim = from_builder(_ramp(gain=3.0))
        first = sim.run()
        sim.set_param("k", "gain", -1.0)
        sim.run()
        assert first["x"][-1] == pytest.approx(6.0)

    def test_nested_block_addressed_by_qualified_name(self):
        sim = load(MASKED_SUBSYSTEM)
        assert "Subsystem1/tranfn0" in sim.blocks
        assert sim.get_param("Subsystem1", "b") == 50.0
        before = sim.run()["speed"][-1]
        sim.set_param("Subsystem1", "b", 500.0)
        after = sim.run()["speed"][-1]
        # Closed loop 800/(1500 s + b + 800) to a step of 20: final 16000/(b+800).
        assert before == pytest.approx(16000 / 850, rel=1e-3)
        assert after == pytest.approx(16000 / 1300, rel=1e-3)

    def test_save_keeps_parameter_changes(self, tmp_path):
        sim = from_builder(_ramp(gain=3.0))
        sim.set_param("k", "gain", 4.0)
        path = str(tmp_path / "ramp.diablos")
        sim.save(path)
        again = load(path)
        assert again.get_param("k", "gain") == 4.0
        assert again.run()["x"][-1] == pytest.approx(8.0)


class TestEngines:
    def test_interpreter_run_reports_no_stale_compiled_diagnostics(self):
        sim = from_builder(_ramp())
        compiled = sim.run()
        assert compiled.engine == "compiled" and compiled.diagnostics
        interp = sim.run(solver="interpreter")
        assert interp.engine == "interpreter"
        assert interp.diagnostics == {}
        assert interp.fallback_reason is None
        assert np.allclose(interp["x"], compiled["x"], atol=1e-6)

    def test_fallback_to_interpreter_says_why(self):
        res = from_builder(_noisy()).run()
        assert res.engine == "interpreter"
        assert res.diagnostics == {}
        assert "Noise" in res.fallback_reason
        assert len(res["y"]) == len(res.time)

    def test_solver_settings_reach_the_compiled_run(self):
        res = from_builder(_ramp()).run(method="Radau", rtol=1e-6, atol=1e-9)
        assert res.diagnostics["method_used"] == "Radau"
        assert res.diagnostics["rtol"] == 1e-6
        assert res.diagnostics["atol"] == 1e-9


class TestBuilder:
    def test_builder_names_and_default_params(self):
        sim = from_builder(_ramp())
        assert sim.blocks == ["u", "k", "x", "y"]
        assert sim.sim_time == 2.0
        # Defaults the builder never set are filled in from the block type.
        assert "init_conds" in sim.params("x")

    def test_duplicate_builder_names_rejected(self):
        b = DiagramBuilder()
        b.add_block("Constant", 0, 0, name="a")
        b.add_block("Gain", 100, 0, name="a")
        with pytest.raises(ValueError, match="Duplicate"):
            from_builder(b)

    def test_to_dict_matches_save(self, tmp_path):
        import json

        b = _ramp()
        path = str(tmp_path / "ramp.dat")
        b.save(path)
        with open(path) as f:
            assert json.load(f) == json.loads(json.dumps(b.to_dict()))


class TestOutput:
    @pytest.mark.parametrize("ext", ["csv", "npz"])
    def test_save_signals(self, tmp_path, ext):
        res = from_builder(_ramp()).run()
        path = str(tmp_path / f"out.{ext}")
        assert res.save(path) == 1
        if ext == "npz":
            data = np.load(path)
            assert np.allclose(data["x"], res["x"])
        else:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            assert np.allclose(data[:, 1], res["x"])


class TestAnalysis:
    def test_monte_carlo_is_reproducible_and_varies(self):
        sim = from_builder(_noisy(sim_time=0.5))
        a = sim.monte_carlo(3, seed=1)
        b = sim.monte_carlo(3, seed=1)
        assert a["n_ok"] == 3
        runs = a["signals"]["y"]["runs"]
        assert np.array_equal(runs, b["signals"]["y"]["runs"])
        assert not np.allclose(runs[0], runs[1])
        assert sim.get_param("n", "seed") == 7

    def test_monte_carlo_lean_and_parallel_match(self):
        sim = from_builder(_noisy(sim_time=0.5))
        full = sim.monte_carlo(4, seed=1)
        lean = sim.monte_carlo(4, seed=1, keep_traces=False, workers=2)
        assert lean["signals"]["y"]["p50"] is None
        assert np.array_equal(lean["signals"]["y"]["mean"], full["signals"]["y"]["mean"])

    def test_sweep_1d(self):
        sim = from_builder(_ramp())
        res = sim.sweep([{"block": "k", "param": "gain", "values": [1.0, 2.0, 4.0]}])
        assert res["mode"] == "1d" and res["n_ok"] == 3
        assert np.allclose(res["signals"]["x"]["metrics"]["final"], [2.0, 4.0, 8.0], rtol=1e-6)
        assert sim.get_param("k", "gain") == 3.0

    def test_linearize_and_operating_point(self):
        sim = load(FIRST_ORDER)
        lin = sim.linearize(inputs=["step0"], outputs=["tranfn1"])
        assert np.allclose(lin["A"], [[-0.5]])
        # DC gain of 1/(2s+1): D - C A^-1 B = 1.
        dc = lin["D"] - lin["C"] @ np.linalg.solve(lin["A"], lin["B"])
        assert np.allclose(dc, [[1.0]])
        trim = sim.find_operating_point()
        assert trim["success"]
        assert np.allclose(trim["y"], [0.0])

    def test_linearize_sees_parameters_set_after_a_run(self):
        # A run leaves flattened copies of the Subsystem contents behind;
        # linearizing must not reuse them with the old mask parameters.
        sim = load(MASKED_SUBSYSTEM)
        sim.run()
        sim.set_param("Subsystem1", "b", 500.0)
        a = sim.linearize()["A"]
        assert np.allclose(a, [[-(500.0 + 800.0) / 1500.0]], rtol=1e-5)

    def test_linearize_without_states_raises(self):
        b = DiagramBuilder()
        c = b.add_block("Constant", 0, 0, name="c")
        y = b.add_block("Scope", 100, 0, name="y")
        b.connect(c, 0, y, 0)
        with pytest.raises(ValueError, match="no continuous states"):
            from_builder(b).linearize()


def test_failed_run_raises_simulation_error():
    b = DiagramBuilder(sim_time=1.0)
    k = b.add_block("Gain", 0, 0, name="k", params={"gain": "undefined_name"})
    c = b.add_block("Constant", -100, 0, name="c")
    y = b.add_block("Scope", 100, 0, name="y")
    b.connect(c, 0, k, 0)
    b.connect(k, 0, y, 0)
    with pytest.raises(SimulationError):
        from_builder(b).run(solver="interpreter")


def test_cold_start_without_a_qapplication():
    """A fresh interpreter with no Qt set up runs a diagram and exits cleanly."""
    code = (
        "from lib.api import load\n"
        f"res = load({FIRST_ORDER!r}).run(t=2.0)\n"
        "from PyQt6.QtGui import QGuiApplication\n"
        "print(res.engine, QGuiApplication.platformName(), round(float(res['output'][-1]), 4))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.split()[-3:] == ["compiled", "offscreen", "0.3935"]
