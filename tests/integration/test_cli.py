"""Headless CLI (``lib/cli.py``) — run a diagram and export Scope traces."""

import json

import numpy as np
import pytest

from lib import cli


def _step_gain_scope(sim_time=1.0, sim_dt=0.1, gain=2.0, step=1.0):
    """Step(value=step) -> Gain(gain) -> Scope, as a .diablos-format dict.

    Output is a constant ``gain*step`` once the step fires, so the exported
    trace is trivially checkable.
    """
    return {
        "sim_data": {
            "wind_width": 1280,
            "wind_height": 770,
            "fps": 60,
            "sim_time": sim_time,
            "sim_dt": sim_dt,
            "sim_trange": 100000,
        },
        "blocks_data": [
            {
                "block_fn": "Step",
                "sid": 0,
                "username": "u",
                "coords_left": 50,
                "coords_top": 200,
                "coords_width": 50,
                "coords_height": 40,
                "coords_height_base": 40,
                "in_ports": 0,
                "out_ports": 1,
                "dragging": False,
                "selected": False,
                "b_color": "#064e3b",
                "b_type": 0,
                "io_edit": "none",
                "fn_name": "step",
                "params": {"value": step, "delay": 0.0, "type": "up"},
                "external": False,
                "flipped": False,
            },
            {
                "block_fn": "Gain",
                "sid": 1,
                "username": "g",
                "coords_left": 200,
                "coords_top": 200,
                "coords_width": 50,
                "coords_height": 40,
                "coords_height_base": 40,
                "in_ports": 1,
                "out_ports": 1,
                "dragging": False,
                "selected": False,
                "b_color": "#1e3a8a",
                "b_type": 0,
                "io_edit": "none",
                "fn_name": "gain",
                "params": {"gain": gain},
                "external": False,
                "flipped": False,
            },
            {
                "block_fn": "Scope",
                "sid": 2,
                "username": "y",
                "coords_left": 350,
                "coords_top": 200,
                "coords_width": 60,
                "coords_height": 50,
                "coords_height_base": 50,
                "in_ports": 1,
                "out_ports": 0,
                "dragging": False,
                "selected": False,
                "b_color": "#FFB6C1",
                "b_type": 3,
                "io_edit": "in",
                "fn_name": "scope",
                "params": {"labels": "y"},
                "external": False,
                "flipped": False,
            },
        ],
        "lines_data": [
            {
                "name": "L0",
                "sid": 0,
                "srcblock": "step0",
                "srcport": 0,
                "dstblock": "gain1",
                "dstport": 0,
                "points": [[110, 220], [200, 220]],
                "cptr": 0,
                "selected": False,
            },
            {
                "name": "L1",
                "sid": 1,
                "srcblock": "gain1",
                "srcport": 0,
                "dstblock": "scope2",
                "dstport": 0,
                "points": [[260, 220], [350, 220]],
                "cptr": 0,
                "selected": False,
            },
        ],
        "version": "2.0",
    }


def _step_integrator_scope(sim_time=1.0, sim_dt=0.1, **solver_settings):
    """Step(1) -> Integrator -> Scope, so the run has an actual ODE state.

    ``solver_settings`` are merged into ``sim_data`` (solver_method, rtol, atol,
    zero_crossing) to stand in for a diagram saved with non-default solver
    settings.
    """
    data = _step_gain_scope(sim_time=sim_time, sim_dt=sim_dt)
    gain = data["blocks_data"][1]
    gain.update(
        {
            "block_fn": "Integrator",
            "username": "i",
            "fn_name": "integrator",
            "params": {"init_conds": 0.0, "method": "SOLVE_IVP"},
        }
    )
    for line in data["lines_data"]:
        if line["dstblock"] == "gain1":
            line["dstblock"] = "integrator1"
        if line["srcblock"] == "gain1":
            line["srcblock"] = "integrator1"
    data["sim_data"].update(solver_settings)
    return data


@pytest.fixture
def diagram(tmp_path):
    path = tmp_path / "step_gain.diablos"
    path.write_text(json.dumps(_step_gain_scope()))
    return str(path)


@pytest.fixture
def make_ode_diagram(tmp_path):
    """Write a Step -> Integrator -> Scope diagram with the given sim_data."""
    counter = {"n": 0}

    def _make(**solver_settings):
        counter["n"] += 1
        path = tmp_path / "ode{}.diablos".format(counter["n"])
        path.write_text(json.dumps(_step_integrator_scope(**solver_settings)))
        return str(path)

    return _make


@pytest.mark.integration
class TestCliRun:
    def test_run_diagram_returns_finished_dsim(self, qapp, diagram):
        dsim = cli.run_diagram(diagram, sim_time=1.0, sim_dt=0.1)
        from lib.analysis.resim import harvest_scope_signals

        result = harvest_scope_signals(dsim)
        assert result is not None and result["signals"]
        # Step(1) * Gain(2) settles at 2.0.
        trace = next(iter(result["signals"].values()))
        assert np.isclose(trace[-1], 2.0, atol=1e-6)

    def test_missing_file_returns_code_2(self, qapp, tmp_path):
        assert cli.main(["run", str(tmp_path / "nope.diablos")]) == 2

    def test_main_writes_csv(self, qapp, diagram, tmp_path):
        out = tmp_path / "out.csv"
        rc = cli.main(["run", diagram, "-o", str(out), "--time", "1", "--dt", "0.1"])
        assert rc == 0 and out.exists()
        rows = out.read_text().strip().splitlines()
        assert rows[0] == "t,y"  # time column + signal label
        last = [float(x) for x in rows[-1].split(",")]
        assert np.isclose(last[1], 2.0, atol=1e-6)  # final Scope value

    def test_main_writes_npz(self, qapp, diagram, tmp_path):
        out = tmp_path / "out.npz"
        rc = cli.main(["run", diagram, "-o", str(out)])
        assert rc == 0 and out.exists()
        data = np.load(out)
        assert "t" in data and "y" in data
        assert np.isclose(data["y"][-1], 2.0, atol=1e-6)

    def test_default_out_path_and_file_sim_params(self, qapp, diagram):
        # No -o: defaults to the diagram name with .csv; no --time/--dt: reads
        # sim_data from the file (sim_time=1.0, sim_dt=0.1 -> 11 samples).
        rc = cli.main(["run", diagram, "-q"])
        assert rc == 0
        import os

        default_csv = os.path.splitext(diagram)[0] + ".csv"
        assert os.path.exists(default_csv)
        rows = open(default_csv).read().strip().splitlines()
        assert len(rows) - 1 == 11  # header + 11 samples (t = 0.0 .. 1.0 step 0.1)

    def test_interpreter_solver_runs(self, qapp, diagram, tmp_path):
        out = tmp_path / "interp.csv"
        rc = cli.main(
            [
                "run",
                diagram,
                "-o",
                str(out),
                "--solver",
                "interpreter",
                "--time",
                "1",
                "--dt",
                "0.1",
            ]
        )
        assert rc == 0 and out.exists()


@pytest.mark.integration
class TestCliHonorsFileSolverSettings:
    """A headless run must reproduce what the GUI would do with the same file.

    ``apply_loaded_data`` returns the diagram's solver settings but does not push
    them onto the DSim facade, so the CLI used to integrate every diagram with
    the built-in default (RK45 / 1e-9 / 1e-12) no matter what was saved.
    """

    def _diagnostics(self, dsim):
        return dsim.engine.get_solver_diagnostics()

    def test_file_solver_method_reaches_the_engine(self, qapp, make_ode_diagram):
        path = make_ode_diagram(solver_method="LSODA", rtol=1e-5, atol=1e-8)
        dsim = cli.run_diagram(path)

        assert dsim.solver_method == "LSODA"
        assert dsim.rtol == 1e-5
        assert dsim.atol == 1e-8
        diag = self._diagnostics(dsim)
        assert diag["method_used"] == "LSODA"
        assert diag["rtol"] == 1e-5
        assert diag["atol"] == 1e-8

    def test_file_zero_crossing_off_reaches_the_engine(self, qapp, make_ode_diagram):
        path = make_ode_diagram(zero_crossing=False)
        dsim = cli.run_diagram(path)
        assert dsim.zero_crossing is False

    def test_file_auto_method_resolves_to_lsoda(self, qapp, make_ode_diagram):
        path = make_ode_diagram(solver_method="auto")
        diag = self._diagnostics(cli.run_diagram(path))
        assert diag["method_requested"] == "auto"
        assert diag["method_used"] == "LSODA"

    def test_legacy_file_without_solver_keys_still_uses_the_defaults(self, qapp, make_ode_diagram):
        path = make_ode_diagram()
        dsim = cli.run_diagram(path)
        assert dsim.solver_method == "RK45"
        assert dsim.rtol == 1e-9
        assert dsim.atol == 1e-12
        assert dsim.zero_crossing is True

    def test_explicit_arguments_override_the_file(self, qapp, make_ode_diagram):
        path = make_ode_diagram(solver_method="LSODA", rtol=1e-5, atol=1e-8)
        dsim = cli.run_diagram(path, solver_method="Radau", rtol=1e-4, atol=1e-7)

        assert dsim.solver_method == "Radau"
        assert self._diagnostics(dsim)["method_used"] == "Radau"
        assert dsim.rtol == 1e-4
        assert dsim.atol == 1e-7

    def test_the_run_is_still_numerically_right(self, qapp, make_ode_diagram):
        # Integral of a unit step over [0, 1] is 1.0, whichever method ran.
        from lib.analysis.resim import harvest_scope_signals

        path = make_ode_diagram(solver_method="BDF")
        result = harvest_scope_signals(cli.run_diagram(path, sim_time=1.0, sim_dt=0.1))
        trace = next(iter(result["signals"].values()))
        assert np.isclose(trace[-1], 1.0, atol=1e-5)


@pytest.mark.integration
class TestCliSolverFlags:
    def _diag_after(self, argv, qapp):
        """Run ``cli.main(argv)`` and return the engine diagnostics it produced."""
        captured = {}
        original = cli.run_diagram

        def spy(*args, **kwargs):
            dsim = original(*args, **kwargs)
            captured["dsim"] = dsim
            return dsim

        cli.run_diagram = spy
        try:
            assert cli.main(argv) == 0
        finally:
            cli.run_diagram = original
        return captured["dsim"].engine.get_solver_diagnostics()

    def test_method_flag_beats_the_file(self, qapp, make_ode_diagram, tmp_path):
        path = make_ode_diagram(solver_method="LSODA")
        out = tmp_path / "m.csv"
        diag = self._diag_after(["run", path, "-o", str(out), "--method", "Radau", "-q"], qapp)
        assert diag["method_used"] == "Radau"

    def test_method_flag_accepts_auto(self, qapp, make_ode_diagram, tmp_path):
        path = make_ode_diagram()
        out = tmp_path / "a.csv"
        diag = self._diag_after(["run", path, "-o", str(out), "--method", "auto", "-q"], qapp)
        assert diag["method_requested"] == "auto"
        assert diag["method_used"] == "LSODA"

    def test_tolerance_flags_beat_the_file(self, qapp, make_ode_diagram, tmp_path):
        path = make_ode_diagram(rtol=1e-5, atol=1e-8)
        out = tmp_path / "t.csv"
        diag = self._diag_after(
            ["run", path, "-o", str(out), "--rtol", "1e-3", "--atol", "1e-6", "-q"], qapp
        )
        assert diag["rtol"] == 1e-3
        assert diag["atol"] == 1e-6

    def test_no_zero_crossing_flag_beats_the_file(self, qapp, make_ode_diagram, tmp_path):
        path = make_ode_diagram(zero_crossing=True)
        out = tmp_path / "z.csv"
        captured = {}
        original = cli.run_diagram

        def spy(*args, **kwargs):
            captured["dsim"] = original(*args, **kwargs)
            return captured["dsim"]

        cli.run_diagram = spy
        try:
            assert cli.main(["run", path, "-o", str(out), "--no-zero-crossing", "-q"]) == 0
        finally:
            cli.run_diagram = original
        assert captured["dsim"].zero_crossing is False

    def test_export_python_bakes_in_the_resolved_auto_method(
        self, qapp, make_ode_diagram, tmp_path
    ):
        # "auto" is a DiaBloS setting, not a scipy method name; the generated
        # script must name the scheme a real run would use.
        path = make_ode_diagram(solver_method="auto")
        out = tmp_path / "model.py"
        assert cli.main(["export-python", path, "-o", str(out), "-q"]) == 0
        assert 'METHOD = "LSODA"' in out.read_text()
