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
        "sim_data": {"wind_width": 1280, "wind_height": 770, "fps": 60,
                     "sim_time": sim_time, "sim_dt": sim_dt, "sim_trange": 100000},
        "blocks_data": [
            {"block_fn": "Step", "sid": 0, "username": "u",
             "coords_left": 50, "coords_top": 200, "coords_width": 50,
             "coords_height": 40, "coords_height_base": 40, "in_ports": 0,
             "out_ports": 1, "dragging": False, "selected": False,
             "b_color": "#064e3b", "b_type": 0, "io_edit": "none",
             "fn_name": "step", "params": {"value": step, "delay": 0.0, "type": "up"},
             "external": False, "flipped": False},
            {"block_fn": "Gain", "sid": 1, "username": "g",
             "coords_left": 200, "coords_top": 200, "coords_width": 50,
             "coords_height": 40, "coords_height_base": 40, "in_ports": 1,
             "out_ports": 1, "dragging": False, "selected": False,
             "b_color": "#1e3a8a", "b_type": 0, "io_edit": "none",
             "fn_name": "gain", "params": {"gain": gain},
             "external": False, "flipped": False},
            {"block_fn": "Scope", "sid": 2, "username": "y",
             "coords_left": 350, "coords_top": 200, "coords_width": 60,
             "coords_height": 50, "coords_height_base": 50, "in_ports": 1,
             "out_ports": 0, "dragging": False, "selected": False,
             "b_color": "#FFB6C1", "b_type": 3, "io_edit": "in",
             "fn_name": "scope", "params": {"labels": "y"},
             "external": False, "flipped": False},
        ],
        "lines_data": [
            {"name": "L0", "sid": 0, "srcblock": "step0", "srcport": 0,
             "dstblock": "gain1", "dstport": 0,
             "points": [[110, 220], [200, 220]], "cptr": 0, "selected": False},
            {"name": "L1", "sid": 1, "srcblock": "gain1", "srcport": 0,
             "dstblock": "scope2", "dstport": 0,
             "points": [[260, 220], [350, 220]], "cptr": 0, "selected": False},
        ],
        "version": "2.0",
    }


@pytest.fixture
def diagram(tmp_path):
    path = tmp_path / "step_gain.diablos"
    path.write_text(json.dumps(_step_gain_scope()))
    return str(path)


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
        assert rows[0] == "t,y"                      # time column + signal label
        last = [float(x) for x in rows[-1].split(",")]
        assert np.isclose(last[1], 2.0, atol=1e-6)   # final Scope value

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
        rc = cli.main(["run", diagram, "-o", str(out), "--solver", "interpreter",
                       "--time", "1", "--dt", "0.1"])
        assert rc == 0 and out.exists()
