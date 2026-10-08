"""Selectable PID anti-windup: interpreter, compiled kernel and exported code agree."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO_ROOT, "examples", "pid_second_order.diablos")
METHODS = ("clamping", "back_calculation", "none")
PROBES = [1.2, 2.0, 3.0, 5.0]


def _diagram(tmp_path, method=None, kb=None, limit=1.0, name="d.diablos"):
    """The pid_second_order example with tight output limits so it saturates."""
    with open(EXAMPLE) as fp:
        data = json.load(fp)
    for b in data["blocks_data"]:
        if b["block_fn"] == "PID":
            b["params"]["u_max"] = limit
            b["params"]["u_min"] = -limit
            b["params"].pop("anti_windup", None)
            if method is not None:
                b["params"]["anti_windup"] = method
            if kb is not None:
                b["params"]["kb"] = kb
    path = str(tmp_path / name)
    with open(path, "w") as fp:
        json.dump(data, fp)
    return path


def _run(path, fast, dt, t_end=6.0):
    from lib import cli
    from lib.analysis.resim import harvest_scope_signals

    dsim = cli.run_diagram(path, sim_time=t_end, sim_dt=dt, use_fast_solver=fast)
    res = harvest_scope_signals(dsim)
    return np.asarray(res["timeline"]), np.asarray(res["signals"]["output"])


@pytest.mark.unit
class TestParamDeclaration:
    def test_choice_param_and_default(self):
        from blocks.pid import PIDBlock

        spec = PIDBlock().params
        assert spec["anti_windup"]["type"] == "choice"
        assert spec["anti_windup"]["default"] == "clamping"
        assert list(spec["anti_windup"]["options"]) == list(METHODS)
        assert spec["kb"]["default"] == 1.0


@pytest.mark.unit
class TestInterpreterBlock:
    def _drive(self, method, dt, steps, **extra):
        from blocks.pid import PIDBlock

        blk = PIDBlock()
        params = {"Kp": 1.0, "Ki": 2.0, "Kd": 0.0, "u_min": -1.0, "u_max": 1.0}
        params.update(extra)
        if method is not None:
            params["anti_windup"] = method
        params.update({"_init_start_": True, "dtime": dt})
        u = None
        for k in range(steps):
            u = blk.execute(k * dt, {0: np.array([5.0]), 1: np.array([0.0])}, params, dtime=dt)
        return float(u[0][0]), params

    def test_missing_param_behaves_as_clamping(self):
        _, p_none = self._drive(None, 0.01, 200)
        _, p_clamp = self._drive("clamping", 0.01, 200)
        assert p_none["_int"] == pytest.approx(p_clamp["_int"])

    def test_clamping_freezes_integrator(self):
        _, p = self._drive("clamping", 0.01, 300)
        assert p["_int"] < 0.5  # stops as soon as Kp*e + Ki*I exceeds 1

    def test_none_keeps_winding_up(self):
        u, p = self._drive("none", 0.01, 300)
        assert u == 1.0
        assert p["_int"] == pytest.approx(5.0 * 0.01 * 299)

    def test_back_calculation_settles_near_saturation_boundary(self):
        _, p = self._drive("back_calculation", 0.01, 2000, kb=5.0)
        # Equilibrium of dI/dt = Ki*e + Kb*(u_sat - u): u -> u_max + Ki*e/Kb
        assert p["Ki"] * p["_int"] + p["Kp"] * 5.0 == pytest.approx(1.0 + 2.0 * 5.0 / 5.0, rel=5e-2)


@pytest.mark.unit
@pytest.mark.slow
class TestPathsAgree:
    @pytest.mark.parametrize("method", METHODS)
    def test_interpreter_converges_to_compiled(self, qapp, tmp_path, method):
        path = _diagram(tmp_path, method)
        tc, yc = _run(path, True, 0.01)
        ref = np.interp(PROBES, tc, yc)
        errs = []
        for dt in (0.01, 0.0025):
            t, y = _run(path, False, dt)
            errs.append(np.max(np.abs(np.interp(PROBES, t, y) - ref)))
        assert errs[1] < 0.005
        assert errs[0] < 0.01

    def test_clamping_compiled_is_unchanged(self, qapp):
        from lib import cli
        from lib.analysis.resim import harvest_scope_signals

        dsim = cli.run_diagram(EXAMPLE, use_fast_solver=True)
        res = harvest_scope_signals(dsim)
        got = np.interp([1.2, 1.6, 2.0, 2.4, 2.8], res["timeline"], res["signals"]["output"])
        np.testing.assert_allclose(
            got, [0.43613937, 0.93256068, 0.85248342, 0.77383648, 0.79228817], atol=1e-6
        )

    def test_default_equals_explicit_clamping(self, qapp, tmp_path):
        _, y_default = _run(_diagram(tmp_path, None, name="a.diablos"), True, 0.01)
        _, y_clamp = _run(_diagram(tmp_path, "clamping", name="b.diablos"), True, 0.01)
        np.testing.assert_array_equal(y_default, y_clamp)

    def test_back_calculation_gain_reduces_overshoot(self, qapp, tmp_path):
        def overshoot(method, kb=None):
            _, y = _run(_diagram(tmp_path, method, kb, name="o.diablos"), True, 0.01)
            return float(np.max(y) - 1.0)

        none = overshoot("none")
        weak = overshoot("back_calculation", 0.5)
        strong = overshoot("back_calculation", 10.0)
        assert strong < weak < none


@pytest.mark.unit
class TestLoading:
    def test_old_diagram_without_param_loads_as_clamping(self, qapp, tmp_path):
        from lib import cli

        path = _diagram(tmp_path, None)
        dsim, _ = cli.load_diagram(path)
        pid = next(b for b in dsim.blocks_list if b.block_fn == "PID")
        assert pid.params.get("anti_windup", "clamping") == "clamping"
        assert pid.params.get("kb", 1.0) == 1.0


@pytest.mark.unit
@pytest.mark.slow
class TestCodegen:
    @pytest.mark.parametrize("method", METHODS)
    def test_exported_script_matches_compiled(self, qapp, tmp_path, method):
        from lib import cli

        path = _diagram(tmp_path, method)
        script = str(tmp_path / "model_{}.py".format(method))
        cli.export_python(path, out_path=script)
        out = str(tmp_path / "gen.npz")
        env = dict(os.environ, MPLBACKEND="Agg")
        proc = subprocess.run(
            [sys.executable, script, "--out", out, "--no-plot"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(tmp_path),
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        gen = np.load(out)
        tc, yc = _run(path, True, 0.01)
        np.testing.assert_allclose(
            np.interp(PROBES, gen["t"], gen["output"]),
            np.interp(PROBES, tc, yc),
            atol=2e-3,
        )
