"""Tests for the parameter-sweep runner (lib/analysis/parameter_sweep.py).

Builds a deterministic Constant -> Gain -> Scope diagram (output = value * gain,
flat in time) so swept-parameter outcomes are exactly predictable, then checks
1-D traces/metrics, the 2-D metric grid, cancellation, and diagram restoration.
Reuses the _params/_load helper pattern from tests/unit/test_monte_carlo.py.
"""

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from lib.analysis.parameter_sweep import ParameterSweepRunner


_BLOCK_INSTANCES = None


def _params(block_type, **overrides):
    global _BLOCK_INSTANCES
    if _BLOCK_INSTANCES is None:
        from lib.block_loader import load_blocks
        _BLOCK_INSTANCES = {}
        for cls in load_blocks():
            try:
                inst = cls()
                _BLOCK_INSTANCES[inst.block_name] = inst
            except Exception:
                pass
    inst = _BLOCK_INSTANCES.get(block_type)
    out = {}
    if inst is not None:
        for k, v in inst.params.items():
            out[k] = v['default'] if isinstance(v, dict) and 'default' in v else v
    out.update(overrides)
    return out


def _load(builder, tmp_path, name):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager
    path = tmp_path / name
    builder.save(str(path))
    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _name_of(dsim, block_fn):
    for b in dsim.blocks_list:
        if b.block_fn == block_fn:
            return b.name
    raise AssertionError(f"No block with block_fn={block_fn!r}")


def _const_gain_scope(tmp_path, name, value=1.0, gain=1.0):
    b = DiagramBuilder()
    c = b.add_block("Constant", 50, 100, params=_params("Constant", value=value))
    g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=gain))
    s = b.add_block("Scope", 350, 100, params=_params("Scope"))
    b.connect(c, 0, g, 0)
    b.connect(g, 0, s, 0)
    return _load(b, tmp_path, name)


def _only_signal(result):
    assert result["signals"], "sweep produced no signals"
    return result["signals"][sorted(result["signals"])[0]]


@pytest.mark.unit
class TestParameterSweep:
    def test_1d_sweep_traces_and_metrics(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sweep1d.diablos", value=1.0, gain=1.0)
        gain = _name_of(dsim, "Gain")
        values = [0.0, 1.0, 2.0, 3.0]

        res = ParameterSweepRunner(dsim).run(
            axes=[{"block": gain, "param": "gain", "values": values}],
            sim_time=0.5, sim_dt=0.05)

        assert res["mode"] == "1d"
        assert res["n_points"] == 4 and res["n_ok"] == 4
        assert res["axis"]["block"] == gain and res["axis"]["param"] == "gain"
        sig = _only_signal(res)
        # Response family: one row per swept value.
        assert sig["traces"].shape[0] == 4
        # output = gain * 1 = gain (flat), so final == swept value.
        assert np.allclose(sig["metrics"]["final"], values)
        assert np.allclose(sig["metrics"]["max"], values)
        assert np.allclose(sig["traces"][:, -1], values)

    def test_1d_sweep_restores_diagram(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sweepr.diablos", gain=2.5)
        gain = _name_of(dsim, "Gain")
        block = {b.name: b for b in dsim.blocks_list}[gain]
        before = block.params["gain"]

        ParameterSweepRunner(dsim).run(
            axes=[{"block": gain, "param": "gain", "values": [0.0, 5.0]}],
            sim_time=0.3, sim_dt=0.05)

        assert block.params["gain"] == before  # never mutates the diagram
        # exec_params must be restored in sync with params (not left desynced).
        if getattr(block, "exec_params", None):
            assert block.exec_params.get("gain") == before

    def test_2d_sweep_metric_grid(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sweep2d.diablos")
        const = _name_of(dsim, "Constant")
        gain = _name_of(dsim, "Gain")
        xv = [1.0, 2.0]          # Constant.value
        yv = [1.0, 2.0, 3.0]     # Gain.gain

        res = ParameterSweepRunner(dsim).run(
            axes=[{"block": const, "param": "value", "values": xv},
                  {"block": gain, "param": "gain", "values": yv}],
            sim_time=0.3, sim_dt=0.05)

        assert res["mode"] == "2d"
        assert res["n_points"] == 6 and res["n_ok"] == 6
        sig = _only_signal(res)
        Z = sig["metrics"]["final"]
        assert Z.shape == (2, 3)
        # output = value * gain -> final[i, j] == xv[i] * yv[j].
        expected = np.outer(xv, yv)
        assert np.allclose(Z, expected)

    def test_cancel_returns_partial_grid(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sweepc.diablos")
        gain = _name_of(dsim, "Gain")
        done = {"n": 0}
        res = ParameterSweepRunner(dsim).run(
            axes=[{"block": gain, "param": "gain", "values": list(range(10))}],
            sim_time=0.3, sim_dt=0.05,
            progress_cb=lambda d, t: done.__setitem__("n", d),
            cancel_cb=lambda: done["n"] >= 3)  # allow 3 points, then cancel
        assert res["n_points"] == 10
        assert res["n_ok"] == 3
        assert res["signals"], "partial sweep should still carry signals"

    def test_bad_axis_raises(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sweepbad.diablos")
        with pytest.raises(ValueError):
            ParameterSweepRunner(dsim).run(
                axes=[{"block": "NoSuchBlock", "param": "gain", "values": [0, 1]}])
        with pytest.raises(ValueError):
            ParameterSweepRunner(dsim).run(axes=[])
