"""Tests for the parameter-sweep run dialog
(modern_ui/widgets/parameter_sweep_dialog.py).

Builds a tiny diagram, constructs the dialog, and checks get_selection() returns
the documented contract for 1-D and 2-D, with values expanded via linspace.
"""

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from modern_ui.widgets.parameter_sweep_dialog import (
    ParameterSweepDialog, numeric_scalar_params, sweepable_blocks,
)


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


def _const_gain_scope(tmp_path, name):
    b = DiagramBuilder()
    c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
    g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=1.0))
    s = b.add_block("Scope", 350, 100, params=_params("Scope"))
    b.connect(c, 0, g, 0)
    b.connect(g, 0, s, 0)
    return _load(b, tmp_path, name)


@pytest.mark.unit
class TestParameterSweepDialog:
    def test_numeric_scalar_param_detection(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "psd_detect.diablos")
        gain_block = {b.name: b for b in dsim.blocks_list}[_name_of(dsim, "Gain")]
        assert "gain" in numeric_scalar_params(gain_block)
        assert sweepable_blocks(dsim), "Constant/Gain expose numeric params"

    def test_default_selection_is_1d(self, qapp, tmp_path):
        from PyQt5.QtWidgets import QDialog
        dsim = _const_gain_scope(tmp_path, "psd_1d.diablos")
        dsim.sim_time = 7.5
        dsim.sim_dt = 0.02

        dialog = ParameterSweepDialog(dsim)
        assert isinstance(dialog, QDialog)

        sel = dialog.get_selection()
        assert set(sel) == {"axes", "sim_time", "sim_dt"}
        assert len(sel["axes"]) == 1  # 1-D by default
        ax = sel["axes"][0]
        assert set(ax) == {"block", "param", "values"}
        assert len(ax["values"]) == 11  # default points
        assert sel["sim_time"] == pytest.approx(7.5)
        assert sel["sim_dt"] == pytest.approx(0.02)

    def test_2d_selection_and_value_range(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "psd_2d.diablos")
        gain = _name_of(dsim, "Gain")
        const = _name_of(dsim, "Constant")

        dialog = ParameterSweepDialog(dsim)
        dialog.mode_combo.setCurrentIndex(1)  # 2-D

        # Configure axis X = Gain.gain over [0, 2] with 3 points -> [0, 1, 2].
        dialog._x["block"].setCurrentText(gain)
        dialog._x["param"].setCurrentText("gain")
        dialog._x["min"].setValue(0.0)
        dialog._x["max"].setValue(2.0)
        dialog._x["points"].setValue(3)

        dialog._y["block"].setCurrentText(const)
        dialog._y["param"].setCurrentText("value")
        dialog._y["min"].setValue(0.0)
        dialog._y["max"].setValue(1.0)
        dialog._y["points"].setValue(2)

        sel = dialog.get_selection()
        assert len(sel["axes"]) == 2
        axx, axy = sel["axes"]
        assert axx["block"] == gain and axx["param"] == "gain"
        assert np.allclose(axx["values"], [0.0, 1.0, 2.0])
        assert axy["block"] == const and axy["param"] == "value"
        assert np.allclose(axy["values"], [0.0, 1.0])
