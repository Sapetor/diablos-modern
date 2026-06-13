"""Tests for the Monte-Carlo ensemble run dialog
(modern_ui/widgets/monte_carlo_dialog.py).

Builds a tiny diagram via DiagramBuilder, loads a DSim (reusing the
_params/_load helper pattern from tests/unit/test_monte_carlo.py), constructs a
MonteCarloDialog, and checks that get_selection() returns the documented
contract dict with the expected defaults. The dialog is never exec_()'d.
"""

import pytest

from lib.diagram_builder import DiagramBuilder
from modern_ui.widgets.monte_carlo_dialog import MonteCarloDialog


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


@pytest.mark.unit
class TestMonteCarloDialog:
    def test_is_qdialog_and_default_selection(self, qapp, tmp_path):
        from PyQt5.QtWidgets import QDialog

        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 250, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "mc_dialog.diablos")

        # Give the diagram known sim_time / sim_dt to verify they seed defaults.
        dsim.sim_time = 7.5
        dsim.sim_dt = 0.02

        dialog = MonteCarloDialog(dsim)
        assert isinstance(dialog, QDialog)

        sel = dialog.get_selection()
        assert set(sel) == {"n_runs", "master_seed", "sim_time", "sim_dt"}

        # Defaults from the spec.
        assert sel["n_runs"] == 100
        assert sel["master_seed"] == 12345

        # sim_time / sim_dt default to the diagram's values.
        assert isinstance(sel["sim_time"], float)
        assert isinstance(sel["sim_dt"], float)
        assert sel["sim_time"] == pytest.approx(7.5)
        assert sel["sim_dt"] == pytest.approx(0.02)

        # Spin-box ranges per spec.
        assert dialog.n_runs_spin.minimum() == 2
        assert dialog.n_runs_spin.maximum() == 100000
        assert dialog.master_seed_spin.minimum() == 0
        assert dialog.master_seed_spin.maximum() == 2_000_000_000

        # Edited values flow through get_selection().
        dialog.n_runs_spin.setValue(250)
        dialog.master_seed_spin.setValue(0)
        assert dialog.get_selection()["n_runs"] == 250
        assert dialog.get_selection()["master_seed"] == 0
