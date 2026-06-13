"""
Tests for LinearizeDialog (modern_ui/widgets/linearize_dialog.py).

Builds a small diagram with DiagramBuilder (reusing the test_linearizer
pattern), loads it into a DSim, constructs the dialog headlessly (no exec_()),
and verifies:
  * it is a QDialog,
  * the input candidate list contains the source block,
  * the output candidate list contains signal-producing blocks,
  * get_selection() returns the shared contract dict shape.
"""

import pytest
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt

from lib.diagram_builder import DiagramBuilder
from modern_ui.widgets.linearize_dialog import LinearizeDialog


_BLOCK_INSTANCES = None


def _params(block_type, **overrides):
    """Full flat params for a block (class defaults + overrides)."""
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
    """Save a built diagram and load it into a fresh DSim."""
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
    raise AssertionError(f"No block with block_fn={block_fn!r} in diagram")


def _item_data(list_widget):
    """All UserRole (block.name) values currently in a QListWidget."""
    return [list_widget.item(i).data(Qt.UserRole) for i in range(list_widget.count())]


@pytest.mark.unit
class TestLinearizeDialog:
    def _simple_dsim(self, tmp_path):
        """Constant -> Integrator -> Scope."""
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        i = b.add_block("Integrator", 200, 100, params=_params("Integrator", init_conds=0.0))
        sc = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, i, 0)
        b.connect(i, 0, sc, 0)
        return _load(b, tmp_path, "dlg.diablos")

    def test_is_qdialog(self, qapp, tmp_path):
        dsim = self._simple_dsim(tmp_path)
        dlg = LinearizeDialog(dsim)
        assert isinstance(dlg, QDialog)

    def test_input_list_populated_with_source(self, qapp, tmp_path):
        dsim = self._simple_dsim(tmp_path)
        dlg = LinearizeDialog(dsim)

        const_name = _name_of(dsim, "Constant")
        input_names = _item_data(dlg.input_list)
        assert const_name in input_names

        # A pure sink (Scope) is NOT an input candidate.
        scope_name = _name_of(dsim, "Scope")
        assert scope_name not in input_names

    def test_output_list_has_signal_blocks(self, qapp, tmp_path):
        dsim = self._simple_dsim(tmp_path)
        dlg = LinearizeDialog(dsim)

        output_names = _item_data(dlg.output_list)
        # Integrator and Constant produce signals (have output ports).
        assert _name_of(dsim, "Integrator") in output_names
        assert _name_of(dsim, "Constant") in output_names
        # Scope has no outputs -> not an output candidate.
        assert _name_of(dsim, "Scope") not in output_names

    def test_get_selection_contract_shape(self, qapp, tmp_path):
        dsim = self._simple_dsim(tmp_path)
        dlg = LinearizeDialog(dsim)

        sel = dlg.get_selection()
        assert set(sel.keys()) == {"input_blocks", "output_blocks", "find_trim"}
        assert isinstance(sel["input_blocks"], list)
        assert isinstance(sel["output_blocks"], list)
        assert isinstance(sel["find_trim"], bool)
        # Nothing selected by default.
        assert sel["input_blocks"] == []
        assert sel["output_blocks"] == []
        assert sel["find_trim"] is False

    def test_get_selection_reflects_user_choices(self, qapp, tmp_path):
        dsim = self._simple_dsim(tmp_path)
        dlg = LinearizeDialog(dsim)

        const_name = _name_of(dsim, "Constant")
        integ_name = _name_of(dsim, "Integrator")

        # Select the source in inputs and the integrator in outputs.
        for i in range(dlg.input_list.count()):
            item = dlg.input_list.item(i)
            if item.data(Qt.UserRole) == const_name:
                item.setSelected(True)
        for i in range(dlg.output_list.count()):
            item = dlg.output_list.item(i)
            if item.data(Qt.UserRole) == integ_name:
                item.setSelected(True)
        dlg.trim_checkbox.setChecked(True)

        sel = dlg.get_selection()
        assert sel["input_blocks"] == [const_name]
        assert sel["output_blocks"] == [integ_name]
        assert sel["find_trim"] is True
