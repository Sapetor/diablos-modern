"""GUI tests for the mask editor dialog (``modern_ui/widgets/mask_editor_dialog.py``).

The dialog is the only place a mask definition is authored, so these tests
drive its table directly (add / remove / reorder / type coercion) and check
that ``get_mask()`` produces a normalized definition -- and refuses to close on
an invalid one.
"""

import pytest
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QDialog

from lib import i18n
from lib.masks import MaskError, get_mask, set_mask
from modern_ui.widgets.mask_editor_dialog import (
    _COL_DEFAULT,
    _COL_DOC,
    _COL_NAME,
    _COL_OPTIONS,
    _COL_TYPE,
    MaskEditorDialog,
)

VEHICLE_MASK = {
    "name": "Vehicle",
    "description": "Force in, speed out.",
    "icon": "1/(ms+b)",
    "shape": "rect",
    "category": "User Library",
    "parameters": [
        {"name": "m", "type": "float", "default": 1500.0, "doc": "Mass [kg]"},
        {"name": "b", "type": "float", "default": 50.0, "doc": "Damping"},
    ],
}


@pytest.fixture
def subsystem(qapp):
    from blocks.subsystem import Subsystem

    block = Subsystem(block_name="Subsystem1", sid=1, coords=QRect(0, 0, 120, 90))
    block.name = "Subsystem1"
    return block


@pytest.fixture
def make_dialog(qapp):
    """Build MaskEditorDialogs and destroy them deterministically.

    Widgets left to Python's garbage collector are destroyed at an arbitrary
    later point -- in a full-suite run that lands inside another test's Qt
    event processing and aborts the interpreter. Deleting them here and
    flushing the deferred deletion keeps the teardown inside this fixture.
    """
    created = []

    def _make(**kwargs):
        dialog = MaskEditorDialog(**kwargs)
        created.append(dialog)
        return dialog

    yield _make

    for dialog in created:
        dialog.close()
        dialog.deleteLater()
    qapp.processEvents()


def _set(dialog, row, col, text):
    dialog.table.item(row, col).setText(text)


@pytest.mark.qt
class TestMaskEditorDialog:
    def test_blank_mask_for_an_unmasked_subsystem(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        assert dialog.table.rowCount() == 0
        # The subsystem's own name is offered as the display name.
        assert dialog.name_edit.text() == "Subsystem1"
        assert dialog.category_edit.text() == "User Library"

    def test_existing_mask_is_loaded_into_the_tabs(self, make_dialog, subsystem):
        set_mask(subsystem, VEHICLE_MASK)
        dialog = make_dialog(block=subsystem)
        assert dialog.name_edit.text() == "Vehicle"
        assert dialog.icon_edit.text() == "1/(ms+b)"
        assert dialog.description_edit.toPlainText() == "Force in, speed out."
        assert dialog.table.rowCount() == 2
        assert dialog.table.item(0, _COL_NAME).text() == "m"
        assert dialog.table.item(0, _COL_DEFAULT).text() == "1500.0"
        assert dialog.table.item(1, _COL_DOC).text() == "Damping"

    def test_add_edit_and_read_back_a_parameter(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        dialog.name_edit.setText("Vehicle")
        row = dialog.add_parameter()
        _set(dialog, row, _COL_NAME, "m")
        _set(dialog, row, _COL_DEFAULT, "1500")
        _set(dialog, row, _COL_DOC, "Mass [kg]")

        mask = dialog.get_mask()
        assert mask["name"] == "Vehicle"
        assert mask["parameters"] == [
            {"name": "m", "type": "float", "default": 1500.0, "doc": "Mass [kg]", "options": []}
        ]

    def test_list_and_choice_parameters(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        dialog.name_edit.setText("Plant")

        row = dialog.add_parameter()
        _set(dialog, row, _COL_NAME, "den")
        dialog.table.cellWidget(row, _COL_TYPE).setCurrentText("list")
        _set(dialog, row, _COL_DEFAULT, "[1, 2]")

        row = dialog.add_parameter()
        _set(dialog, row, _COL_NAME, "mode")
        dialog.table.cellWidget(row, _COL_TYPE).setCurrentText("choice")
        _set(dialog, row, _COL_OPTIONS, "fast, slow")
        _set(dialog, row, _COL_DEFAULT, "slow")

        mask = dialog.get_mask()
        assert mask["parameters"][0]["default"] == [1, 2]
        assert mask["parameters"][1]["options"] == ["fast", "slow"]
        assert mask["parameters"][1]["default"] == "slow"

    def test_an_unparseable_default_is_kept_as_an_expression(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        dialog.name_edit.setText("Plant")
        row = dialog.add_parameter()
        _set(dialog, row, _COL_NAME, "K")
        _set(dialog, row, _COL_DEFAULT, "base_gain")
        assert dialog.get_mask()["parameters"][0]["default"] == "base_gain"

    def test_remove_and_reorder(self, make_dialog, subsystem):
        set_mask(subsystem, VEHICLE_MASK)
        dialog = make_dialog(block=subsystem)
        dialog.table.selectRow(0)
        dialog.move_parameter(1)
        assert [p["name"] for p in dialog.get_mask()["parameters"]] == ["b", "m"]

        dialog.table.selectRow(0)
        dialog.remove_parameter()
        assert [p["name"] for p in dialog.get_mask()["parameters"]] == ["m"]

    def test_invalid_mask_keeps_the_dialog_open(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        dialog.name_edit.setText("Plant")
        row = dialog.add_parameter()
        _set(dialog, row, _COL_NAME, "2bad")

        with pytest.raises(MaskError):
            dialog.get_mask()

        dialog.accept()
        assert dialog.result() != QDialog.Accepted
        assert dialog.error_label.isVisible() or dialog.error_label.text()
        assert "identifier" in dialog.error_label.text()

    def test_accept_closes_on_a_valid_mask(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem)
        dialog.name_edit.setText("Plant")
        dialog.accept()
        assert dialog.result() == QDialog.Accepted

    def test_round_trip_through_set_mask(self, make_dialog, subsystem):
        dialog = make_dialog(block=subsystem, mask=VEHICLE_MASK)
        set_mask(subsystem, dialog.get_mask())
        mask = get_mask(subsystem)
        assert mask["name"] == "Vehicle"
        assert subsystem.params["m"] == 1500.0
        assert subsystem.params["b"] == 50.0


@pytest.mark.qt
class TestMaskEditorDialogLocalization:
    """The dialog is built once (no live retranslation), so the language
    must be switched *before* construction -- see i18n's "Live retranslation"
    note in docs/DEVELOPER_GUIDE.md.
    """

    def test_window_title_is_translated(self, make_dialog, subsystem):
        i18n.set_language("es")
        try:
            dialog = make_dialog(block=subsystem)
            assert dialog.windowTitle() == i18n.tr("Edit Mask")
            assert dialog.windowTitle() != "Edit Mask"
        finally:
            i18n.set_language("en")
