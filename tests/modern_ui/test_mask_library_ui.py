"""GUI tests for masks and user library blocks outside the mask editor.

* the property editor shows a masked subsystem's *mask* parameters (typed
  widgets and the mask's description), not the container's internals;
* the palette lists a discovered library block under the mask's category and
  dropping it instantiates an independent copy;
* a mask edit made through the window is undoable.
"""

import pytest
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QLabel, QLineEdit

from lib.library import LIBRARY_ENV_VAR, get_library_ref, write_library_file
from lib.masks import get_mask, set_mask
from lib.simulation.block import DBlock

VEHICLE_MASK = {
    "name": "Vehicle",
    "description": "Force in, speed out.",
    "icon": "1/(ms+b)",
    "shape": "rect",
    "category": "User Library",
    "parameters": [
        {"name": "m", "type": "float", "default": 1500.0, "doc": "Mass [kg]"},
        {"name": "b", "type": "float", "default": 50.0, "doc": "Damping [N s/m]"},
        {
            "name": "mode",
            "type": "choice",
            "default": "linear",
            "options": ["linear", "saturating"],
            "doc": "Drag model",
        },
    ],
}


def _masked_subsystem(name="Subsystem1", mask=VEHICLE_MASK):
    from blocks.subsystem import Subsystem

    block = Subsystem(block_name=name, sid=1, coords=QRect(100, 100, 120, 90))
    block.name = name
    inner = DBlock(
        block_fn="TranFn",
        sid=0,
        coords=QRect(10, 10, 80, 60),
        color="#4CAF50",
        in_ports=1,
        out_ports=1,
        b_type=1,
        io_edit="none",
        fn_name="tranfn",
        params={"numerator": [1.0], "denominator": "[m, b]"},
    )
    inner.name = "tranfn0"
    block.sub_blocks.append(inner)
    set_mask(block, mask)
    return block


# ---------------------------------------------------------------- property ---


@pytest.mark.qt
class TestPropertyEditorShowsMaskParameters:
    def test_mask_parameters_replace_the_raw_internals(self, qapp):
        from modern_ui.widgets.property_editor import PropertyEditor

        editor = PropertyEditor()
        try:
            editor.set_block(_masked_subsystem())
            assert sorted(editor._widgets) == ["b", "m", "mode"]
            # The mask definition itself is never offered as a parameter.
            assert "_mask" not in editor._widgets
        finally:
            editor.setParent(None)

    def test_widgets_are_typed_from_the_mask_spec(self, qapp):
        from modern_ui.widgets.property_editor import PropertyEditor

        editor = PropertyEditor()
        try:
            editor.set_block(_masked_subsystem())
            mode_editor = editor._widgets["mode"][0]
            assert isinstance(mode_editor, QComboBox)
            assert [mode_editor.itemText(i) for i in range(mode_editor.count())] == [
                "linear",
                "saturating",
            ]
            mass_editor = editor._widgets["m"][0]
            assert isinstance(mass_editor, (QDoubleSpinBox, QLineEdit)) or hasattr(
                mass_editor, "value"
            )
        finally:
            editor.setParent(None)

    def test_docs_come_from_the_mask(self, qapp):
        from modern_ui.widgets.property_editor import PropertyEditor

        editor = PropertyEditor()
        try:
            editor.set_block(_masked_subsystem())
            assert editor._get_param_metadata("m")["doc"] == "Mass [kg]"
            # The mask description is rendered in the panel's Documentation section.
            texts = [label.text() for label in editor.findChildren(QLabel)]
            assert any("Force in, speed out." in text for text in texts)
            # ...and the header names the block by its mask, not "Subsystem".
            assert any(text == "Vehicle" for text in texts)
        finally:
            editor.setParent(None)

    def test_an_unmasked_subsystem_still_uses_the_generic_path(self, qapp):
        from blocks.subsystem import Subsystem
        from modern_ui.widgets.property_editor import PropertyEditor

        editor = PropertyEditor()
        try:
            plain = Subsystem(block_name="Subsystem2", sid=2, coords=QRect(0, 0, 100, 80))
            editor.set_block(plain)
            assert editor._mask is None
            assert editor._widgets == {}
        finally:
            editor.setParent(None)


# ----------------------------------------------------------------- palette ---


@pytest.fixture
def library_dir(tmp_path, monkeypatch, qapp):
    """A temp library folder holding one 'Vehicle' block, wired to the env var."""
    from lib.models.simulation_model import SimulationModel
    from lib.services.file_service import FileService

    folder = tmp_path / "library"
    folder.mkdir()
    model = SimulationModel()
    block_data = FileService(model)._serialize_block(_masked_subsystem())
    write_library_file(block_data, directory=str(folder), block_id="vehicle")
    monkeypatch.setenv(LIBRARY_ENV_VAR, str(folder))
    return folder


@pytest.mark.qt
class TestPaletteLibraryBlocks:
    def test_library_block_is_registered_and_listed(self, qapp, library_dir):
        from lib.models.simulation_model import SimulationModel
        from modern_ui.widgets.modern_palette import CompactBlockRow, ModernBlockPalette

        model = SimulationModel()
        assert model.load_library_blocks() == 1

        class _Dsim:
            pass

        dsim = _Dsim()
        dsim.menu_blocks = model.menu_blocks
        dsim.colors = model.colors
        dsim.model = model

        palette = ModernBlockPalette(dsim)
        try:
            rows = palette.findChildren(CompactBlockRow)
            library_rows = [r for r in rows if getattr(r.menu_block, "library_def", None)]
            assert [r.menu_block.fn_name for r in library_rows] == ["Vehicle"]
            # ...filed under the mask's own category, not keyword-matched.
            assert library_rows[0].category_name == "User Library"
            assert "Force in, speed out." in library_rows[0].toolTip()
        finally:
            palette.setParent(None)

    def test_dropping_a_library_block_creates_an_independent_copy(self, qapp, library_dir):
        from lib.models.simulation_model import SimulationModel

        model = SimulationModel()
        model.load_library_blocks()
        entry = next(mb for mb in model.menu_blocks if getattr(mb, "library_def", None))

        instance = model.add_block(entry, QPoint(400, 300))
        assert instance is not None
        assert instance in model.blocks_list
        assert instance.block_fn == "Subsystem"
        assert instance.username == "Vehicle"

        mask = get_mask(instance)
        assert mask is not None and [p["name"] for p in mask["parameters"]] == ["m", "b", "mode"]
        assert instance.params["m"] == 1500.0
        assert get_library_ref(instance)["id"] == "vehicle"

        # Contents were copied in, unresolved.
        assert [b.block_fn for b in instance.sub_blocks] == ["TranFn"]
        assert instance.sub_blocks[0].params["denominator"] == "[m, b]"

        # A second instance is fully independent of the first.
        other = model.add_block(entry, QPoint(600, 300))
        instance.params["m"] = 900.0
        instance.sub_blocks[0].params["numerator"] = [2.0]
        assert other.params["m"] == 1500.0
        assert other.sub_blocks[0].params["numerator"] == [1.0]
        assert other.name != instance.name


# -------------------------------------------------------------------- undo ---


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    win = ModernDiaBloSWindow()
    yield win
    win.close()


@pytest.fixture(autouse=True)
def _restore_blocks(request):
    if "window" not in request.fixturenames:
        yield
        return
    win = request.getfixturevalue("window")
    dsim = win.canvas.dsim
    saved = list(dsim.blocks_list)
    saved_dirty = dsim.dirty
    yield
    dsim.blocks_list[:] = saved
    dsim.dirty = saved_dirty
    win.canvas.history_manager.undo_stack.clear()
    win.canvas.history_manager.redo_stack.clear()


@pytest.mark.qt
class TestMaskEditUndo:
    def test_applying_a_mask_marks_the_diagram_dirty(self, qapp, window):
        dsim = window.canvas.dsim
        dsim.blocks_list[:] = [_masked_subsystem(mask={"name": "Plain", "parameters": []})]
        dsim.dirty = False

        block = dsim.blocks_list[0]
        assert window.mask_library_manager.apply_mask(block, VEHICLE_MASK)
        assert dsim.dirty is True
        assert get_mask(block)["name"] == "Vehicle"
        assert block.params["m"] == 1500.0

    def test_undo_restores_the_previous_mask_and_contents(self, qapp, window):
        dsim = window.canvas.dsim
        original = _masked_subsystem(mask={"name": "Plain", "parameters": []})
        dsim.blocks_list[:] = [original]

        window.mask_library_manager.apply_mask(original, VEHICLE_MASK)
        assert "m" in dsim.blocks_list[0].params

        window.undo_action()

        restored = dsim.blocks_list[0]
        assert restored.block_fn == "Subsystem"
        assert get_mask(restored)["name"] == "Plain"
        assert "m" not in restored.params
        # Undo must not have flattened the subsystem into a bare DBlock.
        assert [b.block_fn for b in restored.sub_blocks] == ["TranFn"]

    def test_redo_reapplies_the_mask(self, qapp, window):
        dsim = window.canvas.dsim
        original = _masked_subsystem(mask={"name": "Plain", "parameters": []})
        dsim.blocks_list[:] = [original]

        window.mask_library_manager.apply_mask(original, VEHICLE_MASK)
        window.undo_action()
        window.redo_action()

        assert get_mask(dsim.blocks_list[0])["name"] == "Vehicle"
        assert dsim.blocks_list[0].params["m"] == 1500.0


# ------------------------------------------------------------------ library ---


@pytest.mark.qt
class TestSaveAndReloadLibraryBlock:
    def test_save_then_reload_keeps_the_instance_values(self, qapp, window, tmp_path):
        """A library update re-syncs contents but not the instance's tuning."""
        from lib.library import read_library_file

        dsim = window.canvas.dsim
        block = _masked_subsystem()
        dsim.blocks_list[:] = [block]
        block.selected = True

        path = window.save_as_library_block(block, str(tmp_path / "vehicle.diablos"))
        assert path is not None
        saved = read_library_file(path)
        assert saved is not None and saved.name == "Vehicle"
        assert get_library_ref(block)["id"] == "vehicle"

        # Tune this instance, then publish a changed library block.
        block.params["m"] = 900.0
        updated = dict(VEHICLE_MASK)
        updated["description"] = "Updated model."
        from lib.library import write_library_file as _write
        from lib.services.file_service import FileService

        fresh = _masked_subsystem(mask=updated)
        fresh.sub_blocks[0].params["numerator"] = [2.0]
        _write(
            FileService(dsim.model)._serialize_block(fresh),
            directory=str(tmp_path),
            block_id="vehicle",
            mask=updated,
        )

        monkey = str(tmp_path)
        import os

        previous = os.environ.get(LIBRARY_ENV_VAR)
        os.environ[LIBRARY_ENV_VAR] = monkey
        try:
            assert window.reload_from_library(block) is True
        finally:
            if previous is None:
                os.environ.pop(LIBRARY_ENV_VAR, None)
            else:
                os.environ[LIBRARY_ENV_VAR] = previous

        assert get_mask(block)["description"] == "Updated model."
        assert block.sub_blocks[0].params["numerator"] == [2.0]
        # The instance's own value survived the update.
        assert block.params["m"] == 900.0

    def test_reload_without_a_library_ref_is_refused(self, qapp, window):
        dsim = window.canvas.dsim
        block = _masked_subsystem()
        dsim.blocks_list[:] = [block]
        assert window.reload_from_library(block) is False


# -------------------------------------------------------------------- menus ---


@pytest.mark.qt
class TestMaskMenuEntries:
    def test_edit_menu_exposes_the_mask_and_library_actions(self, qapp, window):
        labels = set()
        for action in window.menuBar().actions():
            sub = action.menu()
            if sub is None:
                continue
            for entry in sub.actions():
                labels.add(entry.text())
        assert "Edit &Mask..." in labels
        assert "&Look Under Mask" in labels
        assert "Save as &Library Block..." in labels
        assert "Reload from Li&brary" in labels
        assert "Refresh Block Librar&y" in labels

    def test_context_menu_offers_mask_actions_for_a_subsystem(self, qapp, window, monkeypatch):
        from modern_ui.managers import menu_manager as mm

        dsim = window.canvas.dsim
        block = _masked_subsystem()
        block.selected = True
        dsim.blocks_list[:] = [block]

        shown = {}

        def _capture(self, *args, **kwargs):
            # Rows are QWidgetActions carrying QLabels (see _build_kbd_row),
            # so read the label text where there is one and fall back to the
            # plain action text otherwise.
            labels = []
            for action in self.actions():
                widget = action.defaultWidget() if hasattr(action, "defaultWidget") else None
                label = widget.findChild(QLabel) if widget is not None else None
                labels.append(label.text() if label is not None else action.text())
            shown["labels"] = labels
            return None

        monkeypatch.setattr(mm.QMenu, "exec_", _capture, raising=False)
        window.canvas.menu_manager.show_block_context_menu(block, QPoint(10, 10))

        joined = " | ".join(shown.get("labels", []))
        assert "Edit mask" in joined
        assert "Look under mask" in joined
        assert "Save as library block" in joined
