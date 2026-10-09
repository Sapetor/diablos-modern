"""Dirty state follows the last explicit save, load or new diagram."""

import pytest
from PyQt6.QtCore import QPoint

pytestmark = [pytest.mark.regression, pytest.mark.qt]


@pytest.fixture
def document(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False
    w.close()


def add(w, x=100):
    menu = next(m for m in w.dsim.menu_blocks if m.block_fn == "Gain")
    return w.canvas.add_block_from_palette(menu, QPoint(x, 100))


def save(w, tmp_path):
    target = str(tmp_path / "saved.diablos")
    assert w.project_manager.diagram_service.save_diagram(target)
    return target


def test_undo_to_fresh_document_and_redo_away(document):
    add(document)
    assert document.dsim.dirty
    assert document.canvas.undo()
    assert not document.dsim.dirty
    assert document.canvas.redo()
    assert document.dsim.dirty


def test_save_clears_dirty_and_undo_redo_find_saved_state(document, tmp_path):
    add(document)
    save(document, tmp_path)
    assert not document.dsim.dirty
    add(document, 300)
    assert document.canvas.undo()
    assert not document.dsim.dirty
    assert document.canvas.redo()
    assert document.dsim.dirty


def test_redo_to_saved_state_is_clean(document, tmp_path):
    add(document)
    add(document, 300)
    save(document, tmp_path)
    assert document.canvas.undo()
    assert document.dsim.dirty
    assert document.canvas.redo()
    assert not document.dsim.dirty


@pytest.mark.parametrize("reset", ["load", "new", "canvas_new"])
def test_replacing_document_clears_history_and_sets_clean_origin(document, tmp_path, reset):
    add(document)
    target = save(document, tmp_path)
    add(document, 300)
    assert document.canvas.undo()  # Populate redo as well as undo.
    service = document.project_manager.diagram_service
    if reset == "load":
        assert service.load_diagram(target)
    elif reset == "new":
        service.new_diagram()
    else:
        document.new_diagram()
    hm = document.canvas.history_manager
    assert not hm.undo_stack and not hm.redo_stack
    assert not document.dsim.dirty
    add(document, 500)
    assert document.canvas.undo()
    assert not document.dsim.dirty


@pytest.mark.parametrize("writer", ["dsim", "file_service", "window"])
def test_autosave_does_not_move_saved_state(document, tmp_path, writer):
    add(document)
    save(document, tmp_path)
    add(document, 300)
    target = str(tmp_path / "autosave.diablos")
    if writer == "dsim":
        assert document.dsim.save(autosave=True, filepath=target) == 0
    elif writer == "file_service":
        assert document.dsim.file_service.save(autosave=True, filepath=target) == 0
        document.dsim.dirty = True  # The caller restores this legacy flag.
    else:
        document.autosave_path = target
        document._auto_save()
    assert document.dsim.dirty
    assert document.canvas.undo()
    assert not document.dsim.dirty


@pytest.mark.parametrize("writer", ["dsim", "file_service"])
def test_core_save_also_tracks_saved_state(document, tmp_path, writer):
    add(document)
    target = str(tmp_path / "core.diablos")
    if writer == "dsim":
        assert document.dsim.save(filepath=target) == 0
    else:
        assert document.dsim.file_service.save_to_file(document.dsim.serialize(), target)
    assert document.canvas.undo()
    assert document.dsim.dirty
    assert document.canvas.redo()
    assert not document.dsim.dirty


def test_failed_save_does_not_move_saved_state(document, tmp_path, monkeypatch):
    import lib.services.diagram_service as module

    add(document)
    save(document, tmp_path)
    add(document, 300)

    def fail(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(module.json, "dump", fail)
    assert not document.project_manager.diagram_service.save_diagram(str(tmp_path / "failed"))
    assert document.dsim.dirty
    assert document.canvas.undo()
    assert not document.dsim.dirty


def test_branching_does_not_reuse_saved_history_position(document, tmp_path):
    add(document)
    add(document, 300)
    save(document, tmp_path)
    assert document.canvas.undo()
    add(document, 500)
    assert document.dsim.dirty
    assert document.canvas.undo()
    assert document.dsim.dirty
    assert document.canvas.redo()
    assert document.dsim.dirty


def test_selection_and_runtime_data_do_not_change_saved_identity(document, tmp_path):
    block = add(document)
    save(document, tmp_path)
    block.selected = True
    block.params["_runtime_"] = [42]
    add(document, 300)
    assert document.canvas.undo()
    assert not document.dsim.dirty


def test_non_history_simulation_settings_still_count_as_unsaved(document):
    add(document)
    document.dsim.sim_time += 10
    document.dsim.dirty = True
    assert document.canvas.undo()
    assert document.dsim.dirty


def test_nested_edit_returns_to_whole_saved_document(document, tmp_path):
    block = add(document)
    block.selected = True
    document.canvas._create_subsystem_trigger()
    sub = next(b for b in document.dsim.blocks_list if b.block_fn == "Subsystem")
    save(document, tmp_path)
    document.dsim.enter_subsystem(sub)
    add(document, 300)
    document.dsim.exit_subsystem()
    assert document.canvas.undo()
    assert not document.dsim.dirty
    assert document.canvas.redo()
    assert document.dsim.dirty


@pytest.mark.parametrize("reset", ["load", "new"])
def test_replacing_document_from_subsystem_returns_to_root(document, tmp_path, reset):
    block = add(document)
    block.selected = True
    document.canvas._create_subsystem_trigger()
    sub = next(b for b in document.dsim.blocks_list if b.block_fn == "Subsystem")
    target = save(document, tmp_path)
    document.dsim.enter_subsystem(sub)
    add(document, 300)
    service = document.project_manager.diagram_service
    if reset == "load":
        assert service.load_diagram(target)
    else:
        service.new_diagram()
    assert not document.dsim.navigation_stack
    assert not document.canvas.history_manager.undo_stack
    add(document, 500)
    assert document.canvas.undo()
    assert not document.dsim.dirty


def test_cancelled_save_does_not_move_saved_state(document, tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from PyQt6.QtWidgets import QFileDialog
    import lib.services.diagram_service as module

    add(document)
    save(document, tmp_path)
    add(document, 300)
    dialog = MagicMock()
    dialog.exec.return_value = QFileDialog.DialogCode.Rejected
    monkeypatch.setattr(module, "_create_styled_file_dialog", lambda *a, **k: dialog)
    assert not document.project_manager.diagram_service.save_diagram()
    assert document.dsim.dirty
    assert document.canvas.undo()
    assert not document.dsim.dirty
