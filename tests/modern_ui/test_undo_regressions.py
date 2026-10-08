"""Regression tests for undo/redo, driving the real window and canvas offscreen."""

import gc

import pytest
from PyQt6.QtCore import QPoint


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.close()
    gc.collect()


def _menu_block(canvas, block_fn):
    return next(mb for mb in canvas.dsim.menu_blocks if mb.block_fn == block_fn)


def _add(canvas, block_fn, x=100, y=100):
    blk = canvas.add_block_from_palette(_menu_block(canvas, block_fn), QPoint(x, y))
    assert blk is not None
    return blk


def _names(canvas):
    return sorted(b.name for b in canvas.dsim.blocks_list)


def _get(canvas, name):
    return next(x for x in canvas.dsim.blocks_list if x.name == name)


def test_undo_of_add_block_removes_it(window):
    canvas = window.canvas
    before = _names(canvas)
    _add(canvas, "Gain")
    assert len(canvas.dsim.blocks_list) == len(before) + 1
    assert canvas.undo() is True
    assert _names(canvas) == before
    assert canvas.redo() is True
    assert len(canvas.dsim.blocks_list) == len(before) + 1


def test_undo_preserves_flip_and_username(window):
    canvas = window.canvas
    a = _add(canvas, "Gain", 100, 100)
    b = _add(canvas, "Gain", 300, 100)
    a.flipped = True
    a.username = "MyGain"
    # An unrelated edit, then undo it: a's flip and label must survive.
    canvas._push_undo("unrelated")
    b.params["gain"] = 5.0
    assert canvas.undo()
    restored = _get(canvas, a.name)
    assert restored.flipped is True
    assert restored.username == "MyGain"
    assert canvas.redo()
    restored = _get(canvas, a.name)
    assert restored.flipped is True and restored.username == "MyGain"


def test_flip_is_undoable_and_marks_dirty(window):
    canvas = window.canvas
    a = _add(canvas, "Gain")
    a.selected = True
    canvas.dsim.dirty = False
    canvas.flip_selected_blocks()
    assert a.flipped is True
    assert canvas.dsim.dirty is True
    canvas.undo()
    assert _get(canvas, a.name).flipped is False


def test_rename_via_controller_is_undoable(window):
    canvas = window.canvas
    a = _add(canvas, "Gain")
    old = a.username
    window._on_property_changed(a.name, "_username_", "Renamed")
    assert a.username == "Renamed"
    canvas.undo()
    assert _get(canvas, a.name).username == old


def test_param_edit_is_one_undo_entry_and_noop_is_skipped(window):
    canvas = window.canvas
    a = _add(canvas, "Gain")
    original = a.params["gain"]
    n = len(canvas.history_manager.undo_stack)
    window._on_property_changed(a.name, "gain", original)  # no change
    assert len(canvas.history_manager.undo_stack) == n
    window._on_property_changed(a.name, "gain", 7.5)
    assert len(canvas.history_manager.undo_stack) == n + 1
    canvas.undo()
    assert _get(canvas, a.name).params["gain"] == original


def test_create_subsystem_is_undoable(window):
    canvas = window.canvas
    a = _add(canvas, "Gain")
    a.selected = True
    canvas._create_subsystem_trigger()
    assert any(b.block_fn == "Subsystem" for b in canvas.dsim.blocks_list)
    canvas.undo()
    assert not any(b.block_fn == "Subsystem" for b in canvas.dsim.blocks_list)
    assert any(b.name == a.name for b in canvas.dsim.blocks_list)


def _make_subsystem_and_enter(canvas):
    a = _add(canvas, "Gain")
    a.selected = True
    canvas._create_subsystem_trigger()
    sub = next(b for b in canvas.dsim.blocks_list if b.block_fn == "Subsystem")
    canvas.dsim.enter_subsystem(sub)
    return sub


def test_undo_inside_subsystem_keeps_ports(window):
    canvas = window.canvas
    _make_subsystem_and_enter(canvas)
    ports_before = sorted(
        b.name for b in canvas.dsim.blocks_list if b.block_fn in ("Inport", "Outport")
    )
    assert ports_before
    _add(canvas, "Gain", 400, 300)
    canvas.undo()
    canvas.undo()  # reaches back past subsystem creation's scope; must not corrupt this one
    canvas.dsim.exit_subsystem()
    sub = next((b for b in canvas.dsim.blocks_list if b.block_fn == "Subsystem"), None)
    if sub is not None:
        ports_after = sorted(b.name for b in sub.sub_blocks if b.block_fn in ("Inport", "Outport"))
        assert ports_after == ports_before


def test_undo_across_scopes_navigates_to_snapshot_scope(window):
    canvas = window.canvas
    sub = _make_subsystem_and_enter(canvas)
    n_inner = len(sub.sub_blocks)
    _add(canvas, "Gain", 400, 300)
    canvas.dsim.exit_subsystem()  # now at top level
    assert canvas.undo() is True
    # Undo of the inner add happens in the inner scope.
    assert canvas.dsim.get_current_path()[-1] == sub.name
    assert len(canvas.dsim.blocks_list) == n_inner
    assert canvas.redo() is True
    assert len(canvas.dsim.blocks_list) == n_inner + 1


def test_undo_redo_mark_dirty(window):
    canvas = window.canvas
    _add(canvas, "Gain")
    canvas.dsim.dirty = False
    canvas.undo()
    assert canvas.dsim.dirty is True
    canvas.dsim.dirty = False
    canvas.redo()
    assert canvas.dsim.dirty is True


def test_failed_restore_leaves_diagram_and_stacks_intact(window):
    canvas = window.canvas
    hm = canvas.history_manager
    _add(canvas, "Gain")
    names = _names(canvas)
    undo_n, redo_n = len(hm.undo_stack), len(hm.redo_stack)
    # Corrupt the top entry so rebuilding a block raises.
    hm.undo_stack[-1]["state"]["blocks"].append({"block_fn": "Gain", "name": "Gain99"})
    assert canvas.undo() is False
    assert _names(canvas) == names
    assert (len(hm.undo_stack), len(hm.redo_stack)) == (undo_n, redo_n)


def test_undo_with_empty_stack_returns_false_and_no_toast(window):
    window.canvas.history_manager.undo_stack.clear()
    shown = []
    window.toast.show_message = lambda *a, **k: shown.append(a)
    window.undo_action()
    window.redo_action()
    assert shown == []
