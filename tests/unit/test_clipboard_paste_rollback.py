"""
A paste that fails half-way must leave the diagram and the undo/redo stacks
exactly as they were, while still reporting the error on the status signal.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QRect

from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from modern_ui.managers.clipboard_manager import ClipboardManager
from modern_ui.managers.history_manager import HistoryManager


@pytest.fixture(autouse=True)
def _qt(qapp):
    """DBlock builds a QPixmap, which needs a live QApplication."""
    return qapp


def _block(block_fn, sid, x=100, y=100, in_ports=1, out_ports=1, selected=False):
    block = DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(x, y, 60, 40),
        color="#4CAF50",
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=2,
        io_edit=False,
        fn_name=block_fn,
        params={"gain": 2.0},
        external=False,
        username="",
        block_class=None,
        colors=None,
        category="Math",
    )
    block.selected = selected
    return block


def _block_data(block_fn, x, y, in_ports=1, out_ports=1):
    return {
        "block_fn": block_fn,
        "coords": QRect(x, y, 60, 40),
        "color": "#2196F3",
        "category": "Math",
        "in_ports": in_ports,
        "out_ports": out_ports,
        "b_type": 2,
        "io_edit": False,
        "fn_name": block_fn,
        "params": {"gain": 2.0},
        "external": False,
        "flipped": False,
    }


def _canvas():
    """A canvas stub carrying a real HistoryManager over a real-ish dsim."""
    existing = [
        _block("Gain", 0, selected=True),
        _block("Scope", 0, 300, 100, in_ports=1, out_ports=0),
    ]
    line = DLine(
        sid=0,
        srcblock="gain0",
        srcport=0,
        dstblock="scope0",
        dstport=0,
        points=[existing[0].out_coords[0], existing[1].in_coords[0]],
    )
    dsim = SimpleNamespace(
        blocks_list=existing, line_list=[line], menu_blocks=[], colors=None, dirty=False
    )

    def add_line(src, dst):
        """What SimulationModel.add_line does, enough for HistoryManager._restore_state."""
        new_line = DLine(
            sid=max([ln.sid for ln in dsim.line_list] + [-1]) + 1,
            srcblock=src[0],
            srcport=src[1],
            dstblock=dst[0],
            dstport=dst[1],
            points=[src[2], dst[2]],
        )
        dsim.line_list.append(new_line)
        return new_line

    dsim.add_line = add_line
    canvas = SimpleNamespace(
        dsim=dsim,
        update_calls=0,
        block_selected=SimpleNamespace(emitted=[]),
        simulation_status_changed=SimpleNamespace(emitted=[]),
        interaction_manager=SimpleNamespace(clear_hover=lambda: None),
    )
    canvas.update = lambda: setattr(canvas, "update_calls", canvas.update_calls + 1)
    canvas.block_selected.emit = canvas.block_selected.emitted.append
    canvas.simulation_status_changed.emit = canvas.simulation_status_changed.emitted.append
    canvas.history_manager = HistoryManager(canvas)
    # A history to protect: two undo entries and one redo entry.
    canvas.history_manager.push_undo("Move")
    canvas.history_manager.push_undo("Resize")
    canvas.history_manager.undo()
    assert len(canvas.history_manager.undo_stack) == 1
    assert len(canvas.history_manager.redo_stack) == 1
    return canvas


def _snapshot(canvas):
    dsim = canvas.dsim
    return {
        "blocks": list(dsim.blocks_list),
        "lines": list(dsim.line_list),
        "selected": [b.selected for b in dsim.blocks_list],
        "dirty": dsim.dirty,
        "undo": len(canvas.history_manager.undo_stack),
        "redo": len(canvas.history_manager.redo_stack),
    }


def _manager_with_clipboard(canvas):
    manager = ClipboardManager(canvas)
    manager.clipboard_blocks = [
        _block_data("Constant", 100, 200, in_ports=0),
        _block_data("Gain", 250, 200),
        _block_data("Scope", 400, 200, out_ports=0),
    ]
    manager.clipboard_connections = [
        {"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 0},
        {"start_index": 1, "start_port": 0, "end_index": 2, "end_port": 0},
    ]
    return manager


def _fail_on_second_block(monkeypatch, manager):
    original = manager._instantiate_block
    calls = {"n": 0}

    def flaky(block_data, coords):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom on block 2")
        return original(block_data, coords)

    monkeypatch.setattr(manager, "_instantiate_block", flaky)
    return calls


@pytest.mark.unit
class TestPasteRollback:
    def test_failed_paste_restores_diagram_selection_dirty_and_history(self, monkeypatch):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)
        before = _snapshot(canvas)
        calls = _fail_on_second_block(monkeypatch, manager)

        manager.paste_blocks()

        assert calls["n"] == 2, "the failure must have happened after the first block was added"
        assert _snapshot(canvas) == before
        # Same objects, not merely the same count.
        assert canvas.dsim.blocks_list == before["blocks"]
        assert canvas.dsim.line_list == before["lines"]
        assert canvas.dsim.dirty is False
        assert canvas.block_selected.emitted == []

    def test_failed_paste_still_reports_the_error(self, monkeypatch):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)
        _fail_on_second_block(monkeypatch, manager)

        manager.paste_blocks()

        emitted = canvas.simulation_status_changed.emitted
        assert len(emitted) == 1 and "boom on block 2" in emitted[0]

    def test_failure_after_connections_removes_the_added_lines_too(self, monkeypatch):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)
        before = _snapshot(canvas)
        original_recreate = manager._recreate_connections

        def recreate_then_fail(pasted_blocks):
            original_recreate(pasted_blocks)
            assert len(canvas.dsim.line_list) == len(before["lines"]) + 2
            raise RuntimeError("boom after lines")

        monkeypatch.setattr(manager, "_recreate_connections", recreate_then_fail)

        manager.paste_blocks()

        assert _snapshot(canvas) == before
        assert canvas.dsim.line_list == before["lines"]
        assert "boom after lines" in canvas.simulation_status_changed.emitted[0]

    def test_undo_after_a_failed_paste_undoes_the_previous_action_not_nothing(self, monkeypatch):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)
        _fail_on_second_block(monkeypatch, manager)
        manager.paste_blocks()

        # The one remaining undo entry is "Move" (pushed before the paste).
        assert canvas.history_manager.undo_stack[-1]["description"] == "Move"
        canvas.history_manager.undo()
        assert [b.name for b in canvas.dsim.blocks_list] == ["gain0", "scope0"]
        assert len(canvas.history_manager.undo_stack) == 0


@pytest.mark.unit
class TestSuccessfulPasteHistory:
    def test_successful_paste_pushes_one_entry_and_clears_redo(self):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)

        manager.paste_blocks()

        assert [e["description"] for e in canvas.history_manager.undo_stack] == ["Move", "Paste"]
        assert len(canvas.history_manager.redo_stack) == 0
        assert [b.name for b in canvas.dsim.blocks_list] == [
            "gain0",
            "scope0",
            "constant0",
            "gain1",
            "scope1",
        ]
        assert len(canvas.dsim.line_list) == 3
        assert canvas.dsim.dirty is True
        assert canvas.simulation_status_changed.emitted == []

    def test_undo_of_a_successful_paste_restores_the_pre_paste_diagram(self):
        canvas = _canvas()
        manager = _manager_with_clipboard(canvas)
        manager.paste_blocks()

        canvas.history_manager.undo()

        assert [b.name for b in canvas.dsim.blocks_list] == ["gain0", "scope0"]
        assert [b.selected for b in canvas.dsim.blocks_list] == [True, False]
        assert [(ln.srcblock, ln.dstblock) for ln in canvas.dsim.line_list] == [("gain0", "scope0")]
        assert len(canvas.history_manager.redo_stack) == 1
