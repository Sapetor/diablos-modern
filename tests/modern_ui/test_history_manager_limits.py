"""Undo/redo stack limits and error reporting.

Each history entry deep-copies every block's params, so an *uncapped* redo
stack was an unbounded memory hold on a large diagram (the undo stack was
already capped at 50). ``push_undo`` also swallowed every exception silently,
which is the worst possible failure mode: the next undo then restores the
wrong state with nothing in the log to explain it.
"""

import pytest
from PyQt5.QtCore import QPoint


@pytest.fixture
def canvas(qapp):
    from lib.lib import DSim
    from modern_ui.widgets.modern_canvas import ModernCanvas

    dsim = DSim()
    c = ModernCanvas(dsim)
    yield c
    dsim.dirty = False
    c.deleteLater()


@pytest.mark.qt
class TestStackLimits:
    def test_redo_stack_is_capped_like_undo(self, canvas):
        hm = canvas.history_manager
        assert hm.undo_stack.maxlen == hm.max_undo_steps
        assert hm.redo_stack.maxlen == hm.max_undo_steps

    def test_redo_stack_evicts_instead_of_growing(self, canvas):
        hm = canvas.history_manager
        menu = {b.fn_name: b for b in canvas.dsim.menu_blocks}
        canvas.dsim.add_block(menu["step"], QPoint(50, 50))

        for _ in range(hm.max_undo_steps + 10):
            hm.redo_stack.append({"state": {"blocks": [], "lines": []}, "description": "x"})
        assert len(hm.redo_stack) == hm.max_undo_steps


@pytest.mark.qt
class TestPushUndoErrors:
    def test_capture_failure_is_logged_with_a_traceback(self, canvas, caplog, monkeypatch):
        hm = canvas.history_manager

        def boom():
            raise RuntimeError("snapshot failed")

        monkeypatch.setattr(hm, "_capture_state", boom)
        with caplog.at_level("WARNING"):
            hm.push_undo("Broken action")

        records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert records, "a failed undo push must not be silent"
        assert any(r.exc_info for r in records), "the traceback must reach the log"
