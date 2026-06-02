"""
Characterization tests for the block drag + resize state machine on
ModernCanvas.

These tests exercise the REAL canvas methods (start_drag, _start_resize,
_perform_resize, _finish_drag, _finish_resize) against REAL DBlock objects
and a REAL DSim. Unlike tests/modern_ui/test_interaction_manager.py (which
uses a MagicMock canvas and therefore never runs the real method bodies),
these lock in the actual observable behavior so the drag/resize extraction
refactor can be proven behavior-preserving.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_drag_resize.py -p no:cacheprovider -o addopts="-q"
"""

import pytest
from PyQt5.QtCore import QPoint, QRect

from lib.lib import DSim
from lib.simulation.block import DBlock
from modern_ui.widgets.modern_canvas import ModernCanvas


def _make_block(sid, x, y, w=100, h=80, name=None):
    """Create a real DBlock at the given position/size."""
    block = DBlock(
        block_fn='TestBlock',
        sid=sid,
        coords=QRect(x, y, w, h),
        color='#4CAF50',
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit='none',
        fn_name='testblock',
        params={'gain': 1.0},
        external=False,
        colors=None,
    )
    if name is not None:
        block.name = name
    return block


@pytest.fixture
def canvas(qapp):
    """Build a real ModernCanvas wrapping a real DSim."""
    dsim = DSim()
    c = ModernCanvas(dsim)
    return c


def _add_block(canvas, block):
    canvas.dsim.blocks_list.append(block)
    return block


# ---------------------------------------------------------------------------
# Drag tests
# ---------------------------------------------------------------------------

class TestDrag:
    def test_single_block_drag_moves_block(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100))
        block.selected = True

        # Grab the block at its top-left corner (offset = 0,0).
        canvas.start_drag(block, QPoint(100, 100))

        # Simulate a mouse move of (+50, +30) using the recorded drag_offset,
        # exactly as InteractionManager.handle_mouse_move does.
        new_pos = QPoint(150, 130)
        nx = new_pos.x() - canvas.drag_offset.x()
        ny = new_pos.y() - canvas.drag_offset.y()
        block.relocate_Block(QPoint(int(nx), int(ny)))

        canvas._finish_drag()

        assert block.left == 150
        assert block.top == 130

    def test_multi_select_drag_preserves_relative_offsets(self, canvas):
        b1 = _add_block(canvas, _make_block(0, 100, 100, name='b1'))
        b2 = _add_block(canvas, _make_block(1, 300, 160, name='b2'))
        b1.selected = True
        b2.selected = True

        # Relative offset of b2 from b1 before drag.
        rel_x = b2.left - b1.left  # 200
        rel_y = b2.top - b1.top    # 60

        # Drag clicking on b1 at its corner.
        canvas.start_drag(b1, QPoint(100, 100))

        # Move clicked block to (220, 250) and the other relative to it,
        # replicating the handle_mouse_move group-move logic.
        snapped_x, snapped_y = 220, 250
        b1.relocate_Block(QPoint(snapped_x, snapped_y))
        for blk, off in canvas.drag_offsets.items():
            if blk is not b1:
                blk.relocate_Block(QPoint(snapped_x + off.x(), snapped_y + off.y()))

        canvas._finish_drag()

        assert (b1.left, b1.top) == (220, 250)
        # Relative offset preserved.
        assert b2.left - b1.left == rel_x
        assert b2.top - b1.top == rel_y

    def test_undo_pushed_when_moved_past_threshold(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100))
        block.selected = True

        canvas.start_drag(block, QPoint(100, 100))
        # Move by 10px (>= 5px threshold).
        block.relocate_Block(QPoint(110, 100))

        before = len(canvas.history_manager.undo_stack)
        canvas._finish_drag()
        after = len(canvas.history_manager.undo_stack)

        assert after == before + 1, "Undo should be pushed for a move >= threshold"

    def test_no_undo_when_move_below_threshold(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100))
        block.selected = True

        canvas.start_drag(block, QPoint(100, 100))
        # Move by 2px (< 5px threshold).
        block.relocate_Block(QPoint(102, 100))

        before = len(canvas.history_manager.undo_stack)
        canvas._finish_drag()
        after = len(canvas.history_manager.undo_stack)

        assert after == before, "No undo should be pushed for a sub-threshold move"


# ---------------------------------------------------------------------------
# Resize tests
# ---------------------------------------------------------------------------

class TestResize:
    def test_resize_bottom_right_grows_block(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100, w=100, h=80))
        block.selected = True

        canvas._start_resize(block, 'bottom_right', QPoint(200, 180))
        # Drag the bottom-right handle by (+40, +30).
        canvas._perform_resize(QPoint(240, 210))

        assert block.width == 140
        assert block.height == 110
        # Top-left anchor unchanged for bottom_right.
        assert block.left == 100
        assert block.top == 100

    def test_resize_left_moves_left_edge(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100, w=100, h=80))
        block.selected = True

        canvas._start_resize(block, 'left', QPoint(100, 140))
        # Drag the left handle right by +20: left edge moves in, width shrinks.
        canvas._perform_resize(QPoint(120, 140))

        assert block.left == 120
        assert block.width == 80
        # Height untouched for a pure-left handle.
        assert block.height == 80

    def test_resize_enforces_minimum_size(self, canvas):
        try:
            from config.block_sizes import MIN_BLOCK_WIDTH, MIN_BLOCK_HEIGHT
            min_w, min_h = MIN_BLOCK_WIDTH, MIN_BLOCK_HEIGHT
        except ImportError:
            min_w, min_h = 50, 40

        block = _add_block(canvas, _make_block(0, 100, 100, w=100, h=80))
        block.selected = True
        port_min_h = block.calculate_min_size()
        expected_min_h = max(min_h, port_min_h)

        canvas._start_resize(block, 'bottom_right', QPoint(200, 180))
        # Drag far past the minimum (huge negative delta).
        canvas._perform_resize(QPoint(-500, -500))

        assert block.width >= min_w
        assert block.height >= expected_min_h
        assert canvas.resize_at_limit is True

    def test_finish_resize_pushes_undo_past_threshold(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100, w=100, h=80))
        block.selected = True

        canvas._start_resize(block, 'bottom_right', QPoint(200, 180))
        canvas._perform_resize(QPoint(240, 210))  # +40,+30 -> significant

        before = len(canvas.history_manager.undo_stack)
        canvas._finish_resize()
        after = len(canvas.history_manager.undo_stack)

        assert after == before + 1, "Undo should be pushed for resize >= threshold"
        # State cleared after finish.
        assert canvas.resizing_block is None
        assert canvas.resize_handle is None

    def test_finish_resize_no_undo_below_threshold(self, canvas):
        block = _add_block(canvas, _make_block(0, 100, 100, w=100, h=80))
        block.selected = True

        canvas._start_resize(block, 'bottom_right', QPoint(200, 180))
        canvas._perform_resize(QPoint(202, 181))  # +2,+1 -> sub-threshold

        before = len(canvas.history_manager.undo_stack)
        canvas._finish_resize()
        after = len(canvas.history_manager.undo_stack)

        assert after == before, "No undo should be pushed for sub-threshold resize"
