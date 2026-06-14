"""
Tests for smart-alignment guides shown while dragging a block.

The load-bearing part is the pure ``compute_alignment_guides`` geometry helper,
so the bulk of these tests exercise it directly (no Qt, no canvas state). A
couple of thin integration tests then confirm DragResizeManager applies the
snap to real DBlocks and stashes the guide lines on the canvas, and that the
overlay clears on drag finish.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_alignment_guides.py -p no:cacheprovider -o addopts=""
"""

import pytest
from PyQt5.QtCore import QPoint, QRect

from modern_ui.renderers.canvas_renderer import compute_alignment_guides


# ---------------------------------------------------------------------------
# Pure geometry: compute_alignment_guides
# ---------------------------------------------------------------------------

class TestComputeAlignmentGuides:
    def test_left_edges_snap_when_within_threshold(self):
        # Moving left edge at x=103, other left edge at x=100 -> within 4px.
        moving = (103, 50, 100, 80)
        others = [(100, 200, 100, 80)]
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == pytest.approx(-3.0)  # 100 - 103
        # No vertical relationship within threshold here.
        assert dy == 0.0
        # One vertical guide line at the shared x edge (100).
        assert len(guides) == 1
        (x1, y1), (x2, y2) = guides[0]
        assert x1 == pytest.approx(100.0)
        assert x2 == pytest.approx(100.0)

    def test_no_snap_when_outside_threshold(self):
        # Left edges 10px apart, threshold 4 -> no snap on either axis.
        moving = (110, 300, 100, 80)
        others = [(100, 50, 100, 80)]
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == 0.0
        assert dy == 0.0
        assert guides == []

    def test_center_x_alignment(self):
        # Differing widths so ONLY the center-x edges line up within threshold
        # (left/right edges are far apart). Moving center-x = 80 + 60/2 = 110,
        # other center-x = 50 + 200/2 = 150 ... arrange so centers are 2px apart.
        moving = (98, 300, 100, 80)    # center-x = 148
        others = [(50, 50, 200, 80)]   # center-x = 150; left 50, right 250 (far)
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == pytest.approx(2.0)  # 150 - 148
        # Guide is a vertical line at the shared center x = 150.
        assert len(guides) == 1
        (x1, _), (x2, _) = guides[0]
        assert x1 == pytest.approx(150.0)
        assert x2 == pytest.approx(150.0)

    def test_top_edges_snap_vertically(self):
        # Moving top at 53, other top at 50 -> dy = -3, horizontal guide.
        moving = (400, 53, 100, 80)
        others = [(100, 50, 100, 80)]
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dy == pytest.approx(-3.0)
        assert dx == 0.0
        (x1, y1), (x2, y2) = guides[0]
        assert y1 == pytest.approx(50.0)
        assert y2 == pytest.approx(50.0)

    def test_picks_nearest_x_candidate(self):
        # Two others: one whose left is 3px away, one whose left is 1px away.
        # The nearest (1px) must win.
        moving = (100, 400, 100, 80)            # left = 100
        others = [
            (103, 50, 100, 80),   # left 103 -> dist 3
            (101, 600, 100, 80),  # left 101 -> dist 1 (nearest)
        ]
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == pytest.approx(1.0)  # snaps to 101, not 103
        (x1, _), (x2, _) = guides[0]
        assert x1 == pytest.approx(101.0)

    def test_at_most_one_snap_per_axis(self):
        # Both a left-edge match (dx) and a top-edge match (dy) available; expect
        # exactly one guide per axis -> two guides total.
        moving = (102, 52, 100, 80)
        others = [(100, 50, 100, 80)]
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == pytest.approx(-2.0)
        assert dy == pytest.approx(-2.0)
        assert len(guides) == 2
        orientations = {('v' if g[0][0] == g[1][0] else 'h') for g in guides}
        assert orientations == {'v', 'h'}

    def test_right_to_left_edge_alignment(self):
        # Moving right edge (x=200) snaps to other's left edge (x=202).
        moving = (100, 400, 100, 80)   # right = 200
        others = [(202, 50, 100, 80)]  # left = 202 -> dist 2
        dx, dy, guides = compute_alignment_guides(moving, others, threshold=4)
        assert dx == pytest.approx(2.0)  # 202 - 200
        (x1, _), (x2, _) = guides[0]
        assert x1 == pytest.approx(202.0)

    def test_guide_line_spans_both_blocks_vertically(self):
        # Vertical guide should span from the topmost to the bottommost of the
        # two involved blocks (using the snapped moving rect).
        moving = (103, 400, 100, 80)   # snapped left -> 100, y in [400, 480]
        others = [(100, 50, 100, 80)]  # y in [50, 130]
        _, _, guides = compute_alignment_guides(moving, others, threshold=4)
        (x1, y1), (x2, y2) = guides[0]
        assert y1 == pytest.approx(50.0)    # topmost
        assert y2 == pytest.approx(480.0)   # bottommost

    def test_empty_others_returns_no_snap(self):
        dx, dy, guides = compute_alignment_guides((0, 0, 100, 80), [], threshold=4)
        assert (dx, dy, guides) == (0.0, 0.0, [])


# ---------------------------------------------------------------------------
# Integration: DragResizeManager applies the snap + stashes guides
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


def _make_block(sid, x, y, w=100, h=80, name=None):
    from lib.simulation.block import DBlock
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
    from lib.lib import DSim
    from modern_ui.widgets.modern_canvas import ModernCanvas
    dsim = DSim()
    c = ModernCanvas(dsim)
    # Disable grid snap so smart-alignment guides are active (they intentionally
    # defer to the grid when grid snap is on).
    c.snap_enabled = False
    return c


class TestDragManagerIntegration:
    def test_snap_nudges_dragged_block_onto_neighbor_edge(self, canvas):
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 103, 300, name='mover')  # left 3px off
        canvas.dsim.blocks_list.extend([target, mover])
        mover.selected = True

        canvas.start_drag(mover, QPoint(103, 300))
        canvas.drag_resize_manager.update_drag_alignment()

        # Snapped onto target's left edge.
        assert mover.left == 100
        # Guide line stashed for the painter.
        assert canvas._alignment_guides
        (x1, _), (x2, _) = canvas._alignment_guides[0]
        assert x1 == 100 and x2 == 100

    def test_no_snap_clears_guides(self, canvas):
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 400, 300, name='mover')  # far away
        canvas.dsim.blocks_list.extend([target, mover])
        mover.selected = True

        canvas.start_drag(mover, QPoint(400, 300))
        canvas.drag_resize_manager.update_drag_alignment()

        assert mover.left == 400  # untouched
        assert canvas._alignment_guides == []

    def test_grid_snap_shows_guides_without_nudging(self, canvas):
        # Grid snap on -> guides are pure visual feedback: they DISPLAY when a
        # grid-aligned edge lines up with a neighbour, but the block must stay
        # on the grid, so the alignment nudge is intentionally NOT applied.
        canvas.snap_enabled = True
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 103, 300, name='mover')  # left edge 3px off target
        canvas.dsim.blocks_list.extend([target, mover])
        mover.selected = True

        canvas.start_drag(mover, QPoint(103, 300))
        canvas.drag_resize_manager.update_drag_alignment()

        assert mover.left == 103  # not moved -> stays grid-snapped
        # Guide still lights up at the neighbour's shared edge (x=100).
        assert canvas._alignment_guides
        (x1, _), (x2, _) = canvas._alignment_guides[0]
        assert x1 == 100 and x2 == 100

    def test_guides_cleared_on_finish(self, canvas):
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 103, 300, name='mover')
        canvas.dsim.blocks_list.extend([target, mover])
        mover.selected = True

        canvas.start_drag(mover, QPoint(103, 300))
        canvas.drag_resize_manager.update_drag_alignment()
        assert canvas._alignment_guides  # active mid-drag

        canvas._finish_drag()
        assert canvas._alignment_guides == []

    def test_multi_select_moves_rigidly_on_snap(self, canvas):
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 103, 300, name='mover')   # left 3px off target
        partner = _make_block(2, 300, 300, name='partner')
        canvas.dsim.blocks_list.extend([target, mover, partner])
        mover.selected = True
        partner.selected = True

        rel = partner.left - mover.left  # 197
        canvas.start_drag(mover, QPoint(103, 300))
        canvas.drag_resize_manager.update_drag_alignment()

        assert mover.left == 100              # snapped by -3
        assert partner.left - mover.left == rel  # partner shifted by same delta
