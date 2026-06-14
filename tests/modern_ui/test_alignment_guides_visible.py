"""
Tests for smart-alignment guides being VISIBLE regardless of grid snap.

Grid snap defaults ON, and the old behaviour suppressed the alignment guides
entirely whenever grid snap was on -- so the feature never showed out of the
box. ``DragResizeManager.update_drag_alignment`` now ALWAYS computes the guides
from the dragged block's current rect and stashes them on the canvas so they
display; it only APPLIES the snap nudge when grid snap is OFF (so the two snaps
can't fight and knock the block off the grid).

A tiny pure helper (``_should_apply_alignment_nudge``) captures the apply/skip
decision so it is unit-testable without Qt; the integration tests then confirm
the real DragResizeManager honours it.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_alignment_guides_visible.py -p no:cacheprovider -o addopts=""
"""

import pytest
from PyQt5.QtCore import QPoint, QRect


# ---------------------------------------------------------------------------
# Pure decision logic: when should the alignment nudge actually be applied?
# ---------------------------------------------------------------------------
#
# Mirrors the guard in ``update_drag_alignment``: the visual guides are computed
# and shown in every drag; the position-changing nudge is only applied when grid
# snap is off AND there is a non-zero snap offset on either axis.

def _should_apply_alignment_nudge(grid_snap_on, snap_dx, snap_dy):
    return (not grid_snap_on) and bool(snap_dx or snap_dy)


class TestShouldApplyAlignmentNudge:
    def test_grid_off_with_offset_applies(self):
        assert _should_apply_alignment_nudge(False, -3.0, 0.0) is True

    def test_grid_off_without_offset_does_not_apply(self):
        assert _should_apply_alignment_nudge(False, 0.0, 0.0) is False

    def test_grid_on_with_offset_does_not_apply(self):
        # Guides still show (computed elsewhere); only the nudge is suppressed.
        assert _should_apply_alignment_nudge(True, -3.0, 2.0) is False

    def test_grid_on_without_offset_does_not_apply(self):
        assert _should_apply_alignment_nudge(True, 0.0, 0.0) is False


# ---------------------------------------------------------------------------
# Integration: DragResizeManager shows guides regardless of grid snap
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
    return ModernCanvas(dsim)


def _setup_near_drag(canvas):
    """Target at (100, 50); mover 3px off the target's left edge."""
    target = _make_block(0, 100, 50, name='target')
    mover = _make_block(1, 103, 300, name='mover')  # left edge 3px off target
    canvas.dsim.blocks_list.extend([target, mover])
    mover.selected = True
    canvas.start_drag(mover, QPoint(103, 300))
    return target, mover


class TestGuidesVisibleRegardlessOfGridSnap:
    def test_grid_snap_default_on_shows_guides_but_no_move(self, canvas):
        # Default state: grid snap is ON out of the box.
        assert canvas.snap_enabled is True
        _, mover = _setup_near_drag(canvas)

        canvas.drag_resize_manager.update_drag_alignment()

        # Block stays grid-snapped (no alignment nudge applied)...
        assert mover.left == 103
        # ...but the guide is visible at the neighbour's shared edge.
        assert canvas._alignment_guides
        (x1, _), (x2, _) = canvas._alignment_guides[0]
        assert x1 == 100 and x2 == 100

    def test_grid_snap_off_shows_guides_and_applies_snap(self, canvas):
        canvas.snap_enabled = False
        _, mover = _setup_near_drag(canvas)

        canvas.drag_resize_manager.update_drag_alignment()

        # With grid snap off the nudge IS applied: block lands on the edge...
        assert mover.left == 100
        # ...and the guide is shown for the painter.
        assert canvas._alignment_guides
        (x1, _), (x2, _) = canvas._alignment_guides[0]
        assert x1 == 100 and x2 == 100

    def test_grid_on_no_neighbor_in_range_clears_guides(self, canvas):
        # Far apart -> nothing to align to, on either snap mode.
        assert canvas.snap_enabled is True
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 400, 300, name='mover')
        canvas.dsim.blocks_list.extend([target, mover])
        mover.selected = True
        canvas.start_drag(mover, QPoint(400, 300))

        canvas.drag_resize_manager.update_drag_alignment()

        assert mover.left == 400
        assert canvas._alignment_guides == []

    def test_grid_on_guides_clear_on_finish(self, canvas):
        assert canvas.snap_enabled is True
        _, mover = _setup_near_drag(canvas)

        canvas.drag_resize_manager.update_drag_alignment()
        assert canvas._alignment_guides  # visible mid-drag even with grid on

        canvas._finish_drag()
        assert canvas._alignment_guides == []

    def test_grid_on_multi_select_does_not_shift_partner(self, canvas):
        # With grid snap on, the rigid-group shift must NOT fire (no nudge).
        assert canvas.snap_enabled is True
        target = _make_block(0, 100, 50, name='target')
        mover = _make_block(1, 103, 300, name='mover')
        partner = _make_block(2, 300, 300, name='partner')
        canvas.dsim.blocks_list.extend([target, mover, partner])
        mover.selected = True
        partner.selected = True

        partner_left_before = partner.left
        canvas.start_drag(mover, QPoint(103, 300))
        canvas.drag_resize_manager.update_drag_alignment()

        assert mover.left == 103                 # mover untouched
        assert partner.left == partner_left_before  # partner untouched
        assert canvas._alignment_guides          # guide still shown
