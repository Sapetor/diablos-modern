"""Unit/smoke tests for the empty-canvas placeholder hint.

When the canvas holds no blocks and no simulation is running, ``paintEvent``
draws a centered, dim guidance message (added in ``_draw_empty_hint``). These
tests cover the two halves of that change:

* the hint wording lives in a tiny pure helper (``_empty_hint_lines``) so it is
  testable without a live QPainter, and stays in sync with what users expect;
* painting an empty canvas runs ``_draw_empty_hint`` end-to-end without raising.

The canvas is built against a REAL ``DSim`` (same approach as
tests/modern_ui/test_drag_resize.py) so the real ``paintEvent`` body runs.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_empty_canvas_hint.py -p no:cacheprovider -o addopts=""
"""

import pytest
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap, QPainter, QColor

from lib.lib import DSim
from lib.simulation.block import DBlock
from modern_ui.widgets.modern_canvas import ModernCanvas


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def canvas(qapp):
    """Build a real ModernCanvas wrapping a real DSim."""
    dsim = DSim()
    c = ModernCanvas(dsim)
    c.resize(800, 600)
    return c


def _make_block(sid, x, y, w=100, h=80):
    """Create a real DBlock so the canvas is no longer empty."""
    return DBlock(
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


# ---------------------------------------------------------------------------
# Helper (pure) tests
# ---------------------------------------------------------------------------

class TestEmptyHintLines:
    def test_returns_expected_guidance_lines(self):
        lines = ModernCanvas._empty_hint_lines()
        assert lines == [
            "Double-click to add a block",
            "Drag a block from the palette",
            "Open an example from File ▸ Examples",
        ]

    def test_lines_are_nonempty_strings(self):
        lines = ModernCanvas._empty_hint_lines()
        assert len(lines) == 3
        assert all(isinstance(line, str) and line.strip() for line in lines)

    def test_is_static_callable_without_instance(self):
        # Callable straight off the class — no QPainter / canvas needed.
        assert ModernCanvas._empty_hint_lines() is not None


# ---------------------------------------------------------------------------
# Paint smoke tests
# ---------------------------------------------------------------------------

class TestEmptyHintPainting:
    def _draw_hint_on_pixmap(self, canvas):
        pixmap = QPixmap(canvas.width(), canvas.height())
        pixmap.fill(QColor(0, 0, 0))
        painter = QPainter(pixmap)
        try:
            canvas._draw_empty_hint(painter)
        finally:
            painter.end()

    def test_draw_empty_hint_does_not_raise(self, canvas):
        # Empty, idle canvas: drawing the hint must complete cleanly.
        assert not canvas.dsim.blocks_list
        self._draw_hint_on_pixmap(canvas)

    def test_full_paint_empty_canvas_does_not_raise(self, canvas):
        # Exercise the real paintEvent path (which calls _draw_empty_hint when
        # there are no blocks and no simulation running).
        assert not canvas.dsim.blocks_list
        assert not canvas.is_simulation_running()
        pixmap = QPixmap(canvas.width(), canvas.height())
        pixmap.fill(QColor(0, 0, 0))
        canvas.render(pixmap)

    def test_full_paint_with_blocks_does_not_raise(self, canvas):
        # With a block present the hint branch is skipped; paint must still run.
        canvas.dsim.blocks_list.append(_make_block(0, 100, 100))
        assert canvas.dsim.blocks_list
        pixmap = QPixmap(canvas.width(), canvas.height())
        pixmap.fill(QColor(0, 0, 0))
        canvas.render(pixmap)
