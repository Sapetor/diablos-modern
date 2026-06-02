"""
Characterization tests for ModernDiaBloSWindow's view-actions cluster.

main_window.py historically had zero coverage. These tests build a REAL
ModernDiaBloSWindow under offscreen Qt and pin down the observable behavior of
the view-actions cluster before it is extracted into a dedicated manager:

  * ``set_zoom`` / ``zoom_in`` / ``zoom_out``  (canvas zoom + zoom_status pill)
  * ``toggle_grid``                            (canvas grid + checkable action)
  * ``fit_to_window``                          (bounding-box zoom/pan to fit)
  * ``toggle_minimap``                         (minimap dock visibility toggle)

An autouse fixture restores canvas view state (zoom/pan/grid/blocks) between
tests. The minimap toggle is characterized by spying on isVisible/setVisible
because a dock's real isVisible() is always False while the window is unshown
(headless), which would make the toggle degenerate.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_view_actions.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import pytest
from PyQt5.QtCore import QRect, QPoint

from lib.simulation.block import DBlock


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_canvas_view(window):
    c = window.canvas
    saved = (c.zoom_factor, c.pan_offset, getattr(c, 'grid_visible', None),
             list(getattr(c.dsim, 'blocks_list', []) or []))
    yield
    c.zoom_factor = saved[0]
    c.pan_offset = saved[1]
    if saved[2] is not None:
        c.grid_visible = saved[2]
    c.dsim.blocks_list[:] = saved[3]


def _make_block(name, x, y, w=100, h=80):
    block = DBlock(
        block_fn='Gain', sid=0, coords=QRect(x, y, w, h), color='#4CAF50',
        in_ports=1, out_ports=1, b_type=2, io_edit='none', fn_name='gain',
        params={'gain': 1.0}, external=False, colors=None,
    )
    block.name = name
    return block


def _zoom_pct(window):
    return f"{int(window.canvas.zoom_factor * 100)}%"


# ---------------------------------------------------------------------------
# zoom
# ---------------------------------------------------------------------------

class TestZoom:
    def test_set_zoom_updates_status(self, window):
        window.set_zoom(1.5)
        assert window.canvas.zoom_factor == pytest.approx(1.5)
        assert window.zoom_status.text() == _zoom_pct(window)

    def test_zoom_in_increases(self, window):
        window.set_zoom(0.5)
        before = window.canvas.zoom_factor
        window.zoom_in()
        assert window.canvas.zoom_factor >= before
        assert window.zoom_status.text() == _zoom_pct(window)

    def test_zoom_out_decreases(self, window):
        window.set_zoom(1.5)
        before = window.canvas.zoom_factor
        window.zoom_out()
        assert window.canvas.zoom_factor <= before
        assert window.zoom_status.text() == _zoom_pct(window)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

class TestToggleGrid:
    def test_toggle_flips_and_syncs_action(self, window):
        before = window.canvas.grid_visible
        window.toggle_grid()
        assert window.canvas.grid_visible != before
        # The checkable menu action tracks the canvas grid state.
        assert window.grid_toggle_action.isChecked() == window.canvas.grid_visible
        assert "Grid" in window.status_message.text()
        # Toggling again returns to the original state.
        window.toggle_grid()
        assert window.canvas.grid_visible == before


# ---------------------------------------------------------------------------
# fit_to_window
# ---------------------------------------------------------------------------

class TestFitToWindow:
    def test_no_blocks_message(self, window):
        window.canvas.dsim.blocks_list[:] = []
        window.fit_to_window()
        assert window.status_message.text() == "No blocks to fit"

    def test_fit_sets_zoom_and_status(self, window):
        window.canvas.dsim.blocks_list[:] = [
            _make_block("A", 0, 0), _make_block("B", 400, 300),
        ]
        window.fit_to_window()
        # Zoom is clamped to [0.1, 2.0] and the status pill reflects it.
        assert 0.1 <= window.canvas.zoom_factor <= 2.0
        assert window.zoom_status.text() == _zoom_pct(window)
        assert "Fit 2 block(s) to window" == window.status_message.text()
        assert isinstance(window.canvas.pan_offset, QPoint)


# ---------------------------------------------------------------------------
# minimap
# ---------------------------------------------------------------------------

class TestToggleMinimap:
    def test_toggle_from_hidden_shows_and_refreshes(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.minimap_dock, "isVisible", lambda: False)
        monkeypatch.setattr(window.minimap_dock, "setVisible",
                            lambda v: captured.__setitem__("visible", v))
        monkeypatch.setattr(window.minimap, "refresh",
                            lambda: captured.__setitem__("refreshed", True))
        window.toggle_minimap()
        assert captured["visible"] is True
        assert captured.get("refreshed") is True
        assert window.minimap_action.isChecked() is True

    def test_toggle_from_visible_hides(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.minimap_dock, "isVisible", lambda: True)
        monkeypatch.setattr(window.minimap_dock, "setVisible",
                            lambda v: captured.__setitem__("visible", v))
        window.toggle_minimap()
        assert captured["visible"] is False
        assert window.minimap_action.isChecked() is False
