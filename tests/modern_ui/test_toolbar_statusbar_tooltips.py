"""
Tests for discoverability tooltips + status tips on the toolbar and status bar.

The toolbar's zoom rocker buttons, the command-palette ("Search…") button, and
the transport buttons must carry non-empty tooltips so users can learn what
each control does on hover. Where the toolbar exposes QActions (file / view /
theme), each must also carry a status tip mirrored into the status bar.

The status-bar pills (file, counts, cursor, zoom, theme) are built inside
``StatusBarManager.setup()`` during window construction, so a real
``ModernDiaBloSWindow`` is built once per module to assert their tooltips.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_toolbar_statusbar_tooltips.py -p no:cacheprovider \
        -o addopts=""
"""

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def toolbar():
    from modern_ui.widgets.modern_toolbar import ModernToolBar
    tb = ModernToolBar()
    yield tb
    tb.deleteLater()


# ---------------------------------------------------------------------------
# toolbar — zoom rocker
# ---------------------------------------------------------------------------

class TestZoomRockerTooltips:
    def test_zoom_out_button_has_tooltip(self, toolbar):
        assert toolbar.zoom_rocker.minus_btn.toolTip() == "Zoom out"

    def test_zoom_in_button_has_tooltip(self, toolbar):
        assert toolbar.zoom_rocker.plus_btn.toolTip() == "Zoom in"

    def test_zoom_buttons_tooltips_non_empty(self, toolbar):
        assert toolbar.zoom_rocker.minus_btn.toolTip().strip()
        assert toolbar.zoom_rocker.plus_btn.toolTip().strip()


# ---------------------------------------------------------------------------
# toolbar — command palette / search button
# ---------------------------------------------------------------------------

class TestCommandPaletteTooltip:
    def test_search_button_tooltip_non_empty(self, toolbar):
        assert toolbar.cmdk_btn.toolTip().strip()

    def test_search_button_tooltip_includes_shortcut(self, toolbar):
        # The discoverability shortcut must be advertised in the tooltip.
        assert "Ctrl+K" in toolbar.cmdk_btn.toolTip()


# ---------------------------------------------------------------------------
# toolbar — transport buttons
# ---------------------------------------------------------------------------

class TestTransportTooltips:
    def test_all_transport_buttons_have_tooltips(self, toolbar):
        t = toolbar.transport
        for btn in (t.play_btn, t.pause_btn, t.stop_btn, t.step_btn):
            assert btn.toolTip().strip(), f"missing tooltip on {btn.objectName()}"


# ---------------------------------------------------------------------------
# toolbar — QActions carry status tips
# ---------------------------------------------------------------------------

class TestActionStatusTips:
    def test_actions_have_status_tips(self, toolbar):
        actions = (
            toolbar.new_action, toolbar.open_action, toolbar.save_action,
            toolbar.plot_action, toolbar.capture_action,
            toolbar.auto_route_action, toolbar.theme_action,
        )
        for a in actions:
            assert a.statusTip().strip(), f"missing status tip on {a.text()!r}"

    def test_save_status_tip_includes_shortcut(self, toolbar):
        assert "Ctrl+S" in toolbar.save_action.statusTip()


# ---------------------------------------------------------------------------
# status bar — pills carry tooltips
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


class TestStatusBarPillTooltips:
    def test_counts_pill_tooltip(self, window):
        assert window.counts_status.toolTip() == "Blocks · wires · scopes"

    def test_theme_pill_tooltip(self, window):
        assert window.theme_status.toolTip() == "Click to toggle theme (Ctrl+T)"

    def test_core_pills_have_non_empty_tooltips(self, window):
        for attr in ("status_pill", "file_status", "counts_status",
                     "cursor_status", "zoom_status", "theme_status"):
            pill = getattr(window, attr)
            assert pill.toolTip().strip(), f"missing tooltip on {attr}"
