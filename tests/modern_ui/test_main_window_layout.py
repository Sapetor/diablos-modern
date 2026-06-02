"""
Characterization tests for ModernDiaBloSWindow's layout/panel-construction
cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable result of
the layout cluster before it is extracted into a dedicated manager:

  * ``_setup_layout``               (central widget + nested splitters)
  * ``_create_left_panel``          (block palette)
  * ``_create_canvas_area``         (canvas + breadcrumb + error panel + signal
                                     wiring)
  * ``_create_property_panel``      (property editor in a scroll area)
  * ``_initialize_splitter_sizes``  (width-based splitter sizing w/ min clamp)

These assert on the constructed widget tree and attribute surface (canvas,
property_editor, main_splitter, center_splitter, ... -- several referenced
outside the window, e.g. diagram_service reads main_window.main_splitter).

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_layout.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import pytest
from PyQt5.QtWidgets import QSplitter

from modern_ui.widgets.modern_canvas import ModernCanvas
from modern_ui.widgets.property_editor import PropertyEditor
from modern_ui.platform_config import get_platform_config


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


# ---------------------------------------------------------------------------
# attribute surface
# ---------------------------------------------------------------------------

class TestLayoutAttributes:
    def test_panels_and_splitters_exist(self, window):
        for attr in ("left_panel", "canvas_area", "property_panel",
                     "main_splitter", "center_splitter",
                     "_center_splitter_for_init", "block_palette", "canvas",
                     "breadcrumb_bar", "error_panel", "property_editor",
                     "_prop_scroll_viewport"):
            assert hasattr(window, attr), f"missing layout attribute: {attr}"

    def test_central_widget_set(self, window):
        assert window.centralWidget() is not None

    def test_canvas_is_modern_canvas(self, window):
        assert isinstance(window.canvas, ModernCanvas)

    def test_property_editor_type(self, window):
        assert isinstance(window.property_editor, PropertyEditor)


# ---------------------------------------------------------------------------
# splitter hierarchy
# ---------------------------------------------------------------------------

class TestSplitterHierarchy:
    def test_main_splitter_holds_left_and_center(self, window):
        ms = window.main_splitter
        assert isinstance(ms, QSplitter)
        assert ms.count() == 2
        # widget 0 is the left panel; widget 1 is the center splitter
        assert ms.widget(0) is window.left_panel
        assert ms.widget(1) is window.center_splitter

    def test_center_splitter_holds_canvas_and_property(self, window):
        cs = window.center_splitter
        assert cs.count() == 2
        assert cs.widget(0) is window.canvas_area
        assert cs.widget(1) is window.property_panel

    def test_center_splitter_for_init_alias(self, window):
        assert window._center_splitter_for_init is window.center_splitter

    def test_canvas_inside_canvas_area(self, window):
        # The canvas widget lives within the canvas_area container.
        assert window.canvas_area.isAncestorOf(window.canvas)
        assert window.canvas_area.isAncestorOf(window.breadcrumb_bar)
        assert window.canvas_area.isAncestorOf(window.error_panel)


# ---------------------------------------------------------------------------
# _initialize_splitter_sizes
# ---------------------------------------------------------------------------

class TestInitializeSplitterSizes:
    """Characterize the sizing *computation* by capturing the requested
    setSizes() values. (The post-layout sizes() are mediated by Qt's layout
    engine + widget min/max constraints and are environment-dependent, so they
    are not a stable characterization target.)"""

    def test_requested_sizes_math(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.main_splitter, "setSizes",
                            lambda s: captured.__setitem__("main", list(s)))
        monkeypatch.setattr(window.center_splitter, "setSizes",
                            lambda s: captured.__setitem__("center", list(s)))

        window._initialize_splitter_sizes()
        config = get_platform_config()

        actual_width = window.width()
        left_width = config.splitter_left_width
        center_width = actual_width - left_width

        # Main splitter: fixed left, remainder to center.
        assert captured["main"] == [left_width, center_width]

        # Center splitter: property = percent of center, clamped up to min;
        # canvas takes the rest. Sum must equal the center width.
        expected_property = int(center_width * config.splitter_property_percent)
        if expected_property < config.splitter_property_min_width:
            expected_property = config.splitter_property_min_width
        expected_canvas = center_width - expected_property
        assert captured["center"] == [expected_canvas, expected_property]
        assert sum(captured["center"]) == center_width


# ---------------------------------------------------------------------------
# signal wiring done in _create_canvas_area
# ---------------------------------------------------------------------------

class TestCanvasSignalWiring:
    def test_command_palette_requested_opens_palette(self, window, monkeypatch):
        called = {}
        monkeypatch.setattr(window.command_palette, "show_palette",
                            lambda: called.setdefault("shown", True))
        # Signal was connected to window.show_command_palette in _create_canvas_area.
        window.canvas.command_palette_requested.emit()
        assert called.get("shown") is True
