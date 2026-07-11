"""Tests for the diagram image export / clipboard-copy feature.

Covers the pure exporter functions in
``modern_ui/tools/diagram_image_exporter.py`` and the MainWindow facades
(``export_image`` / ``copy_diagram_image``), all against a real ModernCanvas /
DSim built the same way as the other tests in this directory
(test_empty_canvas_hint.py, test_main_window_view_actions.py).

The exported image renders diagram content only (blocks, connections, ports)
against the theme's canvas background, with no chrome or selection styling.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_diagram_image_export.py -p no:cacheprovider -o addopts=""
"""

import pytest
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QFileDialog

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from modern_ui.themes.theme_manager import theme_manager
from modern_ui.tools import diagram_image_exporter as exporter

pytestmark = pytest.mark.qt


def _make_block(sid, x, y, w=100, h=80, name=None):
    block = DBlock(
        block_fn="Gain",
        sid=sid,
        coords=QRect(x, y, w, h),
        color="#4CAF50",
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit="none",
        fn_name="gain",
        params={"gain": 1.0},
        external=False,
        colors=None,
    )
    block.name = name if name is not None else "Gain%d" % sid
    return block


@pytest.fixture
def canvas(qapp):
    dsim = DSim()
    from modern_ui.widgets.modern_canvas import ModernCanvas

    c = ModernCanvas(dsim)
    c.resize(800, 600)
    return c


def _populate(canvas):
    """Two blocks joined by a connection line."""
    a = _make_block(0, 100, 100, name="A")
    b = _make_block(1, 400, 300, name="B")
    canvas.dsim.blocks_list.extend([a, b])
    line = DLine(0, "A", 0, "B", 0, [QPoint(200, 140), QPoint(400, 340)])
    canvas.dsim.line_list.append(line)
    return a, b, line


# ---------------------------------------------------------------------------
# Bounding rect
# ---------------------------------------------------------------------------


class TestBoundingRect:
    def test_empty_returns_none(self, canvas):
        assert exporter.diagram_bounding_rect(canvas) is None

    def test_covers_all_blocks_with_margin(self, canvas):
        _populate(canvas)
        rect = exporter.diagram_bounding_rect(canvas, margin=exporter.MARGIN)
        # Block A left/top is (100, 100); B extends to (500, 380).
        assert rect.left() <= 100 - exporter.MARGIN
        assert rect.top() <= 100 - exporter.MARGIN
        assert rect.right() >= 500 + exporter.MARGIN
        assert rect.bottom() >= 380 + exporter.MARGIN

    def test_covers_bezier_bow(self, canvas):
        # Vertically aligned ports: the forward Bezier bows sideways past both
        # anchor x's, so the rect must cover the path's true extent, not just
        # the waypoint list.
        line = DLine(0, "A", 0, "B", 0, [QPoint(200, 140), QPoint(200, 340)])
        canvas.dsim.line_list.append(line)
        path_rect = line.path.boundingRect()
        assert path_rect.width() > 10, "expected a real sideways bow"
        rect = exporter.diagram_bounding_rect(canvas, margin=exporter.MARGIN)
        assert rect.contains(path_rect)


# ---------------------------------------------------------------------------
# render_diagram_image
# ---------------------------------------------------------------------------


class TestRenderImage:
    def test_empty_returns_none(self, canvas):
        assert exporter.render_diagram_image(canvas) is None

    def test_image_size_and_non_background_pixels(self, canvas):
        _populate(canvas)
        scale = 3
        image = exporter.render_diagram_image(canvas, scale=scale)
        assert isinstance(image, QImage)
        # Comfortably larger than just the margin at the given scale.
        assert image.width() > exporter.MARGIN * scale
        assert image.height() > exporter.MARGIN * scale

        background = QImage(image.size(), image.format())
        background.fill(theme_manager.get_color("canvas_background"))
        assert image != background, "expected diagram content pixels distinct from background"

    def test_selection_restored_and_not_rendered(self, canvas):
        a, b, line = _populate(canvas)
        a.selected = True
        line.selected = True

        deselected = exporter.render_diagram_image(canvas)

        # Selection state is untouched by the export.
        assert a.selected is True
        assert line.selected is True

        # Clear selection, render again, and compare: selection styling must not
        # leak into the export, so the two images are pixel-identical.
        a.selected = False
        line.selected = False
        clean = exporter.render_diagram_image(canvas)
        assert deselected == clean


# ---------------------------------------------------------------------------
# File export (PNG / SVG) + clipboard
# ---------------------------------------------------------------------------


class TestFileExport:
    def test_png_roundtrip(self, canvas, tmp_path):
        _populate(canvas)
        path = str(tmp_path / "diagram.png")
        assert exporter.export_diagram_to_file(canvas, path) is True
        loaded = QImage(path)
        assert not loaded.isNull()

    def test_svg_writes_xml(self, canvas, tmp_path):
        _populate(canvas)
        path = str(tmp_path / "diagram.svg")
        assert exporter.export_diagram_to_file(canvas, path) is True
        with open(path, "r", encoding="utf-8") as fh:
            head = fh.read(512)
        assert "<?xml" in head or "<svg" in head

    def test_empty_export_writes_nothing(self, canvas, tmp_path):
        path = tmp_path / "empty.png"
        assert exporter.export_diagram_to_file(canvas, str(path)) is False
        assert not path.exists()

    def test_copy_puts_image_on_clipboard(self, canvas):
        _populate(canvas)
        QApplication.clipboard().clear()
        assert exporter.copy_diagram_to_clipboard(canvas) is True
        assert not QApplication.clipboard().image().isNull()

    def test_copy_empty_is_noop(self, canvas):
        QApplication.clipboard().clear()
        assert exporter.copy_diagram_to_clipboard(canvas) is False


# ---------------------------------------------------------------------------
# MainWindow facades
# ---------------------------------------------------------------------------


class TestMainWindowFacades:
    # Window fixtures live on the class so the 11 exporter-level tests above
    # never pay for (or depend on) a full MainWindow construction.
    @pytest.fixture(scope="class")
    def window(self, qapp):
        from modern_ui.main_window import ModernDiaBloSWindow

        w = ModernDiaBloSWindow()
        yield w
        w.close()

    @pytest.fixture(autouse=True)
    def _restore_diagram(self, window):
        saved = (list(window.canvas.dsim.blocks_list), list(window.canvas.dsim.line_list))
        yield
        window.canvas.dsim.blocks_list[:] = saved[0]
        window.canvas.dsim.line_list[:] = saved[1]

    def test_export_image_writes_png(self, window, tmp_path, monkeypatch):
        window.canvas.dsim.blocks_list[:] = [_make_block(0, 50, 50, name="A")]
        window.canvas.dsim.line_list[:] = []
        out = str(tmp_path / "win.png")
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (out, "PNG Image (*.png)")
        )
        window.export_image()
        assert QImage(out).isNull() is False
        assert "Exported image" in window.status_message.text()

    def test_export_image_empty_diagram_noop(self, window, monkeypatch):
        window.canvas.dsim.blocks_list[:] = []
        window.canvas.dsim.line_list[:] = []
        called = {"dialog": False}
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: called.__setitem__("dialog", True) or ("", ""),
        )
        window.export_image()
        assert called["dialog"] is False
        assert "Nothing to export" in window.status_message.text()

    def test_export_image_cancel_dialog(self, window, monkeypatch):
        window.canvas.dsim.blocks_list[:] = [_make_block(0, 50, 50, name="A")]
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        window.export_image()  # should simply return without raising

    def test_copy_diagram_image_facade(self, window):
        window.canvas.dsim.blocks_list[:] = [_make_block(0, 50, 50, name="A")]
        window.canvas.dsim.line_list[:] = []
        QApplication.clipboard().clear()
        window.copy_diagram_image()
        assert not QApplication.clipboard().image().isNull()
        assert "copied" in window.status_message.text().lower()
