"""Display fixes: block name labels, port-label collisions, status-bar pills."""

import os

import pytest
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QApplication

from modern_ui.renderers.block_renderer import (
    BlockRenderer,
    block_label_layout,
)
from modern_ui.themes.theme_manager import theme_manager, ThemeType

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(ROOT, "examples", "pid_second_order.diablos")


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.show()
    yield w
    w.close()


def _find(window, name):
    return next(b for b in window.dsim.blocks_list if b.username == name)


@pytest.mark.qt
class TestBlockNameLabel:
    def test_label_rect_fits_text_and_is_centered(self, window):
        window.open_example(EXAMPLE)
        QApplication.processEvents()
        blk = _find(window, "load disturbance")
        rect, text = block_label_layout(blk)
        adv = QFontMetrics(blk.font).horizontalAdvance(text)
        assert text == blk.username
        assert rect.width() >= adv
        # centred on the block
        assert abs((rect.left() + rect.width() / 2) - (blk.left + blk.width / 2)) <= 1
        assert rect.top() > blk.top + blk.height

    def test_very_long_name_is_elided(self, window):
        blk = _find(window, "load disturbance")
        original = blk.username
        try:
            blk.username = "x" * 200
            rect, text = block_label_layout(blk)
            assert text.endswith("…")
            assert rect.width() <= max(160, 3 * blk.width) + 4
        finally:
            blk.username = original


@pytest.mark.qt
class TestPortLabelCollision:
    def test_pid_port_labels_never_cover_the_face(self, window):
        # Labels that don't fit stay off the face even when selected (they used
        # to stack over the PID text); the port tooltip names the port instead.
        window.open_example(EXAMPLE)
        QApplication.processEvents()
        pid = next(b for b in window.dsim.blocks_list if b.block_fn == "PID")

        class _P:
            def __init__(self):
                self.texts = []

            def setFont(self, *_a):
                pass

            def setPen(self, *_a):
                pass

            def setBrush(self, *_a):
                pass

            def drawRoundedRect(self, *_a):
                pass

            def drawText(self, *a):
                self.texts.append(a[-1])

        renderer = BlockRenderer()
        pid.selected = False
        p = _P()
        renderer.draw_port_labels(pid, p)
        assert p.texts == []
        pid.selected = True
        p = _P()
        renderer.draw_port_labels(pid, p)
        assert p.texts == []
        pid.selected = False

        canvas = window.canvas
        port = canvas.world_to_screen(pid.in_coords[0])
        assert canvas.tooltip_text_at(port).endswith("setpoint")


@pytest.mark.qt
class TestStatusBar:
    def test_counts_pill_current_after_loading_example(self, window):
        window.open_example(EXAMPLE)
        QApplication.processEvents()
        QApplication.processEvents()
        n_blocks = len(window.dsim.blocks_list)
        n_wires = len(window.dsim.line_list)
        assert n_blocks > 0
        text = window.counts_status.text()
        assert str(n_blocks) in text and str(n_wires) in text

    def test_theme_pill_follows_theme_toggle(self, window):
        original = theme_manager.current_theme
        try:
            theme_manager.set_theme(ThemeType.LIGHT)
            QApplication.processEvents()
            assert window.theme_status.text().startswith("Light")
            window.toggle_theme()
            QApplication.processEvents()
            assert window.theme_status.text().startswith("Dark")
        finally:
            theme_manager.set_theme(original)

    def test_status_message_is_elided_not_clipped(self, window):
        window.status_message.setText("A very long status message " * 40)
        QApplication.processEvents()
        pill = window.status_pill
        assert pill._label.text().endswith("…")
        assert pill.toolTip().startswith("A very long")
        window.status_message.setText("Ready")
        QApplication.processEvents()
        assert pill.sizeHint().width() <= pill.maximumWidth()


@pytest.mark.qt
class TestDiagramNameAndInspector:
    """The open file's name and counts reach the status bar and the
    Properties panel (both used to read a dsim attribute nothing set)."""

    def test_file_pill_names_the_open_diagram(self, window):
        window.open_example(EXAMPLE)
        QApplication.processEvents()
        QApplication.processEvents()
        window.status_bar_manager.refresh_file_status()
        assert window.file_status.text() == os.path.basename(EXAMPLE)

    def test_inspector_shows_loaded_diagram(self, window):
        from PyQt6.QtWidgets import QLabel

        window.property_editor.set_block(None)
        window.open_example(EXAMPLE)
        QApplication.processEvents()
        QApplication.processEvents()
        window.status_bar_manager.refresh_counts()
        texts = [lbl.text() for lbl in window.property_editor.findChildren(QLabel)]
        n_blocks = len(window.dsim.blocks_list)
        n_wires = len(window.dsim.line_list)
        assert "pid_second_order" in texts
        assert any(t.startswith(f"{n_blocks} blocks") and f"{n_wires} wires" in t for t in texts)
