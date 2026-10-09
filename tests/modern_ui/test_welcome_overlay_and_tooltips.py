"""Welcome overlay (empty state) and canvas hover tooltips, on the real window."""

import gc
import os

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtTest import QTest

from modern_ui.widgets.welcome_overlay import curated_examples


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    gc.collect()


def _settle(qapp, window):
    window.canvas.update()
    for _ in range(4):
        qapp.processEvents()
        QTest.qWait(5)


def _add(canvas, block_fn, x=300, y=300):
    mb = next(m for m in canvas.dsim.menu_blocks if m.block_fn == block_fn)
    blk = canvas.add_block_from_palette(mb, QPoint(x, y))
    assert blk is not None
    return blk


def test_curated_example_files_exist():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
    for fname, title, desc in curated_examples():
        assert os.path.isfile(os.path.join(root, fname)), fname
        assert title and desc


def test_overlay_visible_on_fresh_window(window, qapp):
    _settle(qapp, window)
    assert window.welcome_overlay.isVisible()
    assert len(window.welcome_overlay.example_buttons) == 3


def test_overlay_hides_on_add_and_returns_on_new(window, qapp):
    _settle(qapp, window)
    _add(window.canvas, "Gain")
    _settle(qapp, window)
    assert not window.welcome_overlay.isVisible()
    window.new_diagram()
    _settle(qapp, window)
    assert window.welcome_overlay.isVisible()


def test_overlay_is_mouse_transparent_outside_buttons(window, qapp):
    from PyQt6.QtCore import Qt

    ov = window.welcome_overlay
    assert ov.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert ov.panel.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert not ov.example_buttons[0].testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_example_card_loads_example(window, qapp):
    _settle(qapp, window)
    card = window.welcome_overlay.example_buttons[1]
    card.click()
    _settle(qapp, window)
    assert window.dsim.blocks_list
    assert "pid_second_order" in window.status_message.text()
    assert not window.welcome_overlay.isVisible()


@pytest.mark.parametrize("fname", [f for f, _, _ in curated_examples()])
def test_each_example_loads(window, qapp, fname):
    window.welcome_overlay.open_example(fname)
    _settle(qapp, window)
    assert window.dsim.blocks_list


def test_overlay_subsystem_mode_shows_light_hint(window, qapp):
    from modern_ui.widgets.welcome_overlay import MODE_SUBSYSTEM

    window.dsim.subsystem_manager.current_subsystem = "sub0"
    window.welcome_overlay.refresh()
    assert window.welcome_overlay.mode == MODE_SUBSYSTEM
    assert window.welcome_overlay.isVisible()
    assert not window.welcome_overlay.actions_box.isVisible()
    window.dsim.subsystem_manager.current_subsystem = None


def test_tooltip_has_full_elided_name(window, qapp):
    canvas = window.canvas
    blk = _add(canvas, "Gain")
    blk.username = "A very long descriptive block name that surely gets elided"
    from modern_ui.renderers.block_renderer import block_label_layout

    rect, shown = block_label_layout(blk)
    assert shown != blk.username  # really elided
    text = canvas.tooltip_text_at(canvas.world_to_screen(rect.center()))
    assert blk.username in text
    assert blk.block_fn in text


def test_tooltip_on_body_and_empty_space(window, qapp):
    canvas = window.canvas
    blk = _add(canvas, "Gain", 400, 300)
    body = canvas.world_to_screen(blk.rect.center())
    assert blk.username in canvas.tooltip_text_at(body)
    assert canvas.tooltip_text_at(QPoint(5, 5)) == ""


def test_layout_survives_the_app_stylesheet(qapp, window):
    """The app QSS (QPushButton min-height/min-width) used to squash the example
    cards, and the palette header clipped "Library"; check under the real QSS."""
    from PyQt6.QtGui import QFontMetrics

    from modern_ui.styles.qss_styles import ModernStyles

    window.setStyleSheet(ModernStyles.get_complete_stylesheet())
    _settle(qapp, window)

    overlay = window.canvas.welcome_overlay
    cards = overlay.findChildren(type(overlay.open_button), "welcomeCard")
    assert len(cards) == 3
    for card in cards:
        line = card.fontMetrics().lineSpacing()
        assert card.height() >= 4 * line
        assert card.width() >= 170

    # The heading must never be clipped, even with a longer translation; the
    # palette's width is fixed by the panel minimum, and platform fonts differ
    # (CI's Linux fonts overflowed a header that fit exactly on macOS).
    title = window.block_palette.title
    for text in ("Library", "Biblioteca"):
        title.setText(text)
        _settle(qapp, window)
        assert title.width() >= QFontMetrics(title.font()).horizontalAdvance(text)
