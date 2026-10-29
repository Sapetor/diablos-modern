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


def test_overlay_hides_on_add_and_returns_on_new(window, qapp, monkeypatch):
    _settle(qapp, window)
    _add(window.canvas, "Gain")
    _settle(qapp, window)
    assert not window.welcome_overlay.isVisible()
    # The added block is unsaved, so New asks first; discard it.
    monkeypatch.setattr(window, "_prompt_unsaved_changes", lambda **_: "discard")
    window.new_diagram()
    _settle(qapp, window)
    assert window.welcome_overlay.isVisible()


def _window_click(window, widget, pos=None):
    """Click through the top-level window, so Qt hit-tests like a real mouse
    (QAbstractButton.click() and QTest on the widget itself skip that)."""
    from PyQt6.QtCore import Qt

    pos = widget.rect().center() if pos is None else pos
    QTest.mouseClick(
        window.windowHandle(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        widget.mapTo(window, pos),
    )


def test_overlay_buttons_receive_real_clicks(window, qapp):
    """WA_TransparentForMouseEvents on a container hides its children from
    hit-testing too, which once made every welcome button dead."""
    _settle(qapp, window)
    ov = window.welcome_overlay
    for btn in [*ov.example_buttons, ov.open_button, ov.browse_button]:
        hit = window.childAt(btn.mapTo(window, btn.rect().center()))
        assert hit is btn or btn.isAncestorOf(hit), (btn.objectName(), hit)


def test_overlay_card_opens_example_on_window_click(window, qapp):
    _settle(qapp, window)
    _window_click(window, window.welcome_overlay.example_buttons[0])
    _settle(qapp, window)
    assert window.dsim.blocks_list
    assert "first_order_step_response" in window.status_message.text()


@pytest.mark.parametrize("where", ["outside_panel", "panel_margin", "hint_label"])
def test_overlay_passes_other_clicks_to_canvas(window, qapp, where):
    from PyQt6.QtCore import QEvent, QObject

    _settle(qapp, window)
    ov = window.welcome_overlay
    target, pos = {
        "outside_panel": (ov, QPoint(10, 10)),
        "panel_margin": (ov.panel, QPoint(4, 4)),
        "hint_label": (ov.hint_label, None),
    }[where]

    class Spy(QObject):
        presses = 0

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.MouseButtonPress:
                Spy.presses += 1
            return False

    spy = Spy()
    window.canvas.installEventFilter(spy)
    _window_click(window, target, pos)
    window.canvas.removeEventFilter(spy)
    assert Spy.presses == 1


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
    # Measuring text needs a real font database: Qt's offscreen platform on
    # Windows has none (QFontDatabase.families() is empty), so every glyph gets
    # placeholder metrics far wider than Segoe UI. Native Windows was checked
    # by hand: "Biblioteca" needs and gets 58px in the 164px palette.
    from PyQt6.QtGui import QFontDatabase

    if not QFontDatabase.families():
        pytest.skip("no font database on this Qt platform; text widths are meaningless")
    title = window.block_palette.title
    for text in ("Library", "Biblioteca"):
        title.setText(text)
        _settle(qapp, window)
        assert title.width() >= QFontMetrics(title.font()).horizontalAdvance(text)


@pytest.mark.parametrize("factor", [1.25, 1.5])
def test_overlay_follows_the_ui_scale(qapp, window, factor):
    """UI scale sets the app font (10pt * factor, see setup_application). The
    overlay's text and cards used fixed TYPE point sizes, so they stayed at
    100% whatever the scale."""
    from PyQt6.QtGui import QFont

    overlay = window.canvas.welcome_overlay
    cards = overlay.findChildren(type(overlay.open_button), "welcomeCard")

    def measure():
        overlay._apply_styling()
        _settle(qapp, window)
        return overlay.title_label.font().pointSizeF(), cards[0].maximumWidth()

    base_title, base_width = measure()
    original = QFont(qapp.font())
    try:
        scaled = QFont(original)
        scaled.setPointSizeF(10 * factor)
        qapp.setFont(scaled)
        title, width = measure()
    finally:
        qapp.setFont(original)
        overlay._apply_styling()
    assert title == pytest.approx(base_title * factor, abs=0.6)
    assert width == pytest.approx(base_width * factor, abs=2)


def test_scaled_cards_fit_a_narrow_canvas(qapp, window):
    """At 150% the scaled card width overflowed a narrow canvas: the cards
    overlapped each other and the Open/Browse buttons."""
    from PyQt6.QtGui import QFont

    overlay = window.canvas.welcome_overlay
    cards = overlay.findChildren(type(overlay.open_button), "welcomeCard")
    original = QFont(qapp.font())
    try:
        scaled = QFont(original)
        scaled.setPointSizeF(15)
        qapp.setFont(scaled)
        overlay._apply_styling()
        window.resize(1000, 700)
        _settle(qapp, window)
        geo = sorted((c.geometry() for c in cards), key=lambda g: g.left())
        for a, b in zip(geo, geo[1:]):
            assert a.right() < b.left(), "example cards overlap"
        panel = overlay.panel.contentsRect()
        assert geo[-1].right() <= panel.right()
    finally:
        qapp.setFont(original)
        overlay._apply_styling()
