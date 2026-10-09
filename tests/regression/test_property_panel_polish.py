"""Properties controls remain legible with the application's full stylesheet."""

from pathlib import Path

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QStyle, QStyleOptionButton

pytestmark = [pytest.mark.regression, pytest.mark.qt]


@pytest.fixture
def panel(window):
    from modern_ui.styles.qss_styles import ModernStyles

    # The main entry point installs this app-wide; window fixtures do not.
    window.setStyleSheet(ModernStyles.get_complete_stylesheet())
    path = Path(__file__).resolve().parents[2] / "examples/c01_tank_feedback.diablos"
    assert window.project_manager.diagram_service.load_diagram(str(path))
    block = next(b for b in window.dsim.blocks_list if b.block_fn == "TranFn")
    window.property_editor.set_block(block)
    window.show()
    QTest.qWait(20)
    return window.property_editor


def test_reset_button_has_room_to_draw_its_symbol(panel):
    _, button, _ = panel._widgets["denominator"]
    button.ensurePolished()
    option = QStyleOptionButton()
    option.initFrom(button)
    rect = button.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, button)
    assert button.width() <= 24
    assert rect.width() >= 14 and rect.height() >= 14


def test_reset_uses_visible_icon_and_still_restores_default(panel, qtbot):
    editor, button, _ = panel._widgets["denominator"]
    assert not button.icon().isNull()
    image = button.icon().pixmap(button.iconSize()).toImage()
    assert any(
        image.pixelColor(x, y).alpha() > 0
        for x in range(image.width())
        for y in range(image.height())
    )
    with qtbot.waitSignal(panel.property_changed) as signal:
        button.click()
    assert signal.args[1:] == ["denominator", panel._defaults["denominator"]]
    assert button.isHidden()
    assert editor.text() == str(panel._defaults["denominator"])
