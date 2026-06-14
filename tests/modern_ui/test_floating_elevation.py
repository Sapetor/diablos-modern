"""Construction smoke tests for soft elevation on floating surfaces.

The command palette overlay frame and the error panel each get a real drop
shadow from the shared ``ELEVATION`` scale via ``make_shadow``. Neither widget
holds any other QGraphicsEffect, so the shadow is attached to the top-level
frame (the palette card / the panel widget itself). These tests confirm the
effect is present and carries the requested elevation token.
"""

import pytest
from PyQt5.QtWidgets import QGraphicsDropShadowEffect

from modern_ui.themes.theme_manager import ELEVATION


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


class TestCommandPaletteElevation:
    def test_card_has_drop_shadow(self):
        from modern_ui.widgets.command_palette import CommandPalette
        palette = CommandPalette()
        try:
            effect = palette._card.graphicsEffect()
            assert isinstance(effect, QGraphicsDropShadowEffect)
        finally:
            palette.deleteLater()

    def test_card_uses_e3_elevation(self):
        from modern_ui.widgets.command_palette import CommandPalette
        palette = CommandPalette()
        try:
            effect = palette._card.graphicsEffect()
            token = ELEVATION['e3']
            assert effect.blurRadius() == token['blur']
            assert effect.yOffset() == token['offset']
            assert effect.color().alpha() == token['alpha']
        finally:
            palette.deleteLater()


class TestErrorPanelElevation:
    def test_panel_has_drop_shadow(self):
        from modern_ui.widgets.error_panel import ErrorPanel
        panel = ErrorPanel()
        try:
            effect = panel.graphicsEffect()
            assert isinstance(effect, QGraphicsDropShadowEffect)
        finally:
            panel.deleteLater()

    def test_panel_uses_e2_elevation(self):
        from modern_ui.widgets.error_panel import ErrorPanel
        panel = ErrorPanel()
        try:
            effect = panel.graphicsEffect()
            token = ELEVATION['e2']
            assert effect.blurRadius() == token['blur']
            assert effect.yOffset() == token['offset']
            assert effect.color().alpha() == token['alpha']
        finally:
            panel.deleteLater()
