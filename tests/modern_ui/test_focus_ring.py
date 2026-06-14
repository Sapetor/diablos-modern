"""
Unit tests for the keyboard-focus ring (accessibility).

Tab-navigation needs a visible focus indicator. Previously only
QLineEdit/QSpinBox showed a ``:focus`` border; these tests pin the new
``:focus`` rules added for QPushButton, the transport cluster
(#TransportPlay/Pause/Stop/Step), and the command-palette button. The ring
is a 2px accent border (distinct from the 1px hover edge) and is expressed
purely with @tokens, so the resolved stylesheet must still carry no
unresolved ``@tokens``.
"""

import re

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


def _focus_blocks(css: str):
    """Return the bodies of every ``... :focus { ... }`` rule in ``css``."""
    return re.findall(r":focus\s*\{([^}]*)\}", css)


class TestButtonFocusRing:
    def test_button_focus_rule_references_accent_value(self):
        from modern_ui.styles.qss_styles import ModernStyles
        from modern_ui.themes.theme_manager import theme_manager

        css = ModernStyles.get_button_style()
        accent = theme_manager.get_color("accent_primary").name().upper()

        # There is at least one :focus rule, and one of them paints the ring
        # with the resolved accent color (border_focus == accent_primary).
        blocks = _focus_blocks(css)
        assert blocks, "button stylesheet has no :focus rule"
        assert any(accent in b.upper() for b in blocks), (
            f"no :focus rule references the accent color {accent}"
        )

    def test_focus_ring_is_two_px_and_distinct_from_hover(self):
        from modern_ui.styles.qss_styles import ModernStyles

        css = ModernStyles.get_button_style()
        blocks = _focus_blocks(css)
        # The ring is a 2px border — wider than the 1px hover edge.
        assert any("2px solid" in b for b in blocks), (
            "focus ring is not a 2px solid border"
        )

    def test_transport_buttons_have_focus_rule(self):
        from modern_ui.styles.qss_styles import ModernStyles

        css = ModernStyles.get_toolbar_style()
        # Each transport object-name carries a :focus selector.
        for name in (
            "TransportPlay",
            "TransportPause",
            "TransportStop",
            "TransportStep",
        ):
            assert f"#{name}:focus" in css, f"#{name} has no :focus rule"

    def test_command_palette_button_has_focus_rule(self):
        from modern_ui.styles.qss_styles import ModernStyles

        css = ModernStyles.get_toolbar_style()
        assert "#CommandPaletteBtn:focus" in css


class TestFocusRingTokensResolved:
    def test_button_style_has_no_unresolved_tokens(self):
        from modern_ui.styles.qss_styles import ModernStyles

        css = ModernStyles.get_button_style()
        assert not re.findall(r"@[a-z_]+", css), "button style has unresolved @tokens"

    def test_complete_stylesheet_has_no_unresolved_tokens(self):
        from modern_ui.styles.qss_styles import ModernStyles

        css = ModernStyles.get_complete_stylesheet()
        assert not re.findall(r"@[a-z_]+", css), "stylesheet has unresolved @tokens"
