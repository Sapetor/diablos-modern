"""Tests for the ThemeManager color / font-metrics memoization.

``get_color`` used to re-parse a hex string on every call and the renderers call
it ~21 times per block per frame. It now memoizes the parsed RGBA channels.
Two properties have to hold:

1. **Freshness** -- the cache must be dropped whenever the active theme,
   palette or solid-fill mode changes, or blocks would keep painting in the
   previous theme's colors.
2. **Isolation** -- callers all over ``modern_ui`` do
   ``c = get_color(...); c.setAlpha(...)``, so every call must still hand back
   its own mutable QColor. The cache stores plain int tuples for that reason.
"""

import pytest
from PyQt5.QtGui import QFont

from lib.theming.theme_manager import (
    DEFAULT_PALETTE,
    ThemeManager,
    ThemeType,
    font_metrics,
    get_ui_font,
    text_width,
)

pytestmark = pytest.mark.qt


@pytest.fixture
def manager(qapp):
    return ThemeManager()


class TestGetColorCache:
    def test_repeated_calls_return_equal_but_distinct_objects(self, manager):
        first = manager.get_color("canvas_background")
        second = manager.get_color("canvas_background")
        assert first == second
        assert first is not second, "callers mutate the result; it must not be shared"

    def test_mutating_a_result_cannot_poison_the_cache(self, manager):
        color = manager.get_color("canvas_background")
        original_alpha = color.alpha()
        color.setAlpha(7)
        color.setRed(3)
        again = manager.get_color("canvas_background")
        assert again.alpha() == original_alpha
        assert again.red() != 3 or original_alpha == 7

    def test_eight_digit_hex_with_alpha_still_parses(self, manager):
        shadow = manager.get_color("block_shadow")
        assert 0 < shadow.alpha() < 255
        assert manager.get_color("block_shadow").alpha() == shadow.alpha()

    def test_unknown_name_falls_back_to_black(self, manager):
        color = manager.get_color("no_such_token_at_all")
        assert (color.red(), color.green(), color.blue()) == (0, 0, 0)

    def test_theme_switch_invalidates(self, manager):
        manager.set_theme(ThemeType.DARK)
        dark = manager.get_color("canvas_background").name()
        manager.set_theme(ThemeType.LIGHT)
        light = manager.get_color("canvas_background").name()
        assert dark != light
        manager.set_theme(ThemeType.DARK)
        assert manager.get_color("canvas_background").name() == dark

    def test_palette_switch_invalidates(self, manager):
        manager.set_palette(DEFAULT_PALETTE)
        before = manager.get_color("block_source").name()
        manager.set_palette("catppuccin")
        after = manager.get_color("block_source").name()
        assert before != after

    def test_solid_fills_switch_invalidates(self, manager):
        manager.set_solid_fills(False)
        gradientish = manager.get_color("block_source").name()
        manager.set_solid_fills(True)
        solid = manager.get_color("block_source").name()
        assert gradientish != solid
        manager.set_solid_fills(False)
        assert manager.get_color("block_source").name() == gradientish

    def test_direct_attribute_assignment_also_invalidates(self, manager):
        """Not everything goes through set_theme(); the property must still flush."""
        manager.current_theme = ThemeType.DARK
        dark = manager.get_color("canvas_background").name()
        manager.current_theme = ThemeType.LIGHT
        assert manager.get_color("canvas_background").name() != dark


class TestFontMetricsCache:
    def test_same_font_returns_same_metrics_object(self, qapp):
        a = font_metrics(get_ui_font(11))
        b = font_metrics(get_ui_font(11))
        assert a is b

    def test_different_size_gets_its_own_metrics(self, qapp):
        assert font_metrics(get_ui_font(11)) is not font_metrics(get_ui_font(17))

    def test_bold_and_italic_are_distinguished(self, qapp):
        plain = get_ui_font(12)
        bold = QFont(plain)
        bold.setWeight(QFont.Bold)
        italic = QFont(plain)
        italic.setItalic(True)
        assert font_metrics(plain) is not font_metrics(bold)
        assert font_metrics(plain) is not font_metrics(italic)

    def test_text_width_matches_qt(self, qapp):
        metrics = font_metrics(get_ui_font(12))
        expected = (
            metrics.horizontalAdvance("hello")
            if hasattr(metrics, "horizontalAdvance")
            else metrics.width("hello")
        )
        assert text_width(metrics, "hello") == expected
        assert text_width(metrics, "") == 0
