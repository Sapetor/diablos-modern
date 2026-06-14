"""
Unit tests for the design-token foundation (Phase 1).

Covers the non-color primitives added to ``theme_manager``: the spacing,
radius, and typography scales, the font stacks + helpers, the elevation
tokens, and their exposure through ``get_qss_variables()`` /
``ModernStyles._replace_theme_variables``.
"""

import pytest

from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QGraphicsDropShadowEffect


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


# ---------------------------------------------------------------------------
# Token scales are well-formed
# ---------------------------------------------------------------------------

class TestTokenScales:
    def test_space_scale_values(self):
        from modern_ui.themes.theme_manager import SPACE
        assert SPACE == {"xs": 2, "sm": 4, "md": 8, "lg": 12, "xl": 16, "2xl": 24}
        assert all(isinstance(v, int) and v >= 0 for v in SPACE.values())

    def test_radius_scale_values(self):
        from modern_ui.themes.theme_manager import RADIUS
        assert RADIUS["sm"] == 4 and RADIUS["md"] == 6 and RADIUS["lg"] == 8
        assert RADIUS["pill"] >= 999  # fully rounded

    def test_type_scale_is_monotonic(self):
        from modern_ui.themes.theme_manager import TYPE
        order = ["caption", "body", "body_strong", "subtitle", "title", "heading"]
        sizes = [TYPE[k] for k in order]
        assert sizes == sorted(sizes)
        assert TYPE["body"] == 9 and TYPE["heading"] == 14

    def test_weight_scale_css_values(self):
        from modern_ui.themes.theme_manager import WEIGHT
        assert WEIGHT["regular"] == 400
        assert WEIGHT["medium"] == 500
        assert WEIGHT["semibold"] == 600

    def test_elevation_tokens_increase(self):
        from modern_ui.themes.theme_manager import ELEVATION
        assert set(ELEVATION) == {"e1", "e2", "e3"}
        blurs = [ELEVATION[k]["blur"] for k in ("e1", "e2", "e3")]
        alphas = [ELEVATION[k]["alpha"] for k in ("e1", "e2", "e3")]
        assert blurs == sorted(blurs)            # deeper elevation = softer/larger
        assert alphas == sorted(alphas)
        assert all(0 <= ELEVATION[k]["alpha"] <= 255 for k in ELEVATION)


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

class TestFontHelpers:
    def test_ui_font_uses_stack_and_size(self):
        from modern_ui.themes.theme_manager import get_ui_font, UI_FONT_STACK, TYPE
        f = get_ui_font(TYPE["body"])
        assert isinstance(f, QFont)
        assert f.pointSize() == 9
        # First family of the stack is the requested family.
        assert f.family() == UI_FONT_STACK[0]

    def test_mono_font_uses_mono_stack(self):
        from modern_ui.themes.theme_manager import get_mono_font, MONO_FONT_STACK
        f = get_mono_font(10)
        assert f.family() == MONO_FONT_STACK[0]
        assert f.pointSize() == 10

    def test_css_weight_maps_to_qt5_enum(self):
        from modern_ui.themes.theme_manager import get_ui_font, WEIGHT
        regular = get_ui_font(9, WEIGHT["regular"]).weight()
        semibold = get_ui_font(9, WEIGHT["semibold"]).weight()
        # Qt5 enum: Normal(50) < DemiBold(63). The mapping must preserve order.
        assert regular == QFont.Normal
        assert semibold == QFont.DemiBold
        assert semibold > regular

    def test_helpers_default_to_no_explicit_size(self):
        from modern_ui.themes.theme_manager import get_ui_font
        # Should not raise when called with no args.
        f = get_ui_font()
        assert isinstance(f, QFont)


# ---------------------------------------------------------------------------
# Drop-shadow factory
# ---------------------------------------------------------------------------

class TestMakeShadow:
    def test_returns_drop_shadow_effect(self):
        from modern_ui.themes.theme_manager import make_shadow
        eff = make_shadow("e2")
        assert isinstance(eff, QGraphicsDropShadowEffect)

    @pytest.mark.parametrize("level", ["e1", "e2", "e3"])
    def test_blur_and_offset_match_elevation_token(self, level):
        from modern_ui.themes.theme_manager import make_shadow, ELEVATION
        eff = make_shadow(level)
        token = ELEVATION[level]
        assert eff.blurRadius() == token["blur"]
        assert eff.yOffset() == token["offset"]
        # Drop-down shadow has no horizontal offset.
        assert eff.xOffset() == 0

    def test_default_level_is_e2(self):
        from modern_ui.themes.theme_manager import make_shadow, ELEVATION
        eff = make_shadow()
        assert eff.blurRadius() == ELEVATION["e2"]["blur"]
        assert eff.yOffset() == ELEVATION["e2"]["offset"]

    def test_color_alpha_from_token_by_default(self):
        from modern_ui.themes.theme_manager import make_shadow, ELEVATION
        eff = make_shadow("e3")
        col = eff.color()
        assert (col.red(), col.green(), col.blue()) == (0, 0, 0)
        assert col.alpha() == ELEVATION["e3"]["alpha"]

    def test_explicit_color_overrides_default(self):
        from modern_ui.themes.theme_manager import make_shadow
        tint = QColor(10, 20, 30, 128)
        eff = make_shadow("e2", color=tint)
        assert eff.color() == tint

    def test_unknown_level_falls_back_to_e2(self):
        from modern_ui.themes.theme_manager import make_shadow, ELEVATION
        eff = make_shadow("nope")
        assert eff.blurRadius() == ELEVATION["e2"]["blur"]
        assert eff.yOffset() == ELEVATION["e2"]["offset"]
        assert eff.color().alpha() == ELEVATION["e2"]["alpha"]


# ---------------------------------------------------------------------------
# QSS variable exposure
# ---------------------------------------------------------------------------

class TestQssVariableExposure:
    def _vars(self):
        from modern_ui.themes.theme_manager import ThemeManager
        return ThemeManager().get_qss_variables()

    def test_spacing_tokens_exposed(self):
        v = self._vars()
        assert v["@space_md"] == "8px"
        assert v["@space_xs"] == "2px"
        assert v["@space_2xl"] == "24px"

    def test_radius_tokens_exposed(self):
        v = self._vars()
        assert v["@radius_md"] == "6px"
        assert v["@radius_lg"] == "8px"

    def test_type_tokens_exposed(self):
        v = self._vars()
        assert v["@font_body"] == "9pt"
        assert v["@font_heading"] == "14pt"

    def test_color_tokens_still_present(self):
        # Adding non-color tokens must not drop the existing color variables.
        v = self._vars()
        assert "@background_primary" in v
        assert "@accent_primary" in v

    def test_replace_theme_variables_substitutes_token(self):
        from modern_ui.styles.qss_styles import ModernStyles
        qss = "QFrame { border-radius: @radius_lg; padding: @space_md; font-size: @font_body; }"
        out = ModernStyles._replace_theme_variables(qss)
        assert "@radius_lg" not in out and "8px" in out
        assert "@space_md" not in out and "8px" in out
        assert "@font_body" not in out and "9pt" in out

    def test_longer_type_token_not_clobbered_by_prefix(self):
        # @font_body_strong must not be partially replaced by @font_body.
        from modern_ui.styles.qss_styles import ModernStyles
        out = ModernStyles._replace_theme_variables("a { font-size: @font_body_strong; }")
        assert "@font_body" not in out
        assert "10pt" in out


class TestCompleteStylesheetUsesTokens:
    def test_complete_stylesheet_has_no_unresolved_tokens(self):
        import re
        from modern_ui.styles.qss_styles import ModernStyles
        css = ModernStyles.get_complete_stylesheet()
        assert not re.findall(r"@[a-z_]+", css), "stylesheet has unresolved @tokens"

    def test_radii_resolve_to_scale_values(self):
        # Radii now come from the RADIUS scale: pill (999) fully rounds the
        # status pill; the old 3/5/11px literals must be gone.
        from modern_ui.styles.qss_styles import ModernStyles
        css = ModernStyles.get_complete_stylesheet()
        assert "border-radius: 999px" in css            # @radius_pill
        assert "border-radius: 11px" not in css
        assert "border-radius: 5px" not in css
        assert "border-radius: 3px" not in css
