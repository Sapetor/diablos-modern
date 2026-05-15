"""
Unit tests for ThemeManager block-palette system.
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_manager():
    """Return a new ThemeManager instance (independent of global singleton)."""
    # We must instantiate a QObject subclass, so a QApplication must exist.
    from modern_ui.themes.theme_manager import ThemeManager
    return ThemeManager()


@pytest.fixture(autouse=True)
def qt_app():
    """Ensure a QApplication exists for QObject construction."""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


# ---------------------------------------------------------------------------
# Default state
# ---------------------------------------------------------------------------

class TestDefaultPalette:
    def test_default_palette_is_solarized(self):
        mgr = _fresh_manager()
        assert mgr.current_palette == "solarized"

    def test_default_theme_is_dark(self):
        mgr = _fresh_manager()
        from modern_ui.themes.theme_manager import ThemeType
        assert mgr.current_theme == ThemeType.DARK

    def test_solarized_dark_source_fill(self):
        mgr = _fresh_manager()
        color = mgr.get_color("block_source")
        assert color.name().lower() == "#073642"

    def test_solarized_dark_source_border(self):
        mgr = _fresh_manager()
        color = mgr.get_color("block_source_border")
        assert color.name().lower() == "#859900"


# ---------------------------------------------------------------------------
# set_palette — Tailwind
# ---------------------------------------------------------------------------

class TestSetPaletteTailwind:
    def test_tailwind_dark_source_fill(self):
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        assert mgr.current_palette == "tailwind"
        color = mgr.get_color("block_source")
        assert color.name().lower() == "#064e3b"

    def test_tailwind_dark_process_fill(self):
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        color = mgr.get_color("block_process")
        assert color.name().lower() == "#1e3a8a"

    def test_tailwind_light_source_fill(self):
        """Switching theme to light with Tailwind palette gives light fill."""
        from modern_ui.themes.theme_manager import ThemeType
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        mgr.set_theme(ThemeType.LIGHT)
        color = mgr.get_color("block_source")
        assert color.name().lower() == "#d1fae5"

    def test_tailwind_light_control_border(self):
        from modern_ui.themes.theme_manager import ThemeType
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        mgr.set_theme(ThemeType.LIGHT)
        color = mgr.get_color("block_control_border")
        assert color.name().lower() == "#c4b5fd"


# ---------------------------------------------------------------------------
# set_palette — Catppuccin
# ---------------------------------------------------------------------------

class TestSetPaletteCatppuccin:
    def test_catppuccin_dark_source_accent(self):
        mgr = _fresh_manager()
        mgr.set_palette("catppuccin")
        color = mgr.get_color("block_source_accent")
        assert color.name().lower() == "#a6d189"

    def test_catppuccin_dark_fill_uniform(self):
        """All dark Catppuccin fills should be the Frappé base (#303446)."""
        mgr = _fresh_manager()
        mgr.set_palette("catppuccin")
        cats = ["source", "process", "control", "sink", "routing",
                "analysis", "pde", "optimization", "other"]
        for cat in cats:
            color = mgr.get_color(f"block_{cat}")
            assert color.name().lower() == "#303446", (
                f"Expected #303446 for block_{cat}, got {color.name()}"
            )

    def test_catppuccin_light_fill_uniform(self):
        """All light Catppuccin fills should be the Latte base (#eff1f5)."""
        from modern_ui.themes.theme_manager import ThemeType
        mgr = _fresh_manager()
        mgr.set_palette("catppuccin")
        mgr.set_theme(ThemeType.LIGHT)
        cats = ["source", "process", "control", "sink", "routing",
                "analysis", "pde", "optimization", "other"]
        for cat in cats:
            color = mgr.get_color(f"block_{cat}")
            assert color.name().lower() == "#eff1f5", (
                f"Expected #eff1f5 for block_{cat}, got {color.name()}"
            )


# ---------------------------------------------------------------------------
# Theme switch preserves palette context
# ---------------------------------------------------------------------------

class TestThemeAndPaletteInteraction:
    def test_dark_to_light_tailwind_source(self):
        from modern_ui.themes.theme_manager import ThemeType
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        # dark first
        assert mgr.get_color("block_source").name().lower() == "#064e3b"
        # switch to light
        mgr.set_theme(ThemeType.LIGHT)
        assert mgr.get_color("block_source").name().lower() == "#d1fae5"

    def test_chrome_keys_unaffected_by_palette(self):
        """Non-block keys (e.g. canvas_background) still come from chrome theme."""
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")
        # canvas_background is a chrome key — must still resolve
        color = mgr.get_color("canvas_background")
        assert color.isValid()


# ---------------------------------------------------------------------------
# Unknown palette fallback
# ---------------------------------------------------------------------------

class TestUnknownPaletteFallback:
    def test_unknown_palette_falls_back_to_solarized(self):
        mgr = _fresh_manager()
        mgr.set_palette("tailwind")  # change away from default
        mgr.set_palette("does_not_exist")
        # Should fall back to solarized
        assert mgr.current_palette == "solarized"
        color = mgr.get_color("block_source")
        assert color.name().lower() == "#073642"


# ---------------------------------------------------------------------------
# Signal fires on set_palette
# ---------------------------------------------------------------------------

class TestSignalOnSetPalette:
    def test_theme_changed_emitted_on_set_palette(self):
        mgr = _fresh_manager()
        received = []
        mgr.theme_changed.connect(lambda v: received.append(v))
        mgr.set_palette("tailwind")
        assert len(received) == 1

    def test_theme_changed_not_emitted_if_palette_unchanged(self):
        mgr = _fresh_manager()
        received = []
        mgr.theme_changed.connect(lambda v: received.append(v))
        mgr.set_palette("solarized")  # already solarized — no change
        assert len(received) == 0

    def test_theme_changed_emitted_on_set_theme(self):
        from modern_ui.themes.theme_manager import ThemeType
        mgr = _fresh_manager()
        received = []
        mgr.theme_changed.connect(lambda v: received.append(v))
        mgr.set_theme(ThemeType.LIGHT)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# All 9 categories defined in every palette × 2 themes
# ---------------------------------------------------------------------------

class TestPaletteCompleteness:
    CATEGORIES = ["source", "process", "control", "sink", "routing",
                  "analysis", "pde", "optimization", "other"]
    SUFFIXES = ["", "_border", "_accent"]

    def test_all_keys_present(self):
        from modern_ui.themes.theme_manager import PALETTES
        for palette_name, variants in PALETTES.items():
            for theme_name, colors in variants.items():
                for cat in self.CATEGORIES:
                    for suffix in self.SUFFIXES:
                        key = f"block_{cat}{suffix}"
                        assert key in colors, (
                            f"Missing {key!r} in palette={palette_name!r} theme={theme_name!r}"
                        )
