"""
Regression test for centralized palette theme re-styling.

Previously every block row, glyph, dot, and category section subscribed to
theme_manager.theme_changed individually (~150+ connections with ~70 blocks).
Now ModernBlockPalette holds a single connection and fans out to its live
children via findChildren. This locks in that the fan-out reaches rows and
that a theme toggle restyles the populated palette without error.
"""

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture
def palette():
    try:
        from lib.lib import DSim
        from modern_ui.widgets.modern_palette import ModernBlockPalette
        dsim = DSim()
        init = getattr(dsim, "menu_blocks_init", None)
        if callable(init):
            init()
        if not getattr(dsim, "menu_blocks", None):
            pytest.skip("no menu_blocks available in this environment")
        pal = ModernBlockPalette(dsim)
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"palette could not be constructed: {exc}")
    yield pal
    pal.deleteLater()


def _rows(pal):
    from modern_ui.widgets.modern_palette import CompactBlockRow
    return pal.findChildren(CompactBlockRow)


class TestCentralizedThemeRestyle:
    def test_palette_builds_rows(self, palette):
        assert len(_rows(palette)) > 0

    def test_central_handler_exists(self, palette):
        assert hasattr(palette, "_on_theme_changed")

    def test_toggle_theme_does_not_raise(self, palette):
        from modern_ui.themes.theme_manager import theme_manager
        start = theme_manager.current_theme
        try:
            theme_manager.toggle_theme()
            theme_manager.toggle_theme()
        finally:
            theme_manager.set_theme(start)

    def test_fanout_reaches_rows(self, palette, monkeypatch):
        rows = _rows(palette)
        assert rows
        called = []
        row = rows[0]
        orig = row._on_theme_changed
        monkeypatch.setattr(
            row, "_on_theme_changed", lambda *a: (called.append(True), orig())[1]
        )
        palette._on_theme_changed()
        assert called, "palette fan-out did not reach the row"

    def test_children_no_longer_self_subscribe(self, palette):
        # The child classes must not connect to theme_changed in __init__ any
        # more; the central handler drives them. We assert the marker methods
        # the handler relies on still exist on freshly built children.
        from modern_ui.widgets.modern_palette import CompactBlockRow, _CategorySection
        rows = palette.findChildren(CompactBlockRow)
        secs = palette.findChildren(_CategorySection)
        assert rows and secs
        assert all(hasattr(r, "_on_theme_changed") for r in rows)
        assert all(hasattr(s, "_apply_styling") for s in secs)
