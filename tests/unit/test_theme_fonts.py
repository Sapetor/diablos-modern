"""Font-helper tests for :mod:`lib.theming.theme_manager`.

``get_ui_font`` / ``get_mono_font`` build a QFont from a whole fallback stack:
the ``QFont(family)`` constructor pins the first entry (what ``family()``
reports) and ``setFamilies`` records the rest for Qt to fall back through.
``QFont.setFamilies`` is unconditionally available on Qt6, so there is no
longer a guarded legacy branch -- these tests pin both halves of the result.
"""

import pytest
from PyQt6.QtGui import QFont

from lib.theming import theme_manager as tm


@pytest.mark.unit
class TestThemeFonts:
    def test_ui_font_normal_path(self, qapp):
        f = tm.get_ui_font(size=tm.TYPE["body"], weight=tm.WEIGHT["semibold"])
        assert isinstance(f, QFont)
        assert f.family() == tm.UI_FONT_STACK[0]
        assert f.pointSize() == tm.TYPE["body"]
        assert f.weight() == tm._qt_weight(tm.WEIGHT["semibold"])

    def test_mono_font_normal_path(self, qapp):
        f = tm.get_mono_font(size=tm.TYPE["caption"])
        assert isinstance(f, QFont)
        assert f.family() == tm.MONO_FONT_STACK[0]
        assert f.pointSize() == tm.TYPE["caption"]

    def test_fonts_without_size_or_weight(self, qapp):
        assert isinstance(tm.get_ui_font(), QFont)
        assert isinstance(tm.get_mono_font(), QFont)

    @pytest.mark.parametrize(
        "builder, stack_name",
        [("get_ui_font", "UI_FONT_STACK"), ("get_mono_font", "MONO_FONT_STACK")],
    )
    def test_whole_fallback_stack_is_applied(self, qapp, builder, stack_name):
        """The helpers hand Qt the entire fallback stack, not just its head."""
        stack = getattr(tm, stack_name)
        f = getattr(tm, builder)(size=tm.TYPE["title"], weight=tm.WEIGHT["bold"])
        assert isinstance(f, QFont)
        assert f.family() == stack[0]
        assert f.families() == list(stack)
        assert f.pointSize() == tm.TYPE["title"]
        assert f.weight() == tm._qt_weight(tm.WEIGHT["bold"])
