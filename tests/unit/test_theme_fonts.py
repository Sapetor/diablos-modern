"""Font-helper tests for :mod:`lib.theming.theme_manager`.

``QFont.setFamilies`` only exists on Qt >= 5.13. Supported release builds use
PyQt5 5.15 (see ``docs/building.md``), but older interpreters still turn up in
dev environments, so ``get_ui_font`` / ``get_mono_font`` guard the call with
``hasattr`` and rely on the ``QFont(family)`` constructor to pin the first
stack entry as the fallback. These tests cover both branches.
"""

import pytest
from PyQt5.QtGui import QFont

from lib.theming import theme_manager as tm


class _LegacyQFont(QFont):
    """A QFont that pretends to be Qt < 5.13 by hiding ``setFamilies``.

    ``hasattr`` swallows the AttributeError, so the guard in the font helpers
    takes the fallback branch exactly as it would on an older Qt.
    """

    def __getattribute__(self, name):
        if name == "setFamilies":
            raise AttributeError(name)
        return super().__getattribute__(name)


@pytest.mark.unit
class TestThemeFonts:
    def test_ui_font_normal_path(self, qapp):
        f = tm.get_ui_font(size=tm.TYPE["body"], weight=tm.WEIGHT["semibold"])
        assert isinstance(f, QFont)
        assert f.family() == tm.UI_FONT_STACK[0]
        assert f.pointSize() == tm.TYPE["body"]
        assert f.weight() == tm._qt5_weight(tm.WEIGHT["semibold"])

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
    def test_fallback_when_setfamilies_missing(self, qapp, monkeypatch, builder, stack_name):
        """On Qt < 5.13 the helpers still return a usable font.

        The fallback is the first family in the stack, pinned by the
        ``QFont(family)`` constructor.
        """
        monkeypatch.setattr(tm, "QFont", _LegacyQFont)
        f = getattr(tm, builder)(size=tm.TYPE["title"], weight=tm.WEIGHT["bold"])
        assert isinstance(f, QFont)
        assert f.family() == getattr(tm, stack_name)[0]
        assert f.pointSize() == tm.TYPE["title"]
        assert f.weight() == tm._qt5_weight(tm.WEIGHT["bold"])
