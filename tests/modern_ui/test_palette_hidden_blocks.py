"""The block palette must not offer blocks that cannot run.

``blocks/external.py`` is an unimplemented stub -- ``execute()`` only ever
returns an error dict and the engine hard-refuses External blocks -- yet it was
listed in the palette like any other block, so dragging it in could only
produce a diagram that refuses to run. Blocks opt out with a plain
``hidden = True`` class attribute.
"""

import types

import pytest

from modern_ui.widgets.modern_palette import is_hidden_block, visible_menu_blocks


def _menu_block(fn_name, cls):
    return types.SimpleNamespace(fn_name=fn_name, block_fn=fn_name, block_class=cls)


@pytest.mark.unit
class TestHiddenFlag:
    def test_external_block_declares_hidden(self):
        from blocks.external import ExternalBlock

        assert getattr(ExternalBlock, "hidden", False) is True

    def test_a_normal_block_does_not(self):
        from blocks.gain import GainBlock

        assert getattr(GainBlock, "hidden", False) is False

    def test_is_hidden_block_reads_the_class_attribute(self):
        hidden_cls = type("H", (), {"hidden": True})
        plain_cls = type("P", (), {})
        assert is_hidden_block(_menu_block("h", hidden_cls)) is True
        assert is_hidden_block(_menu_block("p", plain_cls)) is False
        # A menu entry with no class at all is not hidden.
        assert is_hidden_block(types.SimpleNamespace(fn_name="x")) is False

    def test_visible_menu_blocks_filters(self):
        hidden = _menu_block("external", type("H", (), {"hidden": True}))
        shown = _menu_block("gain", type("P", (), {}))
        assert visible_menu_blocks([hidden, shown]) == [shown]
        assert visible_menu_blocks(None) == []


@pytest.mark.qt
class TestPaletteContents:
    def test_external_is_absent_from_the_real_palette(self, qapp):
        from lib.lib import DSim
        from modern_ui.widgets.modern_palette import ModernBlockPalette

        dsim = DSim()
        assert any(getattr(b, "fn_name", "") == "external" for b in dsim.menu_blocks), (
            "External must still be registered (saved files reference it)"
        )

        palette = ModernBlockPalette(dsim)
        try:
            listed = {
                w.menu_block.fn_name
                for section in palette._sections
                for w in section.findChildren(object)
                if hasattr(w, "menu_block")
            }
            assert "external" not in listed
            assert "gain" in listed
        finally:
            palette.deleteLater()
