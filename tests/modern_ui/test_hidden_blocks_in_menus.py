"""Hidden blocks must be absent from *every* place the user can pick a block.

``blocks/external.py`` is an unimplemented stub (``execute()`` only ever returns
an error dict), so it declares ``hidden = True``. The block palette already
filtered on that flag, but the two other pickers did not:

  * the command palette (``modern_ui/managers/command_palette_manager.py``)
    still emitted an "Add External block" command, and
  * the canvas right-click menu (``modern_ui/managers/menu_manager.py``) still
    fed the whole ``menu_blocks`` list to its inline search / quick-add.

Both now go through ``visible_menu_blocks()`` -- the same helper the palette
uses, so there is one definition of "offerable block".
"""

import types
from unittest.mock import MagicMock

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QWidget

from modern_ui.managers import command_palette_manager as cpm_module
from modern_ui.managers import menu_manager as menu_module


def _menu_block(fn_name, cls):
    return types.SimpleNamespace(fn_name=fn_name, block_fn=fn_name, block_class=cls)


HIDDEN = _menu_block("external", type("HiddenBlock", (), {"hidden": True}))
SHOWN = _menu_block("gain", type("PlainBlock", (), {}))


@pytest.mark.unit
class TestPickersShareTheVisibilityHelper:
    """Both managers use the palette's helper rather than re-implementing it."""

    def test_command_palette_manager_uses_visible_menu_blocks(self):
        from modern_ui.widgets.modern_palette import visible_menu_blocks

        assert cpm_module.visible_menu_blocks is visible_menu_blocks

    def test_menu_manager_uses_visible_menu_blocks(self):
        from modern_ui.widgets.modern_palette import visible_menu_blocks

        assert menu_module.visible_menu_blocks is visible_menu_blocks


@pytest.mark.unit
class TestCommandPaletteIndex:
    def _block_commands(self, menu_blocks):
        """Run setup() against a stub window; return its indexed block commands."""
        window = MagicMock()
        window.canvas.dsim.menu_blocks = menu_blocks
        window._load_recent_files.return_value = []

        cpm_module.CommandPaletteManager(window).setup()

        window.command_palette.set_commands.assert_called_once()
        commands = window.command_palette.set_commands.call_args[0][0]
        return [c for c in commands if c.get("type") == "block"]

    def test_hidden_block_is_not_indexed(self):
        blocks = self._block_commands([HIDDEN, SHOWN])
        assert [c["data"]["block_type"] for c in blocks] == ["gain"]

    def test_visible_blocks_are_still_indexed(self):
        blocks = self._block_commands([SHOWN])
        assert len(blocks) == 1
        assert blocks[0]["name"] == "Add gain block"

    def test_real_menu_blocks_register_external_but_do_not_offer_it(self, qapp):
        """External stays registered (saved files reference it) but is not offered."""
        from lib.lib import DSim

        dsim = DSim()
        assert any(getattr(b, "fn_name", "") == "external" for b in dsim.menu_blocks)

        indexed = {c["data"]["block_type"] for c in self._block_commands(dsim.menu_blocks)}
        assert "external" not in indexed
        assert "gain" in indexed


@pytest.mark.unit
class TestCanvasContextMenu:
    def test_canvas_menu_search_list_filters_hidden_blocks(self, qapp, monkeypatch):
        """The inline search / quick-add list is built from visible blocks only."""
        seen = {}
        real_search_widget = menu_module._CanvasSearchWidget

        def spy_search_widget(on_add, all_blocks):
            seen["all_blocks"] = list(all_blocks)
            return real_search_widget(on_add=on_add, all_blocks=all_blocks)

        monkeypatch.setattr(menu_module, "_CanvasSearchWidget", spy_search_widget)
        # Never pop the modal menu in a test run.
        monkeypatch.setattr(menu_module.QMenu, "exec_", lambda self, *a, **k: None)

        # QMenu needs a real QWidget parent, so stub the canvas as one.
        canvas = QWidget()
        canvas.dsim = types.SimpleNamespace(menu_blocks=[HIDDEN, SHOWN], line_list=[])
        canvas.clipboard_blocks = []
        canvas._paste_blocks = lambda pos: None
        canvas._select_all_blocks = lambda: None
        canvas._clear_selections = lambda: None

        try:
            menu_module.MenuManager(canvas).show_canvas_context_menu(QPoint(10, 10))
        finally:
            canvas.deleteLater()

        assert [mb.fn_name for mb in seen["all_blocks"]] == ["gain"]
