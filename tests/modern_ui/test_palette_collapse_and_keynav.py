"""Tests for collapsible palette categories and keyboard navigation.

Covers two additions to ``modern_ui.widgets.modern_palette``:

1. Collapsible ``_CategorySection`` — clicking the header toggles its rows'
   visibility, flips the chevron, and persists the collapsed flag via QSettings
   keyed by category name; the flag is restored when the palette is rebuilt.
2. Keyboard navigation — the search box installs an event filter so Down/Up
   step into the visible rows, rows are focusable, and Enter/``activate()``
   re-emits the palette's public ``block_drag_started`` signal.

QSettings is redirected to a temporary ini file (by monkeypatching the module's
``QSettings`` symbol) so persistence is exercised without touching the real
user store.
"""

import pytest
from PyQt5.QtCore import Qt, QSettings, QEvent
from PyQt5.QtGui import QKeyEvent

import modern_ui.widgets.modern_palette as mp
from modern_ui.widgets.modern_palette import (
    CompactBlockRow,
    _CategorySection,
    _CHEVRON_EXPANDED,
    _CHEVRON_COLLAPSED,
    _collapsed_settings_key,
)


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Point the module's QSettings at a throwaway ini file.

    Returns a zero-arg factory matching how production code constructs
    ``QSettings(org, app)``, so the test can read back what the palette wrote.
    """
    ini = str(tmp_path / "palette_settings.ini")

    def _factory(*_args, **_kwargs):
        return QSettings(ini, QSettings.IniFormat)

    monkeypatch.setattr(mp, "QSettings", _factory)
    return _factory


@pytest.fixture
def palette(qapp, isolated_settings):
    """Build a real palette, or skip when no menu_blocks are available."""
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
    # Show so child visibility resolves (isVisible() is False under a hidden
    # top-level), matching how visible_rows() behaves in the running app.
    pal.show()
    qapp.processEvents()
    yield pal
    pal.deleteLater()


def _section_with_rows(pal):
    for s in pal.findChildren(_CategorySection):
        if s.rows:
            return s
    pytest.skip("no populated category section available")


# -- Collapse --------------------------------------------------------------


class TestCollapsibleCategory:
    def test_settings_key_includes_category(self):
        assert _collapsed_settings_key("Sources") == "palette/collapsed/Sources"

    def test_toggle_hides_rows_and_flips_chevron(self, palette):
        section = _section_with_rows(palette)
        assert not section.is_collapsed()
        assert all(r.isVisible() for r in section.rows)
        assert section.header.text().startswith(_CHEVRON_EXPANDED)

        section.toggle_collapsed()

        assert section.is_collapsed()
        assert all(not r.isVisible() for r in section.rows)
        assert section.header.text().startswith(_CHEVRON_COLLAPSED)

    def test_toggle_persists_flag(self, palette, isolated_settings):
        section = _section_with_rows(palette)
        cat = section.category_name
        section.toggle_collapsed()

        stored = isolated_settings().value(
            _collapsed_settings_key(cat), False, type=bool
        )
        assert stored is True

    def test_collapsed_state_restored_across_rebuild(self, palette):
        section = _section_with_rows(palette)
        cat = section.category_name
        section.toggle_collapsed()
        assert section.is_collapsed()

        # Rebuild the palette (drops + recreates every section).
        palette.refresh_blocks()

        rebuilt = next(
            s for s in palette.findChildren(_CategorySection)
            if s.category_name == cat
        )
        assert rebuilt.is_collapsed()
        assert all(not r.isVisible() for r in rebuilt.rows)
        assert rebuilt.header.text().startswith(_CHEVRON_COLLAPSED)

    def test_collapsed_section_keeps_header_visible_under_filter(self, palette):
        section = _section_with_rows(palette)
        section.toggle_collapsed()
        # Filtering matches at least one of its rows by an empty query.
        stayed_visible = section.filter("")
        assert stayed_visible is True
        # Rows stay hidden while collapsed, even though they match the filter.
        assert all(not r.isVisible() for r in section.rows)


# -- Keyboard navigation ---------------------------------------------------


class TestKeyboardNavigation:
    def test_rows_are_focusable(self, palette):
        rows = palette.findChildren(CompactBlockRow)
        assert rows
        assert all(r.focusPolicy() == Qt.StrongFocus for r in rows)

    def test_visible_rows_helper_constructs(self, palette):
        rows = palette.visible_rows()
        assert isinstance(rows, list)
        assert all(r.isVisible() for r in rows)

    def test_down_arrow_from_search_focuses_first_row(self, palette):
        rows = palette.visible_rows()
        assert rows
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
        handled = palette.eventFilter(palette.search_bar, ev)
        assert handled is True
        assert rows[0].hasFocus()

    def test_up_arrow_from_search_focuses_last_row(self, palette):
        rows = palette.visible_rows()
        assert rows
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier)
        handled = palette.eventFilter(palette.search_bar, ev)
        assert handled is True
        assert rows[-1].hasFocus()

    def test_enter_activates_focused_row(self, palette):
        rows = palette.visible_rows()
        assert rows
        row = rows[0]
        received = []
        palette.block_drag_started.connect(lambda mb: received.append(mb))
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier)
        row.keyPressEvent(ev)
        assert received and received[0] is row.menu_block

    def test_arrow_moves_focus_between_rows(self, palette):
        rows = palette.visible_rows()
        if len(rows) < 2:
            pytest.skip("need at least two visible rows")
        rows[0].setFocus(Qt.OtherFocusReason)
        ev = QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier)
        rows[0].keyPressEvent(ev)
        assert rows[1].hasFocus()
