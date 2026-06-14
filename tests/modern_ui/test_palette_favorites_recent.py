"""
Favorites + Recent sections at the top of the block palette.

Covers three things:
  * ``update_recents`` — the pure ordering helper (newest-first, de-duplicated,
    capped). Unit-tested in isolation, no Qt needed.
  * Pinning a block persists across a palette rebuild (a favorited fn_name is
    re-rendered in a Favorites section after refresh_blocks()).
  * Adding a block records it into Recent (newest-first, de-duplicated, capped),
    and the Recent section appears.

QSettings is redirected to a temp .ini file so the tests never touch the user's
real palette preferences, and so persistence is observable across two separate
palette instances.
"""

import pytest

from PyQt5.QtCore import QSettings

from modern_ui.widgets import modern_palette as mp
from modern_ui.widgets.modern_palette import (
    ModernBlockPalette, CompactBlockRow, _PinnedSection, update_recents,
)


# -- Pure helper -----------------------------------------------------------

@pytest.mark.unit
class TestUpdateRecents:
    def test_promotes_to_front_newest_first(self):
        assert update_recents(["a", "b", "c"], "d", 6) == ["d", "a", "b", "c"]

    def test_dedupes_moving_existing_to_front(self):
        # Re-adding an existing entry moves it to the front rather than dupes it.
        assert update_recents(["a", "b", "c"], "b", 6) == ["b", "a", "c"]

    def test_caps_to_n(self):
        assert update_recents(["a", "b", "c"], "d", 3) == ["d", "a", "b"]

    def test_empty_fn_name_is_a_noop_copy(self):
        src = ["a", "b"]
        out = update_recents(src, "", 6)
        assert out == ["a", "b"]
        assert out is not src  # must not mutate the input list

    def test_does_not_mutate_input(self):
        src = ["a", "b"]
        update_recents(src, "c", 6)
        assert src == ["a", "b"]


# -- Qt fixtures -----------------------------------------------------------

@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect palette persistence to a private temp .ini store.

    The module reads/writes through the module-level ``ui_settings`` name, so
    patching it here isolates every favorites/recents read and write while still
    letting two palette instances share the same backing file (to prove
    persistence across a rebuild).
    """
    ini = str(tmp_path / "palette_settings.ini")

    def _fake_ui_settings():
        return QSettings(ini, QSettings.IniFormat)

    monkeypatch.setattr(mp, "ui_settings", _fake_ui_settings)
    return _fake_ui_settings


def _make_palette():
    from lib.lib import DSim
    dsim = DSim()
    init = getattr(dsim, "menu_blocks_init", None)
    if callable(init):
        init()
    if not getattr(dsim, "menu_blocks", None):
        pytest.skip("no menu_blocks available in this environment")
    return ModernBlockPalette(dsim)


@pytest.fixture
def palette():
    try:
        pal = _make_palette()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"palette could not be constructed: {exc}")
    yield pal
    pal.deleteLater()


def _pinned_section(palette, title):
    for sec in palette.findChildren(_PinnedSection):
        if sec.category_name == title:
            return sec
    return None


def _fn_names_in_section(section):
    return [getattr(r.menu_block, "fn_name", None) for r in section.rows]


# -- Favorites persistence -------------------------------------------------

def test_pin_persists_across_rebuild(palette):
    # No favorites yet → no Favorites section.
    assert _pinned_section(palette, "Favorites") is None

    rows = palette.findChildren(CompactBlockRow)
    assert rows
    target = getattr(rows[0].menu_block, "fn_name", None)
    assert target

    palette.pin_favorite(target)

    sec = _pinned_section(palette, "Favorites")
    assert sec is not None, "Favorites section should appear once a block is pinned"
    assert target in _fn_names_in_section(sec)
    assert palette.is_favorite(target)


def test_favorite_survives_a_fresh_palette_instance(palette):
    rows = palette.findChildren(CompactBlockRow)
    target = getattr(rows[0].menu_block, "fn_name", None)
    palette.pin_favorite(target)

    # A brand-new palette (sharing the isolated settings file) must reload the
    # pinned block into its Favorites section — proving persistence, not just
    # in-memory state.
    fresh = _make_palette()
    try:
        sec = _pinned_section(fresh, "Favorites")
        assert sec is not None
        assert target in _fn_names_in_section(sec)
    finally:
        fresh.deleteLater()


def test_unpin_removes_section_when_empty(palette):
    rows = palette.findChildren(CompactBlockRow)
    target = getattr(rows[0].menu_block, "fn_name", None)
    palette.pin_favorite(target)
    assert _pinned_section(palette, "Favorites") is not None

    palette.unpin_favorite(target)
    assert _pinned_section(palette, "Favorites") is None
    assert not palette.is_favorite(target)


# -- Recent maintenance ----------------------------------------------------

def test_record_recent_adds_section_newest_first(palette):
    assert _pinned_section(palette, "Recent") is None

    rows = palette.findChildren(CompactBlockRow)
    names = []
    for r in rows:
        n = getattr(r.menu_block, "fn_name", None)
        if n and n not in names:
            names.append(n)
        if len(names) >= 3:
            break
    assert len(names) >= 2, "need at least two distinct blocks for this test"

    for n in names[:3]:
        palette.record_recent(n)

    sec = _pinned_section(palette, "Recent")
    assert sec is not None
    # Newest-first: the last recorded name is at the front.
    assert _fn_names_in_section(sec)[0] == names[:3][-1]


def test_record_recent_is_capped_and_deduped(palette):
    rows = palette.findChildren(CompactBlockRow)
    distinct = []
    for r in rows:
        n = getattr(r.menu_block, "fn_name", None)
        if n and n not in distinct:
            distinct.append(n)
    if len(distinct) <= mp._RECENT_CAP:
        pytest.skip("not enough distinct blocks to exercise the cap")

    # Record more than the cap.
    for n in distinct[: mp._RECENT_CAP + 2]:
        palette.record_recent(n)

    sec = _pinned_section(palette, "Recent")
    assert sec is not None
    section_names = _fn_names_in_section(sec)
    assert len(section_names) == mp._RECENT_CAP, "Recent must be capped"

    # Re-recording an existing block de-dupes (moves to front, no growth).
    before = len(section_names)
    palette.record_recent(section_names[-1])
    sec = _pinned_section(palette, "Recent")
    after = _fn_names_in_section(sec)
    assert len(after) == before
    assert after[0] == section_names[-1]


def test_context_menu_pin_then_unpin(palette, monkeypatch):
    # Drive the right-click menu path: exec_ is monkeypatched to "click" the one
    # action, so we exercise the deferred-after-exec mutation without a display.
    from PyQt5.QtGui import QContextMenuEvent
    from PyQt5.QtWidgets import QMenu

    monkeypatch.setattr(QMenu, "exec_", lambda self, *a, **k: self.actions()[0])

    rows = palette.findChildren(CompactBlockRow)
    assert rows
    row = rows[0]
    target = getattr(row.menu_block, "fn_name", None)
    assert target

    ev = QContextMenuEvent(QContextMenuEvent.Mouse, row.rect().center())
    row.contextMenuEvent(ev)  # menu offers "Pin to Favorites"
    assert palette.is_favorite(target)

    # Re-query: the original row was deleted by the rebuild; pick the favorite's
    # row and unpin it through the menu the same way.
    sec = _pinned_section(palette, "Favorites")
    assert sec is not None
    fav_row = sec.rows[0]
    ev2 = QContextMenuEvent(QContextMenuEvent.Mouse, fav_row.rect().center())
    fav_row.contextMenuEvent(ev2)  # menu now offers "Unpin from Favorites"
    assert not palette.is_favorite(target)
    assert _pinned_section(palette, "Favorites") is None


def test_activate_records_recent(palette):
    rows = palette.findChildren(CompactBlockRow)
    assert rows
    target = getattr(rows[0].menu_block, "fn_name", None)

    rows[0].activate()  # what Enter / click-add does

    sec = _pinned_section(palette, "Recent")
    assert sec is not None
    assert _fn_names_in_section(sec)[0] == target
