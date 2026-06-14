"""
Recents are recorded at the single real chokepoint — the canvas add path —
not on _start_drag.

Background
----------
The palette's Recent section used to record a block unconditionally in
``CompactBlockRow._start_drag`` *after* ``drag.exec_()`` returned. That fired
even when the drag was cancelled with Escape or dropped outside the canvas, so a
block the user never placed still showed up under Recent. Recording now happens
once per *actual* placement in ``ModernCanvas.add_block_from_palette`` (the
chokepoint every add path funnels through: drag-drop, Enter/activate,
double-click, quick-add), reached defensively via ``window.block_palette``.

This file covers:
  * the bug fix — a drag whose ``exec_`` is neutralized (cancelled / off-canvas)
    records NOTHING;
  * the wiring — ``add_block_from_palette`` records the placed block into the
    palette's Recent exactly once, via ``window.block_palette.record_recent``;
  * the helper — ``_record_palette_recent`` is defensive: no window, no palette,
    or a palette without ``record_recent`` is a silent no-op that never raises.

QSettings is redirected to a temp .ini so these tests never touch the user's
real palette preferences.
"""

import pytest

from PyQt5.QtCore import QPoint, QSettings
from PyQt5.QtWidgets import QMainWindow

from modern_ui.widgets import modern_palette as mp
from modern_ui.widgets.modern_palette import (
    ModernBlockPalette, CompactBlockRow, _PinnedSection,
)


# -- Qt fixtures -----------------------------------------------------------

@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect palette persistence to a private temp .ini store.

    The module reads/writes recents through the module-level ``ui_settings``
    name; patching it here isolates every read/write from the real user store.
    """
    ini = str(tmp_path / "palette_settings.ini")

    def _fake_ui_settings():
        return QSettings(ini, QSettings.IniFormat)

    monkeypatch.setattr(mp, "ui_settings", _fake_ui_settings)
    return _fake_ui_settings


def _make_dsim():
    from lib.lib import DSim
    dsim = DSim()
    init = getattr(dsim, "menu_blocks_init", None)
    if callable(init):
        init()
    if not getattr(dsim, "menu_blocks", None):
        pytest.skip("no menu_blocks available in this environment")
    return dsim


def _recent_section(palette):
    for sec in palette.findChildren(_PinnedSection):
        if sec.category_name == "Recent":
            return sec
    return None


def _recent_names(palette):
    sec = _recent_section(palette)
    if sec is None:
        return []
    return [getattr(r.menu_block, "fn_name", None) for r in sec.rows]


@pytest.fixture
def palette():
    try:
        pal = ModernBlockPalette(_make_dsim())
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"palette could not be constructed: {exc}")
    yield pal
    pal.deleteLater()


# -- Bug fix: a cancelled / off-canvas drag records nothing ----------------

def test_cancelled_drag_does_not_record_recent(palette, monkeypatch):
    """_start_drag must not touch recents — only a real add does."""
    assert _recent_section(palette) is None  # nothing recorded yet

    rows = palette.findChildren(CompactBlockRow)
    assert rows
    row = rows[0]

    # Neutralize the blocking QDrag.exec_ so no real drag starts. This models a
    # drag that was cancelled (Escape) or dropped outside the canvas: exec_
    # returns without the canvas drop path ever running.
    from PyQt5.QtGui import QDrag
    monkeypatch.setattr(QDrag, "exec_", lambda *a, **k: 0)

    row._start_drag(None)

    # The old code recorded here unconditionally; the fix means it must not.
    assert _recent_section(palette) is None
    assert _recent_names(palette) == []


# -- Wiring: a real canvas add records into the palette's Recent -----------

@pytest.fixture
def wired(palette):
    """A ModernCanvas and the ``palette`` under one top-level window.

    Both share a single DSim (the palette's), and the window exposes
    ``block_palette`` exactly like LayoutManager does, so the canvas can reach
    the palette through ``self.window()`` — the real production path.
    """
    from modern_ui.widgets.modern_canvas import ModernCanvas

    win = QMainWindow()
    canvas = ModernCanvas(palette.dsim)
    win.setCentralWidget(canvas)
    win.block_palette = palette
    try:
        yield win, canvas, palette
    finally:
        canvas.deleteLater()
        win.deleteLater()


def _first_menu_block(palette):
    rows = palette.findChildren(CompactBlockRow)
    assert rows
    mb = rows[0].menu_block
    assert getattr(mb, "fn_name", None)
    return mb


def test_canvas_add_records_recent_once(wired):
    win, canvas, palette = wired
    mb = _first_menu_block(palette)
    target = mb.fn_name

    new_block = canvas.add_block_from_palette(mb, QPoint(120, 80))
    assert new_block is not None, "add_block_from_palette should create a block"

    # The palette was rebuilt by record_recent; re-query the live Recent section.
    names = _recent_names(canvas.window().block_palette)
    assert names, "Recent section should appear after a real add"
    assert names[0] == target            # newest-first
    assert names.count(target) == 1      # recorded exactly once


def test_canvas_add_records_each_distinct_block_newest_first(wired):
    win, canvas, palette = wired

    rows = palette.findChildren(CompactBlockRow)
    distinct = []
    for r in rows:
        n = getattr(r.menu_block, "fn_name", None)
        if n and n not in distinct:
            distinct.append(r.menu_block)
        if len(distinct) >= 2:
            break
    if len(distinct) < 2:
        pytest.skip("need at least two distinct blocks")

    for mb in distinct:
        # Re-resolve the canvas->palette link each time (the palette rebuilds).
        canvas.add_block_from_palette(mb, QPoint(60, 60))

    names = _recent_names(canvas.window().block_palette)
    # Last added is newest-first; both distinct blocks present, no duplicates.
    assert names[0] == distinct[-1].fn_name
    for mb in distinct:
        assert names.count(mb.fn_name) == 1


def test_add_calls_record_palette_recent_once_per_add(wired, monkeypatch):
    """The chokepoint invokes the recents recorder exactly once per real add."""
    win, canvas, palette = wired
    mb = _first_menu_block(palette)

    calls = []
    monkeypatch.setattr(
        type(canvas), "_record_palette_recent",
        lambda self, menu_block: calls.append(menu_block),
    )

    canvas.add_block_from_palette(mb, QPoint(100, 100))
    assert calls == [mb]


# -- Helper is defensive: never breaks adding a block ----------------------

def test_record_palette_recent_no_window_is_silent(qapp):
    """A detached canvas (top-level window has no block_palette) must not raise."""
    from modern_ui.widgets.modern_canvas import ModernCanvas
    from lib.lib import DSim

    canvas = ModernCanvas(DSim())
    try:
        # No owning window with a block_palette: must be a silent no-op.
        canvas._record_palette_recent(type("MB", (), {"fn_name": "Step"})())
    finally:
        canvas.deleteLater()


def test_record_palette_recent_missing_recorder_is_silent(qapp):
    """A window whose block_palette lacks record_recent must not raise."""
    from modern_ui.widgets.modern_canvas import ModernCanvas
    from lib.lib import DSim

    win = QMainWindow()
    canvas = ModernCanvas(DSim())
    win.setCentralWidget(canvas)
    win.block_palette = object()  # no record_recent attribute
    try:
        canvas._record_palette_recent(type("MB", (), {"fn_name": "Step"})())
    finally:
        canvas.deleteLater()
        win.deleteLater()
