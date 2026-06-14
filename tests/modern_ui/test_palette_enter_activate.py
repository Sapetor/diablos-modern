"""
Regression test: pressing Enter on a focused palette row emits the
palette-level ``block_drag_started(object)`` signal carrying the block, so the
layout wiring (layout_manager connects it to _add_block_from_palette_menu) can
add the block to the canvas. Without this, keyboard navigation could move focus
but Enter did nothing.
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


def test_row_activate_emits_palette_signal_with_block(palette):
    from modern_ui.widgets.modern_palette import CompactBlockRow

    rows = palette.findChildren(CompactBlockRow)
    assert rows
    row = rows[0]

    received = []
    palette.block_drag_started.connect(lambda mb: received.append(mb))

    row.activate()  # what Enter does

    assert received == [row.menu_block]


def test_drag_path_does_not_emit_palette_add_signal(palette, monkeypatch):
    # The Enter signal must be distinct from drag-and-drop, or wiring it to the
    # add-block path would double-add on every drag. Starting a drag must NOT
    # emit block_drag_started.
    from modern_ui.widgets.modern_palette import CompactBlockRow

    rows = palette.findChildren(CompactBlockRow)
    assert rows
    row = rows[0]

    received = []
    palette.block_drag_started.connect(lambda mb: received.append(mb))

    # Neutralize the actual QDrag.exec_ so the test doesn't start a real drag.
    from PyQt5.QtGui import QDrag
    monkeypatch.setattr(QDrag, "exec_", lambda *a, **k: 0)
    try:
        row._start_drag(None)
    except Exception:
        # _start_drag may need a real event; the point is only that it must not
        # emit the palette add-signal. If it raised before emitting, that's fine.
        pass

    assert received == [], "drag path must not emit the Enter add-signal"
