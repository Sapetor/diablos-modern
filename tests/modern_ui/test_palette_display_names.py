"""Palette UX: display names, Recent cap, Enter-to-add, collapse-all."""

import pytest
from PyQt6.QtCore import QPoint, QSettings
from PyQt6.QtWidgets import QMainWindow

from modern_ui.widgets import modern_palette as mp
from modern_ui.widgets.modern_palette import CompactBlockRow, ModernBlockPalette


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    ini = str(tmp_path / "palette.ini")

    def _factory(*_a, **_k):
        return QSettings(ini, QSettings.Format.IniFormat)

    monkeypatch.setattr(mp, "ui_settings", _factory)
    monkeypatch.setattr(mp, "QSettings", _factory)


@pytest.fixture
def palette():
    from lib.lib import DSim

    dsim = DSim()
    pal = ModernBlockPalette(dsim)
    yield pal
    pal.deleteLater()


def _labels(pal):
    return {r.menu_block.fn_name: r.name_label.text() for r in pal.findChildren(CompactBlockRow)}


def _visible_fn_names(pal):
    return [r.menu_block.fn_name for r in pal.visible_rows()]


def test_rows_show_display_names(palette):
    labels = _labels(palette)
    assert labels["transfer_function"] == "Transfer Function"
    assert labels["randomsource"] == "Random Source"
    assert labels["matrixgain"] == "Matrix Gain"
    assert labels["fromfile"] == "From File"
    assert labels["wavegenerator"] == "Wave Generator"
    assert "transfer_function" not in labels.values()


def test_registry_keys_unchanged(palette):
    mbs = {mb.fn_name: mb for mb in palette.dsim.menu_blocks}
    assert mbs["transfer_function"].block_fn == "TranFn"
    assert mbs["randomsource"].block_fn == "RandomSource"


def test_filter_by_display_name_and_old_id(palette):
    palette.show()
    for query in ("transfer function", "transfer_function", "tranfn", "TRANFN"):
        palette.search_bar.setText(query)
        assert "transfer_function" in _visible_fn_names(palette), query
    palette.search_bar.setText("randomsource")
    assert "randomsource" in _visible_fn_names(palette)
    palette.search_bar.setText("random source")
    assert "randomsource" in _visible_fn_names(palette)


def test_recent_capped_at_three(palette):
    names = [mb.fn_name for mb in palette.dsim.menu_blocks][:5]
    for n in names:
        palette.record_recent(n)
    assert len(mp._load_recents()) == 3
    assert mp._load_recents()[0] == names[-1]
    sec = [s for s in palette.findChildren(mp._PinnedSection) if s.category_name == "Recent"][0]
    assert len(sec.rows) == 3


def test_collapse_all_and_expand_all(palette):
    palette.toggle_all_collapsed()
    cats = [s for s in palette._sections if isinstance(s, mp._CategorySection)]
    assert cats and all(s.is_collapsed() for s in cats)
    palette.toggle_all_collapsed()
    assert not any(s.is_collapsed() for s in cats)


def test_enter_adds_first_match_and_clears_filter(palette):
    win = QMainWindow()
    from modern_ui.widgets.modern_canvas import ModernCanvas

    canvas = ModernCanvas(palette.dsim)
    win.setCentralWidget(canvas)
    win.block_palette = palette
    palette.block_drag_started.connect(
        lambda mb: canvas.add_block_from_palette(mb, QPoint(100, 100))
    )
    palette.show()
    before = len(palette.dsim.blocks_list)
    undo_before = len(canvas.history_manager.undo_stack)

    palette.search_bar.setText("transfer function")
    palette.search_bar.returnPressed.emit()

    assert len(palette.dsim.blocks_list) == before + 1
    assert palette.dsim.blocks_list[-1].fn_name == "transfer_function"
    assert len(canvas.history_manager.undo_stack) == undo_before + 1
    assert palette.search_bar.text() == ""
    canvas.deleteLater()
    win.deleteLater()


def test_enter_prefers_prefix_match_over_substring(palette):
    # Row order follows block discovery (filesystem order, platform-dependent);
    # "transfer" must still pick Transfer Function, not a block that merely
    # contains the word, e.g. Discrete Transfer Function.
    added = []
    palette.block_drag_started.connect(lambda mb: added.append(mb.fn_name))
    palette.show()
    palette.search_bar.setText("transfer")
    assert any(r.menu_block.fn_name == "discrete_transfer_function" for r in palette.visible_rows())
    palette.search_bar.returnPressed.emit()
    assert added == ["transfer_function"]


def test_enter_with_no_match_is_noop(palette):
    palette.show()
    before = len(palette.dsim.blocks_list)
    palette.search_bar.setText("zzzzzz-no-such-block")
    palette.search_bar.returnPressed.emit()
    assert len(palette.dsim.blocks_list) == before


def test_long_names_never_widen_the_palette(palette):
    # Regression: an unelidable long name (e.g. a Spanish block or category
    # name) set the content's minimum width above the panel's, so focusing a
    # row scrolled the whole list sideways behind the hidden scrollbar.
    from PyQt6.QtTest import QTest

    row = palette.findChildren(CompactBlockRow)[0]
    row.name_label.setText("An extremely long block name that cannot possibly fit " * 3)
    palette.resize(180, 600)
    palette.show()
    QTest.qWait(50)
    hbar = palette.scroll.horizontalScrollBar()
    assert hbar.maximum() == 0
    assert palette.scroll.widget().width() <= palette.scroll.viewport().width()
    hbar.setValue(40)
    assert hbar.value() == 0
    assert row.name_label.text().startswith("An extremely long")  # full text kept
    assert block_label_in_tooltip(row)


def block_label_in_tooltip(row):
    from modern_ui.widgets.modern_palette import block_label

    return row.toolTip().splitlines()[0] == block_label(row.menu_block)


def test_block_match_rank_tiers():
    import types

    from modern_ui.widgets.modern_palette import block_match_rank

    def mb(display, fn):
        return types.SimpleNamespace(
            display_name=display,
            fn_name=fn,
            block_fn=fn,
            block_class=object,
            search_text=lambda t, d=display, f=fn: (
                f"{d.lower()} {f.lower()} {d.lower().replace(' ', '')}"
            ),
        )

    tf = mb("Transfer Function", "transfer_function")
    dtf = mb("Discrete Transfer Function", "discrete_transfer_function")
    assert block_match_rank(tf, "transfer function") == 0
    assert block_match_rank(dtf, "transfer function") == 2
    assert block_match_rank(tf, "trans") == 1
    assert block_match_rank(tf, "gain") is None
    assert block_match_rank(tf, "") == 0
