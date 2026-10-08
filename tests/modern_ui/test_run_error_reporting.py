"""Run failures reach the user through the error panel + a toast, never a modal.

Drives a REAL window and DSim: a Gain with an unparsable parameter fails when
the run is initialised, once at the top level and once nested in a subsystem.
Also pins click-to-jump (select + centre, entering the subsystem), the canvas
highlight (reusing the validation badge) and its clearing on a good run.
"""

import pytest
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QMessageBox

from lib.error_locator import build_run_errors, find_block_chain, locate_blocks


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1000, 700)
    yield w
    w.close()


@pytest.fixture
def modals(monkeypatch):
    """Record every modal the code under test tries to open."""
    calls = []
    for meth in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(
            QMessageBox,
            meth,
            staticmethod(
                lambda *a, _m=meth, **k: calls.append(_m) or QMessageBox.StandardButton.Ok
            ),
        )
    monkeypatch.setattr(QMessageBox, "exec", lambda self, *a: calls.append("exec") or 0)
    return calls


def _build(window, nested):
    """step -> gain('abc') -> scope; the gain optionally wrapped in a subsystem."""
    canvas = window.canvas
    while len(window.dsim.get_current_path()) > 1:  # a previous test may sit in a subsystem
        window.dsim.exit_subsystem()
    canvas.clear_canvas()
    dsim = window.dsim
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    src = dsim.add_block(menu["step"], QPoint(100, 100))
    gain = dsim.add_block(menu["gain"], QPoint(250, 100))
    scope = dsim.add_block(menu["scope"], QPoint(400, 100))
    dsim.add_line((src.name, 0, src.out_coords[0]), (gain.name, 0, gain.in_coords[0]))
    dsim.add_line((gain.name, 0, gain.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    gain.params["gain"] = "abc"
    sub = None
    if nested:
        for b in dsim.blocks_list:
            b.selected = b is gain
        sub = dsim.create_subsystem_from_selection()
        assert sub is not None
    window.use_fast_solver = False
    return gain, sub


def _entries(window):
    return window.error_panel.errors


@pytest.mark.parametrize("nested", [False, True], ids=["top", "nested"])
def test_failed_run_uses_panel_and_toast_not_modal(window, modals, nested):
    _build(window, nested)
    shown = []
    window.toast.show_message = lambda text, **kw: shown.append((text, kw))

    window.start_simulation()

    assert modals == []
    errs = _entries(window)
    assert errs and "abc" in errs[0].message
    assert shown and "Simulation failed" in shown[-1][0]
    assert shown[-1][1]["is_error"] is True
    assert callable(shown[-1][1]["on_click"])


def test_error_names_the_block_even_without_the_name_in_the_text(window, modals):
    gain, _ = _build(window, nested=True)
    window.start_simulation()
    err = _entries(window)[0]
    assert err.block_name == "Subsystem1/gain0"
    assert err.blocks == [gain]


def test_click_selects_centres_and_enters_subsystem(window, modals):
    gain, sub = _build(window, nested=True)
    window.start_simulation()
    assert window.dsim.get_current_path() == ["Top Level"]

    window.error_panel.error_clicked.emit(_entries(window)[0])

    assert window.dsim.get_current_path()[-1] == sub.name
    assert gain.selected
    assert [b for b in window.dsim.blocks_list if b.selected] == [gain]
    state = window.canvas.zoom_pan_manager.state
    cx = (gain.left + gain.width / 2) * state.zoom_factor + state.pan_offset.x()
    cy = (gain.top + gain.height / 2) * state.zoom_factor + state.pan_offset.y()
    assert abs(cx - window.canvas.width() / 2) <= 1
    assert abs(cy - window.canvas.height() / 2) <= 1


def test_click_on_top_level_block_selects_it(window, modals):
    gain, _ = _build(window, nested=False)
    window.start_simulation()
    window.error_panel.error_clicked.emit(_entries(window)[0])
    assert gain.selected and window.dsim.get_current_path() == ["Top Level"]


def test_offender_and_its_subsystem_are_marked_then_cleared(window, modals):
    gain, sub = _build(window, nested=True)
    window.start_simulation()
    state = window.canvas.rendering_manager.validation_state
    assert state.show_errors
    assert gain in state.blocks_with_errors and sub in state.blocks_with_errors

    # Fix the parameter: the next run clears the marks and the panel.
    gain.params["gain"] = 2.0
    window.start_simulation()
    state = window.canvas.rendering_manager.validation_state
    assert not state.blocks_with_errors
    assert not _entries(window)


def test_validation_block_is_toast_not_modal(window, modals):
    canvas = window.canvas
    while len(window.dsim.get_current_path()) > 1:
        window.dsim.exit_subsystem()
    canvas.clear_canvas()
    menu = {b.fn_name: b for b in window.dsim.menu_blocks}
    window.dsim.add_block(menu["gain"], QPoint(100, 100))  # disconnected input
    shown = []
    window.toast.show_message = lambda text, **kw: shown.append((text, kw))

    window.start_simulation()

    assert modals == []
    assert _entries(window)
    assert shown and "Cannot start simulation" in shown[-1][0]


def test_midrun_block_error_is_reported_after_batch(window, modals):
    """A block failing mid-run ends the worker 'ok'; the controller must still fail."""
    gain, _ = _build(window, nested=False)
    ctl = window.canvas._sim_controller
    window.dsim.error_msg = f"{gain.name}: exploded"
    window.dsim.engine.error_block = gain.name
    shown = []
    window.toast.show_message = lambda text, **kw: shown.append(text)
    try:
        ctl._finish_batch(True, "")
    finally:
        window.dsim.error_msg = ""
    assert modals == []
    assert _entries(window)[0].blocks == [gain]
    assert shown and "Simulation failed" in shown[-1]


def test_toast_click_callback_runs():
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QWidget

    from modern_ui.widgets.toast_notification import ToastNotification

    parent = QWidget()
    toast = ToastNotification(parent)
    hits = []
    toast.show_message("x", on_click=lambda: hits.append(1))
    ev = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(2, 2),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    toast.mousePressEvent(ev)
    assert hits == [1]
    toast.mousePressEvent(ev)  # one-shot
    assert hits == [1]


class _B:
    def __init__(self, name, sub=None, username=""):
        self.name, self.username, self.sub_blocks = name, username, sub or []


class TestLocator:
    def setup_method(self):
        self.g = _B("gain0")
        self.sub = _B("Sub1", [self.g])
        self.top = _B("gain0")  # same local name at the top level
        self.root = [self.top, self.sub]

    def test_flat_name_resolves_to_nested_block(self):
        hits = locate_blocks(self.root, "Block Sub1/gain0 failed")
        assert [b for _n, b in hits] == [self.g]

    def test_whole_token_matching(self):
        assert locate_blocks([_B("gain1")], "gain10: bad") == []

    def test_hint_wins_and_chain(self):
        hits = locate_blocks(self.root, "no names here", hint="Sub1/gain0")
        assert hits[0][1] is self.g
        assert find_block_chain(self.root, self.g) == [self.sub]

    def test_one_entry_per_line(self):
        errs = build_run_errors(self.root, "Sub1/gain0: a\nother")
        assert len(errs) == 2 and errs[0].blocks == [self.g] and errs[1].blocks == []
