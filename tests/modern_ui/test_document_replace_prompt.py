"""New and Open ask about unsaved changes before replacing the diagram.

Only closing the window used to ask: File > New cleared a dirty diagram (and
its undo history) and kept the old file name, and every Open route (dialog,
example, recent file) loaded over it. All of them now go through
``_settle_unsaved_changes``: Cancel, a cancelled Save dialog or a failed save
keep the graph, dirty flag, file identity and history, and leave the block
registry alone; Discard or a completed Save let the replacement happen.
"""

import os
import shutil

import pytest

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "library_block_demo.diablos"
)


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False
    w.close()


@pytest.fixture
def doc(window, tmp_path, monkeypatch):
    """The example opened from ``tmp_path`` and then edited (gain 800 -> 123, dirty)."""
    path = str(tmp_path / "doc.diablos")
    shutil.copy(EXAMPLE, path)
    monkeypatch.setattr(window, "_warn_about_missing_block_types", lambda: None)
    # Opening records a recent file; keep the real config/recent_files.json out of it.
    monkeypatch.setattr(window.recent_files_manager, "add", lambda *_: None)
    window.dsim.dirty = False
    assert window.open_diagram(path) is not None
    window.canvas.history_manager.push_undo("edit gain")
    _gain(window).params["gain"] = 123.0
    window.dsim.dirty = True
    yield path
    window.dsim.dirty = False


def _gain(window):
    return next(b for b in window.dsim.blocks_list if b.name == "gain0")


def _service(window):
    return window.project_manager.diagram_service


ROUTES = {
    "new": lambda w, p: w.new_diagram(),
    "open": lambda w, p: w.open_diagram(p),
    "open_dialog": lambda w, p: w.open_diagram(),
    "example": lambda w, p: w.open_example(p),
    "recent": lambda w, p: w.recent_files_manager.open(p),
}


def _arm(window, monkeypatch, answer, save_ok=False, path=None):
    """Stub the prompt, Save and the Open dialog; return the call log."""
    log = {"asked": 0, "saved": 0, "reloaded": 0}

    def prompt(**_):
        log["asked"] += 1
        return answer

    def save():
        log["saved"] += 1
        if save_ok:
            window.dsim.dirty = False

    real_reload = window.reload_user_blocks

    def reload(*a, **kw):
        log["reloaded"] += 1
        return real_reload(*a, **kw)

    monkeypatch.setattr(window, "_prompt_unsaved_changes", prompt)
    monkeypatch.setattr(window, "save_diagram", save)
    monkeypatch.setattr(window, "reload_user_blocks", reload)
    monkeypatch.setattr(_service(window), "choose_open_file", lambda: path)
    return log


@pytest.mark.qt
class TestReplacingADirtyDiagram:
    @pytest.mark.parametrize("route", sorted(ROUTES))
    @pytest.mark.parametrize("answer, save_ok", [("cancel", False), ("save", False)])
    def test_cancel_or_failed_save_keeps_everything(
        self, window, doc, monkeypatch, route, answer, save_ok
    ):
        history = len(window.canvas.history_manager.undo_stack)
        current = _service(window).current_file
        log = _arm(window, monkeypatch, answer, save_ok, path=doc)

        ROUTES[route](window, doc)

        assert log["asked"] == 1
        assert log["saved"] == (1 if answer == "save" else 0)
        assert log["reloaded"] == 0, "the block registry must not change"
        assert _gain(window).params["gain"] == 123.0
        assert window.dsim.dirty is True
        assert _service(window).current_file == current
        assert len(window.canvas.history_manager.undo_stack) == history

    @pytest.mark.parametrize("route", sorted(ROUTES))
    @pytest.mark.parametrize("answer, save_ok", [("discard", False), ("save", True)])
    def test_discard_or_completed_save_replaces(
        self, window, doc, monkeypatch, route, answer, save_ok
    ):
        log = _arm(window, monkeypatch, answer, save_ok, path=doc)

        ROUTES[route](window, doc)

        assert log["asked"] == 1
        assert window.dsim.dirty is False
        if route == "new":
            assert window.dsim.blocks_list == []
            assert _service(window).current_file is None
        else:
            assert _gain(window).params["gain"] == 800.0  # the saved value

    @pytest.mark.parametrize("route", sorted(ROUTES))
    def test_clean_diagram_is_not_prompted(self, window, doc, monkeypatch, route):
        window.dsim.dirty = False
        log = _arm(window, monkeypatch, "cancel", path=doc)
        ROUTES[route](window, doc)
        assert log["asked"] == 0

    def test_new_forgets_the_previous_file(self, window, doc, monkeypatch):
        _arm(window, monkeypatch, "discard")
        window.new_diagram()
        assert _service(window).current_file is None
        assert window.canvas.history_manager.undo_stack == type(
            window.canvas.history_manager.undo_stack
        )(maxlen=window.canvas.history_manager.undo_stack.maxlen)
