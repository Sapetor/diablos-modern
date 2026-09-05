"""Regression tests for the unsaved-changes prompt on close.

BUG DESCRIPTION
---------------
``closeEvent`` called ``_cleanup_autosave()`` and ``event.accept()``
unconditionally, so closing a window with unsaved edits threw the work away
*and* deleted the autosave that could have recovered it -- with no prompt.

THE FIX
-------
When ``dsim.dirty`` is set, ``closeEvent`` asks Save / Discard / Cancel
(``_prompt_unsaved_changes``, split out so it can be stubbed here -- the suite
neutralizes ``QMessageBox.exec_``). Cancel ignores the event; the autosave is
only removed once the diagram is saved or the changes are discarded.
"""

import pytest
from PyQt5.QtGui import QCloseEvent


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False  # never leave the teardown close() blocked by the prompt
    w.close()


@pytest.fixture(autouse=True)
def _clean_state(window):
    window.dsim.dirty = False
    yield
    window.dsim.dirty = False


@pytest.fixture
def spies(window, monkeypatch):
    """Neutralize teardown side effects and record what closeEvent did."""
    calls = {"cleanup": 0, "save": 0, "stop": 0}
    monkeypatch.setattr(window, "_cleanup_autosave", lambda: calls.__setitem__("cleanup", 1))
    monkeypatch.setattr(window, "stop_simulation", lambda: calls.__setitem__("stop", 1))
    monkeypatch.setattr(window, "_cancel_experiment_workers", lambda: None)
    monkeypatch.setattr(window, "save_diagram", lambda: calls.__setitem__("save", 1))
    return calls


def _close(window, prompt_answer, monkeypatch):
    monkeypatch.setattr(window, "_prompt_unsaved_changes", lambda: prompt_answer)
    event = QCloseEvent()
    window.closeEvent(event)
    return event


@pytest.mark.qt
class TestCloseWithUnsavedChanges:
    def test_clean_diagram_closes_without_prompting(self, window, spies, monkeypatch):
        asked = {"n": 0}
        monkeypatch.setattr(
            window, "_prompt_unsaved_changes", lambda: asked.__setitem__("n", 1) or "cancel"
        )
        window.dsim.dirty = False
        event = QCloseEvent()
        window.closeEvent(event)

        assert asked["n"] == 0
        assert event.isAccepted()
        assert spies["cleanup"] == 1

    def test_cancel_keeps_the_window_and_the_autosave(self, window, spies, monkeypatch):
        window.dsim.dirty = True
        event = _close(window, "cancel", monkeypatch)

        assert not event.isAccepted()
        assert spies["cleanup"] == 0, "autosave must survive a cancelled close"
        assert spies["save"] == 0

    def test_discard_closes_and_removes_the_autosave(self, window, spies, monkeypatch):
        window.dsim.dirty = True
        event = _close(window, "discard", monkeypatch)

        assert event.isAccepted()
        assert spies["save"] == 0
        assert spies["cleanup"] == 1

    def test_save_that_succeeds_closes(self, window, spies, monkeypatch):
        window.dsim.dirty = True

        def _save():
            spies["save"] = 1
            window.dsim.dirty = False  # a completed save clears the flag

        monkeypatch.setattr(window, "save_diagram", _save)
        event = _close(window, "save", monkeypatch)

        assert spies["save"] == 1
        assert event.isAccepted()
        assert spies["cleanup"] == 1

    def test_save_that_is_cancelled_keeps_the_window(self, window, spies, monkeypatch):
        window.dsim.dirty = True  # save_diagram stub leaves it dirty (dialog cancelled)
        event = _close(window, "save", monkeypatch)

        assert spies["save"] == 1
        assert not event.isAccepted()
        assert spies["cleanup"] == 0
