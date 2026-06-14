"""
Tests for ModernDiaBloSWindow's one-time first-run welcome.

The welcome is a brief, NON-modal pointer (palette / File > Examples / F1) shown
through ``self.toast`` on the very first launch, then suppressed forever via the
``ui/first_run_done`` QSettings flag. These tests pin down that it:

  * shows exactly once and then no-ops on subsequent calls (idempotent),
  * is driven strictly by QSettings (a pre-set flag suppresses it),
  * never blocks (it goes through the toast, never ``exec_()``),
  * cannot crash startup when the toast is missing.

QSettings is redirected at the *symbol* in main_window to a throwaway INI file
in ``tmp_path`` so the test never touches the real user config and stays
deterministic across runs.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_first_run.py -p no:cacheprovider \
        -o addopts=""
"""

import pytest
from PyQt5.QtCore import QSettings

import modern_ui.main_window as main_window
from modern_ui.main_window import FIRST_RUN_WELCOME_MESSAGE


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Redirect ``main_window.QSettings`` to a throwaway INI file.

    Every ``QSettings("DiaBloS", "DiaBloS")`` constructed inside the module is
    rerouted to ``tmp_path/diablos.ini`` so the real user config is untouched and
    each test starts from a clean (flag-absent) slate.
    """
    ini_path = str(tmp_path / "diablos.ini")
    monkeypatch.setattr(
        main_window, "QSettings",
        lambda *a, **k: QSettings(ini_path, QSettings.IniFormat),
    )
    return ini_path


class _ToastSpy:
    """Records show_message calls so we can assert without a real fade/animation."""

    def __init__(self):
        self.calls = []

    def show_message(self, message, duration=2000, is_error=False):
        self.calls.append({"message": message, "duration": duration,
                           "is_error": is_error})


class TestFirstRunWelcome:
    def test_message_constant_mentions_the_three_pointers(self):
        msg = FIRST_RUN_WELCOME_MESSAGE.lower()
        assert "palette" in msg
        assert "examples" in msg
        assert "f1" in msg

    def test_shows_once_then_noops(self, window, isolated_settings, monkeypatch):
        spy = _ToastSpy()
        monkeypatch.setattr(window, "toast", spy)

        # First call: flag absent -> welcome shows once.
        window._maybe_show_first_run_welcome()
        assert len(spy.calls) == 1
        assert spy.calls[0]["message"] == FIRST_RUN_WELCOME_MESSAGE
        assert spy.calls[0]["is_error"] is False

        # Second call: flag now set -> no-op (no extra toast).
        window._maybe_show_first_run_welcome()
        assert len(spy.calls) == 1

    def test_idempotent_across_many_calls(self, window, isolated_settings, monkeypatch):
        spy = _ToastSpy()
        monkeypatch.setattr(window, "toast", spy)
        for _ in range(5):
            window._maybe_show_first_run_welcome()
        assert len(spy.calls) == 1

    def test_sets_first_run_done_flag(self, window, isolated_settings, monkeypatch):
        monkeypatch.setattr(window, "toast", _ToastSpy())
        assert QSettings(isolated_settings, QSettings.IniFormat).value(
            "ui/first_run_done", False, type=bool) is False
        window._maybe_show_first_run_welcome()
        assert QSettings(isolated_settings, QSettings.IniFormat).value(
            "ui/first_run_done", False, type=bool) is True

    def test_preexisting_flag_suppresses_welcome(self, window, isolated_settings, monkeypatch):
        # Pre-mark first run done -> the welcome must not show at all.
        QSettings(isolated_settings, QSettings.IniFormat).setValue(
            "ui/first_run_done", True)
        spy = _ToastSpy()
        monkeypatch.setattr(window, "toast", spy)
        window._maybe_show_first_run_welcome()
        assert spy.calls == []

    def test_missing_toast_does_not_crash(self, window, isolated_settings, monkeypatch):
        # A failed/absent toast must never break startup; the flag still flips so
        # the welcome does not retry on every launch.
        monkeypatch.delattr(window, "toast", raising=False)
        window._maybe_show_first_run_welcome()  # must not raise
        assert QSettings(isolated_settings, QSettings.IniFormat).value(
            "ui/first_run_done", False, type=bool) is True

    def test_never_calls_exec(self, window, isolated_settings, monkeypatch):
        # Guard against accidental modality: the welcome path must not invoke any
        # blocking exec_(). A spy toast has no exec_, so simply reaching here
        # without a QDialog is the assertion; we also confirm a non-error toast.
        spy = _ToastSpy()
        monkeypatch.setattr(window, "toast", spy)
        window._maybe_show_first_run_welcome()
        assert all(c["is_error"] is False for c in spy.calls)
