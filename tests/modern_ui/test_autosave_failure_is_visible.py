"""A failed autosave must reach the user, not just the log.

Packaged builds wrote their crash-recovery snapshot to a read-only location and
failed with ``[Errno 30] Read-only file system``. ``FileService.save_to_file``
logged the error and returned False, ``_auto_save`` ignored the return value,
and the app carried on looking healthy while crash recovery was dead --
discovered only from ``~/Library/Logs/DiaBloS``.

The message has to be non-modal: the autosave timer fires every two minutes, so
a dialog would interrupt editing on a loop.
"""

import pytest
from PyQt6.QtCore import QPoint

from lib.i18n import tr

AUTOSAVE_FAILED = "Auto-save failed — could not write the recovery file"


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    # A 2-minute timer firing mid-test would re-enter _auto_save.
    w.autosave_timer.stop()
    yield w
    w.close()


def _add_a_block(window):
    menu = {b.fn_name: b for b in window.dsim.menu_blocks}
    window.dsim.add_block(menu["step"], QPoint(100, 100))


@pytest.mark.qt
class TestAutosaveFailureIsVisible:
    def test_failed_autosave_shows_a_status_message(self, window, monkeypatch):
        _add_a_block(window)

        def _fail(*args, **kwargs):
            window.dsim.file_service.last_write_error = (
                "/read/only/.autosave.diablos",
                OSError(30, "Read-only file system"),
            )
            return 1

        monkeypatch.setattr(window.dsim.file_service, "save", _fail)
        window.status_message.setText("idle")

        window._auto_save()

        assert window.status_message.text() == tr(AUTOSAVE_FAILED), (
            "a failed autosave must say so in the status bar"
        )

    def test_failed_autosave_opens_no_dialog(self, window, monkeypatch):
        """Non-modal: the 2-minute timer must never pop a message box."""
        from PyQt6.QtWidgets import QMessageBox

        _add_a_block(window)
        monkeypatch.setattr(window.dsim.file_service, "save", lambda *a, **k: 1)

        shown = []
        for name in ("critical", "warning", "information"):
            monkeypatch.setattr(
                QMessageBox, name, staticmethod(lambda *a, **k: shown.append(1)), raising=False
            )

        window._auto_save()

        assert not shown, "autosave failures must stay in the status bar"

    def test_successful_autosave_leaves_the_status_alone(self, window, tmp_path):
        _add_a_block(window)
        window.autosave_path = str(tmp_path / "config" / ".autosave.diablos")
        window.status_message.setText("idle")

        window._auto_save()

        assert window.status_message.text() == "idle"
        assert window.dsim.file_service.last_write_error is None
        assert (tmp_path / "config" / ".autosave.diablos").exists()

    def test_an_autosave_exception_also_reaches_the_status_bar(self, window, monkeypatch):
        _add_a_block(window)

        def _boom(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(window.dsim.file_service, "save", _boom)
        window.status_message.setText("idle")

        window._auto_save()

        assert window.status_message.text() == tr(AUTOSAVE_FAILED)
