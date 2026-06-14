"""
Regression tests for ToastNotification.

Bug: three call sites passed ``is_error=True`` to ``show_message`` which did
not accept it, raising a TypeError inside ``except`` blocks and swallowing the
original error. ``show_message`` must accept ``is_error`` and style errors red.
"""

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def toast():
    from modern_ui.widgets.toast_notification import ToastNotification
    t = ToastNotification()
    yield t
    t.hide()


class TestToastIsError:
    def test_show_message_accepts_is_error(self, toast):
        # Must not raise (the original TypeError regression).
        toast.show_message("boom", duration=10, is_error=True)
        assert toast._is_error is True

    def test_default_is_not_error(self, toast):
        toast.show_message("ok", duration=10)
        assert toast._is_error is False

    def test_error_uses_error_border_color(self, toast):
        from modern_ui.themes.theme_manager import theme_manager
        toast.show_message("boom", duration=10, is_error=True)
        error_hex = theme_manager.get_color("error").name().lower()
        assert error_hex in toast.styleSheet().lower()

    def test_non_error_uses_accent_border_color(self, toast):
        from modern_ui.themes.theme_manager import theme_manager
        toast.show_message("hi", duration=10)
        accent_hex = theme_manager.get_color("accent_primary").name().lower()
        assert accent_hex in toast.styleSheet().lower()

    def test_toggle_back_to_non_error_restyles(self, toast):
        from modern_ui.themes.theme_manager import theme_manager
        toast.show_message("boom", duration=10, is_error=True)
        toast.show_message("recovered", duration=10, is_error=False)
        accent_hex = theme_manager.get_color("accent_primary").name().lower()
        assert accent_hex in toast.styleSheet().lower()
