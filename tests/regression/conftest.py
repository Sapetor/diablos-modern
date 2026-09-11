"""Shared fixtures for the regression suite."""

import pytest


@pytest.fixture(scope="module")
def window(qapp):
    """One ModernDiaBloSWindow per test module.

    Building the main window dominates the runtime of any test that needs it,
    so the three local copies of this fixture were collapsed into one.
    """
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1280, 800)
    yield w
    w.close()
