"""Tests for the consolidated UI-preferences QSettings store.

The org/app pair for UI preferences used to be duplicated — ``main_window.py``
inlined ``QSettings("DiaBloS", "DiaBloS")`` (first-run flag) while
``modern_palette.py`` defined its own ``_SETTINGS_ORG``/``_SETTINGS_APP``
(per-category collapsed flags). Both now route through a single source of
truth in ``lib.app_paths``: the ``SETTINGS_ORG``/``SETTINGS_APP`` constants and
the ``ui_settings()`` accessor.

These tests pin down that every call site resolves to the *same* store, and
that the values are kept at the historical ``"DiaBloS"``/``"DiaBloS"`` pair
(changing them would orphan already-written settings).

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_ui_settings_consolidation.py -p no:cacheprovider \
        -o addopts=""
"""

import pytest

from lib.app_paths import SETTINGS_ORG, SETTINGS_APP, ui_settings
import modern_ui.main_window as main_window
import modern_ui.widgets.modern_palette as mp


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


class TestSettingsConstants:
    def test_values_kept_at_historical_pair(self):
        # KEEP these values — they name the existing on-disk store. Changing
        # them would orphan settings written by earlier versions.
        assert SETTINGS_ORG == "DiaBloS"
        assert SETTINGS_APP == "DiaBloS"

    def test_main_window_uses_shared_constants(self):
        # The first-run path constructs QSettings(SETTINGS_ORG, SETTINGS_APP)
        # from the shared module, not its own inlined literals.
        assert main_window.SETTINGS_ORG is SETTINGS_ORG
        assert main_window.SETTINGS_APP is SETTINGS_APP

    def test_palette_uses_shared_constants(self):
        # The palette's module-level names are re-exported from the shared
        # constants (aliased), so its collapsed-flag store matches.
        assert mp._SETTINGS_ORG is SETTINGS_ORG
        assert mp._SETTINGS_APP is SETTINGS_APP


class TestUiSettingsAccessor:
    def test_returns_qsettings(self):
        from PyQt5.QtCore import QSettings
        assert isinstance(ui_settings(), QSettings)

    def test_resolves_to_shared_org_app(self):
        s = ui_settings()
        assert s.organizationName() == SETTINGS_ORG
        assert s.applicationName() == SETTINGS_APP

    def test_all_call_sites_agree_on_org_app(self):
        """``ui_settings()`` and both call sites resolve to one org/app pair."""
        from PyQt5.QtCore import QSettings

        helper = ui_settings()
        # How main_window constructs the first-run store.
        mw_store = QSettings(main_window.SETTINGS_ORG, main_window.SETTINGS_APP)
        # How the palette constructs the collapsed-flag store.
        pal_store = QSettings(mp._SETTINGS_ORG, mp._SETTINGS_APP)

        pairs = {
            (helper.organizationName(), helper.applicationName()),
            (mw_store.organizationName(), mw_store.applicationName()),
            (pal_store.organizationName(), pal_store.applicationName()),
        }
        # One source of truth -> exactly one distinct (org, app) pair.
        assert pairs == {(SETTINGS_ORG, SETTINGS_APP)}

    def test_helper_does_not_import_qt_at_module_top(self):
        """``lib.app_paths`` stays Qt-free at module scope (lazy import)."""
        import lib.app_paths as ap
        # The QSettings import lives inside ui_settings(), not at module top.
        assert not hasattr(ap, "QSettings")
