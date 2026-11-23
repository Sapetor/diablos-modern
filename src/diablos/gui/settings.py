"""The application's preference store (QSettings).

UI preferences (window layout, palette state, language, UI scale, "ask
before every run") live in one QSettings store. It is GUI state: the engine
(``diablos.core``) never reads it. Until 2026-11 this lived in
``diablos.core.app_paths``, which still resolves these names lazily.
"""

import os

# KEEP these values -- they name the existing on-disk store; changing them
# would orphan already-written settings.
SETTINGS_ORG = "DiaBloS"
SETTINGS_APP = "DiaBloS"


def ui_settings():
    """Return the shared ``QSettings`` store for UI preferences.

    ``DIABLOS_SETTINGS_INI`` (a file path) swaps the native store -- the
    registry on Windows -- for a private INI file. ``tests/conftest.py`` sets it
    so the suite neither reads nor overwrites the developer's real preferences.
    """
    from PyQt6.QtCore import QSettings

    ini = os.environ.get("DIABLOS_SETTINGS_INI")
    if ini:
        return QSettings(ini, QSettings.Format.IniFormat)
    return QSettings(SETTINGS_ORG, SETTINGS_APP)
