"""The UI language as the application chooses it: the user's saved choice
(View > Language, stored in QSettings) and the OS locale as Qt reports it.

The engine's ``diablos.core.i18n`` knows only the configured default and the
environment; importing this module gives it Qt's system-locale reader.
Order: saved choice > configured default > system locale > English.
"""

import logging

from diablos.core import i18n

logger = logging.getLogger(__name__)

#: QSettings key holding the user's explicit language choice.
SETTINGS_KEY = i18n.SETTINGS_KEY


def _qt_system_locale():
    from PyQt6.QtCore import QLocale

    return QLocale.system().name()  # e.g. "es_ES"


i18n.set_system_locale_reader(_qt_system_locale)


def stored_language_setting() -> str:
    """The saved language choice, else the configured default, else "system"."""
    try:
        from diablos.gui.settings import ui_settings

        value = ui_settings().value(SETTINGS_KEY, None)
        if value:
            return str(value)
    except Exception:  # pragma: no cover - Qt unavailable
        logger.debug("QSettings unavailable when reading language setting")
    return i18n.config_language_setting()


def store_language_setting(code: str) -> None:
    """Persist the language preference for the next launch."""
    try:
        from diablos.gui.settings import ui_settings

        ui_settings().setValue(SETTINGS_KEY, code)
    except Exception:  # pragma: no cover - Qt unavailable
        logger.warning("Could not persist language setting to QSettings")


def init_language() -> str:
    """Activate the saved (or system) language. Call once at GUI startup."""
    return i18n.set_language(stored_language_setting())
