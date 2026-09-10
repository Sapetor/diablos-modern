"""Lightweight internationalization for DiaBloS.

Design
------
Translation is keyed by the **English source string**, so an untranslated (or
missing) entry simply falls back to the English text that is already in the
code.  Catalogs are flat UTF-8 JSON files in ``locales/<code>.json``::

    {
      "_meta": {"code": "es", "name": "Español", "english_name": "Spanish"},
      "File": "Archivo",
      "Loaded {name}": "Cargado {name}"
    }

Keys starting with ``_`` are metadata and never used as translations.

Usage::

    from lib.i18n import tr
    label.setText(tr("Simulation"))
    status.showMessage(tr("Loaded {name}", name=filename))

Never wrap identifiers (dict keys, theme names, ``objectName``s, settings keys,
block ``block_name``s, parameter keys).  Only user-visible display text.

Formatting must use ``{named}`` placeholders so the catalog key stays stable:
``tr("Loaded {name}", name=x)`` rather than an f-string.

This module deliberately has no Qt import at module scope; ``QLocale`` is
imported lazily inside :func:`system_language` so the core stays unit-testable
and usable from headless code.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Language used as both the fallback and the catalog key language.
DEFAULT_LANGUAGE = "en"

#: Sentinel stored in the config/QSettings meaning "follow the system locale".
SYSTEM_LANGUAGE = "system"

#: QSettings key holding the user's explicit language choice.
SETTINGS_KEY = "ui/language"

#: Config (``config/default_config.json``) dotted key for the same setting.
CONFIG_KEY = "ui.language"

# --- module state -----------------------------------------------------------
_current_language = DEFAULT_LANGUAGE
_current_catalog: Dict[str, str] = {}
_catalog_cache: Dict[str, Dict[str, str]] = {}
_locales_dir_override: Optional[str] = None
_listeners: List[Any] = []


# --- locations --------------------------------------------------------------
def locales_dir() -> str:
    """Return the directory holding ``<code>.json`` catalogs.

    Resolved through :mod:`lib.app_paths` so it works in a PyInstaller bundle
    (``locales`` is listed in ``diablos.spec`` ``datas``).
    """
    if _locales_dir_override is not None:
        return _locales_dir_override
    from lib.app_paths import locales_path

    return locales_path()


def set_locales_dir(path: Optional[str]) -> None:
    """Override the catalog directory (tests only). ``None`` restores default."""
    global _locales_dir_override
    _locales_dir_override = path
    clear_cache()


def clear_cache() -> None:
    """Drop cached catalogs and re-read the active one on next use."""
    global _current_catalog
    _catalog_cache.clear()
    _current_catalog = load_catalog(_current_language)


# --- catalog loading --------------------------------------------------------
def load_catalog(code: str) -> Dict[str, str]:
    """Load (and cache) the catalog for ``code``.

    Returns an empty mapping for English or for any code without a catalog --
    the fallback path, since keys *are* the English strings.
    """
    if not code or code == DEFAULT_LANGUAGE:
        return {}
    if code in _catalog_cache:
        return _catalog_cache[code]

    path = os.path.join(locales_dir(), code + ".json")
    catalog: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key.startswith("_"):
                    continue  # metadata (_meta, _comment, ...)
                if isinstance(value, str) and value:
                    catalog[key] = value
        else:
            logger.warning("Locale catalog %s is not a JSON object; ignoring", path)
    except FileNotFoundError:
        logger.debug("No locale catalog for %r at %s", code, path)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read locale catalog %s: %s", path, exc)

    _catalog_cache[code] = catalog
    return catalog


def catalog_meta(code: str) -> Dict[str, str]:
    """Return the ``_meta`` block of a catalog (``{}`` when absent)."""
    if not code or code == DEFAULT_LANGUAGE:
        return {"code": DEFAULT_LANGUAGE, "name": "English", "english_name": "English"}
    path = os.path.join(locales_dir(), code + ".json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        meta = raw.get("_meta") if isinstance(raw, dict) else None
        if isinstance(meta, dict):
            return {k: str(v) for k, v in meta.items()}
    except (OSError, ValueError):
        pass
    return {}


def available_languages() -> List[Dict[str, str]]:
    """List selectable languages: English plus every ``locales/*.json``.

    Each entry is ``{"code": ..., "name": <native name>, "english_name": ...}``,
    sorted by native name with English first.
    """
    languages = [{"code": DEFAULT_LANGUAGE, "name": "English", "english_name": "English"}]
    directory = locales_dir()
    try:
        names = sorted(f for f in os.listdir(directory) if f.endswith(".json"))
    except OSError:
        names = []
    for filename in names:
        code = os.path.splitext(filename)[0]
        if code == DEFAULT_LANGUAGE:
            continue
        meta = catalog_meta(code)
        languages.append(
            {
                "code": meta.get("code", code) or code,
                "name": meta.get("name", code),
                "english_name": meta.get("english_name", meta.get("name", code)),
            }
        )
    return languages


# --- language selection -----------------------------------------------------
def system_language() -> str:
    """Best-effort two-letter language code for the host system."""
    try:
        from PyQt6.QtCore import QLocale

        name = QLocale.system().name()  # e.g. "es_ES"
        if name:
            return name.split("_")[0].lower()
    except Exception:  # pragma: no cover - Qt missing or headless oddity
        logger.debug("QLocale unavailable; falling back to environment locale")
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var)
        if value and value not in ("C", "POSIX"):
            return value.split(".")[0].split("_")[0].lower()
    return DEFAULT_LANGUAGE


def resolve_language(setting: Optional[str]) -> str:
    """Resolve a stored setting to a concrete catalog code.

    Order: explicit user setting > system locale > ``"en"``.  A requested
    language with no catalog degrades to English rather than raising.
    """
    if setting and setting != SYSTEM_LANGUAGE:
        code = str(setting).split("_")[0].lower()
    else:
        code = system_language()
    if code == DEFAULT_LANGUAGE:
        return DEFAULT_LANGUAGE
    if os.path.exists(os.path.join(locales_dir(), code + ".json")):
        return code
    return DEFAULT_LANGUAGE


def set_language(code: Optional[str]) -> str:
    """Activate a language. Accepts a concrete code or ``"system"``.

    Returns the resolved code actually in effect.
    """
    global _current_language, _current_catalog
    resolved = resolve_language(code)
    _current_language = resolved
    _current_catalog = load_catalog(resolved)
    logger.info("UI language set to %r (requested %r)", resolved, code)
    _notify_listeners(resolved)
    return resolved


def current_language() -> str:
    """Return the active catalog code (``"en"`` when untranslated)."""
    return _current_language


def add_language_listener(callback) -> None:
    """Register ``callback(code)``, called after every :func:`set_language`."""
    if callback not in _listeners:
        _listeners.append(callback)


def remove_language_listener(callback) -> None:
    """Unregister a listener added with :func:`add_language_listener`."""
    if callback in _listeners:
        _listeners.remove(callback)


def _notify_listeners(code: str) -> None:
    for callback in list(_listeners):
        try:
            callback(code)
        except Exception:  # pragma: no cover - a bad listener must not break i18n
            logger.exception("Language listener failed")


# --- persisted setting ------------------------------------------------------
def stored_language_setting() -> str:
    """Read the user's language preference (``"system"`` when unset).

    QSettings (written by the Language menu) wins; the config file provides the
    packaged default.
    """
    try:
        from lib.app_paths import ui_settings

        value = ui_settings().value(SETTINGS_KEY, None)
        if value:
            return str(value)
    except Exception:  # pragma: no cover - Qt unavailable
        logger.debug("QSettings unavailable when reading language setting")
    try:
        value = _config_file_language()
        if value:
            return str(value)
    except Exception:  # pragma: no cover - config unreadable
        logger.debug("Config unavailable when reading language setting")
    return SYSTEM_LANGUAGE


def _config_file_language() -> Optional[str]:
    """Read ``ui.language`` from ``config/default_config.json``.

    The per-user copy (if any) wins over the bundled default. Kept minimal on
    purpose: there is no general config manager, so this is the only reader.
    """
    import json

    from lib.app_paths import resource_path, user_data_path

    for path in (
        user_data_path("config/default_config.json"),
        resource_path("config/default_config.json"),
    ):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        node = data
        for part in CONFIG_KEY.split("."):
            if not isinstance(node, dict) or part not in node:
                node = None
                break
            node = node[part]
        if node:
            return str(node)
    return None


def store_language_setting(code: str) -> None:
    """Persist the language preference for the next launch."""
    try:
        from lib.app_paths import ui_settings

        ui_settings().setValue(SETTINGS_KEY, code)
    except Exception:  # pragma: no cover - Qt unavailable
        logger.warning("Could not persist language setting to QSettings")


def init_language() -> str:
    """Activate the persisted (or system) language. Call once at startup."""
    return set_language(stored_language_setting())


# --- translation ------------------------------------------------------------
def tr(text: str, **fmt: Any) -> str:
    """Translate ``text``, falling back to the English source string.

    ``fmt`` values are substituted with :meth:`str.format` *after* lookup, so
    the catalog key stays the unformatted English sentence::

        tr("Loaded {name}", name="demo.diablos")
    """
    if not isinstance(text, str) or not text:
        return text
    translated = _current_catalog.get(text, text)
    if fmt:
        try:
            return translated.format(**fmt)
        except (KeyError, IndexError, ValueError):
            # Broken placeholder in a translation: fall back to English.
            try:
                return text.format(**fmt)
            except (KeyError, IndexError, ValueError):
                return text
    return translated


def tr_noop(text: str) -> str:
    """Mark ``text`` for extraction without translating it here.

    The gettext ``N_()`` idiom. Use it when a literal is declared far from
    where it is displayed -- a table of menu labels, a badge map, a helper's
    argument -- so ``scripts/extract_strings.py`` still sees the literal while
    the actual lookup happens later at display time::

        _COMMANDS = [(tr_noop("New diagram"), "Ctrl+N"), ...]
        ...
        label.setText(tr(name))     # translated when shown

    Returns ``text`` unchanged, so the stored value keeps its English identity
    and remains usable as a lookup key.
    """
    return text


def has_translation(text: str) -> bool:
    """True when the active catalog contains an entry for ``text``."""
    return text in _current_catalog


# Activate English until the app calls init_language().
_current_catalog = {}
