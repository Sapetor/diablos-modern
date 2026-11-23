"""UI scale preference (File > Preferences > UI Scale).

The scale is applied as Qt's global ``QT_SCALE_FACTOR``, which must be set
before the ``QApplication`` is built, so a change takes effect on the next
launch. Everything scales with it: fonts (including the fixed point sizes in
the ``TYPE`` tokens that the style sheets use), icons, pixel metrics and the
canvas. It multiplies the screen's own device-pixel ratio, so on a Windows
screen at 150% a 125% UI scale gives 187.5%.

Before 2026-11 the setting only enlarged the application font
(``10pt * factor``). Widgets whose style sheet sets a point size from the
``TYPE`` tokens (menu bar, palette, Properties panel, transport, toolbar)
ignored it, so 150% looked patchy. The value lived in
``config/default_config.json``, which in a source checkout is the bundled file
itself, so choosing 125% once had leaked into the shipped default.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

#: QSettings key holding the scale factor.
SETTINGS_KEY = "ui/scale"
#: The factors offered in Preferences.
CHOICES = (1.0, 1.25, 1.5)
#: Stored values outside this range are ignored (a corrupt or hand-edited store
#: must not make the window unusable).
_MIN, _MAX = 0.5, 3.0


def _valid(value):
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return None
    return factor if _MIN <= factor <= _MAX else None


def _legacy_config_scale():
    """``display.scaling_factor`` from the per-user config of a packaged build.

    Only a frozen build has a user copy separate from the bundled file; in a
    source checkout the two paths are the same file, whose value is a packaged
    default rather than the user's choice.
    """
    from diablos.core.app_paths import resource_path, user_data_path

    user_cfg = user_data_path("config/default_config.json")
    if os.path.abspath(user_cfg) == os.path.abspath(resource_path("config/default_config.json")):
        return None
    try:
        with open(user_cfg, "r", encoding="utf-8") as fh:
            return _valid(json.load(fh).get("display", {}).get("scaling_factor"))
    except (OSError, ValueError, AttributeError):
        return None


def stored_ui_scale() -> float:
    """The scale the next launch will use (1.0 when unset or invalid)."""
    try:
        from diablos.gui.settings import ui_settings

        factor = _valid(ui_settings().value(SETTINGS_KEY, None))
        if factor is not None:
            return factor
    except Exception:  # pragma: no cover - Qt unavailable
        logger.debug("QSettings unavailable when reading the UI scale")
    legacy = _legacy_config_scale()
    return legacy if legacy is not None else 1.0


def store_ui_scale(factor: float) -> bool:
    """Persist ``factor`` for the next launch. Returns False if it failed."""
    if _valid(factor) is None:
        raise ValueError(f"UI scale out of range: {factor!r}")
    try:
        from diablos.gui.settings import ui_settings

        settings = ui_settings()
        settings.setValue(SETTINGS_KEY, float(factor))
        settings.sync()
        return settings.status() == settings.Status.NoError
    except Exception:
        logger.exception("Could not store the UI scale")
        return False


def qt_scale_factor() -> float:
    """The global scale Qt runs with in this process (``QT_SCALE_FACTOR``)."""
    return _valid(os.environ.get("QT_SCALE_FACTOR")) or 1.0


def apply_ui_scale() -> float:
    """Set ``QT_SCALE_FACTOR`` from the stored scale. Call before QApplication.

    An explicit ``QT_SCALE_FACTOR`` in the environment wins. Returns the factor
    in effect (as far as this preference is concerned).
    """
    if os.environ.get("QT_SCALE_FACTOR"):
        logger.info(
            "QT_SCALE_FACTOR=%s set in the environment; UI scale preference ignored",
            os.environ["QT_SCALE_FACTOR"],
        )
        return qt_scale_factor()
    factor = stored_ui_scale()
    if factor != 1.0:
        os.environ["QT_SCALE_FACTOR"] = f"{factor:g}"
    return factor
