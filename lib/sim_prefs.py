"""Simulation-related user preferences stored in ``QSettings``.

These are *application* preferences, not diagram data: they follow the user
across diagrams and are therefore kept out of the ``.diablos`` file (which
already stores the per-diagram solver settings -- sim_time, sim_dt, solver
method, tolerances, zero-crossing).

Only one preference lives here today:

``ask_before_run``
    When True, pressing Play re-opens the Simulation-settings dialog before
    every run (the pre-2026-09 behaviour).  The default is False: Play runs
    straight away with the settings stored in the diagram, and the dialog is
    reached on purpose through Simulation > Simulation Settings...

The store is the shared one from :func:`lib.app_paths.ui_settings`, imported at
module scope (not inside the functions) so tests can monkeypatch
``lib.sim_prefs.ui_settings`` the same way the block-palette tests do.
"""

import logging

from lib.app_paths import ui_settings

logger = logging.getLogger(__name__)

# QSettings key. An identifier -- never translated, never renamed (renaming it
# would orphan the value already written to the user's store).
ASK_BEFORE_RUN_KEY = "simulation/ask_before_run"


def _to_bool(value, default: bool) -> bool:
    """Coerce a ``QSettings`` value to a bool.

    QSettings round-trips booleans as the strings "true"/"false" on the INI
    backend, so ``bool(value)`` alone would read a stored False back as True.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def ask_before_run(default: bool = False) -> bool:
    """True when Play should re-open the Simulation-settings dialog first."""
    try:
        return _to_bool(ui_settings().value(ASK_BEFORE_RUN_KEY, None), default)
    except Exception:
        # No Qt available (headless CLI, frozen bootstrap): fall back to the
        # default rather than taking the whole run down with us.
        logger.debug("Could not read %s from QSettings", ASK_BEFORE_RUN_KEY, exc_info=True)
        return default


def set_ask_before_run(value: bool) -> None:
    """Persist the "ask before every run" preference."""
    try:
        ui_settings().setValue(ASK_BEFORE_RUN_KEY, bool(value))
    except Exception:
        logger.debug("Could not write %s to QSettings", ASK_BEFORE_RUN_KEY, exc_info=True)
