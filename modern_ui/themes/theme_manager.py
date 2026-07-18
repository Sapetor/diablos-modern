"""Backward-compat shim.

The theme/design-token system moved to ``lib.theming.theme_manager`` so that
the core ``lib`` package no longer depends on ``modern_ui`` (dependency
inversion — see ``lib/theming/__init__.py``). This module re-exports every
public name from the new location so the ~70 existing
``from modern_ui.themes.theme_manager import ...`` sites keep working unchanged.

CRITICAL: the singleton ``theme_manager`` is imported from the new module, not
re-instantiated here — every consumer shares the exact same ThemeManager object.
"""

from lib.theming.theme_manager import *  # noqa: F401,F403  (re-export the public API)

# Explicit re-exports (belt-and-suspenders — covers names regardless of any
# ``__all__`` and keeps static analyzers/import checkers happy). The singleton
# MUST be the same object as in the new module, never a fresh instance.
from lib.theming.theme_manager import (  # noqa: F401
    theme_manager,
    ThemeManager,
    ThemeType,
    PALETTES,
    PALETTE_DISPLAY_NAMES,
    DEFAULT_PALETTE,
    ELEVATION,
    SPACE,
    RADIUS,
    WEIGHT,
    TYPE,
    get_ui_font,
    get_mono_font,
    make_shadow,
    pulse_alpha,
    MONO_FONT_STACK,
    UI_FONT_STACK,
    PULSE_INTERVAL_MS,
    PULSE_PHASE_STEP,
)
