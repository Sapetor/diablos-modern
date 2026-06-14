"""
AppearanceManager -- owns theme/palette switching, UI-preference persistence,
and the theme-driven restyling of the main window's chrome (status bar, menu
bar, canvas area).

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

Preferences persist atomically to ``user_data_path('user_preferences.json')``,
merging with any keys already present (e.g. window geometry).
"""

import os
import json
import logging

from modern_ui.themes.theme_manager import theme_manager, ThemeType

logger = logging.getLogger(__name__)


class AppearanceManager:
    """Manages theme/palette state, persistence, and chrome restyling."""

    def __init__(self, main_window):
        self.window = main_window

    # -- user actions (mutate theme_manager + persist) ----------------------

    def toggle_theme(self):
        """Toggle theme and persist the choice."""
        theme_manager.toggle_theme()
        self.save_preferences()

    def set_palette(self, palette_key: str):
        """Switch the active block-color palette and persist the choice."""
        theme_manager.set_palette(palette_key)
        self.save_preferences()
        # Refresh canvas so blocks re-render with new palette colors
        if hasattr(self.window, 'canvas'):
            self.window.canvas.update()

    def toggle_solid_fills(self, checked: bool):
        """Toggle solid block fills and persist the choice."""
        theme_manager.set_solid_fills(checked)
        if hasattr(self.window, 'canvas'):
            self.window.canvas.update()
        self.save_preferences()

    # -- persistence --------------------------------------------------------

    def save_preferences(self):
        """Persist all UI preferences (theme, palette, solid_fills) to user_preferences.json."""
        from lib.app_paths import user_data_path
        path = user_data_path("user_preferences.json")
        prefs = {}
        try:
            with open(path, 'r') as f:
                prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prefs = {}
        prefs['theme'] = theme_manager.current_theme.value
        prefs['block_palette'] = theme_manager.current_palette
        prefs['solid_fills'] = theme_manager.solid_fills
        tmp = path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(prefs, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("Could not save user preferences: %s", e)
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass

    # -- theme-driven restyling --------------------------------------------

    def on_theme_changed(self):
        """Handle theme changes."""
        window = self.window
        # Update statusbar label colors (also sets the authoritative theme pill text)
        self.update_statusbar_colors()

        # Update menubar colors
        self.update_menubar_colors()

        # Update canvas area styling
        window.canvas_area.setStyleSheet(f"""
            #CanvasArea {{
                background-color: {theme_manager.get_color('canvas_background').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 6px;
            }}
        """)

    def update_statusbar_colors(self):
        """Apply theme to the status bar shell. Individual pills own their own styles."""
        window = self.window
        bg_color = theme_manager.get_color('statusbar_bg').name()
        border = theme_manager.get_color('border_primary').name()
        window.statusBar().setStyleSheet(
            f"QStatusBar {{ background-color: {bg_color}; border-top: 1px solid {border}; }}"
            f"QStatusBar::item {{ border: 0; }}"
        )
        # Refresh theme pill text
        if hasattr(window, 'theme_status'):
            theme_label = "Dark" if theme_manager.current_theme == ThemeType.DARK else "Light"
            from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES
            palette_label = PALETTE_DISPLAY_NAMES.get(
                theme_manager.current_palette, theme_manager.current_palette
            ).split()[0]
            window.theme_status.setText(f"{theme_label} · {palette_label}")
            window.theme_status.setStyleSheet(
                f"color: {theme_manager.get_color('text_secondary').name()};"
                f" background-color: {theme_manager.get_color('background_tertiary').name()};"
                f" padding: 1px 8px; border-radius: 4px; font-size: 9pt;"
            )

    def update_menubar_colors(self):
        """Re-apply the canonical menubar/menu styling on theme change.

        Single source of truth: ``ModernStyles.get_menubar_style()``. This used
        to build a *second*, divergent menubar stylesheet inline (different
        padding/radius, no role=danger items) and set it directly on the
        menuBar widget, which overrode the cascaded app stylesheet — so the menu
        visibly changed appearance after the first theme toggle. Delegating to
        the shared generator keeps the two in lockstep.
        """
        from modern_ui.styles.qss_styles import ModernStyles
        self.window.menuBar().setStyleSheet(ModernStyles.get_menubar_style())
