"""
Theme Management System for Modern DiaBloS UI
Provides consistent color schemes and styling across the application.
"""

from enum import Enum
from typing import Dict, Any
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import QObject, pyqtSignal


class ThemeType(Enum):
    DARK = "dark"
    LIGHT = "light"


class ThemeManager(QObject):
    """Manages application themes and color schemes."""
    
    theme_changed = pyqtSignal(str)  # Emitted when theme changes
    
    def __init__(self):
        super().__init__()
        self.current_theme = ThemeType.DARK
        self.themes = {
            ThemeType.DARK: self._create_dark_theme(),
            ThemeType.LIGHT: self._create_light_theme()
        }
    
    def _create_dark_theme(self) -> Dict[str, Any]:
        """Create dark theme color scheme with refined, professional colors."""
        return {
            # Main background colors - Softer, less harsh
            'background_primary': '#1A1D23',      # Main window background (slight blue tint)
            'background_secondary': '#252A31',    # Panel backgrounds
            'background_tertiary': '#2F353D',     # Elevated elements

            # Surface colors
            'surface': '#252A31',                  # Cards, dialogs
            'surface_variant': '#2F353D',         # Alternate surface
            'surface_elevated': '#343A43',        # Highly elevated elements

            # Text colors - Better contrast
            'text_primary': '#E8E9ED',            # Primary text (softer white)
            'text_secondary': '#A0A4B0',          # Secondary text
            'text_disabled': '#5F6370',           # Disabled text
            'text_inverse': '#1A1D23',            # Inverse text (for light backgrounds)

            # Accent colors - Refined blue palette
            'accent_primary': '#4C9EFF',          # Primary accent (softer, more professional blue)
            'accent_secondary': '#3B82F6',        # Darker accent
            'accent_hover': '#60A5FF',            # Hover state
            'accent_pressed': '#2563EB',          # Pressed state

            # Status colors - Softer, more professional
            'success': '#3ECF8E',                 # Success (softer green)
            'success_bg': '#1C3D30',              # Success background
            'warning': '#F59E0B',                 # Warning (softer orange)
            'warning_bg': '#3D2F1C',              # Warning background
            'error': '#EF5A6F',                   # Error (softer red)
            'error_bg': '#3D1C24',                # Error background
            'info': '#4C9EFF',                    # Information
            'info_bg': '#1C2F3D',                 # Info background

            # Border colors - Subtle but visible
            'border_primary': '#383E47',          # Primary borders
            'border_secondary': '#484E57',        # Secondary borders
            'border_focus': '#4C9EFF',            # Focused elements
            'border_hover': '#5A6270',            # Hover state

            # Block colors - Sophisticated, muted tones with better contrast
            'block_default': '#2F353D',           # Default block color
            'block_default_border': '#484E57',    # Default border

            # Source blocks - Soft emerald/teal
            'block_source': '#2D4B4A',            # Source blocks
            'block_source_accent': '#3ECF8E',     # Source accent
            'block_source_border': '#3A5F5E',     # Source border

            # Math/Process blocks - Soft blue
            'block_process': '#2D3D55',           # Process blocks
            'block_process_accent': '#4C9EFF',    # Process accent
            'block_process_border': '#3B4D6B',    # Process border

            # Control blocks - Soft purple
            'block_control': '#3D2D55',           # Control blocks
            'block_control_accent': '#A78BFA',    # Control accent
            'block_control_border': '#4D3B6B',    # Control border

            # Sink blocks - Soft red/rose
            'block_sink': '#552D3A',              # Sink blocks
            'block_sink_accent': '#EF5A6F',       # Sink accent
            'block_sink_border': '#6B3B4D',       # Sink border

            # Other blocks - Soft amber
            'block_other': '#554A2D',             # Other blocks
            'block_other_accent': '#F59E0B',      # Other accent
            'block_other_border': '#6B5E3B',      # Other border

            # Selection and interaction
            'block_selected': '#4C9EFF',          # Selected blocks border
            'block_selected_bg': '#2D3D55',       # Selected blocks background
            'block_hover': '#343A43',             # Hover state
            'block_shadow': '#0F1216',            # Shadow color for depth

            # Connection colors - More subtle
            'connection_default': '#6B7280',      # Default connections
            'connection_active': '#4C9EFF',       # Active connections
            'connection_selected': '#60A5FF',     # Selected connections
            'connection_error': '#EF5A6F',        # Error connections
            'connection_preview': '#F59E0B',      # Connection preview
            'connection_shadow': '#0F1216',       # Connection shadow

            # Port colors
            'port_input': '#4C9EFF',              # Input ports
            'port_output': '#3ECF8E',             # Output ports
            'port_hover': '#60A5FF',              # Port hover

            # Grid and canvas
            'grid_dots': '#2F353D',               # Grid dot color (very subtle)
            'grid_lines': '#383E47',              # Grid lines (optional)
            'canvas_background': '#1A1D23',       # Canvas background
            'canvas_background_alt': '#1F2329',   # Alternate canvas (for patterns)
            'selection_rectangle': '#4C9EFF',     # Selection rectangle
            'selection_rectangle_fill': '#4C9EFF1A',  # Selection fill (with alpha)
        }
    
    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme color scheme with refined, professional colors."""
        return {
            # Main background colors - Clean, modern whites and grays
            'background_primary': '#FAFBFC',      # Main window background (very slight gray)
            'background_secondary': '#F3F4F6',    # Panel backgrounds
            'background_tertiary': '#E5E7EB',     # Elevated elements

            # Surface colors
            'surface': '#FFFFFF',                  # Cards, dialogs
            'surface_variant': '#F9FAFB',         # Alternate surface
            'surface_elevated': '#FFFFFF',        # Highly elevated elements

            # Text colors - Excellent contrast
            'text_primary': '#111827',            # Primary text (near black)
            'text_secondary': '#6B7280',          # Secondary text
            'text_disabled': '#9CA3AF',           # Disabled text
            'text_inverse': '#FFFFFF',            # Inverse text (for dark backgrounds)

            # Accent colors - Vibrant but professional blue
            'accent_primary': '#2563EB',          # Primary accent
            'accent_secondary': '#1D4ED8',        # Darker accent
            'accent_hover': '#3B82F6',            # Hover state
            'accent_pressed': '#1E40AF',          # Pressed state

            # Status colors - Clear and distinct
            'success': '#10B981',                 # Success (emerald)
            'success_bg': '#D1FAE5',              # Success background
            'warning': '#F59E0B',                 # Warning (amber)
            'warning_bg': '#FEF3C7',              # Warning background
            'error': '#EF4444',                   # Error (red)
            'error_bg': '#FEE2E2',                # Error background
            'info': '#2563EB',                    # Information
            'info_bg': '#DBEAFE',                 # Info background

            # Border colors - Subtle but clear
            'border_primary': '#E5E7EB',          # Primary borders
            'border_secondary': '#D1D5DB',        # Secondary borders
            'border_focus': '#2563EB',            # Focused elements
            'border_hover': '#9CA3AF',            # Hover state

            # Block colors - Soft, pastel tones with good contrast
            'block_default': '#F9FAFB',           # Default block color
            'block_default_border': '#D1D5DB',    # Default border

            # Source blocks - Soft emerald/teal
            'block_source': '#D1FAE5',            # Source blocks
            'block_source_accent': '#10B981',     # Source accent
            'block_source_border': '#6EE7B7',     # Source border

            # Math/Process blocks - Soft blue
            'block_process': '#DBEAFE',           # Process blocks
            'block_process_accent': '#2563EB',    # Process accent
            'block_process_border': '#93C5FD',    # Process border

            # Control blocks - Soft purple
            'block_control': '#E9D5FF',           # Control blocks
            'block_control_accent': '#9333EA',    # Control accent
            'block_control_border': '#C084FC',    # Control border

            # Sink blocks - Soft red/rose
            'block_sink': '#FEE2E2',              # Sink blocks
            'block_sink_accent': '#EF4444',       # Sink accent
            'block_sink_border': '#FCA5A5',       # Sink border

            # Other blocks - Soft amber
            'block_other': '#FEF3C7',             # Other blocks
            'block_other_accent': '#F59E0B',      # Other accent
            'block_other_border': '#FCD34D',      # Other border

            # Selection and interaction
            'block_selected': '#2563EB',          # Selected blocks border
            'block_selected_bg': '#EFF6FF',       # Selected blocks background
            'block_hover': '#F3F4F6',             # Hover state
            'block_shadow': '#00000015',          # Shadow color for depth (with alpha)

            # Connection colors - Clear and visible
            'connection_default': '#9CA3AF',      # Default connections
            'connection_active': '#2563EB',       # Active connections
            'connection_selected': '#3B82F6',     # Selected connections
            'connection_error': '#EF4444',        # Error connections
            'connection_preview': '#F59E0B',      # Connection preview
            'connection_shadow': '#00000010',     # Connection shadow

            # Port colors
            'port_input': '#2563EB',              # Input ports
            'port_output': '#10B981',             # Output ports
            'port_hover': '#3B82F6',              # Port hover

            # Grid and canvas
            'grid_dots': '#E5E7EB',               # Grid dot color (subtle)
            'grid_lines': '#D1D5DB',              # Grid lines (optional)
            'canvas_background': '#FAFBFC',       # Canvas background
            'canvas_background_alt': '#FFFFFF',   # Alternate canvas (for patterns)
            'selection_rectangle': '#2563EB',     # Selection rectangle
            'selection_rectangle_fill': '#2563EB1A',  # Selection fill (with alpha)
        }
    
    def get_current_theme(self) -> Dict[str, Any]:
        """Get the current theme colors."""
        return self.themes[self.current_theme]
    
    def get_color(self, color_name: str) -> QColor:
        """Get a QColor object for the specified color name."""
        theme = self.get_current_theme()
        color_hex = theme.get(color_name, '#000000')
        return QColor(color_hex)
    
    def set_theme(self, theme_type: ThemeType):
        """Change the current theme."""
        if theme_type != self.current_theme:
            self.current_theme = theme_type
            self.theme_changed.emit(theme_type.value)
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.current_theme == ThemeType.DARK:
            self.set_theme(ThemeType.LIGHT)
        else:
            self.set_theme(ThemeType.DARK)
    
    def get_qss_variables(self) -> Dict[str, str]:
        """Get theme colors as QSS variables for stylesheets."""
        theme = self.get_current_theme()
        qss_vars = {}
        for key, value in theme.items():
            qss_vars[f"@{key}"] = value
        return qss_vars


# Global theme manager instance
theme_manager = ThemeManager()