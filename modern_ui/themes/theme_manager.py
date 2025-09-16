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
        """Create dark theme color scheme."""
        return {
            # Main background colors
            'background_primary': '#2B2B2B',      # Main window background
            'background_secondary': '#3C3C3C',    # Panel backgrounds
            'background_tertiary': '#4A4A4A',     # Elevated elements
            
            # Surface colors
            'surface': '#404040',                  # Cards, dialogs
            'surface_variant': '#4A4A4A',         # Alternate surface
            
            # Text colors
            'text_primary': '#FFFFFF',            # Primary text
            'text_secondary': '#B0B0B0',          # Secondary text
            'text_disabled': '#707070',           # Disabled text
            
            # Accent colors
            'accent_primary': '#0078D4',          # Primary accent (Microsoft Blue)
            'accent_secondary': '#106EBE',        # Darker accent
            'accent_hover': '#429CE3',            # Hover state
            
            # Status colors
            'success': '#107C10',                 # Success/connected
            'warning': '#FF8C00',                 # Warning
            'error': '#D13438',                   # Error/disconnected
            'info': '#0078D4',                    # Information
            
            # Border colors
            'border_primary': '#5A5A5A',          # Primary borders
            'border_secondary': '#707070',        # Secondary borders
            'border_focus': '#0078D4',            # Focused elements
            
            # Block colors
            'block_default': '#4A4A4A',           # Default block color
            'block_source': '#2D5A27',            # Source blocks (green tint)
            'block_process': '#1E3A5F',           # Process blocks (blue tint)
            'block_sink': '#5A2D2D',              # Sink blocks (red tint)
            'block_selected': '#0078D4',          # Selected blocks
            
            # Connection colors
            'connection_default': '#808080',      # Default connections
            'connection_active': '#0078D4',       # Active connections
            'connection_error': '#D13438',        # Error connections
            'connection_preview': '#FFD700',      # Connection preview
            
            # Grid and canvas
            'grid_dots': '#505050',               # Grid dot color
            'canvas_background': '#2B2B2B',       # Canvas background
            'selection_rectangle': '#0078D4',     # Selection rectangle
        }
    
    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme color scheme."""
        return {
            # Main background colors
            'background_primary': '#FFFFFF',      # Main window background
            'background_secondary': '#F5F5F5',    # Panel backgrounds
            'background_tertiary': '#E8E8E8',     # Elevated elements
            
            # Surface colors
            'surface': '#FFFFFF',                  # Cards, dialogs
            'surface_variant': '#F5F5F5',         # Alternate surface
            
            # Text colors
            'text_primary': '#1F1F1F',            # Primary text
            'text_secondary': '#616161',          # Secondary text
            'text_disabled': '#9E9E9E',           # Disabled text
            
            # Accent colors
            'accent_primary': '#0078D4',          # Primary accent
            'accent_secondary': '#106EBE',        # Darker accent
            'accent_hover': '#429CE3',            # Hover state
            
            # Status colors
            'success': '#107C10',                 # Success/connected
            'warning': '#FF8C00',                 # Warning
            'error': '#D13438',                   # Error/disconnected
            'info': '#0078D4',                    # Information
            
            # Border colors
            'border_primary': '#D1D1D1',          # Primary borders
            'border_secondary': '#E0E0E0',        # Secondary borders
            'border_focus': '#0078D4',            # Focused elements
            
            # Block colors
            'block_default': '#F0F0F0',           # Default block color
            'block_source': '#E8F5E8',            # Source blocks (light green)
            'block_process': '#E8F0FF',           # Process blocks (light blue)
            'block_sink': '#FFE8E8',              # Sink blocks (light red)
            'block_selected': '#CCE7FF',          # Selected blocks
            
            # Connection colors
            'connection_default': '#666666',      # Default connections
            'connection_active': '#0078D4',       # Active connections
            'connection_error': '#D13438',        # Error connections
            'connection_preview': '#FFD700',      # Connection preview
            
            # Grid and canvas
            'grid_dots': '#D0D0D0',               # Grid dot color
            'canvas_background': '#FFFFFF',       # Canvas background
            'selection_rectangle': '#0078D4',     # Selection rectangle
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