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
            'background_primary': '#1E1E1E',      # Main window background
            'background_secondary': '#2D2D2D',    # Panel backgrounds
            'background_tertiary': '#3C3C3C',     # Elevated elements
            
            # Surface colors
            'surface': '#2D2D2D',                  # Cards, dialogs
            'surface_variant': '#3C3C3C',         # Alternate surface
            
            # Text colors
            'text_primary': '#EAEAEA',            # Primary text
            'text_secondary': '#A0A0A0',          # Secondary text
            'text_disabled': '#6E6E6E',           # Disabled text
            
            # Accent colors
            'accent_primary': '#0A84FF',          # Primary accent (Vibrant Blue)
            'accent_secondary': '#0059B2',        # Darker accent
            'accent_hover': '#3B9BFF',            # Hover state
            
            # Status colors
            'success': '#34C759',                 # Success/connected
            'warning': '#FF9500',                 # Warning
            'error': '#FF3B30',                   # Error/disconnected
            'info': '#0A84FF',                    # Information
            
            # Border colors
            'border_primary': '#424242',          # Primary borders
            'border_secondary': '#525252',        # Secondary borders
            'border_focus': '#0A84FF',            # Focused elements
            
            # Block colors
            'block_default': '#3C3C3C',           # Default block color
            'block_source': '#2D4A3A',            # Source blocks (green tint)
            'block_process': '#2D3B55',           # Process blocks (blue tint)
            'block_sink': '#552D3A',              # Sink blocks (red tint)
            'block_selected': '#0A84FF',          # Selected blocks
            
            # Connection colors
            'connection_default': '#7A7A7A',      # Default connections
            'connection_active': '#0A84FF',       # Active connections
            'connection_error': '#FF3B30',        # Error connections
            'connection_preview': '#FFCC00',      # Connection preview
            
            # Grid and canvas
            'grid_dots': '#424242',               # Grid dot color
            'canvas_background': '#1E1E1E',       # Canvas background
            'selection_rectangle': '#0A84FF',     # Selection rectangle
        }
    
    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme color scheme."""
        return {
            # Main background colors
            'background_primary': '#FFFFFF',      # Main window background
            'background_secondary': '#F2F2F7',    # Panel backgrounds
            'background_tertiary': '#E5E5EA',     # Elevated elements
            
            # Surface colors
            'surface': '#FFFFFF',                  # Cards, dialogs
            'surface_variant': '#F2F2F7',         # Alternate surface
            
            # Text colors
            'text_primary': '#1C1C1E',            # Primary text
            'text_secondary': '#636366',          # Secondary text
            'text_disabled': '#AEAEB2',           # Disabled text
            
            # Accent colors
            'accent_primary': '#0A84FF',          # Primary accent
            'accent_secondary': '#0059B2',        # Darker accent
            'accent_hover': '#3B9BFF',            # Hover state
            
            # Status colors
            'success': '#34C759',                 # Success/connected
            'warning': '#FF9500',                 # Warning
            'error': '#FF3B30',                   # Error/disconnected
            'info': '#0A84FF',                    # Information
            
            # Border colors
            'border_primary': '#E1E1E1',          # Primary borders
            'border_secondary': '#D1D1D6',        # Secondary borders
            'border_focus': '#0A84FF',            # Focused elements
            
            # Block colors
            'block_default': '#F2F2F7',           # Default block color
            'block_source': '#EBF9EF',            # Source blocks (light green)
            'block_process': '#EBF5FF',           # Process blocks (light blue)
            'block_sink': '#FFF0F0',              # Sink blocks (light red)
            'block_selected': '#CCE7FF',          # Selected blocks
            
            # Connection colors
            'connection_default': '#8A8A8E',      # Default connections
            'connection_active': '#0A84FF',       # Active connections
            'connection_error': '#FF3B30',        # Error connections
            'connection_preview': '#FFCC00',      # Connection preview
            
            # Grid and canvas
            'grid_dots': '#D1D1D6',               # Grid dot color
            'canvas_background': '#FFFFFF',       # Canvas background
            'selection_rectangle': '#0A84FF',     # Selection rectangle
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