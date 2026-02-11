"""
Theme Management System for Modern DiaBloS UI
Provides consistent color schemes and styling across the application.
"""

from enum import Enum
from typing import Dict, Any
from PyQt5.QtGui import QColor
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
        """Create dark theme color scheme with refined, professional colors and better contrast."""
        return {
            # Main background colors - Darker for better contrast
            'background_primary': '#0F1419',      # Main window background (darker)
            'background_secondary': '#1A1F26',    # Panel backgrounds
            'background_tertiary': '#252A32',     # Elevated elements

            # Surface colors
            'surface': '#1A1F26',                  # Cards, dialogs
            'surface_primary': '#1A1F26',          # Primary surface
            'surface_secondary': '#252A32',        # Secondary surface
            'surface_variant': '#252A32',         # Alternate surface
            'surface_elevated': '#2F353D',        # Highly elevated elements

            # Text colors - Excellent contrast
            'text_primary': '#F0F1F4',            # Primary text (brighter white)
            'text_secondary': '#B4B8C0',          # Secondary text (lighter)
            'text_disabled': '#6B7280',           # Disabled text
            'text_inverse': '#0F1419',            # Inverse text (for light backgrounds)

            # Accent colors - Vibrant and clear
            'accent_primary': '#60A5FA',          # Primary accent (brighter blue)
            'accent_secondary': '#3B82F6',        # Darker accent
            'accent_hover': '#7CB8FF',            # Hover state
            'accent_pressed': '#2563EB',          # Pressed state

            # Status colors - Vibrant and clear
            'success': '#10B981',                 # Success (emerald)
            'success_bg': '#064E3B',              # Success background
            'warning': '#F59E0B',                 # Warning (amber)
            'warning_bg': '#78350F',              # Warning background
            'error': '#EF4444',                   # Error (red)
            'error_bg': '#7F1D1D',                # Error background
            'info': '#60A5FA',                    # Information
            'info_bg': '#1E3A8A',                 # Info background

            # Border colors - More visible
            'border_primary': '#374151',          # Primary borders (lighter)
            'border_secondary': '#4B5563',        # Secondary borders (lighter)
            'border_focus': '#60A5FA',            # Focused elements
            'border_hover': '#6B7280',            # Hover state

            # Block colors - Vibrant, clear tones with excellent contrast
            'block_default': '#252A32',           # Default block color
            'block_default_border': '#4B5563',    # Default border

            # Source blocks - Vibrant emerald/teal
            'block_source': '#064E3B',            # Source blocks (darker)
            'block_source_accent': '#10B981',     # Source accent (brighter)
            'block_source_border': '#059669',     # Source border

            # Math/Process blocks - Vibrant blue
            'block_process': '#1E3A8A',           # Process blocks (darker)
            'block_process_accent': '#60A5FA',    # Process accent (brighter)
            'block_process_border': '#3B82F6',    # Process border

            # Control blocks - Vibrant purple
            'block_control': '#5B21B6',           # Control blocks (darker)
            'block_control_accent': '#A78BFA',    # Control accent
            'block_control_border': '#8B5CF6',    # Control border

            # Sink blocks - Vibrant red/rose
            'block_sink': '#7F1D1D',              # Sink blocks (darker)
            'block_sink_accent': '#EF4444',       # Sink accent (brighter)
            'block_sink_border': '#DC2626',       # Sink border

            # Other blocks - Vibrant amber
            'block_other': '#78350F',             # Other blocks (darker)
            'block_other_accent': '#F59E0B',      # Other accent
            'block_other_border': '#D97706',      # Other border

            # Selection and interaction
            'block_selected': '#60A5FA',          # Selected blocks border (brighter)
            'block_selected_bg': '#1E3A8A',       # Selected blocks background
            'block_hover': '#2F353D',             # Hover state
            'block_shadow': '#000000',            # Shadow color for depth

            # Connection colors - Improved visibility and contrast
            'connection_default': '#8B95A5',      # Default connections (lighter gray for better visibility)
            'connection_active': '#4C9EFF',       # Active connections
            'connection_selected': '#60A5FF',     # Selected connections
            'connection_error': '#EF5A6F',        # Error connections
            'connection_preview': '#F59E0B',      # Connection preview
            'connection_shadow': '#0F1216',       # Connection shadow

            # Port colors
            'port_input': '#60A5FA',              # Input ports (brighter)
            'port_output': '#10B981',             # Output ports (brighter)
            'port_hover': '#7CB8FF',              # Port hover

            # Grid and canvas
            'grid_dots': '#252A32',               # Grid dot color (subtle but visible)
            'grid_lines': '#374151',              # Grid lines (optional)
            'canvas_background': '#0F1419',       # Canvas background (darker)
            'canvas_background_alt': '#1A1F26',   # Alternate canvas (for patterns)
            'selection_rectangle': '#60A5FA',     # Selection rectangle
            'selection_rectangle_fill': '#60A5FA1A',  # Selection fill (with alpha)
            
            # Palette specific - elevated surface with jewel tone accents
            'palette_bg': '#1A1F26',               # Palette background (elevated from canvas)
            'palette_item_bg': '#252A32',          # Palette item cards (elevated)
            'palette_item_hover': '#2F3845',       # Palette item hover (even more elevated)
            'palette_item_border': '#374151',      # Subtle border for items
            'palette_item_border_hover': '#60A5FA',  # Accent border on hover (sapphire blue)
            'palette_text': '#FFFFFF',             # Palette text (pure white for contrast)
            'palette_category_bg': '#1E242C',      # Category header background
            'palette_category_text': '#9CA3AF',    # Category header text (muted)
            
            # Block icon/text drawing color (must contrast against block fill)
            'block_icon_color': '#1F2937',         # Dark icons (palette blocks have light fills)

            # Status bar - ensure good contrast
            'statusbar_bg': '#151A20',             # Status bar background
            'statusbar_text': '#E5E7EB',           # Status bar text (brighter white)
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
            'surface_primary': '#FAFBFC',          # Primary surface
            'surface_secondary': '#F3F4F6',        # Secondary surface
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

            # Connection colors - Vibrant and clear (matching preview aesthetics)
            'connection_default': '#2563EB',      # Default connections (blue like preview)
            'connection_active': '#1D4ED8',       # Active connections (darker blue)
            'connection_selected': '#3B82F6',     # Selected connections (bright blue)
            'connection_error': '#EF4444',        # Error connections
            'connection_preview': '#2563EB',      # Connection preview (consistent with default)
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
            
            # Palette specific
            'palette_bg': '#F3F4F6',               # Palette background
            'palette_item_bg': '#FFFFFF',          # Palette item cards
            'palette_item_hover': '#F9FAFB',       # Palette item hover
            'palette_item_border': '#E5E7EB',      # Subtle border for items
            'palette_item_border_hover': '#2563EB',  # Accent border on hover
            'palette_text': '#374151',             # Palette text
            'palette_category_bg': '#E5E7EB',      # Category header background
            'palette_category_text': '#6B7280',    # Category header text
            
            # Block icon/text drawing color (must contrast against block fill)
            'block_icon_color': '#1F2937',         # Dark icons on light block backgrounds

            # Status bar
            'statusbar_bg': '#F3F4F6',             # Status bar background
            'statusbar_text': '#374151',           # Status bar text
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

    def get_icon_size(self):
        """Get the standard icon size for the application."""
        from PyQt5.QtCore import QSize
        return QSize(24, 24)


# Global theme manager instance
theme_manager = ThemeManager()