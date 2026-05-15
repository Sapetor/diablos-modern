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


# ---------------------------------------------------------------------------
# Block-color palettes (independent of dark/light chrome)
# Each palette defines 9 categories × 3 keys (fill, border, accent) × 2 themes.
# ---------------------------------------------------------------------------
PALETTES: Dict[str, Dict[str, Dict[str, str]]] = {
    "solarized": {
        "dark": {
            "block_source":             "#073642",
            "block_source_border":      "#859900",
            "block_source_accent":      "#859900",
            "block_process":            "#073642",
            "block_process_border":     "#268BD2",
            "block_process_accent":     "#268BD2",
            "block_control":            "#073642",
            "block_control_border":     "#6C71C4",
            "block_control_accent":     "#6C71C4",
            "block_sink":               "#073642",
            "block_sink_border":        "#DC322F",
            "block_sink_accent":        "#DC322F",
            "block_routing":            "#073642",
            "block_routing_border":     "#2AA198",
            "block_routing_accent":     "#2AA198",
            "block_analysis":           "#073642",
            "block_analysis_border":    "#D33682",
            "block_analysis_accent":    "#D33682",
            "block_pde":                "#073642",
            "block_pde_border":         "#B58900",
            "block_pde_accent":         "#B58900",
            "block_optimization":       "#073642",
            "block_optimization_border":"#CB4B16",
            "block_optimization_accent":"#CB4B16",
            "block_other":              "#073642",
            "block_other_border":       "#93A1A1",
            "block_other_accent":       "#93A1A1",
        },
        "light": {
            "block_source":             "#EEE8D5",
            "block_source_border":      "#859900",
            "block_source_accent":      "#859900",
            "block_process":            "#EEE8D5",
            "block_process_border":     "#268BD2",
            "block_process_accent":     "#268BD2",
            "block_control":            "#EEE8D5",
            "block_control_border":     "#6C71C4",
            "block_control_accent":     "#6C71C4",
            "block_sink":               "#EEE8D5",
            "block_sink_border":        "#DC322F",
            "block_sink_accent":        "#DC322F",
            "block_routing":            "#EEE8D5",
            "block_routing_border":     "#2AA198",
            "block_routing_accent":     "#2AA198",
            "block_analysis":           "#EEE8D5",
            "block_analysis_border":    "#D33682",
            "block_analysis_accent":    "#D33682",
            "block_pde":                "#EEE8D5",
            "block_pde_border":         "#B58900",
            "block_pde_accent":         "#B58900",
            "block_optimization":       "#EEE8D5",
            "block_optimization_border":"#CB4B16",
            "block_optimization_accent":"#CB4B16",
            "block_other":              "#EEE8D5",
            "block_other_border":       "#93A1A1",
            "block_other_accent":       "#93A1A1",
        },
    },

    "tailwind": {
        "dark": {
            "block_source":             "#064E3B",
            "block_source_border":      "#059669",
            "block_source_accent":      "#10B981",
            "block_process":            "#1E3A8A",
            "block_process_border":     "#2563EB",
            "block_process_accent":     "#60A5FA",
            "block_control":            "#4C1D95",
            "block_control_border":     "#7C3AED",
            "block_control_accent":     "#A78BFA",
            "block_sink":               "#7F1D1D",
            "block_sink_border":        "#DC2626",
            "block_sink_accent":        "#EF4444",
            "block_routing":            "#134E4A",
            "block_routing_border":     "#0D9488",
            "block_routing_accent":     "#2DD4BF",
            "block_analysis":           "#831843",
            "block_analysis_border":    "#DB2777",
            "block_analysis_accent":    "#F472B6",
            "block_pde":                "#713F12",
            "block_pde_border":         "#D97706",
            "block_pde_accent":         "#F59E0B",
            "block_optimization":       "#312E81",
            "block_optimization_border":"#4F46E5",
            "block_optimization_accent":"#818CF8",
            "block_other":              "#1F2937",
            "block_other_border":       "#4B5563",
            "block_other_accent":       "#9CA3AF",
        },
        "light": {
            "block_source":             "#D1FAE5",
            "block_source_border":      "#6EE7B7",
            "block_source_accent":      "#10B981",
            "block_process":            "#DBEAFE",
            "block_process_border":     "#93C5FD",
            "block_process_accent":     "#2563EB",
            "block_control":            "#EDE9FE",
            "block_control_border":     "#C4B5FD",
            "block_control_accent":     "#7C3AED",
            "block_sink":               "#FEE2E2",
            "block_sink_border":        "#FCA5A5",
            "block_sink_accent":        "#DC2626",
            "block_routing":            "#CCFBF1",
            "block_routing_border":     "#5EEAD4",
            "block_routing_accent":     "#0D9488",
            "block_analysis":           "#FCE7F3",
            "block_analysis_border":    "#F9A8D4",
            "block_analysis_accent":    "#DB2777",
            "block_pde":                "#FEF3C7",
            "block_pde_border":         "#FCD34D",
            "block_pde_accent":         "#D97706",
            "block_optimization":       "#E0E7FF",
            "block_optimization_border":"#A5B4FC",
            "block_optimization_accent":"#4F46E5",
            "block_other":              "#F3F4F6",
            "block_other_border":       "#D1D5DB",
            "block_other_accent":       "#6B7280",
        },
    },

    "catppuccin": {
        "dark": {
            # Catppuccin Frappé: uniform fill, per-category accents
            "block_source":             "#303446",
            "block_source_border":      "#A6D189",
            "block_source_accent":      "#A6D189",
            "block_process":            "#303446",
            "block_process_border":     "#8CAAEE",
            "block_process_accent":     "#8CAAEE",
            "block_control":            "#303446",
            "block_control_border":     "#CA9EE6",
            "block_control_accent":     "#CA9EE6",
            "block_sink":               "#303446",
            "block_sink_border":        "#E78284",
            "block_sink_accent":        "#E78284",
            "block_routing":            "#303446",
            "block_routing_border":     "#81C8BE",
            "block_routing_accent":     "#81C8BE",
            "block_analysis":           "#303446",
            "block_analysis_border":    "#F4B8E4",
            "block_analysis_accent":    "#F4B8E4",
            "block_pde":                "#303446",
            "block_pde_border":         "#EF9F76",
            "block_pde_accent":         "#EF9F76",
            "block_optimization":       "#303446",
            "block_optimization_border":"#BABBF1",
            "block_optimization_accent":"#BABBF1",
            "block_other":              "#303446",
            "block_other_border":       "#B5BFE2",
            "block_other_accent":       "#B5BFE2",
        },
        "light": {
            # Catppuccin Latte: uniform fill, per-category accents
            "block_source":             "#EFF1F5",
            "block_source_border":      "#40A02B",
            "block_source_accent":      "#40A02B",
            "block_process":            "#EFF1F5",
            "block_process_border":     "#1E66F5",
            "block_process_accent":     "#1E66F5",
            "block_control":            "#EFF1F5",
            "block_control_border":     "#8839EF",
            "block_control_accent":     "#8839EF",
            "block_sink":               "#EFF1F5",
            "block_sink_border":        "#D20F39",
            "block_sink_accent":        "#D20F39",
            "block_routing":            "#EFF1F5",
            "block_routing_border":     "#179299",
            "block_routing_accent":     "#179299",
            "block_analysis":           "#EFF1F5",
            "block_analysis_border":    "#EA76CB",
            "block_analysis_accent":    "#EA76CB",
            "block_pde":                "#EFF1F5",
            "block_pde_border":         "#FE640B",
            "block_pde_accent":         "#FE640B",
            "block_optimization":       "#EFF1F5",
            "block_optimization_border":"#7287FD",
            "block_optimization_accent":"#7287FD",
            "block_other":              "#EFF1F5",
            "block_other_border":       "#6C6F85",
            "block_other_accent":       "#6C6F85",
        },
    },
}

# Keys that live in PALETTES (not the chrome theme dicts)
_PALETTE_KEYS = frozenset(
    key
    for variants in PALETTES["solarized"].values()
    for key in variants
)

DEFAULT_PALETTE = "solarized"
PALETTE_DISPLAY_NAMES = {
    "solarized": "Solarized",
    "tailwind":  "Tailwind",
    "catppuccin": "Catppuccin Frappé",
}


class ThemeManager(QObject):
    """Manages application themes and color schemes."""

    theme_changed = pyqtSignal(str)  # Emitted when theme or palette changes

    def __init__(self):
        super().__init__()
        self.current_theme = ThemeType.DARK
        self.current_palette = DEFAULT_PALETTE
        self.themes = {
            ThemeType.DARK: self._create_dark_theme(),
            ThemeType.LIGHT: self._create_light_theme()
        }

    # ------------------------------------------------------------------
    # Chrome theme dicts  (block_* keys removed — they live in PALETTES)
    # ------------------------------------------------------------------

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

            # Block defaults (non-category-specific)
            'block_default': '#073642',           # Default block color
            'block_default_border': '#586E75',    # Default border

            # Selection and interaction
            'block_selected': '#60A5FA',          # Selected blocks border (brighter)
            'block_selected_bg': '#1E3A8A',       # Selected blocks background
            'block_hover': '#2F353D',             # Hover state
            'block_shadow': '#000000',            # Shadow color for depth

            # Connection colors - Improved visibility and contrast
            'connection_default': '#8B95A5',      # Default connections
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
            'canvas_background': '#002B36',       # Canvas background (Solarized base03)
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

            # Block defaults (non-category-specific)
            'block_default': '#EEE8D5',           # Default block color (base2)
            'block_default_border': '#586E75',    # Default border (base01)

            # Selection and interaction
            'block_selected': '#2563EB',          # Selected blocks border
            'block_selected_bg': '#EFF6FF',       # Selected blocks background
            'block_hover': '#F3F4F6',             # Hover state
            'block_shadow': '#00000015',          # Shadow color for depth (with alpha)

            # Connection colors - Vibrant and clear
            'connection_default': '#2563EB',      # Default connections
            'connection_active': '#1D4ED8',       # Active connections
            'connection_selected': '#3B82F6',     # Selected connections
            'connection_error': '#EF4444',        # Error connections
            'connection_preview': '#2563EB',      # Connection preview
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

    # ------------------------------------------------------------------
    # Color lookup
    # ------------------------------------------------------------------

    def get_current_theme(self) -> Dict[str, Any]:
        """Get the current chrome theme colors (does not include palette block colors)."""
        return self.themes[self.current_theme]

    def get_color(self, color_name: str) -> QColor:
        """Get a QColor for the specified color name.

        Block-category keys (block_source, block_source_border, etc.) are
        resolved from the active PALETTE + current dark/light variant.
        All other keys come from the chrome theme dict.
        """
        if color_name in _PALETTE_KEYS:
            theme_key = self.current_theme.value  # "dark" or "light"
            palette = PALETTES.get(self.current_palette, PALETTES[DEFAULT_PALETTE])
            color_hex = palette[theme_key].get(color_name, '#000000')
        else:
            theme = self.get_current_theme()
            color_hex = theme.get(color_name, '#000000')
        return QColor(color_hex)

    # ------------------------------------------------------------------
    # Theme control
    # ------------------------------------------------------------------

    def set_theme(self, theme_type: ThemeType):
        """Change the current chrome theme (dark/light)."""
        if theme_type != self.current_theme:
            self.current_theme = theme_type
            self.theme_changed.emit(theme_type.value)

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        if self.current_theme == ThemeType.DARK:
            self.set_theme(ThemeType.LIGHT)
        else:
            self.set_theme(ThemeType.DARK)

    # ------------------------------------------------------------------
    # Palette control
    # ------------------------------------------------------------------

    def set_palette(self, palette_name: str):
        """Change the active block-color palette.

        Falls back to DEFAULT_PALETTE for unknown names (logs a warning).
        Emits theme_changed so consumers re-render.
        """
        if palette_name not in PALETTES:
            import logging
            logging.getLogger(__name__).warning(
                "Unknown palette %r — falling back to %r", palette_name, DEFAULT_PALETTE
            )
            palette_name = DEFAULT_PALETTE
        if palette_name != self.current_palette:
            self.current_palette = palette_name
            self.theme_changed.emit(self.current_theme.value)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_qss_variables(self) -> Dict[str, str]:
        """Get theme colors as QSS variables for stylesheets."""
        theme = self.get_current_theme()
        qss_vars = {}
        for key, value in theme.items():
            qss_vars[f"@{key}"] = value
        # Also expose current palette block colors
        theme_key = self.current_theme.value
        palette = PALETTES.get(self.current_palette, PALETTES[DEFAULT_PALETTE])
        for key, value in palette[theme_key].items():
            qss_vars[f"@{key}"] = value
        return qss_vars

    def get_icon_size(self):
        """Get the standard icon size for the application."""
        from PyQt5.QtCore import QSize
        return QSize(24, 24)


# Global theme manager instance
theme_manager = ThemeManager()
