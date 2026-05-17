"""
Theme Management System for Modern DiaBloS UI
Provides consistent color schemes and styling across the application.

v3 fixup: rebalanced dark theme so canvas + chrome + palette items live in
the same hue family. Previous version mixed Solarized teal (#002B36, #073642)
with neutral gray chrome (#0F1419, #1A1F26) — produced a visible color rift
between canvas and panels.

Changes vs v2:
  * canvas_background       #002B36 → #14181F  (neutral, matches chrome)
  * canvas_background_alt   #1A1F26 → #181C23
  * grid_dots               #252A32 → #2A3038  (slightly more visible)
  * background_primary      #0F1419 → #13171D
  * background_secondary    #1A1F26 → #1C2128
  * background_tertiary     #252A32 → #2B3038
  * surface                 #1A1F26 → #1C2128
  * surface_variant         #252A32 → #2B3038
  * surface_elevated        #2F353D → #363D47
  * border_primary          #374151 → #2D333D
  * border_secondary        #4B5563 → #3A414B
  * palette_item_bg         #252A32 → #232830  (slightly recessed from panel)
  * palette_item_hover      #2F3845 → #2E3540
  * palette_item_border     #374151 → #2A2F38
  * palette_text            #FFFFFF → #E5E9EF  (less harsh)
  * block_shadow            #000000 → #00000066 (40% alpha — softer)
  * DEFAULT_PALETTE         solarized → tailwind  (neutral grays match chrome)
  * Solarized dark fills    #073642 → #1C2128   (matches surface)
  * Catppuccin dark fills   #303446 (unchanged — already neutral)
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
            # Fills retuned to neutral surface — Solarized teal (#073642) was
            # clashing with the neutral chrome. Accents/borders keep the
            # Solarized hues so the palette still reads as "Solarized".
            "block_source":             "#1C2128",
            "block_source_border":      "#859900",
            "block_source_accent":      "#859900",
            "block_process":            "#1C2128",
            "block_process_border":     "#268BD2",
            "block_process_accent":     "#268BD2",
            "block_control":            "#1C2128",
            "block_control_border":     "#6C71C4",
            "block_control_accent":     "#6C71C4",
            "block_sink":               "#1C2128",
            "block_sink_border":        "#DC322F",
            "block_sink_accent":        "#DC322F",
            "block_routing":            "#1C2128",
            "block_routing_border":     "#2AA198",
            "block_routing_accent":     "#2AA198",
            "block_analysis":           "#1C2128",
            "block_analysis_border":    "#D33682",
            "block_analysis_accent":    "#D33682",
            "block_pde":                "#1C2128",
            "block_pde_border":         "#B58900",
            "block_pde_accent":         "#B58900",
            "block_optimization":       "#1C2128",
            "block_optimization_border":"#CB4B16",
            "block_optimization_accent":"#CB4B16",
            "block_other":              "#1C2128",
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
            # Fills at Tailwind 800 level (~5-10% luminance). White icons
            # (#E5E9EF) get 6-15:1 contrast (AA/AAA). Block edge still
            # separates from canvas (#14181F ~1.4% lum) at 3-7× luminance
            # ratio. The +30 RGB gradient brings the top close to 700.
            "block_source":             "#065F46",
            "block_source_border":      "#10B981",
            "block_source_accent":      "#34D399",
            "block_process":            "#1E40AF",
            "block_process_border":     "#3B82F6",
            "block_process_accent":     "#60A5FA",
            "block_control":            "#5B21B6",
            "block_control_border":     "#8B5CF6",
            "block_control_accent":     "#A78BFA",
            "block_sink":               "#991B1B",
            "block_sink_border":        "#EF4444",
            "block_sink_accent":        "#F87171",
            "block_routing":            "#115E59",
            "block_routing_border":     "#14B8A6",
            "block_routing_accent":     "#2DD4BF",
            "block_analysis":           "#9D174D",
            "block_analysis_border":    "#EC4899",
            "block_analysis_accent":    "#F472B6",
            "block_pde":                "#854D0E",
            "block_pde_border":         "#EAB308",
            "block_pde_accent":         "#FACC15",
            "block_optimization":       "#3730A3",
            "block_optimization_border":"#6366F1",
            "block_optimization_accent":"#818CF8",
            "block_other":              "#1F2937",
            "block_other_border":       "#6B7280",
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

_PALETTE_KEYS = frozenset(
    key
    for variants in PALETTES["solarized"].values()
    for key in variants
)

# Tailwind is the default — its dark fills are tuned to neutral chrome.
# (Solarized was the old default; its teal fills clashed with the gray chrome.)
DEFAULT_PALETTE = "tailwind"
PALETTE_DISPLAY_NAMES = {
    "solarized": "Solarized",
    "tailwind":  "Tailwind",
    "catppuccin": "Catppuccin Frappé",
}


class ThemeManager(QObject):
    """Manages application themes and color schemes."""

    theme_changed = pyqtSignal(str)

    _BLOCK_FILL_KEYS = frozenset([
        "block_source", "block_process", "block_control", "block_sink",
        "block_routing", "block_analysis", "block_pde", "block_optimization",
        "block_other",
    ])

    def __init__(self):
        super().__init__()
        self.current_theme = ThemeType.DARK
        self.current_palette = DEFAULT_PALETTE
        self.solid_fills: bool = False
        self.themes = {
            ThemeType.DARK: self._create_dark_theme(),
            ThemeType.LIGHT: self._create_light_theme()
        }

    # ------------------------------------------------------------------
    # Chrome theme dicts
    # ------------------------------------------------------------------

    def _create_dark_theme(self) -> Dict[str, Any]:
        """Dark theme — neutral grays, no teal hue.

        Single hue family across canvas + chrome + palette so adjacent
        surfaces don't visibly clash. Hierarchy comes from luminance, not
        hue: panels are slightly lighter than canvas, palette items are
        slightly recessed from panels, hover lifts them again.
        """
        return {
            # Main backgrounds — single hue, 4-step luminance ramp
            'background_primary':   '#13171D',    # canvas + outermost frame
            'background_secondary': '#1C2128',    # panel containers (palette, inspector)
            'background_tertiary':  '#2B3038',    # elevated widgets, button bg

            # Surface
            'surface':              '#1C2128',
            'surface_primary':      '#1C2128',
            'surface_secondary':    '#2B3038',
            'surface_variant':      '#2B3038',
            'surface_elevated':     '#363D47',

            # Text
            'text_primary':         '#E5E9EF',    # softer than pure white
            'text_secondary':       '#9AA4B2',
            'text_disabled':        '#5B6573',
            'text_inverse':         '#13171D',

            # Accent
            'accent_primary':       '#60A5FA',
            'accent_secondary':     '#3B82F6',
            'accent_hover':         '#7CB8FF',
            'accent_pressed':       '#2563EB',

            # Status
            'success':              '#10B981',
            'success_bg':           '#064E3B',
            'warning':              '#F59E0B',
            'warning_bg':           '#78350F',
            'error':                '#EF4444',
            'error_bg':             '#7F1D1D',
            'info':                 '#60A5FA',
            'info_bg':              '#1E3A8A',

            # Borders — subtle (no more visible separator clutter)
            'border_primary':       '#2D333D',
            'border_secondary':     '#3A414B',
            'border_focus':         '#60A5FA',
            'border_hover':         '#4B5563',

            # Block defaults
            'block_default':        '#1C2128',
            'block_default_border': '#4B5563',

            # Selection / interaction
            'block_selected':       '#60A5FA',
            'block_selected_bg':    '#1E3A8A',
            'block_hover':          '#2F353D',
            'block_shadow':         '#00000066',  # 40% alpha — softer

            # Connections
            'connection_default':   '#8B95A5',
            'connection_active':    '#4C9EFF',
            'connection_selected':  '#60A5FF',
            'connection_error':     '#EF5A6F',
            'connection_preview':   '#F59E0B',
            'connection_shadow':    '#0F1216',

            # Ports
            'port_input':           '#60A5FA',
            'port_output':          '#10B981',
            'port_hover':           '#7CB8FF',

            # Grid + canvas — KEY FIX: canvas is now neutral, matches chrome
            'grid_dots':            '#2A3038',
            'grid_lines':           '#2D333D',
            'canvas_background':    '#14181F',    # neutral — was #002B36 (teal)
            'canvas_background_alt':'#181C23',    # for patterns
            'selection_rectangle':  '#60A5FA',
            'selection_rectangle_fill': '#60A5FA1A',

            # Palette
            'palette_bg':           '#1C2128',
            'palette_item_bg':      '#232830',    # slightly recessed
            'palette_item_hover':   '#2E3540',
            'palette_item_border':  '#2A2F38',
            'palette_item_border_hover': '#60A5FA',
            'palette_text':         '#E5E9EF',    # softer than pure white
            'palette_category_bg':  '#1C2128',
            'palette_category_text':'#9AA4B2',

            # Block icon/text drawing color
            'block_icon_color':     '#E5E9EF',    # light icons on neutral fills

            # Status bar
            'statusbar_bg':         '#1C2128',
            'statusbar_text':       '#E5E9EF',
        }

    def _create_light_theme(self) -> Dict[str, Any]:
        """Light theme — unchanged from v2."""
        return {
            'background_primary':   '#FAFBFC',
            'background_secondary': '#F3F4F6',
            'background_tertiary':  '#E5E7EB',

            'surface':              '#FFFFFF',
            'surface_primary':      '#FAFBFC',
            'surface_secondary':    '#F3F4F6',
            'surface_variant':      '#F9FAFB',
            'surface_elevated':     '#FFFFFF',

            'text_primary':         '#111827',
            'text_secondary':       '#6B7280',
            'text_disabled':        '#9CA3AF',
            'text_inverse':         '#FFFFFF',

            'accent_primary':       '#2563EB',
            'accent_secondary':     '#1D4ED8',
            'accent_hover':         '#3B82F6',
            'accent_pressed':       '#1E40AF',

            'success':              '#10B981',
            'success_bg':           '#D1FAE5',
            'warning':              '#F59E0B',
            'warning_bg':           '#FEF3C7',
            'error':                '#EF4444',
            'error_bg':             '#FEE2E2',
            'info':                 '#2563EB',
            'info_bg':              '#DBEAFE',

            'border_primary':       '#E5E7EB',
            'border_secondary':     '#D1D5DB',
            'border_focus':         '#2563EB',
            'border_hover':         '#9CA3AF',

            'block_default':        '#EEE8D5',
            'block_default_border': '#586E75',

            'block_selected':       '#2563EB',
            'block_selected_bg':    '#EFF6FF',
            'block_hover':          '#F3F4F6',
            'block_shadow':         '#00000015',

            'connection_default':   '#2563EB',
            'connection_active':    '#1D4ED8',
            'connection_selected':  '#3B82F6',
            'connection_error':     '#EF4444',
            'connection_preview':   '#2563EB',
            'connection_shadow':    '#00000010',

            'port_input':           '#2563EB',
            'port_output':          '#10B981',
            'port_hover':           '#3B82F6',

            'grid_dots':            '#E5E7EB',
            'grid_lines':           '#D1D5DB',
            'canvas_background':    '#FAFBFC',
            'canvas_background_alt':'#FFFFFF',
            'selection_rectangle':  '#2563EB',
            'selection_rectangle_fill': '#2563EB1A',

            'palette_bg':           '#F3F4F6',
            'palette_item_bg':      '#FFFFFF',
            'palette_item_hover':   '#F9FAFB',
            'palette_item_border':  '#E5E7EB',
            'palette_item_border_hover': '#2563EB',
            'palette_text':         '#374151',
            'palette_category_bg':  '#E5E7EB',
            'palette_category_text':'#6B7280',

            'block_icon_color':     '#1F2937',

            'statusbar_bg':         '#F3F4F6',
            'statusbar_text':       '#374151',
        }

    # ------------------------------------------------------------------
    # Color lookup
    # ------------------------------------------------------------------

    def get_current_theme(self) -> Dict[str, Any]:
        return self.themes[self.current_theme]

    def get_color(self, color_name: str) -> QColor:
        if color_name in _PALETTE_KEYS:
            theme_key = self.current_theme.value
            palette = PALETTES.get(self.current_palette, PALETTES[DEFAULT_PALETTE])
            lookup = (
                color_name + "_accent"
                if self.solid_fills and color_name in self._BLOCK_FILL_KEYS
                else color_name
            )
            color_hex = palette[theme_key].get(lookup, '#000000')
        else:
            theme = self.get_current_theme()
            color_hex = theme.get(color_name, '#000000')

        # Handle 8-char #RRGGBBAA (e.g. block_shadow with alpha)
        c = QColor(color_hex)
        if len(color_hex) == 9 and color_hex.startswith('#'):
            try:
                alpha = int(color_hex[7:9], 16)
                c.setAlpha(alpha)
            except ValueError:
                pass
        return c

    # ------------------------------------------------------------------
    # Theme control
    # ------------------------------------------------------------------

    def set_theme(self, theme_type: ThemeType):
        if theme_type != self.current_theme:
            self.current_theme = theme_type
            self.theme_changed.emit(theme_type.value)

    def toggle_theme(self):
        if self.current_theme == ThemeType.DARK:
            self.set_theme(ThemeType.LIGHT)
        else:
            self.set_theme(ThemeType.DARK)

    # ------------------------------------------------------------------
    # Palette control
    # ------------------------------------------------------------------

    def set_palette(self, palette_name: str):
        if palette_name not in PALETTES:
            import logging
            logging.getLogger(__name__).warning(
                "Unknown palette %r — falling back to %r", palette_name, DEFAULT_PALETTE
            )
            palette_name = DEFAULT_PALETTE
        if palette_name != self.current_palette:
            self.current_palette = palette_name
            self.theme_changed.emit(self.current_theme.value)

    def set_solid_fills(self, enabled: bool):
        if enabled != self.solid_fills:
            self.solid_fills = enabled
            self.theme_changed.emit(self.current_theme.value)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_qss_variables(self) -> Dict[str, str]:
        theme = self.get_current_theme()
        qss_vars = {}
        for key, value in theme.items():
            qss_vars[f"@{key}"] = value
        theme_key = self.current_theme.value
        palette = PALETTES.get(self.current_palette, PALETTES[DEFAULT_PALETTE])
        for key, value in palette[theme_key].items():
            qss_vars[f"@{key}"] = value
        return qss_vars

    def get_icon_size(self):
        from PyQt5.QtCore import QSize
        return QSize(24, 24)


theme_manager = ThemeManager()
