"""
WindowSetupManager -- configures the main window chrome: window properties
(title/size/font), the menu bar, and the toolbar (creation + signal wiring).

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

``setup_toolbar`` assigns ``window.toolbar`` because the toolbar is referenced
widely (layout/status-bar managers, the variable/workspace editors). The menu
bar itself is built by ``MenuBuilder`` (``window.menu_builder``); this manager
just triggers it.
"""

import logging

from PyQt5.QtGui import QFont

from modern_ui.widgets.modern_toolbar import ModernToolBar
from modern_ui.platform_config import get_platform_config

logger = logging.getLogger(__name__)


class WindowSetupManager:
    """Sets up window properties, menu bar, and toolbar."""

    def __init__(self, main_window):
        self.window = main_window

    def setup_window(self):
        """Setup main window properties with screen-aware sizing."""
        window = self.window
        window.setWindowTitle("DiaBloS - Modern Block Diagram Simulator")

        # Get platform configuration
        config = get_platform_config()

        # Calculate responsive window size based on available screen space
        if window.screen_geometry:
            target_width = int(window.screen_geometry.width() * config.window_width_percent)
            target_height = int(window.screen_geometry.height() * config.window_height_percent)

            # On standard DPI, cap window size to avoid giant windows
            if config.should_cap_window_size:
                target_width = min(target_width, 1600)
                target_height = min(target_height, 1000)

            # Set minimum size
            min_width = max(int(target_width * 0.70), config.window_min_width)
            min_height = max(int(target_height * 0.70), config.window_min_height)

            window.setMinimumSize(min_width, min_height)
            window.resize(target_width, target_height)

            logger.info(f"Window sizing: target={target_width}×{target_height}, min={min_width}×{min_height}")
        else:
            # Fallback to larger sizes
            window.setMinimumSize(1200, 800)
            window.resize(1600, 1000)

        # Set modern font
        font = QFont("Segoe UI", 10)
        window.setFont(font)

        # Apply modern theme
        window.setObjectName("ModernMainWindow")

    def setup_menubar(self):
        """Setup modern menu bar."""
        window = self.window
        if window.menu_builder:
            window.menu_builder.setup_menubar()

    def setup_toolbar(self):
        """Setup modern toolbar."""
        window = self.window
        window.toolbar = ModernToolBar(window)
        window.addToolBar(window.toolbar)

        # Connect toolbar signals
        window.toolbar.new_diagram.connect(window.new_diagram)
        window.toolbar.open_diagram.connect(window.open_diagram)
        window.toolbar.save_diagram.connect(window.save_diagram)
        window.toolbar.play_simulation.connect(window.start_simulation)
        window.toolbar.pause_simulation.connect(window.pause_simulation)
        window.toolbar.stop_simulation.connect(window.stop_simulation)
        window.toolbar.step_simulation.connect(window.step_simulation)
        window.toolbar.plot_results.connect(window.show_plots)
        window.toolbar.capture_screen.connect(window.capture_screen)
        window.toolbar.zoom_changed.connect(window.set_zoom)
        window.toolbar.theme_toggled.connect(window.on_theme_changed)
        window.toolbar.command_palette_requested.connect(window.show_command_palette)
