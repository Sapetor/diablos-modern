"""
Platform-specific configuration and detection for DiaBloS Modern UI.
Centralizes platform detection logic and provides consistent sizing across components.
"""

import platform
import logging
from PyQt5.QtWidgets import QApplication

logger = logging.getLogger(__name__)


class PlatformConfig:
    """
    Centralized platform detection and configuration.
    Provides platform-specific UI sizing and layout parameters.
    """

    def __init__(self):
        """Initialize platform configuration based on current display."""
        self._detect_platform()
        self._log_configuration()

    def _detect_platform(self):
        """Detect platform and screen characteristics."""
        screen = QApplication.primaryScreen()
        self.device_ratio = screen.devicePixelRatio()

        screen_geometry = screen.availableGeometry()
        self.logical_width = screen_geometry.width()
        self.logical_height = screen_geometry.height()

        # Platform detection
        self.is_macos = platform.system() == 'Darwin'
        self.is_windows = platform.system() == 'Windows'
        self.is_linux = platform.system() == 'Linux'

        # Specific screen configurations
        self.is_retina_small = (
            self.is_macos and
            self.device_ratio >= 1.9 and
            self.logical_width < 1500
        )
        self.is_high_dpi = self.device_ratio > 1.25

    def _log_configuration(self):
        """Log detected configuration."""
        logger.info(f"Platform configuration: "
                   f"platform={platform.system()}, "
                   f"devicePixelRatio={self.device_ratio}, "
                   f"logical={self.logical_width}×{self.logical_height}, "
                   f"is_retina_small={self.is_retina_small}")

    # Window sizing

    @property
    def window_width_percent(self) -> float:
        """Percentage of screen width to use for window."""
        if self.is_retina_small:
            return 0.90
        return 0.85

    @property
    def window_height_percent(self) -> float:
        """Percentage of screen height to use for window."""
        if self.is_retina_small:
            return 0.88
        return 0.85

    @property
    def window_min_width(self) -> int:
        """Minimum window width."""
        if self.is_retina_small:
            return 1150
        return 1000

    @property
    def window_min_height(self) -> int:
        """Minimum window height."""
        if self.is_retina_small:
            return 650
        return 700

    @property
    def should_cap_window_size(self) -> bool:
        """Whether to cap window size (on standard DPI only)."""
        return self.device_ratio <= 1.25

    # Left panel (Block Palette)

    @property
    def left_panel_min_width(self) -> int:
        """Minimum width for left panel (block palette)."""
        if self.is_retina_small:
            return 230
        elif self.is_high_dpi:
            return 270
        return 220

    @property
    def left_panel_max_width(self) -> int:
        """Maximum width for left panel."""
        if self.is_retina_small:
            return 330
        elif self.is_high_dpi:
            return 380
        return 320

    # Canvas

    @property
    def canvas_min_width(self) -> int:
        """Minimum width for canvas area."""
        if self.is_retina_small:
            return 650
        elif self.is_high_dpi:
            return 800
        return 700

    @property
    def canvas_min_height(self) -> int:
        """Minimum height for canvas area."""
        if self.is_retina_small:
            return 500
        elif self.is_high_dpi:
            return 600
        return 500

    # Right panel (Properties)

    @property
    def property_panel_min_width(self) -> int:
        """Minimum width for property panel."""
        if self.is_retina_small:
            return 280
        elif self.is_high_dpi:
            return int(280 * 1.3)
        return 280

    @property
    def property_panel_max_width(self) -> int:
        """Maximum width for property panel."""
        if self.is_retina_small:
            return 420
        elif self.is_high_dpi:
            return int(500 * 1.3)
        return 500

    # Splitter sizing

    @property
    def splitter_left_width(self) -> int:
        """Initial width for left splitter panel."""
        if self.is_retina_small:
            return 250
        elif self.is_high_dpi:
            return 300
        return 250

    @property
    def splitter_property_percent(self) -> float:
        """Percentage of center width for property panel."""
        if self.is_retina_small:
            return 0.23
        return 0.25

    @property
    def splitter_property_min_width(self) -> int:
        """Minimum width for property panel in splitter."""
        if self.is_retina_small:
            return 300
        elif self.is_high_dpi:
            return 350
        return 300

    # Block palette blocks

    @property
    def palette_block_size(self) -> int:
        """Size of blocks in the palette."""
        if self.is_retina_small:
            return 95
        elif self.is_high_dpi:
            return int(100 * 1.2)
        return 100


# Global singleton instance
_platform_config = None


def get_platform_config() -> PlatformConfig:
    """
    Get the global platform configuration instance.
    Creates it on first call.

    Returns:
        PlatformConfig: The platform configuration singleton
    """
    global _platform_config
    if _platform_config is None:
        _platform_config = PlatformConfig()
    return _platform_config
