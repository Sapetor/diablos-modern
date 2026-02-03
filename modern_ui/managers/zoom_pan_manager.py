"""
Zoom and Pan Manager for ModernCanvas
Handles view transformation, zooming, and panning operations.
"""

import logging
from PyQt5.QtCore import Qt, QPoint

logger = logging.getLogger(__name__)


class ZoomPanManager:
    """
    Manages zoom and pan transformations for the canvas view.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim

        # Zoom and Pan state
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)

    def screen_to_world(self, pos):
        """Converts screen coordinates to world coordinates.

        Args:
            pos (QPoint): Screen coordinates

        Returns:
            QPoint: World coordinates
        """
        return (pos - self.pan_offset) / self.zoom_factor

    def world_to_screen(self, pos):
        """Converts world coordinates to screen coordinates.

        Args:
            pos (QPoint): World coordinates

        Returns:
            QPoint: Screen coordinates
        """
        return pos * self.zoom_factor + self.pan_offset

    def zoom_in(self):
        """Zoom in by 10%."""
        self.set_zoom(self.zoom_factor * 1.1)

    def zoom_out(self):
        """Zoom out by 10%."""
        self.set_zoom(self.zoom_factor / 1.1)

    def set_zoom(self, factor):
        """Set zoom factor and update canvas.

        Args:
            factor (float): New zoom factor
        """
        self.zoom_factor = factor
        self.canvas.zoom_factor = factor  # Keep canvas.zoom_factor in sync
        self.canvas.update()

    def zoom_to_fit(self):
        """Zoom to fit all blocks in the view."""
        if not self.dsim.blocks_list:
            return

        # Calculate bounding box of all blocks
        min_x = min(block.left for block in self.dsim.blocks_list)
        min_y = min(block.top for block in self.dsim.blocks_list)
        max_x = max(block.left + block.width for block in self.dsim.blocks_list)
        max_y = max(block.top + block.height for block in self.dsim.blocks_list)

        # Add padding
        padding = 50
        bbox_width = max_x - min_x + 2 * padding
        bbox_height = max_y - min_y + 2 * padding

        # Calculate zoom factor to fit
        width_ratio = self.canvas.width() / bbox_width if bbox_width > 0 else 1.0
        height_ratio = self.canvas.height() / bbox_height if bbox_height > 0 else 1.0
        target_zoom = min(width_ratio, height_ratio, 1.0)  # Don't zoom in beyond 100%

        self.set_zoom(target_zoom)
        logger.info(f"Zoomed to fit: {len(self.dsim.blocks_list)} blocks")

    def reset_view(self):
        """Reset zoom and pan to default values."""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.canvas.zoom_factor = 1.0
        self.canvas.pan_offset = QPoint(0, 0)
        self.canvas.update()
        logger.info("View reset to defaults")

    def handle_wheel_event(self, event):
        """Handle mouse wheel events for zooming and scrolling.

        - Plain scroll: Pan the canvas (for MacBook trackpad users)
        - Ctrl/Cmd + scroll: Zoom in/out

        Args:
            event: QWheelEvent
        """
        modifiers = event.modifiers()

        # Check if Ctrl (or Cmd on macOS) is pressed
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            # Zoom mode
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Pan/scroll mode - pan the canvas with touchpad scrolling
            delta_x = event.angleDelta().x()
            delta_y = event.angleDelta().y()

            # Scale the delta for smoother scrolling
            scroll_sensitivity = 0.5
            pan_delta = QPoint(
                int(delta_x * scroll_sensitivity),
                int(delta_y * scroll_sensitivity)
            )

            # Apply panning offset
            self.pan_offset += pan_delta
            self.canvas.pan_offset = self.pan_offset  # Keep canvas.pan_offset in sync
            self.canvas.update()

            logger.debug(f"Canvas panned by {pan_delta}, new offset: {self.pan_offset}")
