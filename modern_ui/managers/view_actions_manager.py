"""
ViewActionsManager -- canvas view actions: zoom (set/in/out), grid toggle,
fit-to-window, and the minimap dock toggle.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

These act on ``window.canvas`` and reflect state into the status-bar zoom pill,
the checkable grid/minimap menu actions, and the minimap dock.
"""

import logging

logger = logging.getLogger(__name__)


class ViewActionsManager:
    """Owns the canvas view actions (zoom, grid, fit, minimap)."""

    def __init__(self, main_window):
        self.window = main_window

    def set_zoom(self, factor: float):
        """Set zoom factor."""
        window = self.window
        if hasattr(window, 'canvas'):
            window.canvas.set_zoom(factor)
            window.zoom_status.setText(f"{int(window.canvas.zoom_factor * 100)}%")

    def zoom_in(self):
        window = self.window
        if hasattr(window, 'canvas'):
            window.canvas.zoom_in()
            window.zoom_status.setText(f"{int(window.canvas.zoom_factor * 100)}%")
            window.toast.show_message(f"🔍 Zoom: {int(window.canvas.zoom_factor * 100)}%", 1500)

    def zoom_out(self):
        window = self.window
        if hasattr(window, 'canvas'):
            window.canvas.zoom_out()
            window.zoom_status.setText(f"{int(window.canvas.zoom_factor * 100)}%")
            window.toast.show_message(f"🔍 Zoom: {int(window.canvas.zoom_factor * 100)}%", 1500)

    def toggle_grid(self):
        """Toggle grid visibility."""
        window = self.window
        if hasattr(window, 'canvas'):
            window.canvas.toggle_grid()
            window.grid_toggle_action.setChecked(window.canvas.grid_visible)
            status = "shown" if window.canvas.grid_visible else "hidden"
            window.status_message.setText(f"Grid {status}")
            icon = "⊞" if window.canvas.grid_visible else "⊟"
            window.toast.show_message(f"{icon} Grid {status.capitalize()}")

    def toggle_minimap(self):
        """Toggle visibility of the minimap dock."""
        window = self.window
        if hasattr(window, 'minimap_dock'):
            visible = not window.minimap_dock.isVisible()
            window.minimap_dock.setVisible(visible)
            if visible:
                # Refresh minimap when shown
                window.minimap.refresh()
            if hasattr(window, 'minimap_action'):
                window.minimap_action.setChecked(visible)

    def fit_to_window(self):
        """Fit all blocks to window by auto-zooming and panning."""
        window = self.window
        if not hasattr(window, 'canvas'):
            return

        from PyQt5.QtCore import QPoint

        # Get all blocks
        blocks = window.canvas.dsim.blocks_list
        if not blocks:
            window.status_message.setText("No blocks to fit")
            return

        # Calculate bounding box of all blocks
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for block in blocks:
            min_x = min(min_x, block.left)
            min_y = min(min_y, block.top)
            max_x = max(max_x, block.left + block.width)
            max_y = max(max_y, block.top + block.height)

        # Add padding around blocks
        padding = 100
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        # Calculate diagram dimensions
        diagram_width = max_x - min_x
        diagram_height = max_y - min_y

        # Get canvas dimensions
        canvas_width = window.canvas.width()
        canvas_height = window.canvas.height()

        # Calculate zoom to fit
        zoom_x = canvas_width / diagram_width if diagram_width > 0 else 1.0
        zoom_y = canvas_height / diagram_height if diagram_height > 0 else 1.0
        new_zoom = min(zoom_x, zoom_y, 2.0)  # Cap at 200% max zoom

        # Apply zoom
        window.canvas.zoom_factor = max(0.1, min(new_zoom, 2.0))  # Clamp between 10% and 200%

        # Calculate center point of diagram
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate pan offset to center the diagram
        offset_x = canvas_width / 2 - center_x * window.canvas.zoom_factor
        offset_y = canvas_height / 2 - center_y * window.canvas.zoom_factor
        window.canvas.pan_offset = QPoint(int(offset_x), int(offset_y))

        # Update display
        window.canvas.update()
        window.zoom_status.setText(f"{int(window.canvas.zoom_factor * 100)}%")
        window.status_message.setText(f"Fit {len(blocks)} block(s) to window")
