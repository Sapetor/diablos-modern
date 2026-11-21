# UI Widgets API

PyQt6 widgets for the DiaBloS Modern interface.

## ModernDiaBloSWindow

Main application window.

::: diablos.gui.main_window.ModernDiaBloSWindow
    options:
      show_root_heading: true

## ModernCanvas

Canvas widget for block diagram editing.

::: diablos.gui.widgets.modern_canvas.ModernCanvas
    options:
      show_root_heading: true
      members:
        - __init__
        - copy_selected_blocks
        - paste_blocks
        - zoom_in
        - zoom_out
        - zoom_to_fit

## Managers

### ClipboardManager

Handles copy/paste operations.

::: diablos.gui.managers.clipboard_manager.ClipboardManager
    options:
      show_root_heading: true

### ZoomPanManager

Handles zoom and pan operations.

::: diablos.gui.managers.zoom_pan_manager.ZoomPanManager
    options:
      show_root_heading: true

### HistoryManager

Handles undo/redo operations.

::: diablos.gui.managers.history_manager.HistoryManager
    options:
      show_root_heading: true

### SelectionManager

Handles block selection.

::: diablos.gui.managers.selection_manager.SelectionManager
    options:
      show_root_heading: true
