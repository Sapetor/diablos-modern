# UI Widgets API

PyQt5 widgets for the DiaBloS Modern interface.

## ModernDiaBloSWindow

Main application window.

::: modern_ui.main_window.ModernDiaBloSWindow
    options:
      show_root_heading: true

## ModernCanvas

Canvas widget for block diagram editing.

::: modern_ui.widgets.modern_canvas.ModernCanvas
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

::: modern_ui.managers.clipboard_manager.ClipboardManager
    options:
      show_root_heading: true

### ZoomPanManager

Handles zoom and pan operations.

::: modern_ui.managers.zoom_pan_manager.ZoomPanManager
    options:
      show_root_heading: true

### HistoryManager

Handles undo/redo operations.

::: modern_ui.managers.history_manager.HistoryManager
    options:
      show_root_heading: true

### SelectionManager

Handles block selection.

::: modern_ui.managers.selection_manager.SelectionManager
    options:
      show_root_heading: true
