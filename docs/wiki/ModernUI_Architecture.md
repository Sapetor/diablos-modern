# Modern UI Architecture

## Overview
The Modern UI is built with PyQt5 and follows a component-based architecture.

## Theme System
Themes are managed by the `ThemeManager` in `modern_ui/themes/theme_manager.py`.
*   **Access**: Use `theme_manager.get_current_theme()` to retrieve the active color dictionary.
*   **Colors**: Define semantic names like `text_primary`, `accent_primary`, `block_control`.
*   **Switching**: Emits `theme_changed` signal when toggled. All modern widgets subscribe to this or check it during paint events.

## Canvas Interaction
The `ModernCanvas` handles rendering, events, and tool interactions.
*   **Subsystems**:
    *   `navigate_scope(index)`: Jumps to a specific hierarchy level.
    *   `scope_changed(path)`: Signal emitted when entering/exiting subsystems.

## Navigation Widgets
### BreadcrumbBar
Located above the canvas, it displays the current scope path (e.g., `Main > Subsystem 1`).
*   **Integration**: Connects to `ModernCanvas.scope_changed` to update labels.
*   **Interaction**: Emits `path_clicked` which triggers `ModernCanvas.navigate_scope`.

## Layout
`MainWindow` uses `QSplitter` to manage:
1.  **Left Panel**: Block Palette.
2.  **Center Area**: Breadcrumb Bar + Modern Canvas (in a VBox).
3.  **Right Panel**: Property Editor.
