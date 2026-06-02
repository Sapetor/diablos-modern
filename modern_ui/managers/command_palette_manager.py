"""
CommandPaletteManager -- builds the command-palette index and handles palette
actions (opening it, placing a block from a palette entry, the execution log
hook).

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

The command callbacks reference the window's own action methods (start/stop
simulation, zoom, file ops, etc.), so they are bound to ``self.window``.
Examples are indexed via ``lib.app_paths.resource_path('examples')`` -- the
canonical bundled-asset resolver -- which matches the previous ``__file__``
based path in development and is correct under PyInstaller too.
"""

import os
import logging

from lib.app_paths import resource_path

logger = logging.getLogger(__name__)


class CommandPaletteManager:
    """Owns the command-palette index and its action handlers."""

    def __init__(self, main_window):
        self.window = main_window

    def show(self):
        """Show the command palette for quick access."""
        window = self.window
        if hasattr(window, 'command_palette'):
            window.command_palette.show_palette()

    def setup(self):
        """Build the command palette index — blocks, sim, view, files, help."""
        window = self.window
        commands: list[dict] = []

        # Block library — typed as 'block' so the BLOCK badge surfaces
        if hasattr(window, 'canvas') and hasattr(window.canvas.dsim, 'menu_blocks'):
            for menu_block in window.canvas.dsim.menu_blocks:
                fn_name = getattr(menu_block, 'fn_name', '') or ''
                block_fn = getattr(menu_block, 'block_fn', '') or fn_name
                commands.append({
                    'name': f'Add {block_fn} block',
                    'type': 'block',
                    'description': f'{block_fn} ({fn_name})',
                    'aliases': [fn_name, block_fn, fn_name.lower()],
                    'callback': lambda mb=menu_block: self.add_block_from_palette_menu(mb),
                    'data': {'block_type': fn_name},
                })

        # Simulation actions (SIM badge)
        for label, kbd, cb in [
            ('Run simulation',   'F5', window.start_simulation),
            ('Pause simulation', 'F6', window.pause_simulation),
            ('Stop simulation',  'F7', window.stop_simulation),
            ('Step simulation',  'F8', window.step_simulation),
            ('Toggle fast solver', '', lambda: window.toggle_fast_solver(not getattr(window, 'use_fast_solver', True))),
        ]:
            commands.append({
                'name': label, 'type': 'sim', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # View toggles (VIEW badge)
        for label, kbd, cb in [
            ('Zoom in',       'Ctrl++',  window.zoom_in),
            ('Zoom out',      'Ctrl+-',  window.zoom_out),
            ('Fit to window', 'Ctrl+0',  window.fit_to_window),
            ('Toggle theme',  'Ctrl+T',  window.toggle_theme),
            ('Toggle grid',   'Ctrl+Shift+G', window.toggle_grid),
            ('Toggle minimap', 'Ctrl+Shift+M', window.toggle_minimap),
            ('Toggle variable editor', 'Ctrl+Shift+V', window.toggle_variable_editor),
            ('Toggle workspace variables', 'Ctrl+Shift+W', window.toggle_workspace_editor),
            ('Toggle tuning panel', '', window.toggle_tuning_panel),
        ]:
            commands.append({
                'name': label, 'type': 'view', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # File actions
        for label, kbd, cb in [
            ('New diagram',  'Ctrl+N', window.new_diagram),
            ('Open diagram', 'Ctrl+O', window.open_diagram),
            ('Save diagram', 'Ctrl+S', window.save_diagram),
            ('Load workspace…', '', window.load_workspace),
            ('Show plots',   '',       window.show_plots),
            ('Export as TikZ…', '',    window.export_tikz),
        ]:
            commands.append({
                'name': label, 'type': 'file', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # Index examples on disk — file paths only, load on click
        try:
            examples_dir = resource_path('examples')
            if os.path.isdir(examples_dir):
                for f in sorted(os.listdir(examples_dir)):
                    if f.endswith(('.json', '.dat', '.diablos')):
                        path = os.path.join(examples_dir, f)
                        commands.append({
                            'name': f'examples / {os.path.splitext(f)[0]}',
                            'type': 'file',
                            'callback': lambda p=path: window.open_example(p),
                            'data': {'path': path},
                        })
        except Exception:
            logger.debug("Could not index examples for command palette", exc_info=True)

        # Recent files
        try:
            recents = window._load_recent_files()
        except Exception:
            recents = []
        for path in recents[:6]:
            commands.append({
                'name': os.path.basename(path),
                'type': 'recent',
                'description': path,
                'callback': lambda p=path: window._open_recent_file(p),
                'data': {'path': path},
            })

        window.command_palette.set_commands(commands)

    def add_block_from_palette_menu(self, menu_block):
        """Add a block to the canvas from command palette."""
        window = self.window
        if not hasattr(window, 'canvas'):
            return

        from PyQt5.QtCore import QPoint
        from PyQt5.QtGui import QCursor

        canvas = window.canvas

        # Get current mouse position in global coordinates
        global_pos = QCursor.pos()

        # Convert to canvas widget coordinates
        canvas_widget_pos = canvas.mapFromGlobal(global_pos)

        # Check if mouse is within canvas bounds
        if canvas.rect().contains(canvas_widget_pos):
            # Convert screen coordinates to canvas coordinates (undo pan and zoom)
            canvas_x = int((canvas_widget_pos.x() - canvas.pan_offset.x()) / canvas.zoom_factor)
            canvas_y = int((canvas_widget_pos.y() - canvas.pan_offset.y()) / canvas.zoom_factor)
            canvas_pos = QPoint(canvas_x, canvas_y)
        else:
            # Fallback: add at center of visible canvas area
            center_x = canvas.width() // 2
            center_y = canvas.height() // 2

            # Convert screen coordinates to canvas coordinates (undo pan and zoom)
            canvas_x = int((center_x - canvas.pan_offset.x()) / canvas.zoom_factor)
            canvas_y = int((center_y - canvas.pan_offset.y()) / canvas.zoom_factor)
            canvas_pos = QPoint(canvas_x, canvas_y)

        # Add the block using the canvas method
        canvas.add_block_from_palette(menu_block, canvas_pos)
        window.toast.show_message(f"✅ Added {menu_block.block_fn} block")

    def on_command_executed(self, command_type: str, data: dict):
        """Handle command palette command execution."""
        logger.info(f"Command executed: {command_type}, data: {data}")
