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
from collections import OrderedDict

from lib.app_paths import resource_path

logger = logging.getLogger(__name__)


# Static (label, shortcut) tables for the action commands, grouped by the badge
# they surface under in the palette. These are the single source of truth for
# both the command palette (which pairs each label with a window callback in
# ``setup``) and the keyboard-shortcuts reference dialog (which reads them via
# ``palette_command_groups`` so it cannot drift). An empty shortcut string means
# "no default binding".
_SIM_COMMANDS: list[tuple[str, str]] = [
    ('Run simulation',     'F5'),
    ('Pause simulation',   'F6'),
    ('Stop simulation',    'F7'),
    ('Step simulation',    'F8'),
    ('Toggle fast solver', ''),
]

_VIEW_COMMANDS: list[tuple[str, str]] = [
    ('Zoom in',       'Ctrl++'),
    ('Zoom out',      'Ctrl+-'),
    ('Fit to window', 'Ctrl+0'),
    ('Toggle theme',  'Ctrl+T'),
    ('Toggle grid',   'Ctrl+Shift+G'),
    ('Toggle minimap', 'Ctrl+Shift+M'),
    ('Toggle variable editor', 'Ctrl+Shift+V'),
    ('Toggle workspace variables', 'Ctrl+Shift+W'),
    ('Toggle tuning panel', ''),
]

_FILE_COMMANDS: list[tuple[str, str]] = [
    ('New diagram',  'Ctrl+N'),
    ('Open diagram', 'Ctrl+O'),
    ('Save diagram', 'Ctrl+S'),
    ('Load workspace…', ''),
    ('Show plots',   ''),
    ('Export as TikZ…', ''),
]


def palette_command_groups() -> "OrderedDict[str, list[tuple[str, str]]]":
    """Return the palette's action commands as group -> [(label, shortcut)].

    Ordered Simulation, View, File to match the palette's build order in
    ``CommandPaletteManager.setup``. The reference dialog consumes this so the
    two can never drift apart. Returns copies so callers can't mutate the
    shared tables.
    """
    return OrderedDict([
        ('Simulation', list(_SIM_COMMANDS)),
        ('View', list(_VIEW_COMMANDS)),
        ('File', list(_FILE_COMMANDS)),
    ])


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

        # Action commands (sim/view/file). Labels and shortcuts live in the
        # module-level tables (shared with the shortcuts dialog); the callbacks
        # are bound here against this window. Each group keeps the same order as
        # its table so the (label, shortcut) and callback rows line up.
        sim_callbacks = [
            window.start_simulation,
            window.pause_simulation,
            window.stop_simulation,
            window.step_simulation,
            lambda: window.toggle_fast_solver(not getattr(window, 'use_fast_solver', True)),
        ]
        view_callbacks = [
            window.zoom_in,
            window.zoom_out,
            window.fit_to_window,
            window.toggle_theme,
            window.toggle_grid,
            window.toggle_minimap,
            window.toggle_variable_editor,
            window.toggle_workspace_editor,
            window.toggle_tuning_panel,
        ]
        file_callbacks = [
            window.new_diagram,
            window.open_diagram,
            window.save_diagram,
            window.load_workspace,
            window.show_plots,
            window.export_tikz,
        ]
        for badge, table, callbacks in [
            ('sim', _SIM_COMMANDS, sim_callbacks),
            ('view', _VIEW_COMMANDS, view_callbacks),
            ('file', _FILE_COMMANDS, file_callbacks),
        ]:
            for (label, kbd), cb in zip(table, callbacks):
                commands.append({
                    'name': label, 'type': badge, 'shortcut': kbd,
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
