"""
Modern Main Window for DiaBloS
Features modern layout, theming, and enhanced user interface.
"""

"""
Modern Main Window for DiaBloS
Features modern layout, theming, and enhanced user interface.
"""

import os
import logging
from typing import Any, Optional
from PyQt5.QtWidgets import (QMainWindow, QWidget,
                             QMessageBox, QFileDialog)
from lib.workspace import WorkspaceManager
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QFont

# Import existing DSim functionality
from lib.lib import DSim
from lib.improvements import (
    PerformanceHelper, SafetyChecks, LoggingHelper,
    SimulationConfig
)

# Import modern UI components
from modern_ui.themes.theme_manager import theme_manager, ThemeType
from modern_ui.widgets.toast_notification import ToastNotification
from modern_ui.widgets.command_palette import CommandPalette
from modern_ui.widgets.variable_editor import VariableEditor
from modern_ui.widgets.workspace_editor import WorkspaceEditor
from modern_ui.widgets.minimap_widget import MinimapWidget
from modern_ui.widgets.waveform_inspector import WaveformInspector
from modern_ui.widgets.tuning_panel import TuningPanel
from modern_ui.controllers.tuning_controller import TuningController

# Setup logging
LoggingHelper.setup_logging(level="INFO", log_file="diablos_modern.log")
logger = logging.getLogger(__name__)


class ModernDiaBloSWindow(QMainWindow):
    """Modern DiaBloS main window with enhanced UI."""
    
    def __init__(self, screen_geometry=None):
        super().__init__()
        logger.info("Starting Modern DiaBloS Application")

        # Store screen geometry for responsive sizing
        self.screen_geometry = screen_geometry

        # Default routing mode for new connections
        self.default_routing_mode = "bezier"
        self.use_fast_solver = True # Enabled by default

        # Core DSim functionality
        self.dsim = DSim()

        # Performance monitoring
        self.perf_helper = PerformanceHelper()

        # Simulation configuration
        self.sim_config = SimulationConfig()

        # Core Managers (Must be before state init)
        from modern_ui.managers.project_manager import ProjectManager
        self.project_manager = ProjectManager(self)

        from modern_ui.managers.recent_files_manager import RecentFilesManager
        self.recent_files_manager = RecentFilesManager(self)

        from modern_ui.managers.appearance_manager import AppearanceManager
        self.appearance_manager = AppearanceManager(self)

        from modern_ui.managers.status_bar_manager import StatusBarManager
        self.status_bar_manager = StatusBarManager(self)

        from modern_ui.managers.property_controller import PropertyController
        self.property_controller = PropertyController(self)

        from modern_ui.managers.command_palette_manager import CommandPaletteManager
        self.command_palette_manager = CommandPaletteManager(self)

        from modern_ui.managers.layout_manager import LayoutManager
        self.layout_manager = LayoutManager(self)

        from modern_ui.managers.window_setup_manager import WindowSetupManager
        self.window_setup_manager = WindowSetupManager(self)

        # Initialize state management (keeping from improved version)
        self._init_state_management()

        # Builders
        from modern_ui.builders.menu_builder import MenuBuilder
        self.menu_builder = MenuBuilder(self)

        # Setup modern UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self.toolbar.set_simulation_state(False, False)
        self._setup_layout()
        self._setup_statusbar()
        
        # Connect property editor signals
        self.property_editor.property_changed.connect(self._on_property_changed)
        self.property_editor.pin_to_tuning.connect(self._add_to_tuning)
        # Wire the inspector to dsim so its empty-state view can render the
        # diagram inspector (V1) rather than a blank placeholder.
        if hasattr(self.property_editor, 'set_diagram_context'):
            self.property_editor.set_diagram_context(self.dsim, self)
        
        # Initialize Variable Editor (Dockable)
        from PyQt5.QtWidgets import QDockWidget
        self.variable_editor = VariableEditor(self)
        self.variable_editor_dock = QDockWidget("Variable Editor", self)
        self.variable_editor_dock.setWidget(self.variable_editor)
        self.variable_editor_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.variable_editor_dock)
        self.variable_editor_dock.hide()  # Hidden by default
        
        # Connect variable editor signals
        self.variable_editor.variables_updated.connect(self._on_variables_updated)

        # Initialize Workspace Editor (Dockable)
        self.workspace_editor = WorkspaceEditor(self)
        self.workspace_editor_dock = QDockWidget("Workspace Variables", self)
        self.workspace_editor_dock.setWidget(self.workspace_editor)
        self.workspace_editor_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.workspace_editor_dock)
        self.workspace_editor_dock.hide()

        # Initialize Minimap (Dockable)
        self.minimap = MinimapWidget(self.canvas, self)
        self.minimap_dock = QDockWidget("Minimap", self)
        self.minimap_dock.setWidget(self.minimap)
        self.minimap_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.minimap_dock)
        self.minimap_dock.hide()  # Hidden by default

        # Create tuning panel and controller
        self.tuning_panel = TuningPanel()
        self.tuning_panel.hide()  # Hidden until first parameter is pinned
        # Insert into canvas area layout (after error panel)
        canvas_layout = self.canvas_area.layout()
        canvas_layout.addWidget(self.tuning_panel, 0)

        self.tuning_controller = TuningController(
            self.dsim, self.dsim.scope_plotter, parent=self
        )
        self.tuning_controller.set_status_callback(
            lambda msg: self.status_message.setText(msg)
        )
        self.tuning_panel.param_changed.connect(self.tuning_controller.on_param_changed)
        self.tuning_panel.panel_cleared.connect(self.tuning_controller.deactivate)

        # Create toast notification (after canvas is created)
        self.toast = ToastNotification(self.canvas)

        # Create command palette
        self.command_palette = CommandPalette(self)
        self.command_palette.command_selected.connect(self._on_command_executed)
        self._setup_command_palette()

        # Global ⌘K / Ctrl+K shortcut so the palette is reachable everywhere.
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        self._cmdk_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self._cmdk_shortcut.setContext(Qt.ApplicationShortcut)
        self._cmdk_shortcut.activated.connect(self.show_command_palette)

        # Initialize DSim components
        self.dsim.main_buttons_init()

        
        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.safe_update)
        self.update_timer.start(int(1000 / self.dsim.FPS))

        # Setup auto-save timer (every 2 minutes)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self._auto_save)
        self.autosave_timer.start(2 * 60 * 1000)  # 2 minutes in milliseconds
        from lib.app_paths import user_data_path
        self.autosave_path = user_data_path('config/.autosave.diablos')

        # Check for auto-save file on startup
        QTimer.singleShot(500, self._check_autosave_recovery)

        # Schedule initial splitter sizing after window is shown
        # This ensures we use actual window dimensions, not screen dimensions
        QTimer.singleShot(0, self._initialize_splitter_sizes)

        logger.info("Modern DiaBloS Window initialized successfully")
    
    def _init_state_management(self):
        """Initialize state management from improved version."""
        from enum import Enum, auto
        
        class State(Enum):
            IDLE = auto()
            DRAGGING = auto()
            CONNECTING = auto()
            CONFIGURING = auto()
        
        # State management
        # State management
        self.State = State
        self.state = State.IDLE
        self.dragging_block = None
        self.drag_offset = None

        # Initialize Services
        self.diagram_service = self.project_manager.diagram_service
        
        # Connection management
        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None
    
    # Window/menubar/toolbar facades -> WindowSetupManager (see managers/window_setup_manager.py)
    def _setup_window(self):
        """Setup main window properties with screen-aware sizing."""
        self.window_setup_manager.setup_window()

    def _setup_menubar(self):
        """Setup modern menu bar."""
        self.window_setup_manager.setup_menubar()

    def create_subsystem(self):
        """Create subsystem from selection (delegate to canvas)."""
        if hasattr(self, 'canvas') and hasattr(self.canvas, '_create_subsystem_trigger'):
            self.canvas._create_subsystem_trigger()

    def toggle_minimap(self):
        """Toggle visibility of the minimap dock."""
        if hasattr(self, 'minimap_dock'):
            visible = not self.minimap_dock.isVisible()
            self.minimap_dock.setVisible(visible)
            if visible:
                # Refresh minimap when shown
                self.minimap.refresh()
            if hasattr(self, 'minimap_action'):
                self.minimap_action.setChecked(visible)

    def _set_scaling(self, factor):
        import json
        from lib.app_paths import user_data_path
        config_path = user_data_path('config/default_config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}

        if 'display' not in config:
            config['display'] = {}
        config['display']['scaling_factor'] = factor

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        QMessageBox.information(self, "UI Scaling",
                                "The UI scaling factor has been changed. Please restart the application for the changes to take effect.")

    def _set_default_routing_mode(self, mode):
        """Set the default routing mode for new connections."""
        if mode in ["bezier", "orthogonal"]:
            self.default_routing_mode = mode

            # Update menu checkmarks
            self.bezier_routing_action.setChecked(mode == "bezier")
            self.orthogonal_routing_action.setChecked(mode == "orthogonal")

            # Pass the setting to the canvas
            if hasattr(self, 'canvas'):
                self.canvas.default_routing_mode = mode

            logger.info(f"Default connection routing mode set to: {mode}")

    def _setup_toolbar(self):
        """Setup modern toolbar."""
        self.window_setup_manager.setup_toolbar()

    # Layout/panel facades -> LayoutManager (see managers/layout_manager.py)
    def _setup_layout(self):
        """Setup modern layout with splitters."""
        self.layout_manager.setup_layout()

    def _initialize_splitter_sizes(self):
        """Initialize splitter sizes based on actual window dimensions."""
        self.layout_manager.initialize_splitter_sizes()

    def _create_left_panel(self) -> QWidget:
        """Create modern left panel for block palette."""
        return self.layout_manager.create_left_panel()

    def _create_canvas_area(self) -> QWidget:
        """Create modern canvas area with responsive sizing and error panel."""
        return self.layout_manager.create_canvas_area()

    def _create_property_panel(self) -> QWidget:
        """Create modern property panel on right side with size constraints."""
        return self.layout_manager.create_property_panel()

    def eventFilter(self, obj, event):
        """Forward focus to child input widgets inside the property scroll area.
        PyQt5 5.15: QScrollArea viewport absorbs clicks without focusing children."""
        if obj is self._prop_scroll_viewport and event.type() == QEvent.MouseButtonPress:
            child = obj.childAt(event.pos())
            # Walk up to find the first focusable widget
            while child and child.focusPolicy() == Qt.NoFocus:
                child = child.parentWidget()
            if child and child is not obj:
                child.setFocus(Qt.MouseFocusReason)
        return super().eventFilter(obj, event)

    # Status-bar facades -> StatusBarManager (see managers/status_bar_manager.py)
    def _setup_statusbar(self):
        """Build the compact pill-style status bar."""
        self.status_bar_manager.setup()

    def _refresh_status_counts(self):
        """Update the counts pill from current dsim state."""
        self.status_bar_manager.refresh_counts()

    def _refresh_file_status(self):
        """Update filename + unsaved indicator in the status bar."""
        self.status_bar_manager.refresh_file_status()
    
    def paintEvent(self, event):
        """Paint event - delegated to canvas widget."""
        pass

    # Toolbar action handlers (delegation to project_manager)
    def open_diagram(self):
        self.project_manager.open_diagram()

    def open_example(self, filename):
        self.project_manager.open_example(filename)

    def save_diagram(self):
        self.project_manager.save_diagram()

    def export_tikz(self):
        """Open the TikZ export dialog."""
        from PyQt5.QtWidgets import QMessageBox
        if not self.dsim.blocks_list:
            QMessageBox.information(self, "Export TikZ", "No blocks to export.")
            return
        from modern_ui.widgets.tikz_export_dialog import TikZExportDialog
        TikZExportDialog(self.dsim.blocks_list, self.dsim.line_list, parent=self).exec_()

    def pause_simulation(self):
        """Pause simulation."""
        if hasattr(self.dsim, 'execution_pause'):
            self.dsim.execution_pause = True
        self.toolbar.set_simulation_state(True, True)

    def step_simulation(self):
        """Execute a single timestep of the simulation.

        If simulation is not running, it will be initialized first,
        allowing step-by-step execution from t=0.
        """
        if not hasattr(self.dsim, 'single_step'):
            self.status_message.setText("Single-step not available")
            return

        # Check if this is the first step (will initialize)
        was_initialized = self.dsim.execution_initialized

        success = self.dsim.single_step()
        if success:
            if not was_initialized:
                self.status_message.setText(f"Started stepping at t={self.dsim.time_step:.4f}s")
            else:
                self.status_message.setText(f"Stepped to t={self.dsim.time_step:.4f}s")
            self.canvas.update()
            # Keep toolbar in paused state (step always pauses)
            self.toolbar.set_simulation_state(True, True)
        else:
            # Check if simulation ended or failed to start
            if not self.dsim.execution_initialized:
                self.toolbar.set_simulation_state(False, False)
                if was_initialized:
                    self.status_message.setText("Simulation finished")
                else:
                    self.status_message.setText("Failed to initialize simulation")
            else:
                self.status_message.setText("Step failed")

    def show_plots(self):
        """Show plots."""
        if not hasattr(self.dsim, 'run_history'):
            return
        history = getattr(self.dsim, 'run_history', [])
        if not history:
            QMessageBox.information(self, "Waveform Inspector", "No scope data available yet.")
            return

        if not hasattr(self, 'waveform_inspector_dock'):
            from PyQt5.QtWidgets import QDockWidget
            self.waveform_inspector = WaveformInspector(self.dsim)
            self.waveform_inspector_dock = QDockWidget("Waveforms", self)
            self.waveform_inspector_dock.setWidget(self.waveform_inspector)
            self.waveform_inspector_dock.setAllowedAreas(
                Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea
            )
            self.addDockWidget(Qt.BottomDockWidgetArea, self.waveform_inspector_dock)

            if hasattr(self, 'variable_editor_dock'):
                self.tabifyDockWidget(self.variable_editor_dock, self.waveform_inspector_dock)
        self.waveform_inspector_dock.show()
        self.waveform_inspector_dock.raise_()
    
    def capture_screen(self):
        """Capture screenshot."""
        if hasattr(self.dsim, 'screenshot'):
            self.dsim.screenshot(self)
    
    def set_zoom(self, factor: float):
        """Set zoom factor."""
        if hasattr(self, 'canvas'):
            self.canvas.set_zoom(factor)
            self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")
    
    # Appearance facades -> AppearanceManager (see managers/appearance_manager.py)
    def toggle_theme(self):
        """Toggle theme and persist the choice."""
        self.appearance_manager.toggle_theme()

    def _set_palette(self, palette_key: str):
        """Switch the active block-color palette and persist the choice."""
        self.appearance_manager.set_palette(palette_key)

    def _toggle_solid_fills(self, checked: bool):
        """Toggle solid block fills and persist the choice."""
        self.appearance_manager.toggle_solid_fills(checked)

    def _save_user_preferences(self):
        """Persist all UI preferences (theme, palette, solid_fills) to user_preferences.json."""
        self.appearance_manager.save_preferences()

    def on_theme_changed(self):
        """Handle theme changes."""
        self.appearance_manager.on_theme_changed()

    def _update_statusbar_colors(self):
        """Apply theme to the status bar shell. Individual pills own their own styles."""
        self.appearance_manager.update_statusbar_colors()

    def _update_menubar_colors(self):
        """Update menubar colors for proper contrast."""
        self.appearance_manager.update_menubar_colors()

    # Menu action handlers (simplified for Phase 1)
    def undo_action(self):
        """Undo last action."""
        if hasattr(self, 'canvas'):
            self.canvas.undo()
            self.status_message.setText("Undo")
            self.toast.show_message("⟲ Undo")

    def redo_action(self):
        """Redo last undone action."""
        if hasattr(self, 'canvas'):
            self.canvas.redo()
            self.status_message.setText("Redo")
            self.toast.show_message("⟳ Redo")

    def select_all(self):
        """Select all blocks in the diagram."""
        if hasattr(self, 'canvas'):
            # Deselect all lines first
            for line in self.canvas.dsim.line_list:
                line.selected = False

            # Select all blocks
            selected_count = 0
            for block in self.canvas.dsim.blocks_list:
                block.selected = True
                selected_count += 1

            # Update canvas to show selection
            self.canvas.update()

            if selected_count > 0:
                self.status_message.setText(f"Selected {selected_count} block(s)")
            else:
                self.status_message.setText("No blocks to select")

    def zoom_in(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_in()
            self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")
            self.toast.show_message(f"🔍 Zoom: {int(self.canvas.zoom_factor * 100)}%", 1500)

    def zoom_out(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_out()
            self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")
            self.toast.show_message(f"🔍 Zoom: {int(self.canvas.zoom_factor * 100)}%", 1500)

    def toggle_grid(self):
        """Toggle grid visibility."""
        if hasattr(self, 'canvas'):
            self.canvas.toggle_grid()
            self.grid_toggle_action.setChecked(self.canvas.grid_visible)
            status = "shown" if self.canvas.grid_visible else "hidden"
            self.status_message.setText(f"Grid {status}")
            icon = "⊞" if self.canvas.grid_visible else "⊟"
            self.toast.show_message(f"{icon} Grid {status.capitalize()}")

    # Command-palette facades -> CommandPaletteManager (see managers/command_palette_manager.py)
    def show_command_palette(self):
        """Show the command palette for quick access."""
        self.command_palette_manager.show()

    def _setup_command_palette(self):
        """Build the command palette index — blocks, sim, view, files, help."""
        self.command_palette_manager.setup()

    def _add_block_from_palette_menu(self, menu_block):
        """Add a block to the canvas from command palette."""
        self.command_palette_manager.add_block_from_palette_menu(menu_block)

    def _on_command_executed(self, command_type: str, data: dict):
        """Handle command palette command execution."""
        self.command_palette_manager.on_command_executed(command_type, data)

    def fit_to_window(self):
        """Fit all blocks to window by auto-zooming and panning."""
        if not hasattr(self, 'canvas'):
            return

        from PyQt5.QtCore import QPoint

        # Get all blocks
        blocks = self.canvas.dsim.blocks_list
        if not blocks:
            self.status_message.setText("No blocks to fit")
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
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()

        # Calculate zoom to fit
        zoom_x = canvas_width / diagram_width if diagram_width > 0 else 1.0
        zoom_y = canvas_height / diagram_height if diagram_height > 0 else 1.0
        new_zoom = min(zoom_x, zoom_y, 2.0)  # Cap at 200% max zoom

        # Apply zoom
        self.canvas.zoom_factor = max(0.1, min(new_zoom, 2.0))  # Clamp between 10% and 200%

        # Calculate center point of diagram
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Calculate pan offset to center the diagram
        offset_x = canvas_width / 2 - center_x * self.canvas.zoom_factor
        offset_y = canvas_height / 2 - center_y * self.canvas.zoom_factor
        self.canvas.pan_offset = QPoint(int(offset_x), int(offset_y))

        # Update display
        self.canvas.update()
        self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")
        self.status_message.setText(f"Fit {len(blocks)} block(s) to window")
    
    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setStretchLastSection(True)

        # Define shortcuts (category, shortcut, description)
        shortcuts = [
            ("File Operations", "", ""),
            ("", "Ctrl+N", "New Diagram"),
            ("", "Ctrl+O", "Open Diagram"),
            ("", "Ctrl+S", "Save Diagram"),
            ("", "", ""),
            ("Editing", "", ""),
            ("", "Ctrl+Z / Cmd+Z", "Undo"),
            ("", "Ctrl+Y / Cmd+Shift+Z", "Redo"),
            ("", "Ctrl+A", "Select All"),
            ("", "Ctrl+C", "Copy"),
            ("", "Ctrl+V", "Paste"),
            ("", "Ctrl+D", "Duplicate"),
            ("", "Delete / Backspace", "Delete Selected"),
            ("", "Ctrl+F", "Flip Block"),
            ("", "", ""),
            ("View", "", ""),
            ("", "Ctrl++", "Zoom In"),
            ("", "Ctrl+-", "Zoom Out"),
            ("", "Ctrl+0", "Fit to Window"),
            ("", "Middle Mouse", "Pan Canvas"),
            ("", "", ""),
            ("Simulation", "", ""),
            ("", "F5", "Run Simulation"),
            ("", "F6", "Pause Simulation"),
            ("", "F7", "Stop Simulation"),
            ("", "", ""),
            ("Canvas", "", ""),
            ("", "Esc", "Cancel Operation"),
            ("", "Right Click", "Context Menu"),
        ]

        table.setRowCount(len(shortcuts))

        for i, (category, shortcut, action) in enumerate(shortcuts):
            if category:  # Category header
                cat_item = QTableWidgetItem(category)
                cat_item.setFont(QFont("Arial", 10, QFont.Bold))
                table.setItem(i, 0, cat_item)
                table.setSpan(i, 0, 1, 2)
            else:
                table.setItem(i, 0, QTableWidgetItem(shortcut))
                table.setItem(i, 1, QTableWidgetItem(action))

        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)

        layout.addWidget(table)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.exec_()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About DiaBloS",
                         "DiaBloS - Modern Block Diagram Simulator\n"
                         "Phase 2: Interactive Block Canvas\n"
                         "✅ Modern UI Foundation\n"
                         "✅ Interactive Block Palette\n"
                         "✅ Drag-and-Drop Block Creation\n"
                         "✅ Modern Canvas with Mouse Events\n"
                         "Built with PyQt5 and modern design principles")
    

    def toggle_variable_editor(self):
        """Toggle Variable Editor visibility."""
        if self.variable_editor_dock.isVisible():
            self.variable_editor_dock.hide()
            self.variable_editor_action.setChecked(False)
            self.toast.show_message("Variable Editor hidden")
        else:
            self.variable_editor_dock.show()
            self.variable_editor_action.setChecked(True)
            self.toast.show_message("Variable Editor shown")

    def toggle_workspace_editor(self):
        """Toggle Workspace Editor visibility."""
        if hasattr(self, 'workspace_editor_dock'):
            if self.workspace_editor_dock.isVisible():
                self.workspace_editor_dock.hide()
                if hasattr(self, 'workspace_editor_action'):
                    self.workspace_editor_action.setChecked(False)
                self.toast.show_message("Workspace Variables hidden")
            else:
                self.workspace_editor_dock.show()
                if hasattr(self, 'workspace_editor_action'):
                    self.workspace_editor_action.setChecked(True)
                self.toast.show_message("Workspace Variables shown")
    
    def _on_variables_updated(self):
        """Handle variable updates from the Variable Editor."""
        try:
            var_count = len(WorkspaceManager().variables)
            
            # Refresh the Workspace Editor (table view)
            if hasattr(self, 'workspace_editor'):
                self.workspace_editor.refresh_variables()

            # Refresh the Property Editor's diagram-inspector Workspace section
            # (only re-renders when no block is selected — block-state view is unaffected)
            if hasattr(self, 'property_editor') and self.property_editor.block is None:
                self.property_editor.set_block(None)

            self.toast.show_message(f"✓ Workspace updated ({var_count} variables)", duration=2000)
            self.status_message.setText(f"Workspace updated with {var_count} variable(s)")
            logger.info(f"Workspace updated from Variable Editor: {var_count} variables")
        except Exception as e:
            logger.error(f"Error handling variable update: {str(e)}")
            self.toast.show_message(f"Error updating workspace: {str(e)}", duration=3000, is_error=True)
    
    def closeEvent(self, event):
        """Handle application shutdown."""
        logger.info("Modern DiaBloS closing...")
        self.stop_simulation()
        self._cleanup_autosave()  # Clean up auto-save file on normal exit
        self.perf_helper.log_stats()
        event.accept()
        logger.info("Modern DiaBloS closed successfully")
    
    # Phase 2 Signal Handlers
    def _on_block_selected(self, block):
        """Handle block selection from canvas."""
        try:
            if block is None:
                self.status_message.setText("")
                self.property_editor.set_block(None)
                return
            block_name = getattr(block, 'fn_name', 'Unknown')
            logger.info(f"Block selected: {block_name}")
            self.status_message.setText(f"Selected: {block_name}")

            # Update property panel with block properties
            self.property_editor.set_block(block)

        except Exception as e:
            logger.error(f"Error handling block selection: {str(e)}")
    
    def _on_connection_created(self, source_block, target_block):
        """Handle connection creation between blocks."""
        try:
            source_name = getattr(source_block, 'fn_name', 'Unknown')
            target_name = getattr(target_block, 'fn_name', 'Unknown')
            logger.info(f"Connection created: {source_name} -> {target_name}")
            self.status_message.setText(f"Connected {source_name} to {target_name}")
            
        except Exception as e:
            logger.error(f"Error handling connection creation: {str(e)}")
    
    def _on_simulation_status_changed(self, status):
        """Handle simulation status changes from canvas."""
        try:
            self.status_message.setText(status)
            logger.info(f"Simulation status: {status}")

            if "finished" in status.lower() or "stopped" in status.lower() or "failed" in status.lower():
                self.toolbar.set_simulation_state(False, False)
            elif "started" in status.lower() or "running" in status.lower():
                self.toolbar.set_simulation_state(True, False)
            
        except Exception as e:
            logger.error(f"Error handling simulation status change: {str(e)}")

    def show_error(self, message):
        """Show an error message to the user."""
        logger.error(message)
        if hasattr(self, 'toast'):
            self.toast.show_message(message, duration=5000, is_error=True)
        else:
            QMessageBox.critical(self, "Error", message)

    # Property/param facades -> PropertyController (see managers/property_controller.py)
    def _convert_param_value(self, new_value, target_type):
        """Convert a parameter value to the target type, handling variables."""
        return self.property_controller.convert_param_value(new_value, target_type)

    def _on_property_changed(self, block_name: str, prop_name: str, new_value: Any) -> None:
        """Handle property changes from the property editor."""
        self.property_controller.on_property_changed(block_name, prop_name, new_value)

    def _add_to_tuning(self, block, param_name):
        """Add a block parameter to the tuning panel."""
        self.property_controller.add_to_tuning(block, param_name)

    def toggle_tuning_panel(self):
        """Toggle the tuning panel visibility."""
        if self.tuning_panel.isVisible():
            self.tuning_panel.hide()
        else:
            self.tuning_panel.show()

    def _on_error_clicked(self, error):
        """Handle error item click - navigate to error location."""
        try:
            from PyQt5.QtCore import QPoint

            # Get affected blocks from the error
            affected_blocks = error.blocks if hasattr(error, 'blocks') else []

            if not affected_blocks:
                logger.warning("No blocks associated with this error")
                return

            # First, deselect all blocks
            for block in self.canvas.dsim.blocks_list:
                block.selected = False

            # Calculate bounding box of all affected blocks
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')

            for block in affected_blocks:
                # Get block position using correct attribute names
                x = block.left
                y = block.top
                w = block.width
                h = block.height

                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

                # Select the affected blocks for visibility
                block.selected = True

            # Add padding
            padding = 50
            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            # Calculate center point
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2

            # Pan canvas to center on the error location
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()

            # Calculate new pan offset to center the error (using QPoint)
            new_offset_x = canvas_width / 2 - center_x * self.canvas.zoom_factor
            new_offset_y = canvas_height / 2 - center_y * self.canvas.zoom_factor
            self.canvas.pan_offset = QPoint(int(new_offset_x), int(new_offset_y))

            # Update canvas to show the changes
            self.canvas.update()

            logger.info(f"Navigated to error location at ({center_x}, {center_y})")
            self.status_message.setText(f"Showing error: {error.message}")

        except Exception as e:
            logger.error(f"Error navigating to error location: {str(e)}")
    
    # Update safe_update to use the new canvas
    def safe_update(self):
        """Safe update with error handling and performance monitoring."""
        try:
            self.perf_helper.start_timer("ui_update")
            
            if hasattr(self, 'canvas'):
                was_running = self.canvas.is_simulation_running()

                if was_running:
                    is_safe, errors = SafetyChecks.check_simulation_state(self.canvas.dsim)
                    if not is_safe:
                        logger.error(f"Simulation state unsafe: {errors}")
                        self.canvas.stop_simulation()
                        self.toolbar.set_simulation_state(False, False)
                        return
                    
                    self.perf_helper.start_timer("simulation_step")
                    if hasattr(self.canvas.dsim, 'execution_loop'):
                        self.canvas.dsim.execution_loop()
                    step_duration = self.perf_helper.end_timer("simulation_step")
                    
                    if step_duration and step_duration > 0.1:
                        logger.warning(f"Slow simulation step: {step_duration:.4f}s")

                self.canvas.update()

                # Refresh minimap if visible
                if hasattr(self, 'minimap_dock') and self.minimap_dock.isVisible():
                    self.minimap.refresh()

                if was_running and hasattr(self.toolbar, 'set_simulation_time'):
                    t_now = getattr(self.dsim, 'time_step', 0.0)
                    t_end = getattr(self.dsim, 'sim_time', 10.0) or 10.0
                    self.toolbar.set_simulation_time(t_now, t_end)

                is_running = self.canvas.is_simulation_running()

                if was_running and not is_running:
                    self.toolbar.set_simulation_state(False, False)
                    self.status_message.setText("Simulation finished")
                    # Arm tuning controller with sim params from completed run
                    self.tuning_controller.store_sim_params(
                        self.dsim.sim_time, self.dsim.sim_dt
                    )
            
            self.perf_helper.end_timer("ui_update")
            
        except Exception as e:
            logger.error(f"Error in safe_update: {str(e)}")
            if hasattr(self, 'canvas'):
                self.canvas.stop_simulation()
                self.toolbar.set_simulation_state(False, False)
    
    # Override toolbar actions to use canvas methods
    def new_diagram(self):
        """Create new diagram."""
        if hasattr(self, 'canvas'):
            self.canvas.clear_canvas()
        self.status_message.setText("New diagram created")
    
    def start_simulation(self) -> None:
        """Start simulation with validation."""
        if not hasattr(self, 'canvas'):
            self.status_message.setText("Canvas not available")
            return

        # Run diagram validation first
        from lib.diagram_validator import ErrorSeverity

        logger.info("Running pre-simulation validation...")
        errors = self.canvas.run_validation()

        # Check for critical errors that block simulation
        has_errors = any(e.severity == ErrorSeverity.ERROR for e in errors)

        if errors:
            # Show error panel with results
            self.error_panel.set_errors(errors)

            if has_errors:
                # Critical errors found - don't start simulation
                error_count = sum(1 for e in errors if e.severity == ErrorSeverity.ERROR)
                self.status_message.setText(f"Cannot start simulation: {error_count} error(s) found")
                logger.warning(f"Simulation blocked by {error_count} validation error(s)")

                # Show a message box for critical errors
                QMessageBox.warning(
                    self,
                    "Validation Errors",
                    f"Cannot start simulation due to {error_count} validation error(s).\n\n"
                    f"Please fix the errors shown in the error panel before running."
                )
                return
            else:
                # Only warnings - allow simulation but notify user
                warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)
                logger.info(f"Starting simulation with {warning_count} warning(s)")
                self.status_message.setText(f"Starting simulation with {warning_count} warning(s)...")
        else:
            # No errors or warnings - clear error panel
            self.error_panel.clear()
            logger.info("Validation passed - no errors or warnings")
            self.status_message.setText("Starting simulation...")

        # Clear validation indicators from canvas before starting
        # (errors will be shown in panel, don't need red borders during simulation)
        self.canvas.clear_validation()

        # Start the simulation
        # Start the simulation
        # Check fast solver preference
        if hasattr(self, 'use_fast_solver'):
             self.dsim.use_fast_solver = self.use_fast_solver
             
        self.canvas.start_simulation()

        # Arm tuning controller after batch simulation completes
        # (safe_update timer can't detect batch completion since it runs synchronously)
        if not self.canvas.is_simulation_running():
            sim_time = getattr(self.dsim, 'sim_time', None)
            sim_dt = getattr(self.dsim, 'sim_dt', None)
            if sim_time and sim_dt:
                self.tuning_controller.store_sim_params(sim_time, sim_dt)

    def stop_simulation(self):
        """Stop simulation."""
        if hasattr(self, 'canvas'):
            self.canvas.stop_simulation()
        self.toolbar.set_simulation_state(False, False)
        self.status_message.setText("Simulation stopped")

    def toggle_fast_solver(self, checked):
        """Toggle fast solver mode."""
        self.use_fast_solver = checked
        if hasattr(self, 'dsim'):
            self.dsim.use_fast_solver = checked
        logger.info(f"Fast Solver enabled: {checked}")

    # Recent Files Management
    # Recent-files facades -> RecentFilesManager (see managers/recent_files_manager.py)
    def _load_recent_files(self):
        """Load recent files list from config."""
        return self.recent_files_manager.load()

    def _save_recent_files(self, recent_files):
        """Save recent files list to config."""
        self.recent_files_manager.save(recent_files)

    def _add_recent_file(self, filepath):
        """Add a file to the recent files list."""
        self.recent_files_manager.add(filepath)

    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_files_manager.update_menu()

    def _open_recent_file(self, filepath):
        """Open a file from the recent files list."""
        self.recent_files_manager.open(filepath)

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self.recent_files_manager.clear()

    # Auto-Save and Recovery
    def _auto_save(self):
        """Auto-save the current diagram to a temporary file."""
        try:
            # Only auto-save if there are blocks or connections
            if not self.canvas.dsim.blocks_list and not self.canvas.dsim.line_list:
                return

            # Create config directory if it doesn't exist
            os.makedirs('config', exist_ok=True)

            # Save using file_service for proper JSON format
            if hasattr(self.dsim, 'file_service'):
                self.dsim.file_service.save(
                    autosave=True, 
                    modern_ui_data={
                        'theme': theme_manager.current_theme.value,
                        'zoom_factor': self.canvas.zoom_factor if hasattr(self.canvas, 'zoom_factor') else 1.0
                    },
                    filepath=self.autosave_path
                )
            else:
                # Fallback: use dsim.save
                self.dsim.save(autosave=True, filepath=self.autosave_path)

            logger.debug("Auto-save completed")

        except Exception as e:
            logger.error(f"Error during auto-save: {str(e)}")

    def _check_autosave_recovery(self):
        self.project_manager.check_autosave_recovery()

    def _cleanup_autosave(self):
        self.project_manager.cleanup_autosave()

    def load_workspace(self):
        """Load variables from a workspace file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Workspace", "", "Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            try:
                WorkspaceManager().load_from_file(filepath)
                self._on_variables_updated()
                self.toast.show_message(f"Workspace loaded from {os.path.basename(filepath)}", duration=3000)
            except Exception as e:
                self.toast.show_message(f"Failed to load workspace: {str(e)}", duration=5000, is_error=True)
                logger.error(f"Failed to load workspace: {str(e)}")
