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
import ast
from typing import Any, Optional
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QLabel, QFrame,
                             QMessageBox, QScrollArea, QFileDialog)
from lib.workspace import WorkspaceManager
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QFont, QColor

# Import existing DSim functionality
from lib.lib import DSim
from lib.improvements import (
    PerformanceHelper, SafetyChecks, LoggingHelper,
    SimulationConfig
)

# Import modern UI components
from modern_ui.themes.theme_manager import theme_manager, ThemeType
from modern_ui.widgets.modern_toolbar import ModernToolBar
from modern_ui.widgets.modern_canvas import ModernCanvas
from modern_ui.widgets.modern_palette import ModernBlockPalette
from modern_ui.widgets.property_editor import PropertyEditor
from modern_ui.widgets.toast_notification import ToastNotification
from modern_ui.widgets.error_panel import ErrorPanel
from modern_ui.widgets.command_palette import CommandPalette
from modern_ui.widgets.variable_editor import VariableEditor
from modern_ui.widgets.workspace_editor import WorkspaceEditor
from modern_ui.widgets.minimap_widget import MinimapWidget
from modern_ui.widgets.waveform_inspector import WaveformInspector
from modern_ui.widgets.breadcrumb_bar import BreadcrumbBar
from modern_ui.widgets.tuning_panel import TuningPanel
from modern_ui.controllers.tuning_controller import TuningController
from modern_ui.platform_config import get_platform_config

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
    
    def _setup_window(self):
        """Setup main window properties with screen-aware sizing."""
        self.setWindowTitle("DiaBloS - Modern Block Diagram Simulator")

        # Get platform configuration
        config = get_platform_config()

        # Calculate responsive window size based on available screen space
        if self.screen_geometry:
            target_width = int(self.screen_geometry.width() * config.window_width_percent)
            target_height = int(self.screen_geometry.height() * config.window_height_percent)

            # On standard DPI, cap window size to avoid giant windows
            if config.should_cap_window_size:
                target_width = min(target_width, 1600)
                target_height = min(target_height, 1000)

            # Set minimum size
            min_width = max(int(target_width * 0.70), config.window_min_width)
            min_height = max(int(target_height * 0.70), config.window_min_height)

            self.setMinimumSize(min_width, min_height)
            self.resize(target_width, target_height)

            logger.info(f"Window sizing: target={target_width}×{target_height}, min={min_width}×{min_height}")
        else:
            # Fallback to larger sizes
            self.setMinimumSize(1200, 800)
            self.resize(1600, 1000)

        # Set modern font
        font = QFont("Segoe UI", 10)
        self.setFont(font)

        # Apply modern theme
        self.setObjectName("ModernMainWindow")
    
    def _setup_menubar(self):
        """Setup modern menu bar."""
        if self.menu_builder:
            self.menu_builder.setup_menubar()

    def create_subsystem(self):
        """Create subsystem from selection (delegate to canvas)."""
        if hasattr(self, 'canvas') and hasattr(self.canvas, '_create_subsystem_trigger'):
            self.canvas._create_subsystem_trigger()

    def toggle_variable_editor(self):
        # Implementation depends on logic elsewhere, ensuring method exists
        if hasattr(self, 'variable_editor'):
             visible = not self.variable_editor.isVisible()
             self.variable_editor.setVisible(visible)
             if hasattr(self, 'variable_editor_action'):
                 self.variable_editor_action.setChecked(visible)

    def toggle_workspace_editor(self):
        """Toggle visibility of the workspace editor dock."""
        if hasattr(self, 'workspace_editor_dock'):
             visible = not self.workspace_editor_dock.isVisible()
             self.workspace_editor_dock.setVisible(visible)
             if hasattr(self, 'workspace_editor_action'):
                 self.workspace_editor_action.setChecked(visible)

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
        self.toolbar = ModernToolBar(self)
        self.addToolBar(self.toolbar)
        
        # Connect toolbar signals (will implement handlers below)
        self.toolbar.new_diagram.connect(self.new_diagram)
        self.toolbar.open_diagram.connect(self.open_diagram)
        self.toolbar.save_diagram.connect(self.save_diagram)
        self.toolbar.play_simulation.connect(self.start_simulation)
        self.toolbar.pause_simulation.connect(self.pause_simulation)
        self.toolbar.stop_simulation.connect(self.stop_simulation)
        self.toolbar.step_simulation.connect(self.step_simulation)
        self.toolbar.plot_results.connect(self.show_plots)
        self.toolbar.capture_screen.connect(self.capture_screen)
        self.toolbar.zoom_changed.connect(self.set_zoom)
        self.toolbar.theme_toggled.connect(self.on_theme_changed)
        self.toolbar.command_palette_requested.connect(self.show_command_palette)
    
    def _setup_layout(self):
        """Setup modern layout with splitters."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_layout.addWidget(main_splitter)

        # Left panel (Block Palette)
        self.left_panel = self._create_left_panel()
        main_splitter.addWidget(self.left_panel)

        # Center area (Canvas + Property Panel on right)
        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.setObjectName("CenterSplitter")
        center_splitter.setChildrenCollapsible(False)
        center_splitter.setHandleWidth(5)

        # Canvas area (will contain the drawing canvas)
        self.canvas_area = self._create_canvas_area()
        center_splitter.addWidget(self.canvas_area)

        # Property panel (right side, vertical)
        self.property_panel = self._create_property_panel()
        center_splitter.addWidget(self.property_panel)

        # Set stretch factors: canvas gets priority when resizing
        center_splitter.setStretchFactor(0, 1)  # Canvas stretches
        center_splitter.setStretchFactor(1, 0)  # Property panel stays fixed size

        # Don't set splitter sizes here - will be set after window is shown
        # This ensures calculations use actual window width, not screen width

        main_splitter.addWidget(center_splitter)

        # Set stretch factors: center (canvas+properties) gets priority when resizing
        main_splitter.setStretchFactor(0, 0)  # Left panel stays fixed size
        main_splitter.setStretchFactor(1, 1)  # Center area stretches

        # Store reference to center splitter for initial sizing
        self._center_splitter_for_init = center_splitter
        
        # Store splitters for theme updates
        self.main_splitter = main_splitter
        self.center_splitter = center_splitter

    def _initialize_splitter_sizes(self):
        """Initialize splitter sizes based on actual window dimensions.
        Called after window is shown to ensure accurate sizing."""
        # Get platform configuration and actual window width
        config = get_platform_config()
        actual_width = self.width()

        logger.info(f"Initializing splitters with actual window width: {actual_width}px")

        # Main splitter: left panel gets fixed width, rest goes to center
        left_width = config.splitter_left_width
        center_width = actual_width - left_width
        self.main_splitter.setSizes([left_width, center_width])

        # Center splitter: canvas gets most space, property panel gets configured percentage
        property_width = int(center_width * config.splitter_property_percent)
        canvas_width = center_width - property_width

        # Ensure property panel is at least minimum width
        min_property_width = config.splitter_property_min_width
        if property_width < min_property_width:
            property_width = min_property_width
            canvas_width = center_width - property_width

        self.center_splitter.setSizes([canvas_width, property_width])

        logger.info(f"Splitter sizes: left={left_width}, canvas={canvas_width}, properties={property_width}")
    
    def _create_left_panel(self) -> QWidget:
        """Create modern left panel for block palette."""
        panel = QFrame()
        panel.setObjectName("ModernPanel")

        # Get platform configuration
        config = get_platform_config()

        panel.setMinimumWidth(config.left_panel_min_width)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Panel title
        title = QLabel("Block Palette")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 4px;")
        layout.addWidget(title)

        # Modern block palette widget (Phase 2)
        self.block_palette = ModernBlockPalette(self.dsim)
        layout.addWidget(self.block_palette)
        return panel
    
    def _create_canvas_area(self) -> QWidget:
        """Create modern canvas area with responsive sizing and error panel."""
        # Create container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        # Create the modern canvas widget
        self.canvas = ModernCanvas(self.dsim)

        # Set default routing mode for new connections
        self.canvas.default_routing_mode = self.default_routing_mode

        # Get platform configuration
        config = get_platform_config()

        self.canvas.setMinimumSize(config.canvas_min_width, config.canvas_min_height)

        logger.info(f"Canvas minimum size: {config.canvas_min_width}×{config.canvas_min_height}")

        # Connect canvas signals
        self.canvas.block_selected.connect(self._on_block_selected)
        self.canvas.connection_created.connect(self._on_connection_created)
        self.canvas.simulation_status_changed.connect(self._on_simulation_status_changed)
        self.canvas.command_palette_requested.connect(self.show_command_palette)
        self.toolbar.auto_route_wires.connect(self.canvas.auto_route_lines)

        # Create breadcrumb bar
        self.breadcrumb_bar = BreadcrumbBar()
        self.breadcrumb_bar.path_clicked.connect(self.canvas.navigate_scope)
        self.canvas.scope_changed.connect(self.breadcrumb_bar.set_path)

        # Add widgets to container
        container_layout.addWidget(self.breadcrumb_bar)
        container_layout.addWidget(self.canvas, 1)  # Canvas gets stretch priority

        # Create error panel
        self.error_panel = ErrorPanel()
        self.error_panel.error_clicked.connect(self._on_error_clicked)
        self.error_panel.setMaximumHeight(200)  # Limit height
        container_layout.addWidget(self.error_panel, 0)  # Error panel doesn't stretch

        return container
    
    def _create_property_panel(self) -> QWidget:
        """Create modern property panel on right side with size constraints."""
        panel = QFrame()
        panel.setObjectName("ModernPanel")
        panel.setFrameStyle(QFrame.StyledPanel)

        # Get platform configuration
        config = get_platform_config()

        panel.setMinimumWidth(config.property_panel_min_width)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Panel title
        title = QLabel("Properties")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 4px;")
        layout.addWidget(title)

        # Scroll area for properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(scroll_area)

        # Property editor
        self.property_editor = PropertyEditor()
        scroll_area.setWidget(self.property_editor)

        # PyQt5 5.15: QScrollArea viewport absorbs mouse clicks and doesn't
        # transfer focus to child widgets. Install event filter to fix this.
        self._prop_scroll_viewport = scroll_area.viewport()
        self._prop_scroll_viewport.installEventFilter(self)

        return panel
    
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

    def _setup_statusbar(self):
        """Compact pill-style status bar (≤ 28px tall).

        Layout left-to-right:  state-pill · file-pill · counts · ⟶ · cursor · zoom · theme-pill
        Segments are separated by 1px vertical dividers (no pipes).
        """
        from modern_ui.widgets.modern_toolbar import _StatusPill  # reuse toolbar pill

        statusbar = self.statusBar()
        statusbar.setSizeGripEnabled(False)
        statusbar.setFixedHeight(26)

        def _vsep():
            f = QFrame()
            f.setFrameShape(QFrame.VLine)
            f.setObjectName("StatusDivider")
            f.setStyleSheet(
                f"color: {theme_manager.get_color('border_primary').name()};"
                f" background: {theme_manager.get_color('border_primary').name()};"
                f" max-width: 1px; min-width: 1px;"
            )
            f.setFixedHeight(14)
            return f

        def _mono_label(text=""):
            from PyQt5.QtGui import QFont as _QF
            lbl = QLabel(text)
            f = _QF("Menlo")
            f.setStyleHint(_QF.Monospace)
            if hasattr(f, 'setFamilies'):
                f.setFamilies(["Menlo", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"])
            f.setPointSize(8)
            lbl.setFont(f)
            return lbl

        # Left: status pill (reused from toolbar)
        self.status_pill = _StatusPill(self)
        statusbar.addWidget(self.status_pill)

        # Hidden compatibility shim — many call sites still call status_message.setText(...)
        self.status_message = QLabel()
        self.status_message.hide()
        # Forward text changes to the pill (idle/running/paused detection)
        def _on_status_text_changed(text):
            try:
                self.toolbar.set_status(text)
            except Exception:
                pass
            t = (text or "").lower()
            if 'run' in t and 'paus' not in t:
                self.status_pill.set_state('running')
            elif 'paus' in t:
                self.status_pill.set_state('paused')
            elif 'error' in t or 'fail' in t:
                self.status_pill.set_state('error', text)
            else:
                self.status_pill.set_state('idle', text if text else None)
        # Replace setText to propagate to the pill
        _orig_setText = self.status_message.setText
        def _propagating_setText(text):
            _orig_setText(text)
            _on_status_text_changed(text)
        self.status_message.setText = _propagating_setText  # type: ignore[attr-defined]

        statusbar.addWidget(_vsep())

        # File info: filename + unsaved indicator
        self.file_status = QLabel("untitled")
        self.file_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_primary').name()};"
        )
        self.file_unsaved_status = QLabel("")
        self.file_unsaved_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_disabled').name()};"
            f" font-size: 9pt;"
        )
        statusbar.addWidget(self.file_status)
        statusbar.addWidget(self.file_unsaved_status)

        statusbar.addWidget(_vsep())

        # Counts pill: blocks N · wires M · scopes K
        self.counts_status = _mono_label("blocks 0 · wires 0 · scopes 0")
        self.counts_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addWidget(self.counts_status)

        # ----- right-aligned permanent widgets -----
        self.cursor_status = _mono_label("cursor 0,0")
        self.cursor_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addPermanentWidget(self.cursor_status)

        statusbar.addPermanentWidget(_vsep())

        self.zoom_status = _mono_label("zoom 100%")
        self.zoom_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
        )
        statusbar.addPermanentWidget(self.zoom_status)

        statusbar.addPermanentWidget(_vsep())

        # Theme + palette tag
        theme_label = "Dark" if theme_manager.current_theme == ThemeType.DARK else "Light"
        from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES
        palette_label = PALETTE_DISPLAY_NAMES.get(theme_manager.current_palette, theme_manager.current_palette).split()[0]
        self.theme_status = QLabel(f"{theme_label} · {palette_label}")
        self.theme_status.setStyleSheet(
            f"color: {theme_manager.get_color('text_secondary').name()};"
            f" background-color: {theme_manager.get_color('background_tertiary').name()};"
            f" padding: 1px 8px; border-radius: 4px; font-size: 9pt;"
        )
        statusbar.addPermanentWidget(self.theme_status)

        # Drive zoom from toolbar's zoom rocker so the two stay in sync
        try:
            self.toolbar.zoom_changed.connect(
                lambda f: self.zoom_status.setText(f"zoom {int(round(f*100))}%")
            )
        except Exception:
            pass

        # Cursor pos from canvas
        try:
            self.canvas.cursor_moved.connect(
                lambda x, y: self.cursor_status.setText(f"cursor {x},{y}")
            )
        except Exception:
            pass

        # Periodic counts refresh (cheap; runs on the same timer that paints)
        self._counts_refresh_timer = QTimer(self)
        self._counts_refresh_timer.timeout.connect(self._refresh_status_counts)
        self._counts_refresh_timer.start(500)

        # Initial state
        self._refresh_status_counts()
        self._refresh_file_status()

        # Apply theme palette to the statusbar host
        self._update_statusbar_colors()

    def _refresh_status_counts(self):
        """Update the counts pill from current dsim state."""
        try:
            dsim = getattr(self, 'dsim', None)
            if dsim is None:
                return
            blocks = list(getattr(dsim, 'blocks_list', []) or [])
            wires = list(getattr(dsim, 'line_list', []) or [])
            scopes = sum(1 for b in blocks if getattr(b, 'block_fn', '') in ('Scope', 'FieldScope'))
            self.counts_status.setText(
                f"blocks {len(blocks)} · wires {len(wires)} · scopes {scopes}"
            )
        except Exception:
            pass

    def _refresh_file_status(self):
        """Update filename + unsaved indicator in the status bar."""
        try:
            path = getattr(self.dsim, 'current_filepath', None) or getattr(self.dsim, 'filepath', None)
            name = os.path.basename(path) if path else "untitled"
            self.file_status.setText(name)
            self.file_unsaved_status.setText("unsaved" if getattr(self.dsim, 'dirty', False) else "")
        except Exception:
            pass
    
    def paintEvent(self, event):
        """Paint event - delegated to canvas widget."""
        pass

    # Toolbar action handlers (delegation to project_manager)
    def open_diagram(self):
        self.project_manager.open_diagram()

    def _update_recent_files_menu(self):
        self.project_manager.update_recent_files_menu()

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
    
    def toggle_theme(self):
        """Toggle theme and persist the choice."""
        theme_manager.toggle_theme()
        self._save_user_preferences()

    def _set_palette(self, palette_key: str):
        """Switch the active block-color palette and persist the choice."""
        theme_manager.set_palette(palette_key)
        self._save_user_preferences()
        # Refresh canvas so blocks re-render with new palette colors
        if hasattr(self, 'canvas'):
            self.canvas.update()

    def _toggle_solid_fills(self, checked: bool):
        """Toggle solid block fills and persist the choice."""
        theme_manager.set_solid_fills(checked)
        if hasattr(self, 'canvas'):
            self.canvas.update()
        self._save_user_preferences()

    def _save_user_preferences(self):
        """Persist all UI preferences (theme, palette, solid_fills) to user_preferences.json."""
        import os
        import json
        from lib.app_paths import user_data_path
        path = user_data_path("user_preferences.json")
        prefs = {}
        try:
            with open(path, 'r') as f:
                prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            prefs = {}
        prefs['theme'] = theme_manager.current_theme.value
        prefs['block_palette'] = theme_manager.current_palette
        prefs['solid_fills'] = theme_manager.solid_fills
        tmp = path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(prefs, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning("Could not save user preferences: %s", e)
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass

    def on_theme_changed(self):
        """Handle theme changes."""
        # Update status bar
        theme_name = "Dark Theme" if theme_manager.current_theme == ThemeType.DARK else "Light Theme"
        self.theme_status.setText(theme_name)

        # Update statusbar label colors
        self._update_statusbar_colors()

        # Update menubar colors
        self._update_menubar_colors()

        # Update canvas area styling
        self.canvas_area.setStyleSheet(f"""
            #CanvasArea {{
                background-color: {theme_manager.get_color('canvas_background').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 6px;
            }}
        """)

    def _update_statusbar_colors(self):
        """Apply theme to the status bar shell. Individual pills own their own styles."""
        bg_color = theme_manager.get_color('statusbar_bg').name()
        border = theme_manager.get_color('border_primary').name()
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background-color: {bg_color}; border-top: 1px solid {border}; }}"
            f"QStatusBar::item {{ border: 0; }}"
        )
        # Refresh theme pill text
        if hasattr(self, 'theme_status'):
            theme_label = "Dark" if theme_manager.current_theme == ThemeType.DARK else "Light"
            from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES
            palette_label = PALETTE_DISPLAY_NAMES.get(
                theme_manager.current_palette, theme_manager.current_palette
            ).split()[0]
            self.theme_status.setText(f"{theme_label} · {palette_label}")
            self.theme_status.setStyleSheet(
                f"color: {theme_manager.get_color('text_secondary').name()};"
                f" background-color: {theme_manager.get_color('background_tertiary').name()};"
                f" padding: 1px 8px; border-radius: 4px; font-size: 9pt;"
            )

    def _update_menubar_colors(self):
        """Update menubar colors for proper contrast."""
        bg_color = theme_manager.get_color('surface').name()
        text_color = theme_manager.get_color('text_primary').name()
        hover_bg = theme_manager.get_color('accent_primary').name()
        hover_bg_alpha = theme_manager.get_color('accent_primary')
        hover_bg_alpha.setAlpha(30)
        disabled_color = theme_manager.get_color('text_disabled').name()

        menubar_style = f"""
            QMenuBar {{
                background-color: {bg_color};
                color: {text_color};
                border-bottom: 1px solid {theme_manager.get_color('border_primary').name()};
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 10px;
            }}
            QMenuBar::item:selected {{
                background-color: {hover_bg_alpha.name(QColor.HexArgb)};
                border-radius: 4px;
            }}
            QMenuBar::item:pressed {{
                background-color: {hover_bg};
                border-radius: 4px;
            }}
            QMenu {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {hover_bg_alpha.name(QColor.HexArgb)};
            }}
            QMenu::item:disabled {{
                color: {disabled_color};
            }}
            QMenu::separator {{
                height: 1px;
                background: {theme_manager.get_color('border_secondary').name()};
                margin: 4px 8px;
            }}
        """

        self.menuBar().setStyleSheet(menubar_style)

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

    def show_command_palette(self):
        """Show the command palette for quick access."""
        if hasattr(self, 'command_palette'):
            self.command_palette.show_palette()

    def _setup_command_palette(self):
        """Build the command palette index — blocks, sim, view, files, help."""
        commands: list[dict] = []

        # Block library — typed as 'block' so the BLOCK badge surfaces
        if hasattr(self, 'canvas') and hasattr(self.canvas.dsim, 'menu_blocks'):
            for menu_block in self.canvas.dsim.menu_blocks:
                fn_name = getattr(menu_block, 'fn_name', '') or ''
                block_fn = getattr(menu_block, 'block_fn', '') or fn_name
                commands.append({
                    'name': f'Add {block_fn} block',
                    'type': 'block',
                    'description': f'{block_fn} ({fn_name})',
                    'aliases': [fn_name, block_fn, fn_name.lower()],
                    'callback': lambda mb=menu_block: self._add_block_from_palette_menu(mb),
                    'data': {'block_type': fn_name},
                })

        # Simulation actions (SIM badge)
        for label, kbd, cb in [
            ('Run simulation',   'F5', self.start_simulation),
            ('Pause simulation', 'F6', self.pause_simulation),
            ('Stop simulation',  'F7', self.stop_simulation),
            ('Step simulation',  'F8', self.step_simulation),
            ('Toggle fast solver', '', lambda: self.toggle_fast_solver(not getattr(self, 'use_fast_solver', True))),
        ]:
            commands.append({
                'name': label, 'type': 'sim', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # View toggles (VIEW badge)
        for label, kbd, cb in [
            ('Zoom in',       'Ctrl++',  self.zoom_in),
            ('Zoom out',      'Ctrl+-',  self.zoom_out),
            ('Fit to window', 'Ctrl+0',  self.fit_to_window),
            ('Toggle theme',  'Ctrl+T',  self.toggle_theme),
            ('Toggle grid',   'Ctrl+Shift+G', self.toggle_grid),
            ('Toggle minimap', 'Ctrl+Shift+M', self.toggle_minimap),
            ('Toggle variable editor', 'Ctrl+Shift+V', self.toggle_variable_editor),
            ('Toggle workspace variables', 'Ctrl+Shift+W', self.toggle_workspace_editor),
            ('Toggle tuning panel', '', self.toggle_tuning_panel),
        ]:
            commands.append({
                'name': label, 'type': 'view', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # File actions
        for label, kbd, cb in [
            ('New diagram',  'Ctrl+N', self.new_diagram),
            ('Open diagram', 'Ctrl+O', self.open_diagram),
            ('Save diagram', 'Ctrl+S', self.save_diagram),
            ('Load workspace…', '', self.load_workspace),
            ('Show plots',   '',       self.show_plots),
            ('Export as TikZ…', '',    self.export_tikz),
        ]:
            commands.append({
                'name': label, 'type': 'file', 'shortcut': kbd,
                'callback': cb, 'data': {},
            })

        # Index examples on disk — file paths only, load on click
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            examples_dir = os.path.join(base_dir, 'examples')
            if os.path.isdir(examples_dir):
                for f in sorted(os.listdir(examples_dir)):
                    if f.endswith(('.json', '.dat', '.diablos')):
                        path = os.path.join(examples_dir, f)
                        commands.append({
                            'name': f'examples / {os.path.splitext(f)[0]}',
                            'type': 'file',
                            'callback': lambda p=path: self.open_example(p),
                            'data': {'path': path},
                        })
        except Exception:
            logger.debug("Could not index examples for command palette", exc_info=True)

        # Recent files
        try:
            recents = self._load_recent_files()
        except Exception:
            recents = []
        for path in recents[:6]:
            commands.append({
                'name': os.path.basename(path),
                'type': 'recent',
                'description': path,
                'callback': lambda p=path: self._open_recent_file(p),
                'data': {'path': path},
            })

        self.command_palette.set_commands(commands)

    def _add_block_from_palette_menu(self, menu_block):
        """Add a block to the canvas from command palette."""
        if not hasattr(self, 'canvas'):
            return

        from PyQt5.QtCore import QPoint
        from PyQt5.QtGui import QCursor

        # Get current mouse position in global coordinates
        global_pos = QCursor.pos()

        # Convert to canvas widget coordinates
        canvas_widget_pos = self.canvas.mapFromGlobal(global_pos)

        # Check if mouse is within canvas bounds
        if self.canvas.rect().contains(canvas_widget_pos):
            # Convert screen coordinates to canvas coordinates (undo pan and zoom)
            canvas_x = int((canvas_widget_pos.x() - self.canvas.pan_offset.x()) / self.canvas.zoom_factor)
            canvas_y = int((canvas_widget_pos.y() - self.canvas.pan_offset.y()) / self.canvas.zoom_factor)
            canvas_pos = QPoint(canvas_x, canvas_y)
        else:
            # Fallback: add at center of visible canvas area
            center_x = self.canvas.width() // 2
            center_y = self.canvas.height() // 2

            # Convert screen coordinates to canvas coordinates (undo pan and zoom)
            canvas_x = int((center_x - self.canvas.pan_offset.x()) / self.canvas.zoom_factor)
            canvas_y = int((center_y - self.canvas.pan_offset.y()) / self.canvas.zoom_factor)
            canvas_pos = QPoint(canvas_x, canvas_y)

        # Add the block using the canvas method
        self.canvas.add_block_from_palette(menu_block, canvas_pos)
        self.toast.show_message(f"✅ Added {menu_block.block_fn} block")

    def _on_command_executed(self, command_type: str, data: dict):
        """Handle command palette command execution."""
        logger.info(f"Command executed: {command_type}, data: {data}")

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

    def _convert_param_value(self, new_value, target_type):
        """
        Convert a parameter value to the target type, handling variables.
        
        Args:
            new_value: The string input from the user.
            target_type: The expected type (int, float, bool, list).
            
        Returns:
            The converted value.
            
        Raises:
            ValueError, TypeError, SyntaxError: If conversion fails and it's not a valid variable name.
        """
        try:
            # Try to convert to the expected type
            if target_type == bool:
                if isinstance(new_value, str):
                    return new_value.lower() == 'true'
                return bool(new_value)
            elif target_type == list:
                converted = ast.literal_eval(new_value)
                if not isinstance(converted, list):
                    raise TypeError("Input must be a list (e.g., [1, 2, 3])")
                return converted
            elif target_type == int:
                return int(new_value)
            elif target_type == float:
                return float(new_value)
            else:
                return str(new_value)
        except (ValueError, TypeError, SyntaxError):
            # If conversion fails, treat as a string (potential variable name or expression)
            # We allow this so that expressions like '[K, K]' or '2*K' can be stored as strings
            # and resolved later by the WorkspaceManager.
            logger.debug(f"Could not convert '{new_value}' to {target_type}, keeping as string.")
            return str(new_value)

    def _on_property_changed(self, block_name: str, prop_name: str, new_value: Any) -> None:
        """Handle property changes from the property editor."""
        try:
            for block in self.canvas.dsim.blocks_list:
                if block.name == block_name:
                    # Handle username change (special case - not in params)
                    if prop_name == '_username_':
                        self.canvas.dsim.dirty = True
                        self.canvas.update()
                        return

                    # Handle port count change from property editor
                    if prop_name in ('_inputs_', '_outputs_'):
                        self.canvas._push_undo("Edit Ports")
                        if prop_name == '_inputs_':
                            block.in_ports = int(new_value)
                        else:
                            block.out_ports = int(new_value)
                        block.update_Block()
                        block.params['_inputs_'] = block.in_ports
                        block.params['_outputs_'] = block.out_ports
                        self.canvas.dsim.dirty = True
                        self.canvas.update()
                        return

                    param_type = type(block.params.get(prop_name))

                    # If the value is already a list or numpy array, preserve it
                    # (property editor may have already converted it for accepts_array params)
                    if isinstance(new_value, (list, tuple)):
                        converted_value = list(new_value)
                    elif hasattr(new_value, 'tolist'):  # numpy array
                        converted_value = new_value
                    else:
                        converted_value = self._convert_param_value(new_value, param_type)

                    logger.debug(f"Updating {block_name}.{prop_name} to {converted_value} (type: {type(converted_value).__name__})")
                    block.update_params({prop_name: converted_value})
                    self.canvas.dsim.dirty = True
                    # For Goto/From blocks, refresh labels and virtual links immediately
                    if block.block_fn in ("Goto", "From") and prop_name in ("tag", "signal_name"):
                        try:
                            self.canvas.dsim.model.link_goto_from()
                        except Exception as e:
                            logger.warning(f"Could not relink Goto/From after property change: {e}")
                    # Refresh canvas to show updated block visuals (Sum signs, labels, etc.)
                    self.canvas.update()
                    break
        except (ValueError, TypeError, SyntaxError) as e:
            logger.error(f"Failed to convert property {prop_name} to type {param_type}: {e}")
            self.show_error(f"Invalid input for {prop_name}: {e}")
        except Exception as e:
            logger.error(f"Error updating property: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def _add_to_tuning(self, block, param_name):
        """Add a block parameter to the tuning panel."""
        self.tuning_panel.add_parameter(block, param_name)

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
    def _load_recent_files(self):
        """Load recent files list from config."""
        import json
        from lib.app_paths import user_data_path
        config_path = user_data_path('config/recent_files.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return data.get('recent_files', [])
        except Exception as e:
            logger.error(f"Error loading recent files: {e}")
        return []

    def _save_recent_files(self, recent_files):
        """Save recent files list to config."""
        import json
        from lib.app_paths import user_data_path
        config_path = user_data_path('config/recent_files.json')
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump({'recent_files': recent_files}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving recent files: {e}")

    def _add_recent_file(self, filepath):
        """Add a file to the recent files list."""
        if not filepath:
            return

        recent_files = self._load_recent_files()

        # Remove if already in list
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to front
        recent_files.insert(0, filepath)

        # Keep only last 10
        recent_files = recent_files[:10]

        self._save_recent_files(recent_files)
        self._update_recent_files_menu()

    def _update_recent_files_menu(self):
        """Update the recent files menu."""
        self.recent_files_menu.clear()

        recent_files = self._load_recent_files()

        if not recent_files:
            action = self.recent_files_menu.addAction("No recent files")
            action.setEnabled(False)
            return

        for filepath in recent_files:
            # Show only filename, but store full path
            filename = os.path.basename(filepath)
            action = self.recent_files_menu.addAction(filename)
            action.setData(filepath)
            action.triggered.connect(lambda checked, path=filepath: self._open_recent_file(path))

        self.recent_files_menu.addSeparator()
        clear_action = self.recent_files_menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self._clear_recent_files)

    def _open_recent_file(self, filepath):
        """Open a file from the recent files list."""
        if os.path.exists(filepath):
            # Open the diagram using the diagram service (or direct load method)
            # Since open_diagram doesn't take arguments in its current form (it opens dialog),
            # we should call dsim.open directly or refactor open_diagram to accept an optional path.
            # Looking at existing code, MainWindow.open_diagram calls self.dsim.open() (which opens dialog)
            # We want to bypass the dialog.
            
            try:
                # Use DSim's open mechanism that supports filepath
                # dsim.open_file(filepath) or dsim.file_service.load(filepath)
                # Let's check DSim.open in lib.py - it takes no args usually?
                # Using the FileService directly via DSim is safer if DSim.open is UI-bound.
                # However, DSim.serialize/deserialize were added.
                
                # Check if we have file_service
                if hasattr(self.dsim, 'file_service'):
                    block_data = self.dsim.file_service.load(filepath=filepath)
                    self.dsim.deserialize(block_data)
                else:
                    # Fallback to older mechanism if needed
                    self.dsim.open(filepath)
                
                self._add_recent_file(filepath)
                self.status_message.setText(f"Opened: {os.path.basename(filepath)}")
                logger.info(f"Opening recent file: {filepath}")
                self.canvas.update()
                
            except Exception as e:
                logger.error(f"Failed to open recent file: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open file:\n{str(e)}")
        else:
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The file '{filepath}' no longer exists."
            )
            # Remove from recent files
            recent_files = self._load_recent_files()
            if filepath in recent_files:
                recent_files.remove(filepath)
                self._save_recent_files(recent_files)
                self._update_recent_files_menu()

    def _clear_recent_files(self):
        """Clear the recent files list."""
        self._save_recent_files([])
        self._update_recent_files_menu()
        self.status_message.setText("Recent files cleared")

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
