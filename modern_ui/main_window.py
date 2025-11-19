"""
Modern Main Window for DiaBloS
Features modern layout, theming, and enhanced user interface.
"""

import sys
import os
import logging
import ast
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QMenuBar, QStatusBar, QLabel, QFrame,
                             QApplication, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Import existing DSim functionality
from lib.lib import DSim
from lib.improvements import (
    ValidationHelper, PerformanceHelper, SafetyChecks, LoggingHelper,
    SimulationConfig, validate_simulation_parameters
)

# Import modern UI components
from modern_ui.themes.theme_manager import theme_manager, ThemeType
from modern_ui.styles.qss_styles import apply_modern_theme, ModernStyles
from modern_ui.widgets.modern_toolbar import ModernToolBar
from modern_ui.widgets.modern_canvas import ModernCanvas
from modern_ui.widgets.modern_palette import ModernBlockPalette
from modern_ui.widgets.property_editor import PropertyEditor
from modern_ui.widgets.toast_notification import ToastNotification
from modern_ui.widgets.error_panel import ErrorPanel
from modern_ui.widgets.command_palette import CommandPalette
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

        # Core DSim functionality
        self.dsim = DSim()

        # Performance monitoring
        self.perf_helper = PerformanceHelper()

        # Simulation configuration
        self.sim_config = SimulationConfig()

        # Initialize state management (keeping from improved version)
        self._init_state_management()

        # Setup modern UI
        self._setup_window()
        self._setup_menubar()
        self._setup_toolbar()
        self.toolbar.set_simulation_state(False, False)
        self._setup_layout()
        self._setup_statusbar()
        self._setup_connections()

        # Create toast notification (after canvas is created)
        self.toast = ToastNotification(self.canvas)

        # Create command palette
        self.command_palette = CommandPalette(self)
        self.command_palette.command_selected.connect(self._on_command_executed)
        self._setup_command_palette()

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
        self.autosave_path = 'config/.autosave.diablos'

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
        self.State = State
        self.state = State.IDLE
        self.dragging_block = None
        self.drag_offset = None
        
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
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("&New\tCtrl+N", self.new_diagram)
        file_menu.addAction("&Open\tCtrl+O", self.open_diagram)
        file_menu.addAction("&Save\tCtrl+S", self.save_diagram)
        file_menu.addSeparator()

        # Recent Files submenu
        self.recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu()

        file_menu.addSeparator()
        file_menu.addAction("E&xit\tAlt+F4", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Undo\tCtrl+Z", self.undo_action)
        edit_menu.addAction("&Redo\tCtrl+Y", self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction("Select &All\tCtrl+A", self.select_all)
        edit_menu.addSeparator()
        edit_menu.addAction("Command &Palette\tCtrl+P", self.show_command_palette)
        
        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        sim_menu.addAction("&Run\tF5", self.start_simulation)
        sim_menu.addAction("&Pause\tF6", self.pause_simulation)
        sim_menu.addAction("&Stop\tF7", self.stop_simulation)
        sim_menu.addSeparator()
        sim_menu.addAction("Show &Plots", self.show_plots)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("&Zoom In\tCtrl++", self.zoom_in)
        view_menu.addAction("Zoom &Out\tCtrl+-", self.zoom_out)
        view_menu.addAction("&Fit to Window\tCtrl+0", self.fit_to_window)
        view_menu.addSeparator()

        # Grid toggle
        self.grid_toggle_action = view_menu.addAction("Show &Grid\tCtrl+G", self.toggle_grid)
        self.grid_toggle_action.setCheckable(True)
        self.grid_toggle_action.setChecked(True)
        view_menu.addSeparator()

        view_menu.addAction("Toggle &Theme\tCtrl+T", self.toggle_theme)
        view_menu.addSeparator()
        scaling_menu = view_menu.addMenu("UI Scale")
        action_100 = scaling_menu.addAction("100%")
        action_100.triggered.connect(lambda: self._set_scaling(1.0))
        action_125 = scaling_menu.addAction("125%")
        action_125.triggered.connect(lambda: self._set_scaling(1.25))
        action_150 = scaling_menu.addAction("150%")
        action_150.triggered.connect(lambda: self._set_scaling(1.5))

        # Default Connection Routing submenu
        view_menu.addSeparator()
        routing_menu = view_menu.addMenu("Default Connection Routing")

        # Bezier mode (default)
        self.bezier_routing_action = routing_menu.addAction("Bezier (Curved)")
        self.bezier_routing_action.setCheckable(True)
        self.bezier_routing_action.setChecked(True)
        self.bezier_routing_action.triggered.connect(lambda: self._set_default_routing_mode("bezier"))

        # Orthogonal mode
        self.orthogonal_routing_action = routing_menu.addAction("Orthogonal (Manhattan)")
        self.orthogonal_routing_action.setCheckable(True)
        self.orthogonal_routing_action.triggered.connect(lambda: self._set_default_routing_mode("orthogonal"))

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&Keyboard Shortcuts", self.show_keyboard_shortcuts)
        help_menu.addSeparator()
        help_menu.addAction("&About DiaBloS", self.show_about)

    def _set_scaling(self, factor):
        import json
        config_path = 'config/default_config.json'
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
        self.toolbar.plot_results.connect(self.show_plots)
        self.toolbar.capture_screen.connect(self.capture_screen)
        self.toolbar.zoom_changed.connect(self.set_zoom)
        self.toolbar.theme_toggled.connect(self.on_theme_changed)
    
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
        main_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        main_layout.addWidget(main_splitter)

        # Left panel (Block Palette)
        self.left_panel = self._create_left_panel()
        main_splitter.addWidget(self.left_panel)

        # Center area (Canvas + Property Panel on right)
        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.setObjectName("CenterSplitter")
        center_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely

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
        panel.setMaximumWidth(config.left_panel_max_width)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
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

        # Add canvas to container
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
        # Remove restrictive max width - let it use 20-30% of screen naturally

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        # Panel title
        title = QLabel("Properties")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 4px;")
        layout.addWidget(title)

        # Scroll area for properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        layout.addWidget(scroll_area)

        # Property editor
        self.property_editor = PropertyEditor()
        scroll_area.setWidget(self.property_editor)

        return panel
    
    def _setup_statusbar(self):
        """Setup modern status bar."""
        statusbar = self.statusBar()

        # Status message
        self.status_message = QLabel("Ready")
        statusbar.addWidget(self.status_message)

        # Add permanent widgets
        statusbar.addPermanentWidget(QLabel("Zoom:"))
        self.zoom_status = QLabel("100%")
        statusbar.addPermanentWidget(self.zoom_status)

        statusbar.addPermanentWidget(QLabel("|"))

        self.cursor_status = QLabel("Cursor: (0, 0)")
        statusbar.addPermanentWidget(self.cursor_status)

        statusbar.addPermanentWidget(QLabel("|"))

        self.theme_status = QLabel("Dark Theme")
        statusbar.addPermanentWidget(self.theme_status)

        # Apply initial theme colors to statusbar labels
        self._update_statusbar_colors()
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Theme change updates
        theme_manager.theme_changed.connect(self.on_theme_changed)
        self.property_editor.property_changed.connect(self._on_property_changed)
    
    # Implementation continues with all the existing functionality from improved version
    # For brevity, I'll add the key methods here and the rest in the next part
    
    def safe_update(self):
        """Safe update with error handling and performance monitoring."""
        try:
            self.perf_helper.start_timer("ui_update")
            
            # Use existing DSim update methods but with safety checks
            if hasattr(self.dsim, 'execution_initialized') and self.dsim.execution_initialized:
                is_safe, errors = SafetyChecks.check_simulation_state(self.dsim)
                if not is_safe:
                    logger.error(f"Simulation state unsafe: {errors}")
                    self.stop_simulation()
                    return
                
                self.perf_helper.start_timer("simulation_step")
                self.dsim.execution_loop()
                step_duration = self.perf_helper.end_timer("simulation_step")
                
                if step_duration and step_duration > 0.1:
                    logger.warning(f"Slow simulation step: {step_duration:.4f}s")
            
            # Update canvas (for now, trigger a repaint)
            self.canvas_area.update()
            
            self.perf_helper.end_timer("ui_update")
            
        except Exception as e:
            logger.error(f"Error in safe_update: {str(e)}")
            self.stop_simulation()
    
    def paintEvent(self, event):
        """Paint event - will be moved to canvas widget in Phase 2."""
        # For Phase 1, keep the existing painting logic from improved version
        # This will be refactored in Phase 2 when we implement the modern canvas
        pass
    
    # Toolbar action handlers
    def new_diagram(self):
        """Create new diagram."""
        if hasattr(self.dsim, 'clear_all'):
            self.dsim.clear_all()
        self.status_message.setText("New diagram created")
    
    def open_diagram(self):
        """Open diagram."""
        if hasattr(self.dsim, 'open'):
            modern_ui_data = self.dsim.open()
            if modern_ui_data:
                theme_manager.set_theme(ThemeType(modern_ui_data.get("theme", "light")))
                self.set_zoom(modern_ui_data.get("zoom_factor", 1.0))
                self.main_splitter.setSizes(modern_ui_data.get("main_splitter_sizes", [250, 950]))
                self.center_splitter.setSizes(modern_ui_data.get("center_splitter_sizes", [600, 200]))
        self.status_message.setText("Diagram opened")
    
    def save_diagram(self):
        """Save diagram."""
        modern_ui_data = {
            "theme": theme_manager.current_theme.value,
            "zoom_factor": self.canvas.zoom_factor,
            "main_splitter_sizes": self.main_splitter.sizes(),
            "center_splitter_sizes": self.center_splitter.sizes()
        }
        if hasattr(self.dsim, 'save'):
            self.dsim.save(modern_ui_data=modern_ui_data)
        self.status_message.setText("Diagram saved")
    
    def start_simulation(self):
        """Start simulation with validation."""
        self.toolbar.set_simulation_state(True, False)
        self.status_message.setText("Starting simulation...")
        # Use existing validation logic from improved version
        # (Implementation details omitted for brevity)
    
    def pause_simulation(self):
        """Pause simulation."""
        if hasattr(self.dsim, 'execution_pause'):
            self.dsim.execution_pause = True
        self.toolbar.set_simulation_state(True, True)
    
    def stop_simulation(self):
        """Stop simulation."""
        if hasattr(self.dsim, 'execution_initialized'):
            self.dsim.execution_initialized = False
        self.toolbar.set_simulation_state(False, False)
    
    def show_plots(self):
        """Show plots."""
        if hasattr(self.dsim, 'plot_again'):
            self.dsim.plot_again()
    
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
        """Toggle theme."""
        theme_manager.toggle_theme()
    
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
        """Update statusbar label colors for proper contrast."""
        text_color = theme_manager.get_color('text_primary').name()
        statusbar_style = f"QLabel {{ color: {text_color}; }}"

        # Update known labels
        self.status_message.setStyleSheet(statusbar_style)
        self.zoom_status.setStyleSheet(statusbar_style)
        self.cursor_status.setStyleSheet(statusbar_style)
        self.theme_status.setStyleSheet(statusbar_style)

        # Apply to all statusbar labels (including "Zoom:" and "|" separators)
        for widget in self.statusBar().findChildren(QLabel):
            widget.setStyleSheet(statusbar_style)

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
        """Setup command palette with available commands."""
        commands = []

        # Add block types
        block_types = [
            ('Constant', 'Constant value source'),
            ('Step', 'Step input signal'),
            ('Ramp', 'Ramp input signal'),
            ('Gain', 'Amplify signal by constant'),
            ('Sum', 'Add or subtract signals'),
            ('Integrator', 'Integrate signal over time'),
            ('Deriv', 'Differentiate signal'),
            ('TranFn', 'Transfer function block'),
            ('Scope', 'Display signal values'),
            ('Mux', 'Multiplex signals'),
            ('Demux', 'Demultiplex signals'),
            ('Abs', 'Absolute value'),
            ('Exp', 'Exponential function'),
            ('Sqrt', 'Square root'),
            ('Sin', 'Sine function'),
            ('Cos', 'Cosine function'),
        ]

        for block_name, description in block_types:
            commands.append({
                'name': f'Add {block_name} Block',
                'type': 'block',
                'description': description,
                'callback': lambda bn=block_name.lower(): self._add_block_from_palette(bn),
                'data': {'block_type': block_name.lower()}
            })

        # Add application actions
        actions = [
            ('New Diagram', 'Create a new diagram', self.new_diagram),
            ('Open Diagram', 'Open an existing diagram', self.open_diagram),
            ('Save Diagram', 'Save current diagram', self.save_diagram),
            ('Run Simulation', 'Start the simulation', self.start_simulation),
            ('Pause Simulation', 'Pause the simulation', self.pause_simulation),
            ('Stop Simulation', 'Stop the simulation', self.stop_simulation),
            ('Show Plots', 'Display simulation plots', self.show_plots),
            ('Zoom In', 'Increase canvas zoom', self.zoom_in),
            ('Zoom Out', 'Decrease canvas zoom', self.zoom_out),
            ('Fit to Window', 'Fit all blocks in view', self.fit_to_window),
            ('Toggle Theme', 'Switch between light/dark mode', self.toggle_theme),
            ('Toggle Grid', 'Show/hide canvas grid', self.toggle_grid),
        ]

        for action_name, description, callback in actions:
            commands.append({
                'name': action_name,
                'type': 'action',
                'description': description,
                'callback': callback,
                'data': {}
            })

        # Add recent files
        if hasattr(self, 'recent_files') and self.recent_files:
            for file_path in self.recent_files[:5]:  # Show top 5 recent files
                import os
                file_name = os.path.basename(file_path)
                commands.append({
                    'name': file_name,
                    'type': 'recent',
                    'description': file_path,
                    'callback': lambda fp=file_path: self._open_file(fp),
                    'data': {'file_path': file_path}
                })

        self.command_palette.set_commands(commands)

    def _add_block_from_palette(self, block_type: str):
        """Add a block to the canvas from command palette."""
        if not hasattr(self, 'canvas'):
            return

        # Find the menu block for this block type
        menu_block = None
        for mb in self.canvas.dsim.menu_blocks:
            if mb.fn_name == block_type:
                menu_block = mb
                break

        if not menu_block:
            logger.warning(f"Menu block not found for type: {block_type}")
            return

        # Add block at center of visible canvas area
        # Account for pan and zoom
        from PyQt5.QtCore import QPoint
        center_x = self.canvas.width() // 2
        center_y = self.canvas.height() // 2

        # Convert screen coordinates to canvas coordinates (undo pan and zoom)
        canvas_x = int((center_x - self.canvas.pan_offset.x()) / self.canvas.zoom_factor)
        canvas_y = int((center_y - self.canvas.pan_offset.y()) / self.canvas.zoom_factor)
        canvas_pos = QPoint(canvas_x, canvas_y)

        # Add the block using the canvas method
        self.canvas.add_block_from_palette(menu_block, canvas_pos)
        self.toast.show_message(f"✅ Added {block_type} block")

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

    def _on_property_changed(self, block_name, prop_name, new_value):
        """Handle property changes from the property editor."""
        try:
            for block in self.canvas.dsim.blocks_list:
                if block.name == block_name:
                    param_type = type(block.params.get(prop_name))

                    try:
                        if param_type == bool:
                            converted_value = bool(new_value)
                        elif param_type == list:
                            converted_value = ast.literal_eval(new_value)
                            if not isinstance(converted_value, list):
                                raise TypeError("Input must be a list (e.g., [1, 2, 3])")
                        elif param_type == int:
                            converted_value = int(new_value)
                        elif param_type == float:
                            converted_value = float(new_value)
                        else:
                            converted_value = str(new_value)

                        block.update_params({prop_name: converted_value})
                        self.canvas.dsim.dirty = True
                        logger.info(f"Updated {block_name}.{prop_name} to {converted_value}")
                    except (ValueError, TypeError, SyntaxError) as e:
                        logger.error(f"Failed to convert property {prop_name} to type {param_type}: {e}")
                    break
        except Exception as e:
            logger.error(f"Error updating property: {e}")

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
    
    def _on_block_drag_started(self, menu_block):
        """Handle block drag started from palette."""
        try:
            block_name = getattr(menu_block, 'fn_name', 'Unknown')
            logger.info(f"Block drag started: {block_name}")
            self.status_message.setText(f"Dragging {block_name}")
            
        except Exception as e:
            logger.error(f"Error handling block drag start: {str(e)}")
    
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

                is_running = self.canvas.is_simulation_running()

                if was_running and not is_running:
                    self.toolbar.set_simulation_state(False, False)
                    self.status_message.setText("Simulation finished")
            
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
    
    def start_simulation(self):
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
        self.canvas.start_simulation()
    
    def stop_simulation(self):
        """Stop simulation."""
        if hasattr(self, 'canvas'):
            self.canvas.stop_simulation()
        self.toolbar.set_simulation_state(False, False)
        self.status_message.setText("Simulation stopped")

    # Recent Files Management
    def _load_recent_files(self):
        """Load recent files list from config."""
        import json
        config_path = 'config/recent_files.json'
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
        config_path = 'config/recent_files.json'
        try:
            os.makedirs('config', exist_ok=True)
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
            # TODO: Implement actual file opening logic
            # For now, just add to recent files
            self._add_recent_file(filepath)
            self.status_message.setText(f"Opened: {os.path.basename(filepath)}")
            logger.info(f"Opening recent file: {filepath}")
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

            # Save current state to autosave file
            import pickle
            with open(self.autosave_path, 'wb') as f:
                state = {
                    'blocks': [],
                    'lines': []
                }

                # Capture blocks
                for block in self.canvas.dsim.blocks_list:
                    block_data = {
                        'name': block.name,
                        'block_fn': block.block_fn,
                        'coords': (block.left, block.top, block.width, block.height),
                        'color': block.b_color.name() if hasattr(block.b_color, 'name') else str(block.b_color),
                        'in_ports': block.in_ports,
                        'out_ports': block.out_ports,
                        'b_type': block.b_type,
                        'io_edit': block.io_edit,
                        'fn_name': block.fn_name,
                        'params': block.params.copy() if hasattr(block, 'params') and block.params else {},
                        'external': block.external
                    }
                    state['blocks'].append(block_data)

                # Capture connections
                for line in self.canvas.dsim.line_list:
                    line_data = {
                        'name': line.name,
                        'srcblock': line.srcblock,
                        'srcport': line.srcport,
                        'dstblock': line.dstblock,
                        'dstport': line.dstport
                    }
                    state['lines'].append(line_data)

                pickle.dump(state, f)

            logger.debug("Auto-save completed")

        except Exception as e:
            logger.error(f"Error during auto-save: {str(e)}")

    def _check_autosave_recovery(self):
        """Check if there's an auto-save file and offer recovery."""
        try:
            if not os.path.exists(self.autosave_path):
                return

            # Ask user if they want to recover
            reply = QMessageBox.question(
                self,
                "Recover Auto-Saved Diagram?",
                "An auto-saved diagram was found. This may be from an unexpected shutdown.\n\n"
                "Would you like to recover it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Yes:
                self._recover_autosave()
            else:
                # User declined, remove autosave file
                os.remove(self.autosave_path)
                logger.info("Auto-save file deleted (user declined recovery)")

        except Exception as e:
            logger.error(f"Error checking auto-save recovery: {str(e)}")

    def _recover_autosave(self):
        """Recover diagram from auto-save file."""
        try:
            import pickle
            from PyQt5.QtCore import QRect
            from PyQt5.QtGui import QColor

            with open(self.autosave_path, 'rb') as f:
                state = pickle.load(f)

            # Clear current diagram
            self.canvas.dsim.blocks_list.clear()
            self.canvas.dsim.line_list.clear()

            # Restore blocks
            for block_data in state['blocks']:
                coords = QRect(*block_data['coords'])
                block = self.canvas.dsim.add_new_block(
                    block_data['block_fn'],
                    coords,
                    QColor(block_data['color']),
                    block_data['in_ports'],
                    block_data['out_ports'],
                    block_data['b_type'],
                    block_data['io_edit'],
                    block_data['fn_name'],
                    block_data['params'],
                    block_data['external']
                )
                if block:
                    block.name = block_data['name']

            # Restore connections
            for line_data in state['lines']:
                # Find blocks by name
                src_block = None
                dst_block = None
                for block in self.canvas.dsim.blocks_list:
                    if block.name == line_data['srcblock']:
                        src_block = block
                    if block.name == line_data['dstblock']:
                        dst_block = block

                if src_block and dst_block:
                    src_port_pos = src_block.out_coords[line_data['srcport']]
                    dst_port_pos = dst_block.in_coords[line_data['dstport']]

                    line = self.canvas.dsim.add_line(
                        (line_data['srcblock'], line_data['srcport'], src_port_pos),
                        (line_data['dstblock'], line_data['dstport'], dst_port_pos)
                    )
                    if line:
                        line.name = line_data['name']

            # Remove autosave file after successful recovery
            os.remove(self.autosave_path)

            self.canvas.update()
            self.status_message.setText("Diagram recovered from auto-save")
            logger.info("Diagram successfully recovered from auto-save")

            QMessageBox.information(
                self,
                "Recovery Successful",
                "Your diagram has been successfully recovered from the auto-save file."
            )

        except Exception as e:
            logger.error(f"Error recovering from auto-save: {str(e)}")
            QMessageBox.warning(
                self,
                "Recovery Failed",
                f"Failed to recover diagram from auto-save:\n{str(e)}"
            )

    def _cleanup_autosave(self):
        """Clean up auto-save file on normal shutdown."""
        try:
            if os.path.exists(self.autosave_path):
                os.remove(self.autosave_path)
                logger.debug("Auto-save file cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up auto-save: {str(e)}")