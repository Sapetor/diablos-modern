"""
Modern Main Window for DiaBloS
Features modern layout, theming, and enhanced user interface.
"""

import sys
import logging
import ast
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QMenuBar, QStatusBar, QLabel, QFrame,
                             QApplication, QMessageBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

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
        
        # Initialize DSim components
        self.dsim.main_buttons_init()

        
        # Setup update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.safe_update)
        self.update_timer.start(int(1000 / self.dsim.FPS))
        
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

        # Calculate responsive window size based on available screen space
        if self.screen_geometry:
            # Use 90% of screen width and 90% of screen height for larger window
            target_width = min(int(self.screen_geometry.width() * 0.90), 1600)
            target_height = min(int(self.screen_geometry.height() * 0.90), 1000)

            # Set minimum size to 70% of target, but not less than 800x600
            min_width = max(int(target_width * 0.70), 800)
            min_height = max(int(target_height * 0.70), 600)

            self.setMinimumSize(min_width, min_height)
            self.resize(target_width, target_height)
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
        file_menu.addAction("E&xit\tAlt+F4", self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Undo\tCtrl+Z", self.undo_action)
        edit_menu.addAction("&Redo\tCtrl+Y", self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction("Select &All\tCtrl+A", self.select_all)
        
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
        view_menu.addAction("Toggle &Theme\tCtrl+T", self.toggle_theme)
        view_menu.addSeparator()
        scaling_menu = view_menu.addMenu("UI Scale")
        action_100 = scaling_menu.addAction("100%")
        action_100.triggered.connect(lambda: self._set_scaling(1.0))
        action_125 = scaling_menu.addAction("125%")
        action_125.triggered.connect(lambda: self._set_scaling(1.25))
        action_150 = scaling_menu.addAction("150%")
        action_150.triggered.connect(lambda: self._set_scaling(1.5))
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
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

        # Set initial splitter sizes based on window width
        # Property panel should be ~20% of width, canvas gets the rest
        if self.screen_geometry:
            total_width = int(self.screen_geometry.width() * 0.90)  # Account for window chrome
            property_width = int(total_width * 0.20)
            canvas_width = total_width - property_width
            center_splitter.setSizes([canvas_width, property_width])
        else:
            center_splitter.setSizes([1000, 300])  # Fallback for old behavior

        main_splitter.addWidget(center_splitter)

        # Set stretch factors: center (canvas+properties) gets priority when resizing
        main_splitter.setStretchFactor(0, 0)  # Left panel stays fixed size
        main_splitter.setStretchFactor(1, 1)  # Center area stretches

        # Set main splitter sizes based on window width
        # Left panel should be ~15% of width, center gets the rest
        if self.screen_geometry:
            total_width = int(self.screen_geometry.width() * 0.90)
            left_width = int(total_width * 0.15)
            center_width = total_width - left_width
            main_splitter.setSizes([left_width, center_width])
        else:
            main_splitter.setSizes([250, 1350])  # Fallback for old behavior
        
        # Store splitters for theme updates
        self.main_splitter = main_splitter
        self.center_splitter = center_splitter
    
    def _create_left_panel(self) -> QWidget:
        """Create modern left panel for block palette."""
        panel = QFrame()
        panel.setObjectName("ModernPanel")
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(300)
        
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
        """Create modern canvas area with responsive sizing."""
        # Create the modern canvas widget
        self.canvas = ModernCanvas(self.dsim)

        # Set responsive minimum size for canvas
        if self.screen_geometry:
            # Minimum size should be reasonable for drawing
            min_width = max(int(self.screen_geometry.width() * 0.50), 700)
            min_height = max(int(self.screen_geometry.height() * 0.60), 500)
            self.canvas.setMinimumSize(min_width, min_height)
        else:
            self.canvas.setMinimumSize(900, 700)  # Fallback - larger canvas
        
        # Connect canvas signals
        self.canvas.block_selected.connect(self._on_block_selected)
        self.canvas.connection_created.connect(self._on_connection_created)
        self.canvas.simulation_status_changed.connect(self._on_simulation_status_changed)
        
        return self.canvas
    
    def _create_property_panel(self) -> QWidget:
        """Create modern property panel on right side with size constraints."""
        panel = QFrame()
        panel.setObjectName("ModernPanel")
        panel.setFrameStyle(QFrame.StyledPanel)

        # Set size constraints for vertical panel on right side
        # Minimum width: enough to show property labels and controls
        # Maximum width: 30% of window width to prevent covering canvas
        panel.setMinimumWidth(250)
        if self.screen_geometry:
            max_width = int(self.screen_geometry.width() * 0.30)
            panel.setMaximumWidth(max_width)
        else:
            panel.setMaximumWidth(400)  # Fallback max width

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
        
        # Update canvas area styling
        self.canvas_area.setStyleSheet(f"""
            #CanvasArea {{
                background-color: {theme_manager.get_color('canvas_background').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 6px;
            }}
        """)
    
    # Menu action handlers (simplified for Phase 1)
    def undo_action(self): pass
    def redo_action(self): pass
    def select_all(self): pass
    def zoom_in(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_in()
            self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")

    def zoom_out(self):
        if hasattr(self, 'canvas'):
            self.canvas.zoom_out()
            self.zoom_status.setText(f"{int(self.canvas.zoom_factor * 100)}%")
    def fit_to_window(self): pass
    
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
        if hasattr(self, 'canvas'):
            self.canvas.start_simulation()
        else:
            self.status_message.setText("Canvas not available")
    
    def stop_simulation(self):
        """Stop simulation."""
        if hasattr(self, 'canvas'):
            self.canvas.stop_simulation()
        self.toolbar.set_simulation_state(False, False)
        self.status_message.setText("Simulation stopped")