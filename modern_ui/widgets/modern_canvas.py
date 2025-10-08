"""Modern Canvas Widget for DiaBloS Phase 2
Handles block rendering, mouse interactions, and drag-and-drop functionality.
"""

import logging
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QMenu, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, pyqtSignal
import pyqtgraph as pg
from scipy import signal
import numpy as np
from PyQt5.QtGui import QPainter, QPen, QColor

# Import DSim and helper modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.lib import DSim
from lib.improvements import PerformanceHelper, SafetyChecks, ValidationHelper, SimulationConfig
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class State:
    """State enumeration for canvas interactions."""
    IDLE = "idle"
    DRAGGING = "dragging"
    DRAGGING_BLOCK = "dragging_block"
    DRAGGING_LINE_POINT = "dragging_line_point"
    DRAGGING_LINE_SEGMENT = "dragging_line_segment"
    CONNECTING = "connecting"
    CONFIGURING = "configuring"


class ModernCanvas(QWidget):
    """Modern canvas widget for DiaBloS block diagram editing."""
    
    # Signals
    block_selected = pyqtSignal(object)  # Emitted when a block is selected
    connection_created = pyqtSignal(object, object)  # Emitted when a connection is made
    simulation_status_changed = pyqtSignal(str)  # Emitted when simulation status changes
    
    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        
        # Initialize core DSim functionality
        self.dsim = dsim
        
        # Performance monitoring
        self.perf_helper = PerformanceHelper()
        
        # Simulation configuration
        self.sim_config = SimulationConfig()
        
        # State management
        self.state = State.IDLE
        self.dragging_block = None
        self.drag_offset = None
        
        # Connection management
        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None
        self.source_block_for_connection = None

        # Zoom and Pan
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.panning = False
        self.last_pan_pos = QPoint(0, 0)

        # Clipboard for copy-paste
        self.clipboard_blocks = []

        # Setup canvas
        self._setup_canvas()
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        logger.info("Modern canvas initialized successfully")
    
    def _setup_canvas(self):
        """Setup canvas properties and styling."""
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)  # Enable mouse tracking for hover effects
        self.setFocusPolicy(Qt.StrongFocus)  # Allow keyboard focus
        
        # Apply theme-aware styling
        self._update_theme_styling()
        
        # Connect to theme changes
        theme_manager.theme_changed.connect(self._update_theme_styling)
    
    
    
    def _update_theme_styling(self):
        """Update canvas styling based on current theme."""
        canvas_bg = theme_manager.get_color('canvas_background')
        border_color = theme_manager.get_color('border_primary')
        
        self.setStyleSheet(f"""
            ModernCanvas {{
                background-color: {canvas_bg.name()};
                border: 1px solid {border_color.name()};
                border-radius: 6px;
            }}
        """)
    
    def add_block_from_palette(self, menu_block, position):
        """Add a new block from the palette at the specified position."""
        try:
            block_name = getattr(menu_block, 'fn_name', 'Unknown')
            logger.info(f"Adding block from palette: {block_name} at ({position.x()}, {position.y()})")
            
            # Add new block using DSim
            if hasattr(self.dsim, 'add_block'):
                new_block = self.dsim.add_block(menu_block, position)
                if new_block:
                    logger.info(f"Successfully added {block_name}")
                    
                    # Emit signal
                    self.block_selected.emit(new_block)
                    
                    # Trigger repaint
                    self.update()
                    
                    return new_block
                else:
                    logger.error(f"Failed to create block {block_name}")
            else:
                logger.error("DSim does not have add_block method")
                
        except Exception as e:
            logger.error(f"Error adding block from palette: {str(e)}")
        
        return None
    
    def start_simulation(self):
        """Start simulation with validation."""
        try:
            logger.info("Starting simulation from canvas")
            
            # Run validation first
            is_valid, errors = ValidationHelper.validate_block_connections(
                self.dsim.blocks_list, self.dsim.line_list
            )
            
            if not is_valid:
                error_msg = "\n".join(errors)
                logger.error(f"Simulation validation failed: {error_msg}")
                self.simulation_status_changed.emit(f"Validation failed: {error_msg}")
                return False
            
            # Check simulation state safety
            is_safe, safety_errors = SafetyChecks.check_simulation_state(self.dsim)
            if not is_safe:
                error_msg = "\n".join(safety_errors)
                logger.error(f"Simulation safety check failed: {error_msg}")
                self.simulation_status_changed.emit(f"Safety check failed: {error_msg}")
                return False
            
            # Start simulation
            if hasattr(self.dsim, 'execution_init'):
                success = self.dsim.execution_init()
                if success:
                    if self.dsim.real_time:
                        self.simulation_status_changed.emit("Simulation started")
                        logger.info("Simulation started successfully")
                        return True
                    else:
                        self.run_batch_simulation()
                        return True
                else:
                    error_msg = self.dsim.error_msg if hasattr(self.dsim, 'error_msg') and self.dsim.error_msg else "Algebraic loop detected or invalid diagram."
                    logger.error(f"Simulation initialization failed. {error_msg}")
                    self.simulation_status_changed.emit(f"Simulation failed to start. {error_msg}")
                    # Also pop up a message box
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Critical)
                    msgBox.setText("Simulation Failed to Start")
                    msgBox.setInformativeText(error_msg)
                    msgBox.setWindowTitle("Simulation Error")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec_()
                    return False
            else:
                logger.error("DSim does not have execution_initialize method")
                self.simulation_status_changed.emit("Simulation start failed")
                return False
                
        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            self.simulation_status_changed.emit(f"Error: {str(e)}")
            return False
    
    def run_batch_simulation(self):
        """Run the simulation in batch mode (as fast as possible)."""
        logger.info("Running simulation in batch mode.")
        self.simulation_status_changed.emit("Running simulation...")
        
        # This will block the UI. In a real application, this should be run in a separate thread.
        self.dsim.execution_batch()
        
        self.simulation_status_changed.emit("Simulation finished")
        logger.info("Batch simulation finished.")
        self.dsim.plot_again()
    
    def stop_simulation(self):
        """Stop simulation safely."""
        try:
            if hasattr(self.dsim, 'execution_initialized'):
                self.dsim.execution_initialized = False
            
            self.simulation_status_changed.emit("Simulation stopped")
            logger.info("Simulation stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")
    
    def get_simulation_time(self):
        """Get current simulation time."""
        if hasattr(self.dsim, 't'):
            return getattr(self.dsim, 't', 0.0)
        return 0.0
    
    def is_simulation_running(self):
        """Check if simulation is running."""
        return getattr(self.dsim, 'execution_initialized', False)
    
    def paintEvent(self, event):
        """Paint the canvas with blocks, connections, and other elements."""
        try:
            self.perf_helper.start_timer("canvas_paint")
            
            # Create painter
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)
            
            # Clear canvas with theme-appropriate background
            painter.fillRect(self.rect(), theme_manager.get_color('canvas_background'))

            # Draw grid
            grid_size = 10
            grid_color = theme_manager.get_color('grid_dots')
            painter.setPen(QPen(grid_color, 1)) # 1 pixel wide dots

            for x in range(0, self.width(), grid_size):
                for y in range(0, self.height(), grid_size):
                    painter.drawPoint(x, y)
            
            # Draw DSim elements
            if hasattr(self.dsim, 'display_blocks'):
                self.dsim.display_blocks(painter)
            
            if hasattr(self.dsim, 'display_lines'):
                self.dsim.display_lines(painter)
            
            # Draw temporary connection line
            if self.line_creation_state == 'start' and self.temp_line:
                pen = QPen(theme_manager.get_color('accent_primary'), 2, Qt.DashLine)
                painter.setPen(pen)
                start_point, end_point = self.temp_line
                painter.drawLine(start_point, end_point)
            
            painter.end()
            
            paint_duration = self.perf_helper.end_timer("canvas_paint")
            
            # Log slow paint events
            if paint_duration and paint_duration > 0.05:
                logger.warning(f"Slow canvas paint: {paint_duration:.4f}s")
                
        except Exception as e:
            logger.error(f"Error in canvas paintEvent: {str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        try:
            if event.button() == Qt.MiddleButton:
                self.panning = True
                self.last_pan_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                return

            pos = self.screen_to_world(event.pos())
            logger.debug(f"Canvas mouse press at ({pos.x()}, {pos.y()})")
            
            if event.button() == Qt.LeftButton:
                self._handle_left_click(pos)
            elif event.button() == Qt.RightButton:
                self._handle_right_click(pos)
                
        except Exception as e:
            logger.error(f"Error in canvas mousePressEvent: {str(e)}")
    
    def _handle_left_click(self, pos):
        """Handle left mouse button clicks."""
        try:
            # IMPORTANT: Check for port clicks FIRST, before block clicks
            # This allows clicking on ports to create connections instead of dragging blocks
            port_clicked = self._check_port_clicks(pos)
            if port_clicked:
                return

            # Check for block clicks (only if no port was clicked)
            clicked_block = self._get_clicked_block(pos)
            if clicked_block:
                self._handle_block_click(clicked_block, pos)
                return

            # Check for line clicks
            clicked_line, collision_result = self._get_clicked_line(pos)
            if clicked_line:
                self._handle_line_click(clicked_line, collision_result, pos)
                return

            # Cancel any ongoing line creation if clicking on empty area
            if self.line_creation_state:
                self._cancel_line_creation()
            else:
                # If no item was clicked, clear all selections
                self._clear_selections()

        except Exception as e:
            logger.error(f"Error in _handle_left_click: {str(e)}")
    
    def _handle_right_click(self, pos):
        """Handle right mouse button clicks."""
        try:
            clicked_block = self._get_clicked_block(pos)
            if clicked_block:
                if clicked_block.block_fn == "BodeMag":
                    self.show_bode_plot_menu(clicked_block, self.mapToGlobal(pos))
                elif clicked_block.block_fn == "RootLocus":
                    self.show_root_locus_menu(clicked_block, self.mapToGlobal(pos))
                else:
                    # For other blocks, you might want a different context menu
                    # or the parameter dialog from older versions.
                    pass
                return

            # Cancel any ongoing line creation
            if self.line_creation_state:
                self._cancel_line_creation()

        except Exception as e:
            logger.error(f"Error in _handle_right_click: {str(e)}")

    def show_bode_plot_menu(self, block, pos):
        """Show context menu for the BodeMag block."""
        menu = QMenu(self)
        plot_action = menu.addAction("Generate Bode Plot")
        action = menu.exec_(pos)
        if action == plot_action:
            self.generate_bode_plot(block)

    def generate_bode_plot(self, bode_block):
        """Find the connected transfer function, calculate, and plot the Bode diagram."""
        # 1. Find the connected transfer function block
        source_block = None
        for line in self.dsim.line_list:
            if line.dstblock == bode_block.name:
                for block in self.dsim.blocks_list:
                    if block.name == line.srcblock:
                        if block.block_fn == 'TranFn':
                            source_block = block
                            break
                if source_block:
                    break
        
        if not source_block:
            QMessageBox.warning(self, "Bode Plot Error", "BodeMag block must be connected to the output of a Transfer Function block.")
            return

        # 2. Get numerator and denominator
        num = source_block.params.get('numerator')
        den = source_block.params.get('denominator')

        if not num or not den:
            QMessageBox.warning(self, "Bode Plot Error", "Connected Transfer Function has invalid parameters.")
            return

        # 3. Calculate Bode plot data
        w, mag, phase = signal.bode((num, den))

        # 4. Display the plot
        plot_window = QWidget()
        plot_window.setWindowTitle(f"Bode Plot: {source_block.name}")
        layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)
        plot_window.setLayout(layout)

        plot_widget.setLogMode(x=True, y=False)
        plot_widget.setLabel('left', 'Magnitude', units='dB')
        plot_widget.setLabel('bottom', 'Frequency', units='rad/s')
        plot_widget.setTitle(f"Bode Magnitude Plot: {source_block.name}")
        plot_widget.plot(w, mag)
        plot_widget.showGrid(x=True, y=True)

        if not hasattr(self, 'plot_windows'):
            self.plot_windows = []
        self.plot_windows.append(plot_window)
        plot_window.show()

    def show_root_locus_menu(self, block, pos):
        """Show context menu for the RootLocus block."""
        menu = QMenu(self)
        plot_action = menu.addAction("Generate Root Locus Plot")
        action = menu.exec_(pos)
        if action == plot_action:
            self.generate_root_locus(block)

    def generate_root_locus(self, rootlocus_block):
        """Find the connected transfer function, calculate, and plot the root locus."""
        # 1. Find the connected transfer function block
        source_block = None
        for line in self.dsim.line_list:
            if line.dstblock == rootlocus_block.name:
                for block in self.dsim.blocks_list:
                    if block.name == line.srcblock:
                        if block.block_fn == 'TranFn':
                            source_block = block
                            break
                if source_block:
                    break

        if not source_block:
            QMessageBox.warning(self, "Root Locus Error", "RootLocus block must be connected to the output of a Transfer Function block.")
            return

        # 2. Get numerator and denominator
        num = source_block.params.get('numerator')
        den = source_block.params.get('denominator')

        if not num or not den:
            QMessageBox.warning(self, "Root Locus Error", "Connected Transfer Function has invalid parameters.")
            return

        # 3. Calculate root locus using scipy
        try:
            # Ensure num and den are numpy arrays
            num = np.atleast_1d(num)
            den = np.atleast_1d(den)

            # Pad numerator to same length as denominator
            if len(num) < len(den):
                num = np.pad(num, (len(den) - len(num), 0), 'constant')
            elif len(den) < len(num):
                den = np.pad(den, (len(num) - len(den), 0), 'constant')

            # Calculate root locus for gains from 0 to a reasonable maximum
            # Characteristic equation: 1 + K*G(s) = 0
            # => den(s) + K*num(s) = 0
            k_values = np.concatenate([
                np.linspace(0, 1, 150),           # Fine detail near K=0
                np.linspace(1, 10, 150),          # Fine detail 1-10
                np.logspace(1, 4, 400)            # Logarithmic from 10 to 10000
            ])

            # Store roots for each K value to track branches
            all_roots = []
            for k in k_values:
                # Characteristic equation: den(s) + k*num(s) = 0
                char_poly = den + k * num
                roots = np.roots(char_poly)
                all_roots.append(roots)

            # Convert to numpy array for easier manipulation
            all_roots = np.array(all_roots)  # Shape: (num_k_values, num_poles)

            # Track branches: for each pole, follow its path as K increases
            num_poles = all_roots.shape[1]

            # Sort roots at each K to maintain branch continuity
            # Start with the first set of roots
            sorted_roots = [all_roots[0]]
            for i in range(1, len(all_roots)):
                current = all_roots[i]
                previous = sorted_roots[-1]

                # Match each current root to the closest previous root
                # This helps maintain branch continuity
                matched = []
                available = list(range(len(current)))

                for prev_root in previous:
                    if not available:
                        break
                    # Find closest root
                    distances = [abs(current[j] - prev_root) for j in available]
                    closest_idx = available[np.argmin(distances)]
                    matched.append(current[closest_idx])
                    available.remove(closest_idx)

                # Add any remaining unmatched roots
                for idx in available:
                    matched.append(current[idx])

                sorted_roots.append(np.array(matched))

            sorted_roots = np.array(sorted_roots)

            # 4. Display the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Root Locus: {source_block.name}")
            layout = QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)

            plot_widget.setLabel('left', 'Imaginary Axis', units='rad/s')
            plot_widget.setLabel('bottom', 'Real Axis', units='1/s')
            plot_widget.setTitle(f"Root Locus Plot: {source_block.name}")

            # Plot each branch as a continuous line
            colors = [(100, 149, 237), (237, 100, 100), (100, 237, 149),
                     (237, 149, 100), (149, 100, 237), (237, 237, 100)]

            for pole_idx in range(num_poles):
                branch_real = sorted_roots[:, pole_idx].real
                branch_imag = sorted_roots[:, pole_idx].imag
                color = colors[pole_idx % len(colors)]
                plot_widget.plot(branch_real, branch_imag,
                               pen=pg.mkPen(color, width=2),
                               name=f'Branch {pole_idx+1}' if pole_idx == 0 else None)

            # Mark the open-loop poles (roots of denominator at K=0)
            poles = np.roots(den)
            plot_widget.plot(poles.real, poles.imag, pen=None, symbol='x', symbolSize=12,
                           symbolPen=pg.mkPen('r', width=3), name='Poles')

            # Mark the open-loop zeros (roots of numerator)
            if np.any(num):  # Only if numerator is not all zeros
                zeros = np.roots(num)
                if len(zeros) > 0:
                    plot_widget.plot(zeros.real, zeros.imag, pen=None, symbol='o', symbolSize=12,
                                   symbolPen=pg.mkPen('g', width=3), symbolBrush=None, name='Zeros')

            # Add axes lines
            plot_widget.addLine(x=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
            plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))

            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()

            if not hasattr(self, 'plot_windows'):
                self.plot_windows = []
            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            QMessageBox.critical(self, "Root Locus Error", f"Error calculating root locus: {str(e)}")
            logger.error(f"Root locus calculation error: {str(e)}")


    def _get_clicked_block(self, pos):
        for block in reversed(getattr(self.dsim, 'blocks_list', [])):
            if hasattr(block, 'rect') and block.rect.contains(pos):
                return block
        return None

    def _get_clicked_line(self, pos):
        for line in getattr(self.dsim, 'line_list', []):
            if hasattr(line, 'collision'):
                result = line.collision(pos)
                if result:
                    return line, result
        return None, None

    def _clear_selections(self):
        for block in getattr(self.dsim, 'blocks_list', []):
            block.selected = False
        for line in getattr(self.dsim, 'line_list', []):
            line.selected = False
            if hasattr(line, 'selected_segment'):
                line.selected_segment = -1
        self.source_block_for_connection = None
        self.update()

    def _handle_block_click(self, block, pos):
        """Handle clicking on a block."""
        try:
            logger.debug(f"Block clicked: {getattr(block, 'fn_name', 'Unknown')}")

            # NEW: Connection logic with Ctrl+Click
            if QApplication.keyboardModifiers() & Qt.ControlModifier and self.source_block_for_connection and self.source_block_for_connection is not block:
                source_block = self.source_block_for_connection
                target_block = block

                if source_block.out_ports > 0:
                    source_port_index = 0  # Use the first output port

                    # Find an available input port on the target block
                    connected_input_ports = {line.dstport for line in self.dsim.line_list if line.dstblock == target_block.name}
                    target_port_index = -1
                    for i in range(target_block.in_ports):
                        if i not in connected_input_ports:
                            target_port_index = i
                            break
                    
                    if target_port_index != -1:
                        logger.info(f"Creating connection from {source_block.name} to {target_block.name}")
                        self.line_start_block = source_block
                        self.line_start_port = source_port_index
                        self._finish_line_creation(target_block, target_port_index)
                        # Make target block selected and source for next connection
                        self.source_block_for_connection.selected = False
                        target_block.selected = True
                        self.source_block_for_connection = target_block
                        self.update()
                    else:
                        logger.warning(f"Could not connect: No available input ports on {target_block.name}")
                    
                    return # End of connection logic for this click

            # Original logic for selection and dragging
            if QApplication.keyboardModifiers() & Qt.ControlModifier:
                block.toggle_selection()
            else:
                self._clear_selections()
                block.selected = True
                self.source_block_for_connection = block # Set source for connection

            # Start dragging the block
            self.start_drag(block, pos)

            # Emit selection signal
            self.block_selected.emit(block)
            self.update()
        except Exception as e:
            logger.error(f"Error in _handle_block_click: {str(e)}")

    def _check_port_clicks(self, pos):
        """Check for port clicks to create connections. Returns True if a port was clicked."""
        try:
            # Check all blocks for port collisions
            for block in getattr(self.dsim, 'blocks_list', []):
                if hasattr(block, 'port_collision'):
                    # Convert QPoint to tuple for collision detection
                    point_tuple = (pos.x(), pos.y())
                    port_result = block.port_collision(point_tuple)
                    if port_result != (-1, -1):
                        port_type, port_index = port_result
                        logger.debug(f"Port clicked: {port_type}{port_index} on block {getattr(block, 'name', 'Unknown')}")
                        self._handle_port_click(block, port_type, port_index, pos)
                        return True # Port was clicked
            return False # No port was clicked
        except Exception as e:
            logger.error(f"Error in _check_port_clicks: {str(e)}")
            return False

    def _handle_port_click(self, block, port_type, port_index, pos):
        """Handle port click for connection creation."""
        try:
            block_name = getattr(block, 'name', 'Unknown')
            logger.debug(f"Port clicked on block {block_name}, port: {port_type}{port_index}")

            if self.line_creation_state is None:
                if port_type == 'o': # Start line from output port
                    self.state = State.CONNECTING
                    self.line_creation_state = 'start'
                    self.line_start_block = block
                    self.line_start_port = port_index
                    # Get the output port coordinates
                    if hasattr(block, 'out_coords') and port_index < len(block.out_coords):
                        start_point = block.out_coords[port_index]
                        self.temp_line = (start_point, pos)
                    logger.info(f"Started line creation from {block_name} output port {port_index}")
            elif self.line_creation_state == 'start':
                if port_type == 'i': # End line at input port
                    logger.info(f"Completing line to {block_name} input port {port_index}")
                    self._finish_line_creation(block, port_index)
                else:
                    logger.info("Canceling line creation - clicked on output port")
                    self._cancel_line_creation()
            self.update()
        except Exception as e:
            logger.error(f"Error in _handle_port_click: {str(e)}")

    def _finish_line_creation(self, end_block, end_port):
        """Complete line creation between two blocks."""
        try:
            start_block_name = getattr(self.line_start_block, 'name', 'Unknown')
            end_block_name = getattr(end_block, 'name', 'Unknown')
            logger.debug(f"Finishing line creation from {start_block_name} to {end_block_name}")

            if hasattr(self.dsim, 'add_line'):
                # Get coordinates for the line
                start_coords = None
                end_coords = None
                if (hasattr(self.line_start_block, 'out_coords') and
                    self.line_start_port < len(self.line_start_block.out_coords)):
                    start_coords = self.line_start_block.out_coords[self.line_start_port]
                if (hasattr(end_block, 'in_coords') and
                    end_port < len(end_block.in_coords)):
                    end_coords = end_block.in_coords[end_port]

                if start_coords and end_coords:
                    # Validate connection before creating
                    is_valid, validation_errors = self._validate_connection(
                        self.line_start_block, self.line_start_port,
                        end_block, end_port
                    )
                    if not is_valid:
                        error_msg = "\n".join(validation_errors)
                        logger.warning(f"Connection validation failed: {error_msg}")
                        self.simulation_status_changed.emit(f"Connection invalid: {error_msg}")
                        self._cancel_line_creation()
                        return

                    # Create line using DSim's add_line method
                    new_line = self.dsim.add_line(
                        (start_block_name, self.line_start_port, start_coords),
                        (end_block_name, end_port, end_coords)
                    )
                    if new_line:
                        logger.info(f"Line created: {start_block_name} -> {end_block_name}")
                        self.dsim.update_lines()
                        self.connection_created.emit(self.line_start_block, end_block)
                    else:
                        logger.warning("Failed to create line")
                else:
                    logger.error("Could not get port coordinates for line creation")
            self._cancel_line_creation()
        except Exception as e:
            logger.error(f"Error in _finish_line_creation: {str(e)}")
            self._cancel_line_creation()

    def _check_line_clicks(self, pos):
        """Check for clicks on connection lines."""
        try:
            # Check for clicks near existing lines
            for line in getattr(self.dsim, 'line_list', []):
                if hasattr(line, 'points') and self._point_near_line(pos, line):
                    self._handle_line_click(line, pos)
                    return
        except Exception as e:
            logger.error(f"Error in _check_line_clicks: {str(e)}")

    def _point_near_line(self, pos, line):
        """Check if a point is near a line."""
        try:
            if not hasattr(line, 'points') or len(line.points) < 2:
                return False

            threshold = 10 # pixels
            point_tuple = (pos.x(), pos.y())

            # Check each segment of the line
            for i in range(len(line.points) - 1):
                start = line.points[i]
                end = line.points[i + 1]

                # Convert QPoint to tuple if needed
                if hasattr(start, 'x'):
                    start = (start.x(), start.y())
                if hasattr(end, 'x'):
                    end = (end.x(), end.y())

                # Calculate distance from point to line segment
                distance = self._point_to_line_distance(point_tuple, start, end)
                if distance <= threshold:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in _point_near_line: {str(e)}")
            return False

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate minimum distance from point to line segment."""
        try:
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            # Calculate the distance
            A = x0 - x1
            B = y0 - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D

            if len_sq == 0: # Line start and end are the same point
                return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

            param = dot / len_sq

            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dx = x0 - xx
            dy = y0 - yy
            return (dx * dx + dy * dy) ** 0.5
        except Exception as e:
            logger.error(f"Error calculating point to line distance: {str(e)}")
            return float('inf')

    def _handle_line_click(self, line, collision_result, pos):
        """Handle clicking on a connection line."""
        try:
            line_name = getattr(line, 'name', 'Unknown')
            logger.info(f"Line clicked: {line_name}")

            collision_type, collision_index = collision_result

            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                self._clear_selections()
            
            line.selected = True
            line.modified = True # Always allow modification on click

            if collision_type == "point":
                self.state = State.DRAGGING_LINE_POINT
                self.dragging_item = (line, collision_index)
                self.drag_offset = pos
                line.selected_segment = -1 # A point is selected, not a segment
                logger.info(f"Dragging point {collision_index} of line {line_name}")
            elif collision_type == "segment":
                self.state = State.DRAGGING_LINE_SEGMENT
                self.dragging_item = (line, collision_index)
                self.drag_offset = pos
                line.selected_segment = collision_index # A segment is selected
                logger.info(f"Dragging segment {collision_index} of line {line_name}")
            else: # "line" or None
                line.selected_segment = -1 # The whole line is selected
            
            self.update()
        except Exception as e:
            logger.error(f"Error in _handle_line_click: {str(e)}")
        except Exception as e:
            logger.error(f"Error in _handle_line_click: {str(e)}")

    def _configure_block(self, block):
        """Configure a block (right-click action)."""
        try:
            logger.info(f"Configuring block: {getattr(block, 'fn_name', 'Unknown')}")
            # Use DSim's configuration dialog
            if hasattr(self.dsim, 'configure_block'):
                self.dsim.configure_block(block)
        except Exception as e:
            logger.error(f"Error configuring block: {str(e)}")

    def start_drag(self, block, pos):
        """Start dragging a block."""
        try:
            self.state = State.DRAGGING
            self.dragging_block = block
            # Calculate drag offset based on block's top-left corner
            self.drag_offset = QPoint(pos.x() - block.left, pos.y() - block.top)
            logger.debug(f"Started dragging block: {getattr(block, 'fn_name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        try:
            if self.panning:
                delta = event.pos() - self.last_pan_pos
                self.pan_offset += delta
                self.last_pan_pos = event.pos()
                self.update()
                return

            pos = self.screen_to_world(event.pos())

            if self.state == State.DRAGGING and self.dragging_block:
                # Update block position
                # Calculate new top-left position for the block
                new_x = pos.x() - self.drag_offset.x()
                new_y = pos.y() - self.drag_offset.y()

                # Snap to grid
                grid_size = 10
                snapped_x = round(new_x / grid_size) * grid_size
                snapped_y = round(new_y / grid_size) * grid_size

                # Update block position using DBlock's relocate_Block method
                self.dragging_block.relocate_Block(QPoint(snapped_x, snapped_y))
                self.dsim.update_lines() # Update lines dynamically during drag
                self.update() # Trigger repaint
            elif self.state == State.DRAGGING_LINE_POINT and self.dragging_item:
                line, point_index = self.dragging_item
                line.points[point_index] = pos
                line.path, line.points, line.segments = line.create_trajectory(line.points[0], line.points[-1], self.dsim.blocks_list, line.points)
                self.update()
            elif self.state == State.DRAGGING_LINE_SEGMENT and self.dragging_item:
                line, segment_index = self.dragging_item
                p1 = line.points[segment_index]
                p2 = line.points[segment_index + 1]
                is_horizontal = abs(p1.x() - p2.x()) > abs(p1.y() - p2.y())

                # If the line has only two points and we are dragging a segment, we need to add a new point to create a corner
                if len(line.points) == 2:
                    if is_horizontal:
                        new_point = QPoint(int(p1.x() + (p2.x() - p1.x()) // 2), int(pos.y()))
                    else:
                        new_point = QPoint(int(pos.x()), int(p1.y() + (p2.y() - p1.y()) // 2))
                    line.points.insert(1, new_point)
                    self.dragging_item = (line, 1 if pos.y() > p1.y() else 0)
                    segment_index = self.dragging_item[1]

                if is_horizontal:
                    # Move horizontal segment vertically
                    line.points[segment_index].setY(pos.y())
                    line.points[segment_index + 1].setY(pos.y())
                else:
                    # Move vertical segment horizontally
                    line.points[segment_index].setX(pos.x())
                    line.points[segment_index + 1].setX(pos.x())
                
                # Regenerate the trajectory with the updated points
                line.path, line.points, line.segments = line.create_trajectory(line.points[0], line.points[-1], self.dsim.blocks_list, line.points)
                self.update()
            elif self.line_creation_state == 'start' and self.temp_line:
                # Update temporary line endpoint
                self.temp_line = (self.temp_line[0], pos)
                self.update()
        except Exception as e:
            logger.error(f"Error in mouseMoveEvent: {str(e)}")

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        try:
            if event.button() == Qt.MiddleButton:
                self.panning = False
                self.setCursor(Qt.ArrowCursor)

            if self.state == State.DRAGGING:
                self._finish_drag()
            elif self.state in [State.DRAGGING_LINE_POINT, State.DRAGGING_LINE_SEGMENT]:
                if self.dragging_item:
                    line, _ = self.dragging_item
                    if hasattr(line, '_stub_created'):
                        del line._stub_created
                self.state = State.IDLE
                self.dragging_item = None
                self.update()

        except Exception as e:
            logger.error(f"Error in mouseReleaseEvent: {str(e)}")

    def _finish_drag(self):
        """Finish dragging operation."""
        try:
            if self.dragging_block:
                logger.debug(f"Finished dragging block: {getattr(self.dragging_block, 'fn_name', 'Unknown')}")
                # Reset drag state
                self.state = State.IDLE
                self.dragging_block = None
                self.drag_offset = None
                self.dsim.update_lines() # Ensure lines are updated after drag finishes
                self.update() # Trigger a final repaint
        except Exception as e:
            logger.error(f"Error finishing drag: {str(e)}")

    def _cancel_line_creation(self):
        """Cancel line creation process."""
        try:
            self.line_creation_state = None
            self.line_start_block = None
            self.line_start_port = None
            self.temp_line = None
            self.state = State.IDLE
            self.update()
            logger.debug("Line creation cancelled")
        except Exception as e:
            logger.error(f"Error cancelling line creation: {str(e)}")

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        try:
            # Check for Control/Command modifier (works on both Mac and Windows/Linux)
            ctrl_pressed = event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)

            if event.key() == Qt.Key_Escape:
                # Cancel any ongoing operations
                if self.line_creation_state:
                    self._cancel_line_creation()
                elif self.state == State.DRAGGING:
                    self._finish_drag()
            elif event.key() == Qt.Key_Delete:
                self.remove_selected_items()
            elif event.key() == Qt.Key_F and ctrl_pressed:
                self.flip_selected_blocks()
            elif event.key() == Qt.Key_C and ctrl_pressed:
                self.copy_selected_blocks()
            elif event.key() == Qt.Key_V and ctrl_pressed:
                self.paste_blocks()
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {str(e)}")

    def flip_selected_blocks(self):
        """Flip selected blocks horizontally."""
        try:
            for block in self.dsim.blocks_list:
                if block.selected:
                    block.flipped = not block.flipped
                    block.update_Block() # Recalculate port positions
            self.dsim.update_lines()
            self.update() # Redraw canvas
            logger.info("Flipped selected blocks")
        except Exception as e:
            logger.error(f"Error flipping blocks: {str(e)}")

    def copy_selected_blocks(self):
        """Copy selected blocks to clipboard."""
        try:
            import copy

            # Find all selected blocks
            selected_blocks = [block for block in self.dsim.blocks_list if block.selected]

            if not selected_blocks:
                logger.info("No blocks selected to copy")
                return

            # Deep copy the block data (not the actual block objects)
            self.clipboard_blocks = []
            for block in selected_blocks:
                block_data = {
                    'block_fn': block.block_fn,
                    'username': block.username,
                    'coords': QRect(block.left, block.top, block.width, block.height_base),
                    'color': block.b_color.name(),
                    'in_ports': block.in_ports,
                    'out_ports': block.out_ports,
                    'b_type': block.b_type,
                    'io_edit': block.io_edit,
                    'fn_name': block.fn_name,
                    'params': copy.deepcopy(block.params),
                    'external': block.external,
                    'flipped': getattr(block, 'flipped', False)
                }
                self.clipboard_blocks.append(block_data)

            logger.info(f"Copied {len(self.clipboard_blocks)} block(s) to clipboard")
        except Exception as e:
            logger.error(f"Error copying blocks: {str(e)}")

    def paste_blocks(self):
        """Paste blocks from clipboard."""
        try:
            if not self.clipboard_blocks:
                logger.info("Clipboard is empty")
                return

            # Deselect all current blocks
            for block in self.dsim.blocks_list:
                block.selected = False

            # Paste offset (so pasted blocks don't overlap exactly)
            paste_offset = QPoint(30, 30)

            # Create new blocks from clipboard
            pasted_blocks = []
            for block_data in self.clipboard_blocks:
                # Offset the position
                new_coords = block_data['coords'].translated(paste_offset)

                # Import DBlock class
                from lib.simulation.block import DBlock

                # Calculate unique ID for this block type (same logic as SimulationModel.add_block)
                block_fn = block_data['block_fn']
                id_list = [int(b_elem.name[len(b_elem.block_fn):])
                           for b_elem in self.dsim.blocks_list
                           if b_elem.block_fn == block_fn]
                sid = max(id_list) + 1 if id_list else 0

                # Find the corresponding MenuBlock to get block_class
                block_class = None
                for menu_block in self.dsim.menu_blocks:
                    if menu_block.block_fn == block_fn:
                        block_class = menu_block.block_class
                        break

                # Create new block with unique ID
                new_block = DBlock(
                    block_fn=block_fn,
                    sid=sid,
                    coords=new_coords,
                    color=block_data['color'],
                    in_ports=block_data['in_ports'],
                    out_ports=block_data['out_ports'],
                    b_type=block_data['b_type'],
                    io_edit=block_data['io_edit'],
                    fn_name=block_data['fn_name'],
                    params=block_data['params'].copy(),
                    external=block_data['external'],
                    username=block_data['username'],
                    block_class=block_class,
                    colors=self.dsim.colors
                )
                new_block.flipped = block_data['flipped']
                new_block.selected = True  # Select the pasted blocks

                # Add to blocks list
                self.dsim.blocks_list.append(new_block)
                pasted_blocks.append(new_block)

            # Mark as dirty
            self.dsim.dirty = True

            # Redraw canvas
            self.update()

            logger.info(f"Pasted {len(pasted_blocks)} block(s)")

            # Emit signal for first pasted block if any
            if pasted_blocks:
                self.block_selected.emit(pasted_blocks[0])

        except Exception as e:
            logger.error(f"Error pasting blocks: {str(e)}")

    def remove_selected_items(self):
        """Remove all selected blocks and lines."""
        try:
            blocks_to_remove = [block for block in self.dsim.blocks_list if block.selected]
            lines_to_remove = [line for line in self.dsim.line_list if line.selected]

            for block in blocks_to_remove:
                self.dsim.remove_block_and_lines(block)
            for line in lines_to_remove:
                if line in self.dsim.line_list:
                    self.dsim.line_list.remove(line)
            self.update()
            logger.info(f"Removed {len(blocks_to_remove)} blocks and {len(lines_to_remove)} lines")
        except Exception as e:
            logger.error(f"Error removing selected items: {str(e)}")

    def clear_canvas(self):
        """Clear all blocks and connections from the canvas."""
        try:
            if hasattr(self.dsim, 'clear_all'):
                self.dsim.clear_all()
                self.update()
                logger.info("Canvas cleared")
        except Exception as e:
            logger.error(f"Error clearing canvas: {str(e)}")

    def get_blocks(self):
        """Get all blocks on the canvas."""
        return getattr(self.dsim, 'blocks_list', [])

    def get_connections(self):
        """Get all connections on the canvas."""
        return getattr(self.dsim, 'line_list', [])

    def screen_to_world(self, pos):
        """Converts screen coordinates to world coordinates."""
        return (pos - self.pan_offset) / self.zoom_factor

    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.update()

    def zoom_in(self):
        self.set_zoom(self.zoom_factor * 1.1)

    def zoom_out(self):
        self.set_zoom(self.zoom_factor / 1.1)

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    # Drag and Drop Events
    def dragEnterEvent(self, event):
        """Handle drag enter events."""
        try:
            if event.mimeData().hasText():
                mime_text = event.mimeData().text()
                if mime_text.startswith("diablo_block:"):
                    event.acceptProposedAction()
                    logger.debug("Drag enter accepted for DiaBloS block")
                else:
                    event.ignore()
            else:
                event.ignore()
        except Exception as e:
            logger.error(f"Error in dragEnterEvent: {str(e)}")
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move events."""
        try:
            if event.mimeData().hasText():
                mime_text = event.mimeData().text()
                if mime_text.startswith("diablo_block:"):
                    event.acceptProposedAction()
                else:
                    event.ignore()
            else:
                event.ignore()
        except Exception as e:
            logger.error(f"Error in dragMoveEvent: {str(e)}")
            event.ignore()

    def dropEvent(self, event):
        """Handle drop events to create blocks."""
        try:
            if event.mimeData().hasText():
                mime_text = event.mimeData().text()
                if mime_text.startswith("diablo_block:"):
                    block_name = mime_text.split(":", 1)[1]
                    drop_pos = self.screen_to_world(event.pos())
                    logger.info(f"Drop event: Creating {block_name} at ({drop_pos.x()}, {drop_pos.y()})")

                    # Find the corresponding menu block
                    menu_block = self._find_menu_block_by_name(block_name)
                    if menu_block:
                        # Create the block at the drop position
                        new_block = self.add_block_from_palette(menu_block, drop_pos)
                        if new_block:
                            event.acceptProposedAction()
                            logger.info(f"Successfully created {block_name} via drag-and-drop")
                        else:
                            logger.error(f"Failed to create {block_name}")
                            event.ignore()
                    else:
                        logger.error(f"Menu block not found: {block_name}")
                        event.ignore()
                else:
                    event.ignore()
            else:
                event.ignore()
        except Exception as e:
            logger.error(f"Error in dropEvent: {str(e)}")
            event.ignore()

    def _find_menu_block_by_name(self, block_name):
        """Find a menu block by its function name."""
        try:
            menu_blocks = getattr(self.dsim, 'menu_blocks', [])
            for menu_block in menu_blocks:
                if getattr(menu_block, 'fn_name', '') == block_name:
                    return menu_block
            return None
        except Exception as e:
            logger.error(f"Error finding menu block {block_name}: {str(e)}")
            return None

    def _validate_connection(self, start_block, start_port, end_block, end_port):
        """Validate a connection between two blocks."""
        try:
            validation_errors = []

            # Basic validation checks
            if start_block == end_block:
                validation_errors.append("Cannot connect a block to itself")

            if end_block.block_fn == "BodeMag" and start_block.block_fn != "TranFn":
                validation_errors.append("BodeMag block can only be connected to a Transfer Function.")

            if end_block.block_fn == "RootLocus" and start_block.block_fn != "TranFn":
                validation_errors.append("RootLocus block can only be connected to a Transfer Function.")

            # Check if connection already exists
            existing_lines = getattr(self.dsim, 'line_list', [])
            for line in existing_lines:
                if (hasattr(line, 'srcblock') and hasattr(line, 'dstblock') and
                    hasattr(line, 'srcport') and hasattr(line, 'dstport')):
                    start_name = getattr(start_block, 'name', '')
                    end_name = getattr(end_block, 'name', '')
                    if (line.srcblock == start_name and line.srcport == start_port and
                        line.dstblock == end_name and line.dstport == end_port):
                        validation_errors.append("Connection already exists")
                        break

            # Check if input port is already connected
            for line in existing_lines:
                if (hasattr(line, 'dstblock') and hasattr(line, 'dstport')):
                    end_name = getattr(end_block, 'name', '')
                    if line.dstblock == end_name and line.dstport == end_port:
                        validation_errors.append("Input port already connected")
                        break

            # Use ValidationHelper if available
            try:
                all_blocks = getattr(self.dsim, 'blocks_list', [])
                all_lines = getattr(self.dsim, 'line_list', [])
                # Create a temporary line list for validation
                temp_lines = list(all_lines)
                # Add our proposed connection for validation
                temp_line = type('TempLine', (), {
                    'srcblock': getattr(start_block, 'name', ''),
                    'srcport': start_port,
                    'dstblock': getattr(end_block, 'name', ''),
                    'dstport': end_port
                })()
                temp_lines.append(temp_line)

                is_valid, helper_errors = ValidationHelper.validate_block_connections(
                    all_blocks, temp_lines
                )
                if not is_valid:
                    validation_errors.extend(helper_errors)
            except Exception as e:
                logger.debug(f"ValidationHelper not available or failed: {str(e)}")


            return len(validation_errors) == 0, validation_errors
        except Exception as e:
            logger.error(f"Error validating connection: {str(e)}")
            return False, [f"Validation error: {str(e)}"]
