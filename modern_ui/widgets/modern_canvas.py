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
        self.drag_offsets = {}  # For multi-block dragging
        
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

        # Rectangle selection
        self.selection_rect_start = None
        self.selection_rect_end = None
        self.is_rect_selecting = False

        # Hover tracking for visual feedback
        self.hovered_block = None
        self.hovered_port = None  # Tuple: (block, port_index, is_output)
        self.hovered_line = None

        # Grid snapping
        self.grid_size = 20  # Snap to 20px grid
        self.snap_enabled = True

        # Undo/Redo system
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50

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

            # Draw sophisticated grid system
            self._draw_grid(painter)
            
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

            # Draw rectangle selection
            if self.is_rect_selecting and self.selection_rect_start and self.selection_rect_end:
                # Save painter state before drawing selection
                painter.save()

                # Calculate normalized rectangle
                x1 = min(self.selection_rect_start.x(), self.selection_rect_end.x())
                y1 = min(self.selection_rect_start.y(), self.selection_rect_end.y())
                x2 = max(self.selection_rect_start.x(), self.selection_rect_end.x())
                y2 = max(self.selection_rect_start.y(), self.selection_rect_end.y())

                selection_rect = QRect(x1, y1, x2 - x1, y2 - y1)

                # Draw semi-transparent blue fill
                fill_color = QColor(100, 149, 237, 50)  # Cornflower blue with alpha
                painter.fillRect(selection_rect, fill_color)

                # Draw blue border
                border_pen = QPen(QColor(100, 149, 237), 2, Qt.DashLine)
                painter.setPen(border_pen)
                painter.setBrush(Qt.NoBrush)  # Explicitly set no brush for border
                painter.drawRect(selection_rect)

                # Restore painter state
                painter.restore()

            # Draw hover effects
            self._draw_hover_effects(painter)

            painter.end()
            
            paint_duration = self.perf_helper.end_timer("canvas_paint")
            
            # Log slow paint events
            if paint_duration and paint_duration > 0.05:
                logger.warning(f"Slow canvas paint: {paint_duration:.4f}s")
                
        except Exception as e:
            logger.error(f"Error in canvas paintEvent: {str(e)}")

    def _draw_grid(self, painter):
        """Draw a sophisticated grid system with dots at intervals."""
        try:
            # Grid configuration
            small_grid_size = 20  # Small dot spacing (20px)
            large_grid_size = 100  # Large dot spacing (100px for emphasis)

            # Get theme colors
            small_dot_color = theme_manager.get_color('grid_dots')
            large_dot_color = theme_manager.get_color('grid_dots')
            large_dot_color.setAlpha(180)  # Make large dots slightly more visible

            # Calculate visible area bounds (considering zoom and pan)
            visible_rect = self.rect()

            # Draw small dots
            painter.setPen(Qt.NoPen)
            painter.setBrush(small_dot_color)
            for x in range(0, self.width(), small_grid_size):
                for y in range(0, self.height(), small_grid_size):
                    # Only draw small dots if not on a large grid intersection
                    if x % large_grid_size != 0 or y % large_grid_size != 0:
                        painter.drawEllipse(QPoint(x, y), 1, 1)

            # Draw larger dots at major grid intersections
            painter.setBrush(large_dot_color)
            for x in range(0, self.width(), large_grid_size):
                for y in range(0, self.height(), large_grid_size):
                    painter.drawEllipse(QPoint(x, y), 2, 2)

        except Exception as e:
            logger.error(f"Error drawing grid: {str(e)}")

    def _draw_hover_effects(self, painter):
        """Draw hover effects for ports, blocks, and connections."""
        try:
            painter.save()

            # Draw hovered port (highest priority)
            if self.hovered_port:
                block, port_idx, is_output = self.hovered_port
                port_list = block.out_coords if is_output else block.in_coords
                if port_idx < len(port_list):
                    port_pos = port_list[port_idx]

                    # Draw pulsing glow around hovered port
                    glow_color = theme_manager.get_color('accent_primary')
                    glow_color.setAlpha(100)
                    painter.setBrush(glow_color)
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(port_pos, 12, 12)

                    # Draw brighter center
                    center_color = theme_manager.get_color('accent_primary')
                    center_color.setAlpha(180)
                    painter.setBrush(center_color)
                    painter.drawEllipse(port_pos, 8, 8)

            # Draw hovered block outline
            elif self.hovered_block and not self.hovered_block.selected:
                block = self.hovered_block
                hover_color = theme_manager.get_color('accent_secondary')
                hover_color.setAlpha(120)

                # Draw glowing outline
                painter.setPen(QPen(hover_color, 2.5, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(
                    block.left - 2,
                    block.top - 2,
                    block.width + 4,
                    block.height + 4,
                    8, 8
                )

            # Draw hovered connection highlight
            elif self.hovered_line and not self.hovered_line.selected:
                line = self.hovered_line
                if line.path and not line.path.isEmpty():
                    hover_color = theme_manager.get_color('accent_secondary')
                    hover_color.setAlpha(150)

                    # Draw thicker line underneath
                    painter.setPen(QPen(hover_color, 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(line.path)

            painter.restore()

        except Exception as e:
            logger.error(f"Error drawing hover effects: {str(e)}")

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
                # If no item was clicked, start rectangle selection
                modifiers = QApplication.keyboardModifiers()

                # Start rectangle selection
                self.is_rect_selecting = True
                self.selection_rect_start = pos
                self.selection_rect_end = pos

                # Clear existing selection unless Shift is held
                if not (modifiers & Qt.ShiftModifier):
                    self._clear_selections()

                logger.debug(f"Started rectangle selection at ({pos.x()}, {pos.y()})")

        except Exception as e:
            logger.error(f"Error in _handle_left_click: {str(e)}")
    
    def _handle_right_click(self, pos):
        """Handle right mouse button clicks - show context menus."""
        try:
            # Check what was clicked and show appropriate context menu
            clicked_block = self._get_clicked_block(pos)
            clicked_line = self._get_clicked_line(pos)

            if clicked_block:
                # Show block context menu
                self._show_block_context_menu(clicked_block, pos)
            elif clicked_line:
                # Show connection context menu
                self._show_connection_context_menu(clicked_line, pos)
            else:
                # Show canvas context menu
                self._show_canvas_context_menu(pos)

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

    def _finalize_rect_selection(self):
        """Finalize rectangle selection and select blocks within the rectangle."""
        try:
            if not self.selection_rect_start or not self.selection_rect_end:
                return

            # Create QRect from start and end points
            # Normalize the rectangle (in case user dragged from bottom-right to top-left)
            x1 = min(self.selection_rect_start.x(), self.selection_rect_end.x())
            y1 = min(self.selection_rect_start.y(), self.selection_rect_end.y())
            x2 = max(self.selection_rect_start.x(), self.selection_rect_end.x())
            y2 = max(self.selection_rect_start.y(), self.selection_rect_end.y())

            selection_rect = QRect(x1, y1, x2 - x1, y2 - y1)

            # Select all blocks whose rectangles intersect with the selection rectangle
            selected_count = 0
            for block in getattr(self.dsim, 'blocks_list', []):
                if hasattr(block, 'rect') and selection_rect.intersects(block.rect):
                    block.selected = True
                    selected_count += 1

            logger.info(f"Rectangle selection completed: {selected_count} block(s) selected")

            # Reset rectangle selection state
            self.is_rect_selecting = False
            self.selection_rect_start = None
            self.selection_rect_end = None

            # Redraw canvas
            self.update()

        except Exception as e:
            logger.error(f"Error finalizing rectangle selection: {str(e)}")

    def _handle_block_click(self, block, pos):
        """Handle clicking on a block."""
        try:
            logger.debug(f"Block clicked: {getattr(block, 'fn_name', 'Unknown')}")

            modifiers = QApplication.keyboardModifiers()

            # NEW: Connection logic with Ctrl+Click (only when a source is already selected)
            if (modifiers & Qt.ControlModifier) and self.source_block_for_connection and self.source_block_for_connection is not block:
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

            # Selection logic based on modifiers
            if modifiers & Qt.ShiftModifier:
                # Shift+Click: Add to selection (don't clear others)
                block.selected = True
                logger.info(f"Added {block.name} to selection (multi-select)")
            elif modifiers & Qt.ControlModifier:
                # Ctrl+Click (when no source block): Toggle selection
                block.toggle_selection()
                if block.selected:
                    self.source_block_for_connection = block
                logger.info(f"Toggled selection for {block.name}")
            else:
                # Normal click: If clicking on unselected block, clear all and select only this block
                # If clicking on already-selected block, keep all selections (for multi-block drag)
                if not block.selected:
                    self._clear_selections()
                    block.selected = True
                    logger.info(f"Selected {block.name}")
                else:
                    logger.info(f"Clicked on already-selected block {block.name}, keeping selection for drag")
                self.source_block_for_connection = block # Set source for connection

            # Start dragging the block (or all selected blocks)
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

                    # Push undo state before creating connection
                    self._push_undo("Connect")

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
        """Start dragging a block (or multiple selected blocks)."""
        try:
            self.state = State.DRAGGING
            self.dragging_block = block
            # Calculate drag offset based on block's top-left corner
            self.drag_offset = QPoint(pos.x() - block.left, pos.y() - block.top)

            # Store RELATIVE offsets from the clicked block to all other selected blocks
            # This maintains relative positions when dragging multiple blocks
            self.drag_offsets = {}
            for b in self.dsim.blocks_list:
                if b.selected:
                    # Store offset from clicked block to this block
                    self.drag_offsets[b] = QPoint(b.left - block.left, b.top - block.top)

            logger.debug(f"Started dragging {len(self.drag_offsets)} block(s)")
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")

    def mouseMoveEvent(self, event):
        """Handle mouse move events with hover detection."""
        try:
            if self.panning:
                delta = event.pos() - self.last_pan_pos
                self.pan_offset += delta
                self.last_pan_pos = event.pos()
                self.update()
                return

            pos = self.screen_to_world(event.pos())

            # Update hover states (only when not dragging/selecting)
            if self.state == State.IDLE and not self.is_rect_selecting:
                self._update_hover_states(pos)

            # Update rectangle selection
            if self.is_rect_selecting:
                self.selection_rect_end = pos
                self.update()
                return

            if self.state == State.DRAGGING and self.dragging_block:
                # Update all selected block positions with grid snapping
                # Calculate new position for the clicked block
                new_x = pos.x() - self.drag_offset.x()
                new_y = pos.y() - self.drag_offset.y()

                # Snap to grid if enabled
                if self.snap_enabled:
                    snapped_x = round(new_x / self.grid_size) * self.grid_size
                    snapped_y = round(new_y / self.grid_size) * self.grid_size
                else:
                    snapped_x = new_x
                    snapped_y = new_y

                # Move clicked block to snapped position
                self.dragging_block.relocate_Block(QPoint(snapped_x, snapped_y))

                # If multiple blocks are selected, move them all by maintaining relative positions
                if hasattr(self, 'drag_offsets') and len(self.drag_offsets) > 1:
                    for block, relative_offset in self.drag_offsets.items():
                        if block is not self.dragging_block:
                            # Position this block relative to the clicked block
                            block_x = snapped_x + relative_offset.x()
                            block_y = snapped_y + relative_offset.y()
                            block.relocate_Block(QPoint(block_x, block_y))

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

            # Finalize rectangle selection
            if self.is_rect_selecting and event.button() == Qt.LeftButton:
                self._finalize_rect_selection()
                return

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

                # Push undo state after moving blocks
                self._push_undo("Move")

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
            shift_pressed = event.modifiers() & Qt.ShiftModifier

            if event.key() == Qt.Key_Escape:
                # Cancel any ongoing operations
                if self.line_creation_state:
                    self._cancel_line_creation()
                elif self.state == State.DRAGGING:
                    self._finish_drag()
            elif event.key() == Qt.Key_Delete:
                self.remove_selected_items()
            elif event.key() == Qt.Key_Z and ctrl_pressed and shift_pressed:
                # Ctrl+Shift+Z: Redo (alternative to Ctrl+Y)
                self.redo()
            elif event.key() == Qt.Key_Z and ctrl_pressed:
                # Ctrl+Z: Undo
                self.undo()
            elif event.key() == Qt.Key_Y and ctrl_pressed:
                # Ctrl+Y: Redo
                self.redo()
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

            # Push undo state before pasting
            self._push_undo("Paste")

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
                # Note: username='' makes it default to the new name (block_fn + sid)
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
                    username='',  # Let it default to new name
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

    def _show_block_context_menu(self, block, pos):
        """Show context menu for a block."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme_manager.get_color('surface_secondary').name()};
                color: {theme_manager.get_color('text_primary').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {theme_manager.get_color('accent_primary').name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {theme_manager.get_color('border_secondary').name()};
                margin: 4px 8px;
            }}
        """)

        # Ensure block is selected
        if not block.selected:
            self._clear_selections()
            block.selected = True
            self.update()

        # Block actions
        delete_action = menu.addAction("Delete")
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(self.remove_selected_items)

        duplicate_action = menu.addAction("Duplicate")
        duplicate_action.setShortcut("Ctrl+D")
        duplicate_action.triggered.connect(lambda: self._duplicate_block(block))

        menu.addSeparator()

        copy_action = menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self._copy_selected_blocks)

        cut_action = menu.addAction("Cut")
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self._cut_selected_blocks)

        menu.addSeparator()

        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(lambda: self._show_block_properties(block))

        # Show menu at cursor position (convert world to screen coordinates)
        screen_pos = self.mapToGlobal(self.world_to_screen(pos))
        menu.exec_(screen_pos)

    def _show_connection_context_menu(self, line, pos):
        """Show context menu for a connection line."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme_manager.get_color('surface_secondary').name()};
                color: {theme_manager.get_color('text_primary').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {theme_manager.get_color('accent_primary').name()};
            }}
        """)

        # Ensure line is selected
        if not line.selected:
            self._clear_line_selections()
            line.selected = True
            self.update()

        delete_action = menu.addAction("Delete Connection")
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(lambda: self._delete_line(line))

        highlight_action = menu.addAction("Highlight Path")
        highlight_action.triggered.connect(lambda: self._highlight_connection_path(line))

        # Show menu at cursor position
        screen_pos = self.mapToGlobal(self.world_to_screen(pos))
        menu.exec_(screen_pos)

    def _show_canvas_context_menu(self, pos):
        """Show context menu for empty canvas area."""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {theme_manager.get_color('surface_secondary').name()};
                color: {theme_manager.get_color('text_primary').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {theme_manager.get_color('accent_primary').name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {theme_manager.get_color('border_secondary').name()};
                margin: 4px 8px;
            }}
        """)

        # Paste (only enabled if clipboard has blocks)
        paste_action = menu.addAction("Paste")
        paste_action.setShortcut("Ctrl+V")
        paste_action.setEnabled(len(self.clipboard_blocks) > 0)
        paste_action.triggered.connect(lambda: self._paste_blocks(pos))

        menu.addSeparator()

        select_all_action = menu.addAction("Select All")
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self._select_all_blocks)

        clear_selection_action = menu.addAction("Clear Selection")
        clear_selection_action.setShortcut("Esc")
        clear_selection_action.triggered.connect(self._clear_selections)

        menu.addSeparator()

        # Zoom submenu
        zoom_menu = menu.addMenu("Zoom")
        zoom_in_action = zoom_menu.addAction("Zoom In")
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)

        zoom_out_action = zoom_menu.addAction("Zoom Out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)

        zoom_fit_action = zoom_menu.addAction("Fit All")
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.zoom_to_fit)

        # Show menu at cursor position
        screen_pos = self.mapToGlobal(self.world_to_screen(pos))
        menu.exec_(screen_pos)

    # Helper methods for context menu actions
    def _duplicate_block(self, block):
        """Duplicate a block."""
        try:
            # Push undo state before duplication
            self._push_undo("Duplicate")

            offset = 30  # Offset for duplicated block
            new_coords = QRect(
                block.coords.x() + offset,
                block.coords.y() + offset,
                block.coords.width(),
                block.coords.height()
            )

            new_block = self.dsim.add_new_block(
                block.block_fn,
                new_coords,
                block.b_color,
                block.in_ports,
                block.out_ports,
                block.b_type,
                block.io_edit,
                block.fn_name,
                block.params.copy() if hasattr(block, 'params') else {},
                block.external
            )

            if new_block:
                logger.info(f"Duplicated block: {block.fn_name} -> {new_block.name}")
                self._clear_selections()
                new_block.selected = True
                self.update()

        except Exception as e:
            logger.error(f"Error duplicating block: {str(e)}")

    def _copy_selected_blocks(self):
        """Copy selected blocks to clipboard."""
        try:
            self.clipboard_blocks = []
            for block in self.dsim.blocks_list:
                if block.selected:
                    self.clipboard_blocks.append({
                        'block_fn': block.block_fn,
                        'color': block.b_color,
                        'in_ports': block.in_ports,
                        'out_ports': block.out_ports,
                        'b_type': block.b_type,
                        'io_edit': block.io_edit,
                        'fn_name': block.fn_name,
                        'params': block.params.copy() if hasattr(block, 'params') else {},
                        'external': block.external,
                        'coords': block.coords
                    })
            logger.info(f"Copied {len(self.clipboard_blocks)} blocks to clipboard")
        except Exception as e:
            logger.error(f"Error copying blocks: {str(e)}")

    def _cut_selected_blocks(self):
        """Cut selected blocks to clipboard."""
        self._copy_selected_blocks()
        self.remove_selected_items()

    def _paste_blocks(self, pos):
        """Paste blocks from clipboard at specified position."""
        try:
            if not self.clipboard_blocks:
                return

            # Push undo state before pasting
            self._push_undo("Paste")

            # Calculate offset from first block's position to paste position
            if self.clipboard_blocks:
                first_block_coords = self.clipboard_blocks[0]['coords']
                offset_x = pos.x() - first_block_coords.x()
                offset_y = pos.y() - first_block_coords.y()

                self._clear_selections()
                for block_data in self.clipboard_blocks:
                    new_coords = QRect(
                        block_data['coords'].x() + offset_x,
                        block_data['coords'].y() + offset_y,
                        block_data['coords'].width(),
                        block_data['coords'].height()
                    )

                    new_block = self.dsim.add_new_block(
                        block_data['block_fn'],
                        new_coords,
                        block_data['color'],
                        block_data['in_ports'],
                        block_data['out_ports'],
                        block_data['b_type'],
                        block_data['io_edit'],
                        block_data['fn_name'],
                        block_data['params'],
                        block_data['external']
                    )

                    if new_block:
                        new_block.selected = True

                logger.info(f"Pasted {len(self.clipboard_blocks)} blocks")
                self.update()

        except Exception as e:
            logger.error(f"Error pasting blocks: {str(e)}")

    def _select_all_blocks(self):
        """Select all blocks on canvas."""
        for block in self.dsim.blocks_list:
            block.selected = True
        self.update()

    def _show_block_properties(self, block):
        """Show properties dialog for a block."""
        # Emit signal to show properties in property panel
        self.block_selected.emit(block)

    def _delete_line(self, line):
        """Delete a specific connection line."""
        try:
            if line in self.dsim.line_list:
                # Push undo state before deleting line
                self._push_undo("Delete Connection")

                self.dsim.line_list.remove(line)
                logger.info(f"Deleted connection: {line.name}")
                self.update()
        except Exception as e:
            logger.error(f"Error deleting line: {str(e)}")

    def _clear_line_selections(self):
        """Clear all line selections."""
        for line in self.dsim.line_list:
            line.selected = False

    def _highlight_connection_path(self, line):
        """Temporarily highlight a connection path."""
        # This could be enhanced with animation
        line.selected = True
        self.update()

    def _update_hover_states(self, pos):
        """Update hover states for blocks, ports, and connections."""
        needs_repaint = False

        # Check for hovered port first (highest priority)
        new_hovered_port = None
        for block in self.dsim.blocks_list:
            # Check output ports
            for i, port_pos in enumerate(block.out_coords):
                if self._point_near_port(pos, port_pos):
                    new_hovered_port = (block, i, True)  # True = output port
                    break
            # Check input ports
            if not new_hovered_port:
                for i, port_pos in enumerate(block.in_coords):
                    if self._point_near_port(pos, port_pos):
                        new_hovered_port = (block, i, False)  # False = input port
                        break
            if new_hovered_port:
                break

        if new_hovered_port != self.hovered_port:
            self.hovered_port = new_hovered_port
            needs_repaint = True

        # Check for hovered block (if no port is hovered)
        if not new_hovered_port:
            new_hovered_block = self._get_clicked_block(pos)
            if new_hovered_block != self.hovered_block:
                self.hovered_block = new_hovered_block
                needs_repaint = True
        else:
            if self.hovered_block is not None:
                self.hovered_block = None
                needs_repaint = True

        # Check for hovered line (if no block/port is hovered)
        if not new_hovered_port and not self.hovered_block:
            new_hovered_line = self._get_clicked_line(pos)
            if new_hovered_line != self.hovered_line:
                self.hovered_line = new_hovered_line
                needs_repaint = True
        else:
            if self.hovered_line is not None:
                self.hovered_line = None
                needs_repaint = True

        if needs_repaint:
            self.update()

    def _point_near_port(self, point, port_pos, threshold=12):
        """Check if a point is near a port position."""
        dx = point.x() - port_pos.x()
        dy = point.y() - port_pos.y()
        return (dx * dx + dy * dy) < (threshold * threshold)

    # Undo/Redo System
    def _capture_state(self):
        """Capture current diagram state for undo/redo."""
        try:
            state = {
                'blocks': [],
                'lines': []
            }

            # Capture all blocks
            for block in self.dsim.blocks_list:
                block_data = {
                    'name': block.name,
                    'block_fn': block.block_fn,
                    'coords': (block.coords.x(), block.coords.y(), block.coords.width(), block.coords.height()),
                    'color': block.b_color.name() if hasattr(block.b_color, 'name') else str(block.b_color),
                    'in_ports': block.in_ports,
                    'out_ports': block.out_ports,
                    'b_type': block.b_type,
                    'io_edit': block.io_edit,
                    'fn_name': block.fn_name,
                    'params': block.params.copy() if hasattr(block, 'params') and block.params else {},
                    'external': block.external,
                    'selected': block.selected
                }
                state['blocks'].append(block_data)

            # Capture all connections
            for line in self.dsim.line_list:
                line_data = {
                    'name': line.name,
                    'srcblock': line.srcblock,
                    'srcport': line.srcport,
                    'dstblock': line.dstblock,
                    'dstport': line.dstport,
                    'selected': line.selected
                }
                state['lines'].append(line_data)

            return state

        except Exception as e:
            logger.error(f"Error capturing state: {str(e)}")
            return None

    def _restore_state(self, state):
        """Restore diagram state from snapshot."""
        try:
            if not state:
                return False

            # Clear current diagram
            self.dsim.blocks_list.clear()
            self.dsim.line_list.clear()

            # Restore blocks
            for block_data in state['blocks']:
                coords = QRect(*block_data['coords'])
                block = self.dsim.add_new_block(
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
                    block.selected = block_data.get('selected', False)
                    # Restore original name
                    block.name = block_data['name']

            # Restore connections
            for line_data in state['lines']:
                # Find blocks by name
                src_block = None
                dst_block = None
                for block in self.dsim.blocks_list:
                    if block.name == line_data['srcblock']:
                        src_block = block
                    if block.name == line_data['dstblock']:
                        dst_block = block

                if src_block and dst_block:
                    src_port_pos = src_block.out_coords[line_data['srcport']]
                    dst_port_pos = dst_block.in_coords[line_data['dstport']]

                    line = self.dsim.add_line(
                        (line_data['srcblock'], line_data['srcport'], src_port_pos),
                        (line_data['dstblock'], line_data['dstport'], dst_port_pos)
                    )
                    if line:
                        line.selected = line_data.get('selected', False)
                        line.name = line_data['name']

            self.update()
            return True

        except Exception as e:
            logger.error(f"Error restoring state: {str(e)}")
            return False

    def _push_undo(self, description="Action"):
        """Push current state to undo stack."""
        try:
            state = self._capture_state()
            if state:
                self.undo_stack.append({'state': state, 'description': description})

                # Limit stack size
                if len(self.undo_stack) > self.max_undo_steps:
                    self.undo_stack.pop(0)

                # Clear redo stack when new action is performed
                self.redo_stack.clear()

                logger.debug(f"Pushed to undo stack: {description} (stack size: {len(self.undo_stack)})")

        except Exception as e:
            logger.error(f"Error pushing undo: {str(e)}")

    def undo(self):
        """Undo the last action."""
        try:
            if not self.undo_stack:
                logger.info("Nothing to undo")
                return

            # Capture current state for redo
            current_state = self._capture_state()
            if current_state:
                self.redo_stack.append({'state': current_state, 'description': 'Redo'})

            # Pop and restore previous state
            undo_item = self.undo_stack.pop()
            if self._restore_state(undo_item['state']):
                logger.info(f"Undone: {undo_item['description']}")
            else:
                logger.error("Failed to undo")

        except Exception as e:
            logger.error(f"Error in undo: {str(e)}")

    def redo(self):
        """Redo the last undone action."""
        try:
            if not self.redo_stack:
                logger.info("Nothing to redo")
                return

            # Capture current state for undo
            current_state = self._capture_state()
            if current_state:
                self.undo_stack.append({'state': current_state, 'description': 'Undo'})

            # Pop and restore redo state
            redo_item = self.redo_stack.pop()
            if self._restore_state(redo_item['state']):
                logger.info(f"Redone: {redo_item['description']}")
            else:
                logger.error("Failed to redo")

        except Exception as e:
            logger.error(f"Error in redo: {str(e)}")

    def remove_selected_items(self):
        """Remove all selected blocks and lines."""
        try:
            blocks_to_remove = [block for block in self.dsim.blocks_list if block.selected]
            lines_to_remove = [line for line in self.dsim.line_list if line.selected]

            # Push undo state before deletion
            if blocks_to_remove or lines_to_remove:
                self._push_undo("Delete")

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
