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
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.lib import DSim
from lib.improvements import PerformanceHelper, SafetyChecks, ValidationHelper, SimulationConfig
from modern_ui.themes.theme_manager import theme_manager
from lib.analysis.control_system_analyzer import ControlSystemAnalyzer
from modern_ui.renderers.block_renderer import BlockRenderer
from modern_ui.renderers.connection_renderer import ConnectionRenderer
from modern_ui.renderers.canvas_renderer import CanvasRenderer
from modern_ui.interactions.interaction_manager import InteractionManager, State
from modern_ui.managers.history_manager import HistoryManager
from modern_ui.managers.menu_manager import MenuManager
from modern_ui.managers.selection_manager import SelectionManager

logger = logging.getLogger(__name__)


class ModernCanvas(QWidget):
    """Modern canvas widget for DiaBloS block diagram editing."""

    # Signals
    block_selected = pyqtSignal(object)  # Emitted when a block is selected
    connection_created = pyqtSignal(object, object)  # Emitted when a connection is made
    simulation_status_changed = pyqtSignal(str)  # Emitted when simulation status changes
    command_palette_requested = pyqtSignal()  # Emitted when command palette should open
    scope_changed = pyqtSignal(list)  # Emitted when navigation scope changes (path)
    
    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Initialize core DSim functionality
        self.dsim = dsim
        
        # Performance monitoring
        self.perf_helper = PerformanceHelper()
        
        # Simulation configuration
        self.sim_config = SimulationConfig()
        
        # State management
        self.state = State.IDLE
        self.interaction_manager = InteractionManager(self)
        
        # Interactions (Migrating to InteractionManager)
        # self.dragging_block = None -> interaction_manager.dragging_block
        self.drag_offset = None
        self.drag_offsets = {}  # For multi-block dragging
        self.drag_start_positions = {}  # Track starting positions for undo
        
        # Connection management
        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None
        self.source_block_for_connection = None
        self.default_routing_mode = "bezier"  # Default routing mode for new connections

        # Zoom and Pan
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.panning = False
        self.last_pan_pos = QPoint(0, 0)

        # Grid visibility
        self.grid_visible = True

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

        # Block resizing
        self.resizing_block = None
        self.resize_handle = None  # Which handle is being dragged
        self.resize_start_rect = None  # Original block rect before resize
        self.resize_start_pos = None  # Mouse position at start of resize

        # Validation system
        self.validation_errors = []
        self.blocks_with_errors = set()
        self.blocks_with_warnings = set()
        self.show_validation_errors = False

        # Initialize helpers
        self.performance = PerformanceHelper()
        self.validator = ValidationHelper()
        self.safety = SafetyChecks()
        
        # State
        self.state = State.IDLE
        
        # Managers
        self.interaction_manager = InteractionManager(self)
        self.history_manager = HistoryManager(self)
        self.menu_manager = MenuManager(self)
        self.selection_manager = SelectionManager(self)
        
        # Initialize Analysis Tool
        self.analyzer = ControlSystemAnalyzer(self, parent=self)
        
        # Initialize Renderer
        self.block_renderer = BlockRenderer()
        self.connection_renderer = ConnectionRenderer()
        self.canvas_renderer = CanvasRenderer()

        # Setup UI
        self._setup_canvas()
        
        # Initialize state
        self.zoom_level = 1.0
        self.offset = QPoint(0, 0)
        
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
                    # Apply dynamic sizing based on port count
                    if hasattr(new_block, 'calculate_min_size'):
                        min_height = new_block.calculate_min_size()
                        if min_height > new_block.height:
                            logger.info(f"Resizing {block_name} to min_height {min_height}")
                            new_block.height = min_height
                            # Update rect
                            new_block.rect.setHeight(min_height)
                            new_block.update_Block()

                    logger.info(f"Successfully added {block_name}")

                    # Capture state for undo
                    self._push_undo("Add Block")

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
        
        solver_type = getattr(self.dsim, 'last_solver_type', 'Standard')
        self.simulation_status_changed.emit(f"Simulation finished [{solver_type}]")
        logger.info(f"Batch simulation finished. Solver: {solver_type}")
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
        painter = QPainter()
        begun = painter.begin(self)
        if not begun:
            return
        try:
            self.perf_helper.start_timer("canvas_paint")
            painter.setRenderHint(QPainter.Antialiasing)

            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)
            
            # Clear canvas with theme-appropriate background
            painter.fillRect(self.rect(), theme_manager.get_color('canvas_background'))

            # Draw sophisticated grid system
            self.canvas_renderer.draw_grid(painter, self.rect(), self.width(), self.height(), self.grid_visible)

            # Draw DSim elements in proper order: blocks -> lines -> ports
            # This ensures ports appear on top of connection lines
            self._render_blocks(painter, draw_ports=False)
            self._render_lines(painter)
            self._render_ports(painter)
            
            # Draw temporary connection line (with enhanced preview)
            if self.line_creation_state == 'start' and self.temp_line:
                start_point, end_point = self.temp_line

                # Check if hovering over valid target port
                is_valid_target = False
                if self.hovered_port:
                    hovered_block, port_idx, is_output = self.hovered_port
                    # Valid if hovering over an input port (not output)
                    if not is_output:
                        is_valid_target = True

                self.canvas_renderer.draw_temp_line(painter, start_point, end_point, is_valid_target)

            # Draw rectangle selection
            if self.is_rect_selecting and self.selection_rect_start and self.selection_rect_end:
                self.canvas_renderer.draw_selection_rect(painter, self.selection_rect_start, self.selection_rect_end)

            # Draw hover effects
            self.canvas_renderer.draw_hover_effects(painter, self.hovered_port, self.hovered_block, self.hovered_line)

            # Draw validation error indicators
            if self.show_validation_errors:
                self.canvas_renderer.draw_validation_errors(painter, self.blocks_with_errors, self.blocks_with_warnings)

            # Draw routing tag HUD (Goto/From overview)
            self.canvas_renderer.draw_tag_hud(painter, self.dsim)

            painter.end()
            paint_duration = self.perf_helper.end_timer("canvas_paint")
            
            # Log slow paint events
            if paint_duration and paint_duration > 0.05:
                logger.warning(f"Slow canvas paint: {paint_duration:.4f}s")
                
        except Exception as e:
            logger.error(f"Error in canvas paintEvent: {str(e)}")
        finally:
            if painter.isActive():
                try:
                    painter.end()
                except Exception:
                    pass

    # ===== Rendering Methods (moved from DSim) =====
    
    def _render_blocks(self, painter, draw_ports=True):
        """Render all blocks to canvas.
        
        This replaces DSim.display_blocks() - rendering logic belongs in canvas.
        """
        if painter is None:
            return
        for block in self.dsim.blocks_list:
            self.block_renderer.draw_block(block, painter, draw_ports=draw_ports)
            if block.selected:
                self.block_renderer.draw_resize_handles(block, painter)

    def _render_lines(self, painter):
        """Render all connection lines.
        
        This replaces DSim.display_lines() - rendering logic belongs in canvas.
        """
        if painter is None:
            return
        for line in self.dsim.line_list:
            if not getattr(line, "hidden", False):
                self.connection_renderer.draw_line(line, painter)

    def _render_ports(self, painter):
        """Render all ports on top of lines for better visibility.
        
        This replaces DSim.display_ports() - rendering logic belongs in canvas.
        """
        for block in self.dsim.blocks_list:
            self.block_renderer.draw_ports(block, painter)

    def _update_line_positions(self):
        """Update line positions after block movement.
        
        This replaces DSim.update_lines() - line position logic belongs in canvas.
        """
        for line in self.dsim.line_list:
            line.update_line(self.dsim.blocks_list)




    def mouseDoubleClickEvent(self, event):
        """Handle double click events."""
        m_pos = event.pos()
        
        # Check if clicked on a block
        clicked_block = None
        
        # Check logical blocks first (z-order top)
        for block in reversed(self.blocks_renderer_order):
            if block.rect.contains(m_pos):
                clicked_block = block
                break
                
        if clicked_block:
            # Special handling for Analysis blocks - Double click generates plot
            if clicked_block.block_fn in ['BodePhase', 'Nyquist', 'RootLocus', 'BodeMag']:
                logger.info(f"Double-click analysis trigger for {clicked_block.name}")
                if clicked_block.block_fn == 'BodePhase':
                     self.generate_bode_phase_plot(clicked_block)
                elif clicked_block.block_fn == 'Nyquist':
                     self.generate_nyquist_plot(clicked_block)
                elif clicked_block.block_fn == 'RootLocus':
                     self.generate_root_locus(clicked_block)
                elif clicked_block.block_fn == 'BodeMag':
                     self.generate_bode_plot(clicked_block)
                return

            # Default: Open parameter dialog
            self.show_param_dialog(clicked_block)
        
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        try:
            # Delegate completely to InteractionManager
            self.interaction_manager.handle_mouse_press(event)

        except Exception as e:
            logger.error(f"Error in canvas mousePressEvent: {str(e)}")

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events."""
        try:
            if event.button() == Qt.LeftButton:
                pos = self.screen_to_world(event.pos())

                # Check if double-clicked on empty space (not on block or line)
                clicked_block = self._get_clicked_block(pos)
                clicked_line, _ = self._get_clicked_line(pos)

                if clicked_block:
                    # Check if it's a subsystem to enter
                    if getattr(clicked_block, 'block_type', '') == 'Subsystem':
                        self.dsim.enter_subsystem(clicked_block)
                        self.update()
                        logger.info(f"Entered subsystem: {clicked_block.name}")
                        
                        # Reset view to ensure blocks are visible
                        self.pan_offset = QPoint(0, 0)
                        self.zoom_factor = 1.0
                        self.zoom_to_fit()
                        
                        self.scope_changed.emit(self.dsim.get_current_path())
                        return
                    else:
                        # Open block properties otherwise
                        self._show_block_properties(clicked_block)

                if not clicked_block and not clicked_line:
                    # Double-clicked on empty space - open command palette
                    logger.info("Double-clicked on empty canvas - emitting command_palette_requested")
                    self.command_palette_requested.emit()

        except Exception as e:
            logger.error(f"Error in canvas mouseDoubleClickEvent: {str(e)}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        # logger.info(f"Key press: {event.key()} Modifiers: {event.modifiers()}")
        try:
            if event.key() == Qt.Key_Escape:
                # If selection exists, clear it
                has_selection = any(b.selected for b in getattr(self.dsim, 'blocks_list', [])) or \
                                any(l.selected for l in getattr(self.dsim, 'line_list', []))
                
                if has_selection:
                    self._clear_selections()
                else:
                    # If no selection, try to go up directory/subsystem
                    if self.dsim.current_subsystem:
                        self.dsim.exit_subsystem()
                        self.update()
                        
                        # Reset view on exit too
                        self.pan_offset = QPoint(0, 0)
                        self.zoom_factor = 1.0
                        self.zoom_to_fit()
                        
                        self.scope_changed.emit(self.dsim.get_current_path())
                        
            elif event.key() == Qt.Key_Delete:
                 self.remove_selected_items()
            
            elif event.key() == Qt.Key_G and (event.modifiers() & Qt.ControlModifier):
                 # Ctrl+G to Group/Create Subsystem
                 self._create_subsystem_trigger()
                 
            else:
                 super().keyPressEvent(event)
                 
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {str(e)}")

    def navigate_scope(self, path_str):
        """Navigate to a specific scope path (e.g. via BreadcrumbBar)."""
        logger.info(f"Navigating to scope: {path_str}")
        if hasattr(self.dsim, 'navigate_scope'):
            self.dsim.navigate_scope(path_str)
            self.update()
            
            # Reset view
            self.pan_offset = QPoint(0, 0)
            self.zoom_factor = 1.0
            self.zoom_to_fit()
            
            self.scope_changed.emit(self.dsim.get_current_path())
        else:
            logger.warning("DSim does not support navigate_scope")
            
    # Cleaned up dangling except block here
    
    def _handle_right_click(self, pos):
        """Handle right mouse button clicks - show context menus."""
        try:
            # Check what was clicked and show appropriate context menu
            clicked_block = self._get_clicked_block(pos)
            clicked_line, _ = self._get_clicked_line(pos)  # Returns (line, collision_type) tuple

            if clicked_block:
                # Show block context menu - needs global screen position
                from PyQt5.QtGui import QCursor
                self._show_block_context_menu(clicked_block, QCursor.pos())
            elif clicked_line:
                # Show connection context menu
                from PyQt5.QtGui import QCursor
                self._show_connection_context_menu(clicked_line, QCursor.pos())
            else:
                # Show canvas context menu
                from PyQt5.QtGui import QCursor
                self._show_canvas_context_menu(QCursor.pos())

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
        self.analyzer.generate_bode_plot(bode_block)

    def show_root_locus_menu(self, block, pos):
        """Show context menu for the RootLocus block."""
        menu = QMenu(self)
        plot_action = menu.addAction("Generate Root Locus Plot")
        action = menu.exec_(pos)
        if action == plot_action:
            self.generate_root_locus(block)

    def generate_root_locus(self, rootlocus_block):
        """Find the connected transfer function, calculate, and plot the root locus."""
        self.analyzer.generate_root_locus(rootlocus_block)

    def generate_root_locus_plot(self, rootlocus_block):
        """Wrapper method to maintain naming consistency with generate_bode_plot."""
        self.generate_root_locus(rootlocus_block)

    def _get_clicked_block(self, pos):
        # logger.info(f"Checking click at {pos}")
        for block in reversed(getattr(self.dsim, 'blocks_list', [])):
            if hasattr(block, 'rect') and block.rect.contains(pos):
                # logger.info(f"Hit block: {block.name}")
                return block
            # else:
            #     logger.debug(f"Miss: {block.name} rect={block.rect}")
        return None

    def _get_clicked_line(self, pos):
        for line in getattr(self.dsim, 'line_list', []):
            if getattr(line, "hidden", False):
                continue
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
                    # Find first free output port
                    connected_output_ports = {line.srcport for line in self.dsim.line_list if line.srcblock == source_block.name}
                    source_port_index = 0
                    for i in range(source_block.out_ports):
                        if i not in connected_output_ports:
                            source_port_index = i
                            break
                    # If all ports connected, source_port_index remains 0 (fan-out allowed)

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
                        # Set the default routing mode for the new connection
                        new_line.routing_mode = self.default_routing_mode
                        logger.info(f"Line created: {start_block_name} -> {end_block_name} (routing: {self.default_routing_mode})")
                        # If Goto/From involved, relink to sync labels/virtual lines
                        if getattr(self.line_start_block, "block_fn", "") in ("Goto", "From") or getattr(end_block, "block_fn", "") in ("Goto", "From"):
                            try:
                                self.dsim.model.link_goto_from()
                            except Exception as e:
                                logger.warning(f"Could not relink Goto/From after connection: {e}")
                        self._update_line_positions()
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
            self.drag_start_positions = {}  # Track starting positions for undo threshold
            for b in self.dsim.blocks_list:
                if b.selected:
                    # Store offset from clicked block to this block
                    self.drag_offsets[b] = QPoint(b.left - block.left, b.top - block.top)
                    # Store starting position
                    self.drag_start_positions[b] = (b.left, b.top)

            logger.debug(f"Started dragging {len(self.drag_offsets)} block(s)")
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")

    def _start_resize(self, block, handle, pos):
        """Start resizing a block."""
        try:
            self.state = State.RESIZING
            self.resizing_block = block
            self.resize_handle = handle
            self.resize_start_pos = pos
            self.resize_start_rect = QRect(block.left, block.top, block.width, block.height)

            logger.debug(f"Started resizing block {block.name} from handle {handle}")
        except Exception as e:
            logger.error(f"Error starting resize: {str(e)}")

    def _perform_resize(self, pos):
        """Perform the resize operation based on current mouse position."""
        try:
            if not self.resizing_block or not self.resize_handle:
                return

            block = self.resizing_block
            handle = self.resize_handle
            start_rect = self.resize_start_rect

            # Calculate delta from start position
            delta_x = pos.x() - self.resize_start_pos.x()
            delta_y = pos.y() - self.resize_start_pos.y()

            # Calculate new position and size based on handle
            new_left = start_rect.left()
            new_top = start_rect.top()
            new_width = start_rect.width()
            new_height = start_rect.height()

            if 'left' in handle:
                new_left = start_rect.left() + delta_x
                new_width = start_rect.width() - delta_x
            elif 'right' in handle:
                new_width = start_rect.width() + delta_x

            if 'top' in handle:
                new_top = start_rect.top() + delta_y
                new_height = start_rect.height() - delta_y
            elif 'bottom' in handle:
                new_height = start_rect.height() + delta_y

            # Apply minimum size constraints
            try:
                from config.block_sizes import MIN_BLOCK_WIDTH, MIN_BLOCK_HEIGHT
                min_width = MIN_BLOCK_WIDTH
                min_height = MIN_BLOCK_HEIGHT
            except ImportError:
                min_width = 50
                min_height = 40

            # Ensure minimum size
            if new_width < min_width:
                if 'left' in handle:
                    new_left = start_rect.right() - min_width
                new_width = min_width

            if new_height < min_height:
                if 'top' in handle:
                    new_top = start_rect.bottom() - min_height
                new_height = min_height

            # Update block position and size
            block.left = new_left
            block.top = new_top
            block.resize_Block(new_width, new_height)
            block.rect.moveTo(new_left, new_top)

            # Update connected lines
            self._update_line_positions()
            self.update()

        except Exception as e:
            logger.error(f"Error performing resize: {str(e)}")

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        # Delegate to InteractionManager
        self.interaction_manager.handle_mouse_move(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        # Delegate to InteractionManager
        self.interaction_manager.handle_mouse_release(event)

    def _finish_drag(self):
        """Finish dragging operation."""
        try:
            if self.dragging_block:
                logger.debug(f"Finished dragging block: {getattr(self.dragging_block, 'fn_name', 'Unknown')}")

                # Only push undo if blocks actually moved significantly (threshold: 5 pixels)
                moved_significantly = False
                move_threshold = 5  # pixels

                for block, start_pos in self.drag_start_positions.items():
                    start_left, start_top = start_pos
                    distance = abs(block.left - start_left) + abs(block.top - start_top)
                    if distance >= move_threshold:
                        moved_significantly = True
                        break

                if moved_significantly:
                    self._push_undo("Move")
                else:
                    logger.debug("Block moved less than threshold, not capturing undo")

                # Reset drag state
                self.state = State.IDLE
                self.dragging_block = None
                self.drag_offset = None
                self.drag_start_positions = {}
                self._update_line_positions() # Ensure lines are updated after drag finishes
                self.update() # Trigger a final repaint
        except Exception as e:
            logger.error(f"Error finishing drag: {str(e)}")

    def _finish_resize(self):
        """Finish resizing operation."""
        try:
            if self.resizing_block and self.resize_start_rect:
                logger.debug(f"Finished resizing block: {getattr(self.resizing_block, 'fn_name', 'Unknown')}")

                # Only push undo if block actually resized significantly (threshold: 5 pixels)
                block = self.resizing_block
                start_rect = self.resize_start_rect
                resize_threshold = 5  # pixels

                # Check if size or position changed significantly
                size_change = (abs(block.width - start_rect.width()) +
                              abs(block.height - start_rect.height()))
                pos_change = (abs(block.left - start_rect.left()) +
                             abs(block.top - start_rect.top()))

                if size_change >= resize_threshold or pos_change >= resize_threshold:
                    self._push_undo("Resize")
                else:
                    logger.debug("Block resized less than threshold, not capturing undo")

                # Reset resize state
                self.state = State.IDLE
                self.resizing_block = None
                self.resize_handle = None
                self.resize_start_rect = None
                self.resize_start_pos = None

                # Ensure lines are updated after resize
                self._update_line_positions()
                self.update()
        except Exception as e:
            logger.error(f"Error finishing resize: {str(e)}")

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
                else:
                    # Clear selections if no other operation is active
                    self._clear_selections()
                    self.update()
            elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                # Delete or Backspace - works on both Mac (Delete key) and Windows/Linux (Del key)
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
            elif event.key() == Qt.Key_A and ctrl_pressed:
                # Ctrl+A: Select all blocks
                self._select_all_blocks()
            elif event.key() == Qt.Key_F5:
                if shift_pressed:
                    # Shift+F5: Stop simulation
                    self.stop_simulation()
                    logger.info("F5: Stopped simulation")
                else:
                    # F5: Start/run simulation
                    self.start_simulation()
                    logger.info("F5: Started simulation")
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {str(e)}")

    def flip_selected_blocks(self):
        """Flip selected blocks horizontally."""
        try:
            for block in self.dsim.blocks_list:
                if block.selected:
                    block.flipped = not block.flipped
                    block.update_Block() # Recalculate port positions
            self._update_line_positions()
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
            selected_indices = {} # Map original block object to its index in clipboard

            # 1. Copy Blocks
            for i, block in enumerate(selected_blocks):
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
                
                # SPECIAL HANDLING FOR SUBSYSTEM COPY
                if block.block_fn == 'Subsystem':
                    # We need to deepcopy the internal structure!
                    # block.sub_blocks and block.sub_lines contain DBlock/DLine objects.
                    # We can use copy.deepcopy for this as they should be pickleable (mostly).
                    try:
                        block_data['sub_blocks'] = copy.deepcopy(block.sub_blocks)
                        block_data['sub_lines'] = copy.deepcopy(block.sub_lines)
                        block_data['ports'] = copy.deepcopy(block.ports) if hasattr(block, 'ports') else {}
                        block_data['ports_map'] = copy.deepcopy(block.ports_map) if hasattr(block, 'ports_map') else {}
                    except Exception as e:
                        logger.error(f"Error deepcopying subsystem contents for {block.name}: {e}")
                        # Fallback? If we fail, paste will create empty subsystem.
                
                self.clipboard_blocks.append(block_data)
                selected_indices[block] = i

            # 2. Copy Internal Connections
            self.clipboard_connections = []
            
            # Map names to objects for lookup
            name_to_block = {b.name: b for b in self.dsim.blocks_list}
            
            for line in self.dsim.connections_list:
                # Resolve block objects from names
                src_obj = name_to_block.get(line.srcblock)
                dst_obj = name_to_block.get(line.dstblock)
                
                if src_obj and dst_obj and src_obj in selected_indices and dst_obj in selected_indices:
                    conn_data = {
                        'start_index': selected_indices[src_obj],
                        'start_port': line.srcport,
                        'end_index': selected_indices[dst_obj],
                        'end_port': line.dstport
                    }
                    self.clipboard_connections.append(conn_data)

            logger.info(f"Copied {len(self.clipboard_blocks)} blocks and {len(self.clipboard_connections)} connections")
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
                # Create new block with unique ID
                # Note: username='' makes it default to the new name (block_fn + sid)
                
                # SPECIAL HANDLING FOR SUBSYSTEM
                if block_fn == 'Subsystem':
                    from blocks.subsystem import Subsystem
                    new_block = Subsystem(
                        block_name=f"Subsystem{sid}", # Default name will be corrected by DBlock init logic or manually set below
                        sid=sid,
                        coords=new_coords,
                        color=block_data['color']
                    )
                    # Restore other attributes
                    new_block.io_edit = block_data['io_edit']
                    new_block.fn_name = block_data['fn_name']
                    new_block.params = block_data['params'].copy()
                    new_block.params['_name_'] = new_block.name # Ensure params name matches
                    new_block.external = block_data['external']
                    
                    # Restore internal structure if available
                    if 'sub_blocks' in block_data:
                        try:
                            new_block.sub_blocks = copy.deepcopy(block_data['sub_blocks'])
                            new_block.sub_lines = copy.deepcopy(block_data['sub_lines'])
                            new_block.ports = copy.deepcopy(block_data.get('ports', {}))
                            new_block.ports_map = copy.deepcopy(block_data.get('ports_map', {}))
                            
                            # Recursively update names/SIDs of internal blocks? 
                            # If we just copy them, they might have old names like "Subsystem1/Gain1".
                            # But now they are inside "Subsystem2".
                            # The 'name' attribute of sub-blocks usually doesn't include path? 
                            # Let's check: in 'blocks.subsystem', sub_blocks usually have simple names?
                            # Actually, Flattener relies on recursive structure, names inside are usually just "Gain1".
                            # Flattener builds full path.
                            # So deepcopy should be fine mostly, UNLESS SIDs conflict?
                            # Internal SIDs are local to the subsystem presumably?
                            # Standard simulation engine doesn't enforce global uniqueness for internal blocks until flattening.
                            # However, 'update_Block' might rely on things.
                            
                            # Let's trust deepcopy for now.
                            logger.info(f"Restored {len(new_block.sub_blocks)} internal blocks for {new_block.name}")
                        except Exception as e:
                            logger.error(f"Error restoring subsystem contents for {new_block.name}: {e}")
                    
                else: 
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

            # Recreate Connections
            from lib.simulation.connection import DLine
            if hasattr(self, 'clipboard_connections'):
                for conn_data in self.clipboard_connections:
                    try:
                        start_block = pasted_blocks[conn_data['start_index']]
                        end_block = pasted_blocks[conn_data['end_index']]
                        
                        start_port = conn_data['start_port']
                        end_port = conn_data['end_port']

                        # Create new line
                        # Need new SID
                        line_ids = [l.sid for l in self.dsim.connections_list]
                        new_sid = max(line_ids) + 1 if line_ids else 0
                        
                        # Minimal points (start/end)
                        p1 = start_block.out_coords[start_port]
                        p2 = end_block.in_coords[end_port]

                        new_line = DLine(
                            sid=new_sid,
                            srcblock=start_block.name,
                            srcport=start_port,
                            dstblock=end_block.name,
                            dstport=end_port,
                            points=[p1, p2]
                        )
                        
                        # Ensure path is calculated
                        # Add to list FIRST so update_line can see it if it needs to check existence (though it mainly checks blocks)
                        self.dsim.connections_list.append(new_line)
                        
                        try:
                            # Pass the full block list including new ones
                            new_line.update_line(self.dsim.blocks_list)
                        except Exception as e:
                            logger.warning(f"Failed to update trajectory for pasted line {new_sid}: {e}")
                            
                    except IndexError:
                        logger.warning("Skipping connection: Block index out of range")
                    except Exception as e:
                        logger.error(f"Error pasting connection: {e}")

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



    # Helper methods for context menu actions
    def _duplicate_block(self, block):
        """Duplicate a block."""
        try:
            from lib.simulation.menu_block import MenuBlocks
            from PyQt5.QtCore import QPoint

            # Push undo state before duplication
            self._push_undo("Duplicate")

            offset = 30  # Offset for duplicated block
            new_position = QPoint(
                block.rect.x() + block.rect.width() // 2 + offset,
                block.rect.y() + block.rect.height() // 2 + offset
            )

            # Create a MenuBlocks object from the existing block
            io_params = {
                'inputs': block.in_ports,
                'outputs': block.out_ports,
                'b_type': block.b_type,
                'io_edit': block.io_edit
            }

            menu_block = MenuBlocks(
                block_fn=block.block_fn,
                fn_name=block.fn_name,
                io_params=io_params,
                ex_params=block.params.copy() if hasattr(block, 'params') else {},
                b_color=block.b_color,
                coords=(block.rect.width(), block.rect.height()),
                external=block.external,
                block_class=getattr(block, 'block_class', None)
            )

            # Use add_block with the MenuBlocks object
            new_block = self.dsim.add_block(menu_block, new_position)

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
                        'coords': block.rect,
                        'flipped': getattr(block, 'flipped', False)
                    })
            logger.info(f"Copied {len(self.clipboard_blocks)} blocks to clipboard")
        except Exception as e:
            logger.error(f"Error copying blocks: {str(e)}")

    def _cut_selected_blocks(self):
        """Cut selected blocks to clipboard."""
        self._copy_selected_blocks()
        self.remove_selected_items()

    def navigate_scope(self, index):
        """
        Navigate to a specific depth in the hierarchy.
        Args:
            index: The index in the path list to navigate to (0 = Top Level).
        """
        current_path = self.dsim.get_current_path()
        current_depth = len(current_path) - 1 # 0-indexed index of current scope
        
        target_depth = index
        
        if target_depth < 0: return
        if target_depth >= current_depth: return # Already there or invalid
        
        # Pop scopes until we reach target
        # Calculate how many times to pop
        # If current is depth 2 (Main > Sub1 > Sub2), index 0 (Main) -> pop 2 times
        pops = current_depth - target_depth
        
        for _ in range(pops):
            self.dsim.exit_subsystem()
            
        self.update()
        self.scope_changed.emit(self.dsim.get_current_path())
        logger.info(f"Navigated to scope index {index}")

    def contextMenuEvent(self, event):
        """Handle context menu events."""
        # This might be called automatically by Qt on right click.
        # Check if we should delegate or handle here.
        # Since _handle_right_click also exists (called by interaction manager), 
        # let's try to map event position to items.
        
        pos = event.pos()
        # Convert to world pos if needed? 
        # _get_clicked_block expects... screen pos?
        # _get_clicked_block uses block.rect (world coords usually? no rect is screen/canvas coords?).
        # Blocks are stored in world coords? Need validation.
        # Mouse events usually give widget coords.
        # _get_clicked_block expects 'pos' in same system as block.rect.
        # If block.rect is in logical coords, we need screen_to_world.
        
        world_pos = self.screen_to_world(pos)
        block = self._get_clicked_block(world_pos)
        
        if block:
            self._show_block_context_menu(block, event.globalPos())
        else:
             # Check line?
             line, _ = self._get_clicked_line(world_pos)
             if line:
                 self._show_connection_context_menu(line, event.globalPos())
             else:
                 self._show_canvas_context_menu(event.globalPos())

    def _show_block_context_menu(self, block, global_pos):
        """Show context menu for a block."""
        from PyQt5.QtWidgets import QMenu, QAction
        
        context_menu = QMenu(self)
        
        # DEBUG LOGGING for context menu logic
        logger.info(f"Opening context menu for block: {block.name}")
        logger.info(f"  block_fn: {getattr(block, 'block_fn', 'N/A')}")
        logger.info(f"  type(block): {type(block)}")
        
        # Check selection (might be multiple)
        selected_blocks = [b for b in self.dsim.blocks_list if b.selected]
        if block not in selected_blocks:
            # Right clicked on unselected block?
            # Maybe select it alone?
            pass # Usually right click doesn't change selection unless separate logic
            
        # If multiple selected, actions apply to all
        target_blocks = selected_blocks if selected_blocks else [block]

        # Duplicate
        action_duplicate = QAction("Duplicate", self)
        action_duplicate.triggered.connect(lambda: [self._duplicate_block(b) for b in target_blocks])
        context_menu.addAction(action_duplicate)
        
        # Delete
        action_delete = QAction("Delete", self)
        action_delete.triggered.connect(self.remove_selected_items)
        context_menu.addAction(action_delete)
        
        context_menu.addSeparator()
        
        # Copy
        action_copy = QAction("Copy", self)
        action_copy.triggered.connect(self._copy_selected_blocks)
        context_menu.addAction(action_copy)

        context_menu.addSeparator()
        
        # Subsystem Creation
        if len(target_blocks) > 0:
            action_create_subsystem = QAction("Create Subsystem from Selection", self)
            action_create_subsystem.triggered.connect(self._create_subsystem_trigger)
            context_menu.addAction(action_create_subsystem)
            
        # Specific Block Actions (e.g. Bode)
        if hasattr(block, 'block_fn'):
             logger.info(f"Checking specific block actions for {block.block_fn}")
             if block.block_fn == 'BodeMag':
                 logger.info("Adding BodeMag action")
                 context_menu.addSeparator()
                 action_bode = QAction("Generate Bode Plot", self)
                 action_bode.triggered.connect(lambda: self.generate_bode_plot(block))
                 context_menu.addAction(action_bode)
             elif block.block_fn == 'RootLocus':
                 logger.info("Adding RootLocus action")
                 context_menu.addSeparator()
                 action_rl = QAction("Generate Root Locus", self)
                 action_rl.triggered.connect(lambda: self.generate_root_locus(block))
                 context_menu.addAction(action_rl)
             elif block.block_fn == 'Nyquist':
                 logger.info("Adding Nyquist action")
                 context_menu.addSeparator()
                 action_nyq = QAction("Generate Nyquist Plot", self)
                 action_nyq.triggered.connect(lambda: self.generate_nyquist_plot(block))
                 context_menu.addAction(action_nyq)
             elif block.block_fn == 'BodePhase':
                 logger.info("Adding BodePhase action")
                 context_menu.addSeparator()
                 action_bp = QAction("Generate Bode Phase Plot", self)
                 action_bp.triggered.connect(lambda: self.generate_bode_phase_plot(block))
                 context_menu.addAction(action_bp)
             else:
                 logger.info(f"No specific actions for {block.block_fn}")

        context_menu.exec_(global_pos)

    def _show_canvas_context_menu(self, global_pos):
        """Show context menu for empty canvas."""
        from PyQt5.QtWidgets import QMenu, QAction
        context_menu = QMenu(self)
        
        action_paste = QAction("Paste", self)
        action_paste.triggered.connect(lambda: self._paste_blocks(self.mapFromGlobal(global_pos)))
        if not getattr(self, 'clipboard_blocks', None):
            action_paste.setEnabled(False)
        context_menu.addAction(action_paste)
        
        context_menu.exec_(global_pos)

    def _show_connection_context_menu(self, line, global_pos):
        """Show context menu for connection."""
        from PyQt5.QtWidgets import QMenu, QAction
        context_menu = QMenu(self)
        
        action_delete = QAction("Delete Connection", self)
        # Assuming line is selected or we target it specifically
        # For now reusing remove_selected_items if line selected
        action_delete.triggered.connect(self.remove_selected_items)
        context_menu.addAction(action_delete)
        
        context_menu.exec_(global_pos)
        
    def _create_subsystem_trigger(self):
        """Trigger subsystem creation."""
        logger.info("Create subsystem trigger called")
        subsys = self.dsim.create_subsystem_from_selection()
        if subsys:
            self.update()

    def generate_bode_plot(self, block):
        """Generate Bode plot for the given block."""
        if hasattr(self.dsim, 'analyzer'):
            self.dsim.analyzer.generate_bode_plot(block)

    def generate_bode_phase_plot(self, block):
        """Generate Bode Phase plot for the given block."""
        if hasattr(self.dsim, 'analyzer'):
            self.dsim.analyzer.generate_bode_phase_plot(block)

    def generate_nyquist_plot(self, block):
        """Generate Nyquist plot for the given block."""
        if hasattr(self.dsim, 'analyzer'):
            self.dsim.analyzer.generate_nyquist_plot(block)
            
    def generate_root_locus(self, block):
        """Generate Root Locus for the given block."""
        if hasattr(self.dsim, 'analyzer'):
            self.dsim.analyzer.generate_root_locus(block)

    def _paste_blocks(self, pos):
        """Paste blocks from clipboard at specified position."""
        try:
            from lib.simulation.menu_block import MenuBlocks
            from PyQt5.QtCore import QPoint

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
                    # Calculate new position (center of block)
                    new_position = QPoint(
                        block_data['coords'].x() + block_data['coords'].width() // 2 + offset_x,
                        block_data['coords'].y() + block_data['coords'].height() // 2 + offset_y
                    )

                    # Create MenuBlocks object from clipboard data
                    io_params = {
                        'inputs': block_data['in_ports'],
                        'outputs': block_data['out_ports'],
                        'b_type': block_data['b_type'],
                        'io_edit': block_data['io_edit']
                    }

                    menu_block = MenuBlocks(
                        block_fn=block_data['block_fn'],
                        fn_name=block_data['fn_name'],
                        io_params=io_params,
                        ex_params=block_data['params'],
                        b_color=block_data['color'],
                        coords=(block_data['coords'].width(), block_data['coords'].height()),
                        external=block_data['external'],
                        block_class=block_data.get('block_class', None)
                    )

                    # Use add_block with the MenuBlocks object
                    new_block = self.dsim.add_block(menu_block, new_position)

                    if new_block:
                        new_block.selected = True

                logger.info(f"Pasted {len(self.clipboard_blocks)} blocks")
                self.update()

        except Exception as e:
            logger.error(f"Error pasting blocks: {str(e)}")



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

    def _highlight_connection_path(self, line):
        """Temporarily highlight a connection path."""
        # This could be enhanced with animation
        line.selected = True
        self.update()

    def _edit_connection_label(self, line):
        """Edit the label of a connection."""
        from PyQt5.QtWidgets import QInputDialog

        # Get current label
        current_label = line.label if hasattr(line, 'label') else ""

        # Show input dialog
        text, ok = QInputDialog.getText(
            self,
            "Edit Connection Label",
            f"Enter label for connection {line.srcblock} → {line.dstblock}:",
            text=current_label
        )

        if ok:
            line.label = str(text)
            self.update()
            logger.info(f"Updated connection label: {line.name} -> '{text}'")

    def _set_connection_routing_mode(self, line, mode):
        """Change the routing mode for a connection."""
        if mode in ["bezier", "orthogonal"]:
            line.set_routing_mode(mode)
            # Force update of the line path
            line.update_line(self.dsim.blocks_list)
            self._capture_state()  # Capture state for undo
            self.update()
            logger.info(f"Changed routing mode for {line.name} to {mode}")

    def _update_hover_states(self, pos):
        """Update hover states for blocks, ports, and connections."""
        needs_repaint = False

        # Check for resize handles on selected blocks (highest priority)
        resize_handle = None
        for block in self.dsim.blocks_list:
            if block.selected:
                handle = self.block_renderer.get_resize_handle_at(block, pos)
                if handle:
                    resize_handle = handle
                    self._set_resize_cursor(handle)
                    break

        # Initialize variables
        new_hovered_port = None

        if not resize_handle:
            # Check for hovered port
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

            # Reset cursor if not over resize handle or port
            if not new_hovered_port:
                self.setCursor(Qt.ArrowCursor)

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
            line_result, _ = self._get_clicked_line(pos)  # Returns (line, collision_type) tuple
            if line_result != self.hovered_line:
                self.hovered_line = line_result
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

    def _set_resize_cursor(self, handle):
        """Set the appropriate cursor for a resize handle."""
        cursor_map = {
            'top_left': Qt.SizeFDiagCursor,
            'top_right': Qt.SizeBDiagCursor,
            'bottom_left': Qt.SizeBDiagCursor,
            'bottom_right': Qt.SizeFDiagCursor,
            'top': Qt.SizeVerCursor,
            'bottom': Qt.SizeVerCursor,
            'left': Qt.SizeHorCursor,
            'right': Qt.SizeHorCursor,
        }
        self.setCursor(cursor_map.get(handle, Qt.ArrowCursor))

    # Validation System
    def run_validation(self):
        """Run diagram validation and update error visualization."""
        try:
            from lib.diagram_validator import DiagramValidator

            validator = DiagramValidator(self.dsim)
            self.validation_errors = validator.validate()

            # Update sets of blocks with errors/warnings
            self.blocks_with_errors = validator.get_blocks_with_errors()
            self.blocks_with_warnings = validator.get_blocks_with_warnings()

            # Enable error visualization
            self.show_validation_errors = True

            # Trigger repaint
            self.update()

            logger.info(f"Validation complete: {len(self.validation_errors)} issues found")
            return self.validation_errors

        except Exception as e:
            logger.error(f"Error running validation: {str(e)}")
            return []

    def clear_validation(self):
        """Clear validation errors and hide indicators."""
        self.validation_errors = []
        self.blocks_with_errors = set()
        self.blocks_with_warnings = set()
        self.show_validation_errors = False
        self.update()



    def _draw_block_error_indicator(self, painter, block, is_error=True):
        """Draw error/warning indicator on a specific block."""
        try:
            # Choose color based on severity
            if is_error:
                indicator_color = QColor(220, 53, 69)  # Red for errors
                border_width = 3
            else:
                indicator_color = QColor(255, 193, 7)  # Yellow/orange for warnings
                border_width = 2

            # Draw pulsing border around block
            painter.setPen(QPen(indicator_color, border_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)

            # Draw outline around block
            padding = 4
            error_rect = QRect(
                block.left - padding,
                block.top - padding,
                block.width + 2 * padding,
                block.height + 2 * padding
            )
            painter.drawRoundedRect(error_rect, 10, 10)

            # Draw error/warning icon in top-right corner
            icon_size = 16
            icon_x = block.left + block.width - icon_size - 2
            icon_y = block.top + 2

            # Draw icon background circle
            icon_bg = QColor(indicator_color)
            icon_bg.setAlpha(200)
            painter.setBrush(icon_bg)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(icon_x, icon_y, icon_size, icon_size)

            # Draw exclamation mark
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            # Vertical line
            painter.drawLine(
                icon_x + icon_size // 2, icon_y + 3,
                icon_x + icon_size // 2, icon_y + icon_size - 6
            )
            # Dot
            painter.drawPoint(icon_x + icon_size // 2, icon_y + icon_size - 3)

        except Exception as e:
            logger.error(f"Error drawing block error indicator: {str(e)}")

    # Undo/Redo System


    def undo(self):
        """Undo the last action."""
        self.history_manager.undo()

    def redo(self):
        """Redo the last undone action."""
        self.history_manager.redo()

    def _push_undo(self, description="Action"):
        """Push current state to undo stack. (Internal helper wrapper)"""
        self.history_manager.push_undo(description)

    def _select_all_blocks(self):
        self.selection_manager.select_all_blocks()

    def _clear_line_selections(self):
        self.selection_manager.clear_line_selections()

    def remove_selected_items(self):
        self.selection_manager.remove_selected_items()

    def clear_canvas(self):
        """Clear all blocks and connections from the canvas."""
        try:
            if hasattr(self.dsim, 'clear_all'):
                self.dsim.clear_all()
                # Clear validation errors when canvas is cleared
                self.clear_validation()
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

    def zoom_to_fit(self):
        """Zoom to fit all blocks in the view."""
        if not self.dsim.blocks_list:
            return

        # Calculate bounding box of all blocks
        min_x = min(block.left for block in self.dsim.blocks_list)
        min_y = min(block.top for block in self.dsim.blocks_list)
        max_x = max(block.left + block.width for block in self.dsim.blocks_list)
        max_y = max(block.top + block.height for block in self.dsim.blocks_list)

        # Add padding
        padding = 50
        bbox_width = max_x - min_x + 2 * padding
        bbox_height = max_y - min_y + 2 * padding

        # Calculate zoom factor to fit
        width_ratio = self.width() / bbox_width if bbox_width > 0 else 1.0
        height_ratio = self.height() / bbox_height if bbox_height > 0 else 1.0
        target_zoom = min(width_ratio, height_ratio, 1.0)  # Don't zoom in beyond 100%

        self.set_zoom(target_zoom)
        logger.info(f"Zoomed to fit: {len(self.dsim.blocks_list)} blocks")

    def toggle_grid(self):
        """Toggle grid visibility."""
        self.grid_visible = not self.grid_visible
        self.update()
        logger.info(f"Grid visibility: {self.grid_visible}")

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming and scrolling.

        - Plain scroll: Pan the canvas (for MacBook trackpad users)
        - Ctrl/Cmd + scroll: Zoom in/out
        """
        modifiers = event.modifiers()

        # Check if Ctrl (or Cmd on macOS) is pressed
        if modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            # Zoom mode
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
        else:
            # Pan/scroll mode - pan the canvas with touchpad scrolling
            delta_x = event.angleDelta().x()
            delta_y = event.angleDelta().y()

            # Scale the delta for smoother scrolling
            scroll_sensitivity = 0.5
            pan_delta = QPoint(
                int(delta_x * scroll_sensitivity),
                int(delta_y * scroll_sensitivity)
            )

            # Apply panning offset
            self.pan_offset += pan_delta
            self.update()

            logger.debug(f"Canvas panned by {pan_delta}, new offset: {self.pan_offset}")

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

            # BodeMag and RootLocus connections logic
            allowed_bode_blocks = ['TranFn', 'DiscreteTranFn', 'StateSpace', 'DiscreteStateSpace', 'PID']
            
            if end_block.block_fn in ["BodeMag", "BodePhase", "Nyquist"] and start_block.block_fn not in allowed_bode_blocks:
                validation_errors.append(f"{end_block.block_fn} block can only be connected to: {', '.join(allowed_bode_blocks)}")

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

    # Analysis Methods
    def generate_bode_plot(self, block):
        """Delegate Bode plot generation to analyzer."""
        if hasattr(self, 'analyzer'):
            self.analyzer.generate_bode_plot(block)
        else:
            logger.error("Analyzer not initialized")

    def generate_root_locus(self, block):
        """Delegate Root Locus generation to analyzer."""
        if hasattr(self, 'analyzer'):
            self.analyzer.generate_root_locus(block)
        else:
            logger.error("Analyzer not initialized")

    def generate_nyquist_plot(self, block):
        """Delegate Nyquist plot generation to analyzer."""
        if hasattr(self, 'analyzer'):
            self.analyzer.generate_nyquist_plot(block)
        else:
            logger.error("Analyzer not initialized")

    def generate_bode_phase_plot(self, block):
        """Delegate Bode Phase plot generation to analyzer."""
        if hasattr(self, 'analyzer'):
            self.analyzer.generate_bode_phase_plot(block)
        else:
            logger.error("Analyzer not initialized")
