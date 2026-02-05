"""Modern Canvas Widget for DiaBloS Phase 2
Handles block rendering, mouse interactions, and drag-and-drop functionality.
"""

import logging
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox, QMenu, QToolTip
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor

# Import DSim and helper modules
import sys
import os
import copy

# Add project root to path (idempotent check)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.append(_project_root)

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
from modern_ui.managers.clipboard_manager import ClipboardManager
from modern_ui.managers.zoom_pan_manager import ZoomPanManager
from modern_ui.managers.connection_manager import ConnectionManager
from modern_ui.managers.rendering_manager import RenderingManager

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
        self.resize_at_limit = False  # True when resize hits minimum size

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
        self.clipboard_manager = ClipboardManager(self)
        self.zoom_pan_manager = ZoomPanManager(self)
        self.connection_manager = ConnectionManager(self)
        self.rendering_manager = RenderingManager(self)
        
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

        # Print verification results to terminal
        self._print_terminal_verification()

    def _print_terminal_verification(self):
        """Print verification results to terminal after simulation completes."""
        import numpy as np
        try:
            # Use active blocks from engine if available, otherwise fall back to blocks_list
            has_engine = hasattr(self.dsim, 'engine') and self.dsim.engine is not None
            use_active = has_engine and len(self.dsim.engine.active_blocks_list) > 0
            blocks_source = self.dsim.engine.active_blocks_list if use_active else self.dsim.blocks_list

            # Collect Display block values
            display_values = {}
            for block in blocks_source:
                if block.block_fn == 'Display':
                    params = block.params or {}
                    display_val = params.get('_display_value_', '---')
                    label = params.get('label', '')
                    block_name = label if label else block.username
                    display_values[block_name] = display_val

            # Collect StateVariable final states (optimization convergence)
            state_values = {}
            for block in blocks_source:
                if block.block_fn == 'StateVariable':
                    exec_params = getattr(block, 'exec_params', {}) or {}
                    state = exec_params.get('_state_')
                    initial = exec_params.get('initial_value')
                    if state is not None:
                        state_arr = np.atleast_1d(state)
                        initial_arr = np.atleast_1d(initial) if initial is not None else None
                        block_name = block.username if block.username else block.name
                        state_values[block_name] = {'final': state_arr, 'initial': initial_arr}

            # Collect Scope convergence info (first/last values)
            scope_convergence = {}
            for block in blocks_source:
                if block.block_fn == 'Scope':
                    exec_params = getattr(block, 'exec_params', {}) or {}
                    vec = exec_params.get('vector')
                    if vec is not None and hasattr(vec, '__len__') and len(vec) > 0:
                        arr = np.array(vec)
                        vec_dim = exec_params.get('vec_dim', 1)
                        labels = exec_params.get('vec_labels', block.username)

                        # Reshape if interleaved multi-dimensional
                        if arr.ndim == 1 and vec_dim > 1 and len(arr) >= vec_dim:
                            num_samples = len(arr) // vec_dim
                            arr = arr[:num_samples * vec_dim].reshape(num_samples, vec_dim)

                        block_name = block.username if block.username else block.name
                        if arr.ndim == 2:
                            first_val = arr[0, :]
                            last_val = arr[-1, :]
                        else:
                            first_val = arr[0] if len(arr) > 0 else None
                            last_val = arr[-1] if len(arr) > 0 else None
                        scope_convergence[block_name] = {
                            'labels': labels,
                            'first': first_val,
                            'last': last_val,
                            'samples': len(arr),
                            'data': arr,
                            'verify_mode': exec_params.get('verify_mode', 'auto'),
                        }

            # Build output with verification checks
            has_output = display_values or state_values or scope_convergence
            all_checks_passed = True
            check_results = []

            if has_output:
                print("\n" + "=" * 60, flush=True)
                print("VERIFICATION RESULTS", flush=True)
                print("=" * 60, flush=True)

                # Display block values
                if display_values:
                    print("\n📊 Display Values:", flush=True)
                    for name, value in display_values.items():
                        print(f"   {name}: {value}", flush=True)

                # StateVariable convergence check
                if state_values:
                    print("\n🎯 Optimization Convergence:", flush=True)
                    for name, info in state_values.items():
                        final = info['final']
                        initial = info['initial']

                        # Check if converged to near zero (common for quadratic minimization)
                        final_norm = np.linalg.norm(final)
                        converged_to_zero = final_norm < 1e-3

                        # Check if state changed from initial
                        if initial is not None:
                            initial_norm = np.linalg.norm(initial)
                            state_changed = not np.allclose(final, initial, rtol=1e-2)
                            reduction = (initial_norm - final_norm) / initial_norm if initial_norm > 0 else 0
                        else:
                            state_changed = True
                            reduction = None

                        # Format output
                        if len(final) <= 4:
                            final_str = np.array2string(final, precision=6, suppress_small=True)
                        else:
                            final_str = f"[{final[0]:.4g}, ..., {final[-1]:.4g}]"

                        status = "✓" if (converged_to_zero or state_changed) else "✗"
                        if not (converged_to_zero or state_changed):
                            all_checks_passed = False

                        print(f"   {status} {name}: {final_str}", flush=True)
                        if reduction is not None and reduction > 0:
                            print(f"      ‖x‖ reduced by {reduction*100:.1f}%", flush=True)
                        if converged_to_zero:
                            print(f"      Converged to ‖x‖ = {final_norm:.2e}", flush=True)

                # Scope convergence verification
                if scope_convergence:
                    print("\n📈 Signal Convergence:", flush=True)
                    for name, info in scope_convergence.items():
                        first = info['first']
                        last = info['last']
                        samples = info['samples']
                        data = info['data']

                        def format_val(v):
                            if v is None:
                                return "N/A"
                            v = np.atleast_1d(v)
                            if len(v) == 1:
                                return f"{float(v[0]):.6g}"
                            elif len(v) <= 3:
                                return np.array2string(v, precision=4, suppress_small=True)
                            else:
                                return f"[{v[0]:.4g}, {v[1]:.4g}, ...]"

                        # Check convergence criteria
                        first_norm = np.linalg.norm(np.atleast_1d(first))
                        last_norm = np.linalg.norm(np.atleast_1d(last))

                        # Get explicit verification mode or fall back to heuristics
                        verify_mode = info.get('verify_mode', 'auto')

                        if verify_mode == "none":
                            # Skip this scope entirely
                            continue

                        if verify_mode == "auto":
                            # Fall back to name-based heuristics (current behavior)
                            # Note: removed 'error' from is_objective keywords to avoid false positives
                            is_objective = any(kw in name.lower() for kw in ['f_', 'cost', 'obj', 'norm', 'value'])
                            is_state = any(kw in name.lower() for kw in ['x_', 'state', 'traj', 'position'])
                        elif verify_mode == "objective":
                            is_objective = True
                            is_state = False
                        elif verify_mode == "trajectory":
                            is_objective = False
                            is_state = True
                        else:  # "comparison" or unknown
                            is_objective = False
                            is_state = False

                        if is_objective and first_norm > 0:
                            # Objective should decrease significantly
                            reduction = (first_norm - last_norm) / first_norm
                            converged = reduction > 0.9 or last_norm < 1e-6
                            status = "✓" if converged else "✗"
                            if not converged:
                                all_checks_passed = False
                            print(f"   {status} {name}: {format_val(first)} → {format_val(last)}", flush=True)
                            if reduction > 0:
                                print(f"      Reduced by {reduction*100:.1f}%", flush=True)
                        elif is_state:
                            # State should change and ideally converge
                            changed = not np.allclose(first, last, rtol=0.01)
                            status = "✓" if changed else "✗"
                            if not changed:
                                all_checks_passed = False
                            print(f"   {status} {name}: {format_val(first)} → {format_val(last)}", flush=True)
                        else:
                            # Generic scope or comparison mode - just show values (no pass/fail)
                            print(f"   • {name} ({samples} pts): {format_val(first)} → {format_val(last)}", flush=True)

                # Final verdict
                print("\n" + "-" * 60, flush=True)
                if all_checks_passed:
                    print("✓ VERIFICATION PASSED", flush=True)
                else:
                    print("✗ VERIFICATION FAILED - Check values above", flush=True)
                print("=" * 60 + "\n", flush=True)
                sys.stdout.flush()
            else:
                print("\n[Simulation completed - no verification data]", flush=True)
                sys.stdout.flush()

        except Exception as e:
            # Log the actual error for debugging
            print(f"\n[Could not print verification results: {e}]", file=sys.stderr, flush=True)
            logger.warning(f"Could not print verification results: {e}")

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

    # ===== Rendering Methods =====

    def _render_blocks(self, painter, draw_ports=True):
        """Render all blocks to canvas."""
        self.rendering_manager.render_blocks(painter, draw_ports)

    def _render_lines(self, painter):
        """Render all connection lines."""
        self.rendering_manager.render_lines(painter)

    def _render_ports(self, painter):
        """Render all ports on top of lines for better visibility."""
        self.rendering_manager.render_ports(painter)

    def _update_line_positions(self):
        """Update line positions after block movement."""
        self.connection_manager.update_line_positions()




    def mousePressEvent(self, event):
        """Handle mouse press events."""
        try:
            # Delegate completely to InteractionManager
            self.interaction_manager.handle_mouse_press(event)

        except Exception as e:
            logger.error(f"Error in canvas mousePressEvent: {str(e)}")

    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events."""
        # Force reset of any pending selection/drag state that might have started on press
        # RESET CANVAS ATTRIBUTES DIRECTLY
        self.is_rect_selecting = False
        self.selection_rect_start = None
        self.selection_rect_end = None
        self.state = State.IDLE
        self.update()

        try:
            if event.button() == Qt.LeftButton:
                pos = self.screen_to_world(event.pos())

                # Check if double-clicked on empty space (not on block or line)
                clicked_block = self._get_clicked_block(pos)
                clicked_line, _ = self._get_clicked_line(pos)

                if clicked_block:
                    logger.info(f"Double-clicked block: {clicked_block.name}, fn: {clicked_block.block_fn}")
                    
                    # 1. SPECIAL: Analysis Blocks -> Trigger Plot
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

                    # 2. SPECIAL: Subsystems -> Enter
                    # Check both block_type (legacy) and block_fn
                    is_subsystem = (getattr(clicked_block, 'block_type', '') == 'subsystem' or 
                                   clicked_block.block_fn == 'Subsystem')
                                   
                    if is_subsystem:
                        self.dsim.enter_subsystem(clicked_block)
                        self.update()
                        logger.info(f"Entered subsystem: {clicked_block.name}")
                        
                        # Reset view to ensure blocks are visible
                        self.pan_offset = QPoint(0, 0)
                        self.zoom_factor = 1.0
                        self.zoom_to_fit()
                        
                        self.scope_changed.emit(self.dsim.get_current_path())
                        return
                    
                    # 3. DEFAULT: Properties Dialog
                    self._show_block_properties(clicked_block)

                if not clicked_block and not clicked_line:
                    # Double-clicked on empty space - open command palette
                    logger.info("Double-clicked on empty canvas - emitting command_palette_requested")
                    self.command_palette_requested.emit()
                    
        except Exception as e:
            logger.error(f"Error in mouseDoubleClickEvent: {e}")

        # Accept the event to prevent propagation issues
        event.accept()

    def focusInEvent(self, event):
        """Reset selection state when focus returns to canvas without mouse button pressed."""
        super().focusInEvent(event)
        # If focus returns without left mouse button pressed, reset any pending rect selection
        # This handles cases where a popup (like command palette) closed and focus returned
        from PyQt5.QtWidgets import QApplication
        if not (QApplication.mouseButtons() & Qt.LeftButton):
            if self.is_rect_selecting:
                logger.debug("Resetting rect selection on focus return (no mouse button pressed)")
                self.is_rect_selecting = False
                self.selection_rect_start = None
                self.selection_rect_end = None
                self.update()

    def navigate_scope_by_path(self, path_str):
        """Navigate to a specific scope path (e.g. via BreadcrumbBar string)."""
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
        """Handle right mouse button clicks - delegate to MenuManager."""
        try:
            # Delegate to MenuManager for rich context menus
            self.menu_manager.handle_context_menu(pos)
        except Exception as e:
            logger.error(f"Error in _handle_right_click: {str(e)}")

    def show_bode_plot_menu(self, block, pos):
        """Show context menu for the BodeMag block."""
        menu = QMenu(self)
        plot_action = menu.addAction("Generate Bode Plot")
        action = menu.exec_(pos)
        if action == plot_action:
            self.generate_bode_plot(block)

    def show_root_locus_menu(self, block, pos):
        """Show context menu for the RootLocus block."""
        menu = QMenu(self)
        plot_action = menu.addAction("Generate Root Locus Plot")
        action = menu.exec_(pos)
        if action == plot_action:
            self.generate_root_locus(block)

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
        """Get the line at the given position."""
        return self.connection_manager.get_clicked_line(pos)

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
                # Early return - but still reset state below via finally
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
        except Exception as e:
            logger.error(f"Error finalizing rectangle selection: {str(e)}")
        finally:
            # Always reset rectangle selection state, even on early return or exception
            self.is_rect_selecting = False
            self.selection_rect_start = None
            self.selection_rect_end = None
            self.update()

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
        return self.connection_manager.check_port_clicks(pos)

    def _handle_port_click(self, block, port_type, port_index, pos):
        """Handle port click for connection creation."""
        self.connection_manager.handle_port_click(block, port_type, port_index, pos)

    def _finish_line_creation(self, end_block, end_port):
        """Complete line creation between two blocks."""
        self.connection_manager.finish_line_creation(end_block, end_port)

    def _check_line_clicks(self, pos):
        """Check for clicks on connection lines."""
        self.connection_manager.check_line_clicks(pos)

    def _point_near_line(self, pos, line):
        """Check if a point is near a line."""
        return self.connection_manager.point_near_line(pos, line)

    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate minimum distance from point to line segment."""
        return self.connection_manager.point_to_line_distance(point, line_start, line_end)

    def _handle_line_click(self, line, collision_result, pos):
        """Handle clicking on a connection line."""
        self.connection_manager.handle_line_click(line, collision_result, pos)

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

            # Also check block's port-based minimum height (for multi-port blocks)
            if hasattr(block, 'calculate_min_size'):
                port_min_height = block.calculate_min_size()
                min_height = max(min_height, port_min_height)

            # Track if we're hitting the resize limit
            at_width_limit = new_width <= min_width
            at_height_limit = new_height <= min_height

            # Ensure minimum size
            if new_width < min_width:
                if 'left' in handle:
                    new_left = start_rect.right() - min_width
                new_width = min_width

            if new_height < min_height:
                if 'top' in handle:
                    new_top = start_rect.bottom() - min_height
                new_height = min_height

            # Visual feedback: change cursor when at limit
            if at_width_limit or at_height_limit:
                self.setCursor(Qt.ForbiddenCursor)
                self.resize_at_limit = True
            else:
                # Restore appropriate resize cursor
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
                self.resize_at_limit = False

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
                self.resize_at_limit = False
                self.setCursor(Qt.ArrowCursor)

                # Ensure lines are updated after resize
                self._update_line_positions()
                self.update()
        except Exception as e:
            logger.error(f"Error finishing resize: {str(e)}")

    def _cancel_line_creation(self):
        """Cancel line creation process."""
        self.connection_manager.cancel_line_creation()

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
                    # Check if anything is selected
                    has_selection = any(b.selected for b in getattr(self.dsim, 'blocks_list', [])) or \
                                    any(l.selected for l in getattr(self.dsim, 'line_list', []))
                    if has_selection:
                        self._clear_selections()
                    elif self.dsim.current_subsystem:
                        # Exit subsystem if no selection and inside one
                        self.dsim.exit_subsystem()
                        self.pan_offset = QPoint(0, 0)
                        self.zoom_factor = 1.0
                        self.zoom_to_fit()
                        self.scope_changed.emit(self.dsim.get_current_path())
                    self.update()
            elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                # Delete or Backspace - works on both Mac (Delete key) and Windows/Linux (Del key)
                self.remove_selected_items()
            elif event.key() == Qt.Key_G and ctrl_pressed:
                # Ctrl+G: Create subsystem from selection
                self._create_subsystem_trigger()
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
            # Alignment shortcuts (Ctrl+Shift+key)
            elif event.key() == Qt.Key_L and ctrl_pressed and shift_pressed:
                self.align_left()
            elif event.key() == Qt.Key_R and ctrl_pressed and shift_pressed:
                self.align_right()
            elif event.key() == Qt.Key_H and ctrl_pressed and shift_pressed:
                self.align_center_horizontal()
            elif event.key() == Qt.Key_T and ctrl_pressed and shift_pressed:
                self.align_top()
            elif event.key() == Qt.Key_B and ctrl_pressed and shift_pressed:
                self.align_bottom()
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
        self.clipboard_manager.copy_selected_blocks()
        # Keep clipboard_blocks accessible for backward compatibility
        self.clipboard_blocks = self.clipboard_manager.clipboard_blocks

    def paste_blocks(self):
        """Paste blocks from clipboard."""
        self.clipboard_manager.paste_blocks()



    # Helper methods for context menu actions
    def _duplicate_block(self, block):
        """Duplicate a block."""
        self.clipboard_manager._duplicate_block(block)

    def _copy_selected_blocks(self):
        """Copy selected blocks to clipboard (legacy method, use copy_selected_blocks instead)."""
        self.copy_selected_blocks()

    def _cut_selected_blocks(self):
        """Cut selected blocks to clipboard."""
        self.clipboard_manager.cut_selected_blocks()

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
        # Context menu is already handled by _handle_right_click via interaction_manager.
        # Just accept the event to prevent duplicate menu from appearing.
        event.accept()

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
            self._update_line_positions()  # Recalculate line paths to connect to new subsystem ports
            self.update()

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
        self.connection_manager.delete_line(line)

    def _highlight_connection_path(self, line):
        """Temporarily highlight a connection path."""
        self.connection_manager.highlight_connection_path(line)

    def _edit_connection_label(self, line):
        """Edit the label of a connection."""
        self.connection_manager.edit_connection_label(line)

    def _set_connection_routing_mode(self, line, mode):
        """Change the routing mode for a connection."""
        self.connection_manager.set_connection_routing_mode(line, mode)

    def _update_hover_states(self, pos):
        """Update hover states for blocks, ports, and connections."""
        self.rendering_manager.update_hover_states(pos)

    def _point_near_port(self, point, port_pos, threshold=12):
        """Check if a point is near a port position."""
        return self.rendering_manager.point_near_port(point, port_pos, threshold)

    def _set_resize_cursor(self, handle):
        """Set the appropriate cursor for a resize handle."""
        self.rendering_manager.set_resize_cursor(handle)

    # Validation System
    def run_validation(self):
        """Run diagram validation and update error visualization."""
        return self.rendering_manager.run_validation()

    def clear_validation(self):
        """Clear validation errors and hide indicators."""
        self.rendering_manager.clear_validation()

    def _draw_block_error_indicator(self, painter, block, is_error=True):
        """Draw error/warning indicator on a specific block."""
        self.rendering_manager.draw_block_error_indicator(painter, block, is_error)

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
        return self.zoom_pan_manager.screen_to_world(pos)

    def world_to_screen(self, pos):
        """Converts world coordinates to screen coordinates."""
        return self.zoom_pan_manager.world_to_screen(pos)

    def set_zoom(self, factor):
        """Set zoom factor."""
        self.zoom_pan_manager.set_zoom(factor)

    def zoom_in(self):
        """Zoom in by 10%."""
        self.zoom_pan_manager.zoom_in()

    def zoom_out(self):
        """Zoom out by 10%."""
        self.zoom_pan_manager.zoom_out()

    def zoom_to_fit(self):
        """Zoom to fit all blocks in the view."""
        self.zoom_pan_manager.zoom_to_fit()

    def reset_view(self):
        """Reset zoom and pan to default values."""
        self.zoom_pan_manager.reset_view()

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
        self.zoom_pan_manager.handle_wheel_event(event)

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

    # ===== Alignment Methods =====

    def align_left(self):
        """Align selected blocks to the leftmost block's left edge."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Left")
        AlignmentTools.align_left(blocks)
        self._update_line_positions()
        self.update()

    def align_right(self):
        """Align selected blocks to the rightmost block's right edge."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Right")
        AlignmentTools.align_right(blocks)
        self._update_line_positions()
        self.update()

    def align_center_horizontal(self):
        """Align selected blocks to horizontal center."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Center Horizontal")
        AlignmentTools.align_center_horizontal(blocks)
        self._update_line_positions()
        self.update()

    def align_top(self):
        """Align selected blocks to the topmost block's top edge."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Top")
        AlignmentTools.align_top(blocks)
        self._update_line_positions()
        self.update()

    def align_bottom(self):
        """Align selected blocks to the bottommost block's bottom edge."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Bottom")
        AlignmentTools.align_bottom(blocks)
        self._update_line_positions()
        self.update()

    def align_center_vertical(self):
        """Align selected blocks to vertical center."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 2:
            logger.info("Need at least 2 blocks selected to align")
            return
        self._push_undo("Align Center Vertical")
        AlignmentTools.align_center_vertical(blocks)
        self._update_line_positions()
        self.update()

    def distribute_horizontal(self):
        """Distribute selected blocks evenly horizontally."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 3:
            logger.info("Need at least 3 blocks selected to distribute")
            return
        self._push_undo("Distribute Horizontal")
        AlignmentTools.distribute_horizontal(blocks)
        self._update_line_positions()
        self.update()

    def distribute_vertical(self):
        """Distribute selected blocks evenly vertically."""
        from modern_ui.tools.alignment_tools import AlignmentTools
        blocks = self.selection_manager.get_selected_blocks()
        if len(blocks) < 3:
            logger.info("Need at least 3 blocks selected to distribute")
            return
        self._push_undo("Distribute Vertical")
        AlignmentTools.distribute_vertical(blocks)
        self._update_line_positions()
        self.update()
