"""Modern Canvas Widget for DiaBloS Phase 2
Handles block rendering, mouse interactions, and drag-and-drop functionality.
"""

import logging
import math
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QEvent, pyqtSignal
from PyQt5.QtGui import QPainter, QPen

# Import DSim and helper modules
import sys
import os
import types

# Add project root to path (idempotent check)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.append(_project_root)

from lib.improvements import PerformanceHelper, SafetyChecks, ValidationHelper, SimulationConfig
from modern_ui.themes.theme_manager import (
    theme_manager, get_ui_font, TYPE, SPACE,
    pulse_alpha, PULSE_INTERVAL_MS, PULSE_PHASE_STEP,
)
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
from modern_ui.managers.drag_resize_manager import DragResizeManager
from modern_ui.controllers.simulation_controller import SimulationController
from modern_ui.widgets.canvas_state import CanvasState

logger = logging.getLogger(__name__)


class ModernCanvas(QWidget):
    """Modern canvas widget for DiaBloS block diagram editing."""

    # Idle-OFF animation tuning. Cadence is shared with the toolbar status dot
    # via theme_manager.PULSE_* so the two pulses read as one effect.
    _ANIM_INTERVAL_MS = PULSE_INTERVAL_MS
    _ANIM_PHASE_STEP = PULSE_PHASE_STEP

    # Signals
    block_selected = pyqtSignal(object)  # Emitted when a block is selected
    connection_created = pyqtSignal(object, object)  # Emitted when a connection is made
    simulation_status_changed = pyqtSignal(str)  # Emitted when simulation status changes
    command_palette_requested = pyqtSignal()  # Emitted when command palette should open
    scope_changed = pyqtSignal(list)  # Emitted when navigation scope changes (path)
    cursor_moved = pyqtSignal(int, int)  # (x, y) in canvas coordinates — drives status bar
    
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
        
        # State management - Unified state object
        self.canvas_state = CanvasState()

        # Connection routing default
        self.default_routing_mode = "bezier"

        # Live overlay toggles (View → Live overlay submenu)
        self.show_live_chips = True

        # Clipboard and undo/redo state live in their managers
        # (clipboard_manager / history_manager) — the single source of truth.

        # Initialize helpers
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
        self.drag_resize_manager = DragResizeManager(self)

        # Plain attribute set only in start_drag (never via canvas_state); guard
        # against AttributeError when _finish_drag reads it before any drag.
        self.dragging_block = None

        # Smart-alignment guide lines (canvas coords) active during a block
        # drag; recomputed each move by DragResizeManager, drawn in paintEvent.
        self._alignment_guides = []

        # Simulation lifecycle controller (re-emits status as our own signal)
        self._sim_controller = SimulationController(self.dsim, parent=self)
        self._sim_controller.status_changed.connect(self.simulation_status_changed)

        # Initialize Analysis Tool
        self.analyzer = ControlSystemAnalyzer(self, parent=self)
        
        # Initialize Renderer
        self.block_renderer = BlockRenderer()
        self.connection_renderer = ConnectionRenderer()
        self.canvas_renderer = CanvasRenderer()

        # Idle-OFF animation: a single ~30fps timer drives subtle glow pulses on
        # the hovered port / active wire / running simulation. It stays STOPPED
        # while the canvas is idle so large diagrams never repaint continuously
        # (see _evaluate_animation_state). The phase is a monotonic float fed
        # through sin() by renderers via glow_pulse_alpha().
        self._animation_phase = 0.0
        self._animation_timer = QTimer(self)  # parented -> no leak on destroy
        self._animation_timer.setInterval(self._ANIM_INTERVAL_MS)
        self._animation_timer.timeout.connect(self._on_animation_tick)
        # Let the connection renderer pulse its active-wire glow via our phase.
        self.connection_renderer.pulse_alpha = self.glow_pulse_alpha
        # Re-evaluate the gate whenever simulation status flips (start/stop).
        self.simulation_status_changed.connect(
            lambda _msg: self._evaluate_animation_state()
        )

        # Setup UI
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
        # Force a repaint so blocks pick up the new theme colors immediately
        # (BlockRenderer resolves fill/border via theme_manager each paint).
        self.update()
    
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

                    # The block now exists in dsim. Keep the return value
                    # consistent with dsim state even if a post-add step
                    # (undo capture, signal emit, repaint) raises.
                    try:
                        # Capture state for undo
                        self._push_undo("Add Block")

                        # Emit signal
                        self.block_selected.emit(new_block)

                        # Record this real placement into the palette's Recent
                        # section. This is the single chokepoint every add path
                        # (drag-drop, Enter/activate, double-click, quick-add)
                        # funnels through, so a drag cancelled with Escape or
                        # dropped outside the canvas never records a phantom
                        # recent — only actual adds do.
                        self._record_palette_recent(menu_block)

                        # Trigger repaint
                        self.update()
                    except Exception as e:
                        logger.error(f"Post-add step failed for {block_name}: {str(e)}")

                    return new_block
                else:
                    logger.error(f"Failed to create block {block_name}")
            else:
                logger.error("DSim does not have add_block method")
                
        except Exception as e:
            logger.error(f"Error adding block from palette: {str(e)}")

        return None

    def _record_palette_recent(self, menu_block):
        """Record a real block placement into the palette's Recent section.

        Reaches the palette via the top-level window's ``block_palette`` (set in
        LayoutManager). Fully defensive: a missing window, missing palette, or a
        palette without record_recent is silently ignored, so adding a block
        never fails just because the palette is absent (e.g. in tests/headless).
        """
        try:
            fn_name = getattr(menu_block, 'fn_name', None)
            if not fn_name:
                return
            window = self.window()
            palette = getattr(window, 'block_palette', None) if window is not None else None
            recorder = getattr(palette, 'record_recent', None)
            if callable(recorder):
                recorder(fn_name)
        except Exception as e:
            # Recents is a cosmetic convenience; never let it break adding a block.
            logger.debug(f"Recording palette recent failed: {e}")

    def start_simulation(self):
        """Start simulation with validation (delegates to SimulationController)."""
        return self._sim_controller.start()

    def run_batch_simulation(self):
        """Run the simulation in batch mode (delegates to SimulationController)."""
        return self._sim_controller.run_batch()

    def stop_simulation(self):
        """Stop simulation safely (delegates to SimulationController)."""
        return self._sim_controller.stop()

    def get_simulation_time(self):
        """Get current simulation time (delegates to SimulationController)."""
        return self._sim_controller.current_time()

    def is_simulation_running(self):
        """Check if simulation is running (delegates to SimulationController)."""
        return self._sim_controller.is_running()

    # ===== Idle-OFF glow animation =====

    def _animation_should_run(self) -> bool:
        """True while any state that warrants a live glow pulse is active.

        Gated to: a hovered port, an in-progress connection drag, or a running
        simulation. When none hold the timer is stopped so an idle diagram —
        however large — never repaints continuously.
        """
        return bool(
            self.hovered_port is not None
            or self.line_creation_state
            or self.is_simulation_running()
        )

    def _evaluate_animation_state(self):
        """Start/stop the animation timer to match the current gate.

        Called from the hover/connection state setters and on simulation status
        changes. Idempotent: starting an already-running timer (or stopping an
        already-stopped one) is a no-op, so it is cheap to call on every hover
        transition. On the idle edge it issues one final repaint so the pulse
        settles back to its resting alpha instead of freezing mid-cycle.
        """
        # The property setters that call us can, in principle, fire before the
        # timer is constructed in __init__; tolerate that gracefully.
        timer = getattr(self, '_animation_timer', None)
        if timer is None:
            return
        try:
            should_run = self._animation_should_run()
            active = timer.isActive()
            if should_run and not active:
                timer.start()
            elif not should_run and active:
                timer.stop()
                self._animation_phase = 0.0  # rest at the resting alpha
                self.update()  # final repaint so the glow settles, not freezes
        except Exception as e:
            # Animation is purely cosmetic; never let a gate hiccup break input.
            logger.debug(f"Animation state evaluation failed: {e}")

    def _on_animation_tick(self):
        """Advance the sine phase one step and request a repaint."""
        # Wrap to keep the float bounded over long-running sessions.
        self._animation_phase = (self._animation_phase + self._ANIM_PHASE_STEP) % (2 * math.pi)
        self.update()

    @property
    def animation_phase(self) -> float:
        """Current monotonic sine phase (radians) for renderers to read."""
        return self._animation_phase

    def glow_pulse_alpha(self, base_alpha: int, depth: float = 0.4) -> int:
        """Modulate a hovered-port / active-wire glow by the canvas phase.

        Renderers call this to make the glow breathe while the animation timer
        runs; when idle the phase is 0 so it resolves to a stable ``base_alpha``.
        Delegates to the shared ``theme_manager.pulse_alpha`` so the canvas and
        the toolbar status dot use one pulse formula.
        """
        return pulse_alpha(self._animation_phase, base_alpha, depth)

    def paintEvent(self, event):
        """Paint the canvas with blocks, connections, and other elements."""
        painter = QPainter()
        begun = painter.begin(self)
        if not begun:
            return
        try:
            self.perf_helper.start_timer("canvas_paint")
            painter.setRenderHint(QPainter.Antialiasing)

            # Fill viewport background in WIDGET coordinates so panning
            # doesn't expose unfilled areas at the edges.
            painter.fillRect(self.rect(), theme_manager.get_color('canvas_background'))

            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)

            # Compute the visible region in SCENE coordinates so the grid
            # tiles across whatever the user has scrolled into view, instead
            # of looking like a finite sheet that drags around with the pan.
            zoom = self.zoom_factor or 1.0
            sx0 = int(-self.pan_offset.x() / zoom) - 1
            sy0 = int(-self.pan_offset.y() / zoom) - 1
            sw = int(self.width() / zoom) + 2
            sh = int(self.height() / zoom) + 2
            scene_rect = QRect(sx0, sy0, sw, sh)
            self.canvas_renderer.draw_grid(painter, scene_rect, sw, sh, self.grid_visible)

            # Empty-canvas placeholder hint: when nothing has been placed yet
            # and the user isn't mid-simulation, draw dim guidance on how to get
            # started. Purely additive — drawn over the grid, under everything
            # else, so it never collides with real diagram content.
            if not getattr(self.dsim, 'blocks_list', []) and not self.is_simulation_running():
                self._draw_empty_hint(painter)

            # Draw DSim elements in proper order: blocks -> lines -> ports
            # This ensures ports appear on top of connection lines
            self._render_blocks(painter, draw_ports=False)
            self._render_lines(painter)
            self._render_ports(painter)

            # Smart-alignment guides while dragging a block. Drawn in SCENE
            # space (the pan/zoom transform is still active) so the hairlines
            # land exactly on the shared block edges.
            if self.state == State.DRAGGING and self._alignment_guides:
                self.canvas_renderer.draw_alignment_guides(painter, self._alignment_guides)

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

            # Draw hover effects. Pass our glow_pulse_alpha so the renderer can
            # gently pulse the hovered-port glow while the animation timer runs
            # (it resolves to a stable alpha when idle).
            self.canvas_renderer.draw_hover_effects(
                painter, self.hovered_port, self.hovered_block, self.hovered_line,
                pulse_alpha=self.glow_pulse_alpha,
            )

            # Draw validation error indicators
            if self.show_validation_errors:
                self.canvas_renderer.draw_validation_errors(painter, self.blocks_with_errors, self.blocks_with_warnings)

            # Draw routing tag HUD (Goto/From overview)
            self.canvas_renderer.draw_tag_hud(painter, self.dsim)

            # Live overlay: V1 port-value chips while simulation is running.
            # Hidden during drag/zoom for perf (the chips would jitter anyway).
            if (self.is_simulation_running()
                    and getattr(self, 'show_live_chips', True)
                    and self.state != State.DRAGGING):
                try:
                    self.connection_renderer.draw_port_value_chips(
                        painter, getattr(self.dsim, 'blocks_list', []) or []
                    )
                except Exception as e:
                    # Non-critical live overlay: log at debug so a genuine
                    # chip-render bug is diagnosable without spamming each frame.
                    logger.debug(f"Port-value chip render failed: {e}")

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
                    logger.debug("Failed to end painter in paintEvent finally block", exc_info=True)

    @staticmethod
    def _empty_hint_lines():
        """Guidance lines shown on an empty canvas.

        Kept as a tiny pure helper so the wording is unit-testable without a
        live QPainter. Order is top-to-bottom as drawn.
        """
        return [
            "Double-click to add a block",
            "Drag a block from the palette",
            "Open an example from File ▸ Examples",
        ]

    def _draw_empty_hint(self, painter):
        """Draw centered, dim guidance text for an empty, idle canvas.

        Drawn in WIDGET coordinates (the painter is reset for the duration) so
        the hint stays centered in the viewport regardless of pan or zoom.
        """
        lines = self._empty_hint_lines()
        painter.save()
        try:
            # Drop the scene transform so the hint is laid out against the
            # widget viewport, then center it there.
            painter.resetTransform()
            painter.setPen(QPen(theme_manager.get_color('text_disabled')))
            painter.setFont(get_ui_font(TYPE['subtitle']))

            fm = painter.fontMetrics()
            line_h = fm.height() + SPACE['sm']
            block_h = line_h * len(lines)
            y = (self.height() - block_h) // 2 + fm.ascent()
            for line in lines:
                rect = QRect(0, int(y - fm.ascent()), self.width(), line_h)
                painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, line)
                y += line_h
        finally:
            painter.restore()

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
        """Update line positions after block movement.

        During an active block drag this is the per-move hook InteractionManager
        calls right after relocating the block, so we first let the drag manager
        apply smart-alignment snapping (which may nudge the block onto a shared
        edge) before the wires are recomputed against the final position.
        """
        if self.state == State.DRAGGING and self.dragging_block is not None:
            self.drag_resize_manager.update_drag_alignment()
        self.connection_manager.update_line_positions()

    def _reroute_affected_lines(self, block_names):
        """Re-run A* routing on lines connected to the given block names."""
        try:
            from lib.simulation.wire_router import route_all_lines
            lines = self.dsim.line_list
            affected = [l for l in lines if l.srcblock in block_names or l.dstblock in block_names]
            if affected:
                route_all_lines(affected, self.dsim.blocks_list)
                self.dsim.dirty = True
                # Request a repaint so rerouted wires show immediately, even if
                # the caller's interaction flow doesn't trigger update().
                self.update()
        except Exception as e:
            logger.error(f"Reroute after move failed: {e}")

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
        had_selection = any(
            b.selected for b in getattr(self.dsim, 'blocks_list', [])
        )
        for block in getattr(self.dsim, 'blocks_list', []):
            block.selected = False
        for line in getattr(self.dsim, 'line_list', []):
            line.selected = False
            if hasattr(line, 'selected_segment'):
                line.selected_segment = -1
        self.source_block_for_connection = None
        if had_selection:
            self.block_selected.emit(None)
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

    def _finish_line_creation(self, end_block, end_port):
        """Complete line creation between two blocks."""
        self.connection_manager.finish_line_creation(end_block, end_port)

    def _handle_line_click(self, line, collision_result, pos):
        """Handle clicking on a connection line."""
        self.connection_manager.handle_line_click(line, collision_result, pos)

    def start_drag(self, block, pos):
        """Start dragging a block (or multiple selected blocks)."""
        return self.drag_resize_manager.start_drag(block, pos)

    def _start_resize(self, block, handle, pos):
        """Start resizing a block."""
        return self.drag_resize_manager._start_resize(block, handle, pos)

    def _perform_resize(self, pos):
        """Perform the resize operation based on current mouse position."""
        return self.drag_resize_manager._perform_resize(pos)

    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        # Surface canvas-coords cursor for the status bar
        try:
            world = self.screen_to_world(event.pos())
            self.cursor_moved.emit(world.x(), world.y())
        except Exception as e:
            # Status-bar cursor readout only; log at debug so a broken
            # coordinate transform stays observable without disrupting moves.
            logger.debug(f"Cursor coordinate emit failed: {e}")
        # Delegate to InteractionManager
        self.interaction_manager.handle_mouse_move(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        # Delegate to InteractionManager
        self.interaction_manager.handle_mouse_release(event)

    def _finish_drag(self):
        """Finish dragging operation."""
        return self.drag_resize_manager._finish_drag()

    def _finish_resize(self):
        """Finish resizing operation."""
        return self.drag_resize_manager._finish_resize()

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

            clipboard_blocks = self.clipboard_manager.clipboard_blocks
            if not clipboard_blocks:
                return

            # Push undo state before pasting (snapshot must be the pre-paste
            # state). If nothing is actually created we roll this back below so
            # the user doesn't get a no-op 'Paste' entry on the undo stack.
            self._push_undo("Paste")

            # Calculate offset from first block's position to paste position
            first_block_coords = clipboard_blocks[0]['coords']
            offset_x = pos.x() - first_block_coords.x()
            offset_y = pos.y() - first_block_coords.y()

            self._clear_selections()
            created_count = 0
            for block_data in clipboard_blocks:
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
                # Preserve category so add_block resolves the correct
                # theme color (otherwise it defaults to 'Other' -> grey).
                menu_block.category = block_data.get('category', 'Other')

                # Use add_block with the MenuBlocks object
                new_block = self.dsim.add_block(menu_block, new_position)

                if new_block:
                    new_block.selected = True
                    created_count += 1

            if created_count:
                logger.info(f"Pasted {created_count} blocks")
                self.update()
            else:
                # Nothing was created: drop the no-op 'Paste' undo entry so the
                # user isn't forced to undo twice for no visible change.
                if self.history_manager.undo_stack:
                    self.history_manager.undo_stack.pop()
                logger.warning("Paste produced no new blocks; skipped undo entry")

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

    def auto_route_lines(self):
        """Recompute every connection's path with the orthogonal A* router."""
        try:
            from lib.simulation.wire_router import route_all_lines
            blocks = self.dsim.blocks_list
            lines = self.dsim.line_list
            if not lines:
                logger.info("Auto-route: no connections to route")
                return
            self._push_undo("Auto-route wires")
            n = route_all_lines(lines, blocks)
            self.dsim.dirty = True
            self.update()
            logger.info(f"Auto-route: rerouted {n} of {len(lines)} connections")
        except Exception as e:
            logger.error(f"Auto-route failed: {e}")

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

    def event(self, event):
        """Intercept macOS trackpad pinch-to-zoom native gestures.

        PyQt5 has no dedicated nativeGestureEvent override; QNativeGestureEvent
        is delivered here with type QEvent.NativeGesture.
        """
        if event.type() == QEvent.NativeGesture:
            if self.zoom_pan_manager.handle_native_gesture(event):
                event.accept()
                return True
        return super().event(event)

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

            # Check if the destination input port is already connected.
            # An exact-duplicate connection (same src+srcport+dst+dstport) is a
            # strict subset of this case, so a single pass over the destination
            # port covers both without emitting two overlapping messages.
            existing_lines = getattr(self.dsim, 'line_list', [])
            end_name = getattr(end_block, 'name', '')
            for line in existing_lines:
                if (hasattr(line, 'dstblock') and hasattr(line, 'dstport')):
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
                temp_line = types.SimpleNamespace(
                    srcblock=getattr(start_block, 'name', ''),
                    srcport=start_port,
                    dstblock=getattr(end_block, 'name', ''),
                    dstport=end_port
                )
                temp_lines.append(temp_line)

                is_valid, helper_errors = ValidationHelper.validate_block_connections(
                    all_blocks, temp_lines
                )
                if not is_valid:
                    validation_errors.extend(helper_errors)
            except AttributeError as e:
                # Helper genuinely unavailable (method missing) — expected on
                # builds without the extended validator; keep it quiet.
                logger.debug(f"ValidationHelper not available: {str(e)}")
            except Exception as e:
                # The helper exists but raised while validating: that's a real
                # bug in the validator, not an absent feature. Surface it so a
                # broken validator isn't mistaken for a passing connection.
                logger.warning(f"ValidationHelper execution failed: {str(e)}")


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

    def compute_lqr(self, block):
        """Delegate LQR computation to analyzer."""
        if hasattr(self, 'analyzer'):
            self.analyzer.compute_lqr(block)
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

    # =========================================================================
    # Backward-compatible properties delegating to canvas_state
    # These allow existing code to work while migrating to the new state object
    # =========================================================================

    # Zoom/Pan properties
    @property
    def zoom_factor(self):
        return self.canvas_state.zoom_pan.zoom_factor

    @zoom_factor.setter
    def zoom_factor(self, value):
        self.canvas_state.zoom_pan.zoom_factor = value

    @property
    def zoom_level(self):
        return self.canvas_state.zoom_pan.zoom_factor

    @zoom_level.setter
    def zoom_level(self, value):
        self.canvas_state.zoom_pan.zoom_factor = value

    @property
    def pan_offset(self):
        return self.canvas_state.zoom_pan.pan_offset

    @pan_offset.setter
    def pan_offset(self, value):
        self.canvas_state.zoom_pan.pan_offset = value

    @property
    def panning(self):
        return self.canvas_state.zoom_pan.is_panning

    @panning.setter
    def panning(self, value):
        self.canvas_state.zoom_pan.is_panning = value

    @property
    def last_pan_pos(self):
        return self.canvas_state.zoom_pan.last_pan_pos

    @last_pan_pos.setter
    def last_pan_pos(self, value):
        self.canvas_state.zoom_pan.last_pan_pos = value

    # Grid properties
    @property
    def grid_visible(self):
        return self.canvas_state.grid.visible

    @grid_visible.setter
    def grid_visible(self, value):
        self.canvas_state.grid.visible = value

    @property
    def grid_size(self):
        return self.canvas_state.grid.size

    @grid_size.setter
    def grid_size(self, value):
        self.canvas_state.grid.size = value

    @property
    def snap_enabled(self):
        return self.canvas_state.grid.snap_enabled

    @snap_enabled.setter
    def snap_enabled(self, value):
        self.canvas_state.grid.snap_enabled = value

    # Selection properties
    @property
    def selection_rect_start(self):
        return self.canvas_state.selection.rect_start

    @selection_rect_start.setter
    def selection_rect_start(self, value):
        self.canvas_state.selection.rect_start = value

    @property
    def selection_rect_end(self):
        return self.canvas_state.selection.rect_end

    @selection_rect_end.setter
    def selection_rect_end(self, value):
        self.canvas_state.selection.rect_end = value

    @property
    def is_rect_selecting(self):
        return self.canvas_state.selection.is_selecting

    @is_rect_selecting.setter
    def is_rect_selecting(self, value):
        self.canvas_state.selection.is_selecting = value

    # Hover properties
    @property
    def hovered_block(self):
        return self.canvas_state.hover.block

    @hovered_block.setter
    def hovered_block(self, value):
        self.canvas_state.hover.block = value

    @property
    def hovered_port(self):
        return self.canvas_state.hover.port

    @hovered_port.setter
    def hovered_port(self, value):
        self.canvas_state.hover.port = value
        # Drive the idle-OFF glow timer: starts on the first hover, stops when
        # the port (and other gated states) clear.
        self._evaluate_animation_state()

    @property
    def hovered_line(self):
        return self.canvas_state.hover.line

    @hovered_line.setter
    def hovered_line(self, value):
        self.canvas_state.hover.line = value

    # Drag properties
    @property
    def drag_offset(self):
        return self.canvas_state.drag.offset

    @drag_offset.setter
    def drag_offset(self, value):
        self.canvas_state.drag.offset = value

    @property
    def drag_offsets(self):
        return self.canvas_state.drag.offsets

    @drag_offsets.setter
    def drag_offsets(self, value):
        self.canvas_state.drag.offsets = value

    @property
    def drag_start_positions(self):
        return self.canvas_state.drag.start_positions

    @drag_start_positions.setter
    def drag_start_positions(self, value):
        self.canvas_state.drag.start_positions = value

    # Resize properties
    @property
    def resizing_block(self):
        return self.canvas_state.resize.block

    @resizing_block.setter
    def resizing_block(self, value):
        self.canvas_state.resize.block = value

    @property
    def resize_handle(self):
        return self.canvas_state.resize.handle

    @resize_handle.setter
    def resize_handle(self, value):
        self.canvas_state.resize.handle = value

    @property
    def resize_start_rect(self):
        return self.canvas_state.resize.start_rect

    @resize_start_rect.setter
    def resize_start_rect(self, value):
        self.canvas_state.resize.start_rect = value

    @property
    def resize_start_pos(self):
        return self.canvas_state.resize.start_pos

    @resize_start_pos.setter
    def resize_start_pos(self, value):
        self.canvas_state.resize.start_pos = value

    @property
    def resize_at_limit(self):
        return self.canvas_state.resize.at_limit

    @resize_at_limit.setter
    def resize_at_limit(self, value):
        self.canvas_state.resize.at_limit = value

    # Connection properties
    @property
    def line_creation_state(self):
        return self.canvas_state.connection.creation_state

    @line_creation_state.setter
    def line_creation_state(self, value):
        self.canvas_state.connection.creation_state = value
        # A connection drag is one of the gated states; (re)evaluate the timer
        # when it begins ('start') or clears (None).
        self._evaluate_animation_state()

    @property
    def line_start_block(self):
        return self.canvas_state.connection.start_block

    @line_start_block.setter
    def line_start_block(self, value):
        self.canvas_state.connection.start_block = value

    @property
    def line_start_port(self):
        return self.canvas_state.connection.start_port

    @line_start_port.setter
    def line_start_port(self, value):
        self.canvas_state.connection.start_port = value

    @property
    def temp_line(self):
        return self.canvas_state.connection.temp_line

    @temp_line.setter
    def temp_line(self, value):
        self.canvas_state.connection.temp_line = value

    @property
    def source_block_for_connection(self):
        return self.canvas_state.connection.source_block

    @source_block_for_connection.setter
    def source_block_for_connection(self, value):
        self.canvas_state.connection.source_block = value

    # Validation properties
    @property
    def validation_errors(self):
        return self.canvas_state.validation.errors

    @validation_errors.setter
    def validation_errors(self, value):
        self.canvas_state.validation.errors = value

    @property
    def blocks_with_errors(self):
        return self.canvas_state.validation.blocks_with_errors

    @blocks_with_errors.setter
    def blocks_with_errors(self, value):
        self.canvas_state.validation.blocks_with_errors = value

    @property
    def blocks_with_warnings(self):
        return self.canvas_state.validation.blocks_with_warnings

    @blocks_with_warnings.setter
    def blocks_with_warnings(self, value):
        self.canvas_state.validation.blocks_with_warnings = value

    @property
    def show_validation_errors(self):
        return self.canvas_state.validation.show_errors

    @show_validation_errors.setter
    def show_validation_errors(self, value):
        self.canvas_state.validation.show_errors = value
