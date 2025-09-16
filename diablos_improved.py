"""
DiaBloS with incremental improvements - A practical approach.

This version shows how to gradually improve the existing codebase without
breaking changes, using helper functions and better practices.
"""

import sys
import logging
import time
from enum import Enum, auto
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPen, QColor

# Import existing code
from lib.lib import DSim
from lib.improvements import (
    ValidationHelper, PerformanceHelper, SafetyChecks, LoggingHelper,
    SimulationConfig, validate_simulation_parameters, safe_execute_block_function
)

# Setup enhanced logging
LoggingHelper.setup_logging(level="INFO", log_file="diablos.log")
logger = logging.getLogger(__name__)


class State(Enum):
    """Application state management."""
    IDLE = auto()
    DRAGGING = auto()
    CONNECTING = auto()
    CONFIGURING = auto()


class ImprovedDiaBloSWindow(QMainWindow):
    """
    Improved DiaBloS window that uses existing DSim but adds safety checks,
    better logging, and validation helpers.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("Starting Improved DiaBloS Application")
        
        # Create the original DSim instance
        self.dsim = DSim()
        
        # Add performance monitoring
        self.perf_helper = PerformanceHelper()
        
        # Current simulation config
        self.sim_config = SimulationConfig()
        
        # Dragging state management
        self.state = State.IDLE
        self.dragging_block = None
        self.drag_offset = None
        
        # Connection state management
        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None
        
        # Setup UI (using existing DSim display methods)
        self.init_ui()
        
        # Setup timer for updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.safe_update)
        self.update_timer.start(int(1000 / self.dsim.FPS))
        
        logger.info("Improved DiaBloS Window initialized successfully")
    
    def init_ui(self):
        """Initialize UI using existing DSim methods."""
        self.setWindowTitle("DiaBloS - Improved Version")
        self.setGeometry(100, 100, self.dsim.SCREEN_WIDTH, self.dsim.SCREEN_HEIGHT)
        
        # Create central widget and use DSim's display methods
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Initialize DSim's components
        self.dsim.main_buttons_init()
        self.dsim.menu_blocks_init()
    
    def safe_update(self):
        """Safe update with error handling and performance monitoring."""
        try:
            self.perf_helper.start_timer("ui_update")
            
            # Use existing DSim update methods but with safety checks
            if hasattr(self.dsim, 'execution_initialized') and self.dsim.execution_initialized:
                # Check if we can safely continue simulation
                is_safe, errors = SafetyChecks.check_simulation_state(self.dsim)
                if not is_safe:
                    logger.error(f"Simulation state unsafe: {errors}")
                    self.stop_simulation_safely()
                    return
                
                # Perform simulation step with performance monitoring
                self.perf_helper.start_timer("simulation_step")
                self.dsim.execution_loop()
                step_duration = self.perf_helper.end_timer("simulation_step")
                
                if step_duration and step_duration > 0.1:  # Log slow steps
                    logger.warning(f"Slow simulation step: {step_duration:.4f}s")
            
            # Update display
            self.update()
            
            self.perf_helper.end_timer("ui_update")
            
        except Exception as e:
            logger.error(f"Error in safe_update: {str(e)}")
            self.stop_simulation_safely()
    
    def start_simulation_with_validation(self):
        """Start simulation with comprehensive validation."""
        try:
            logger.info("Starting simulation with validation...")
            
            # Validate simulation parameters
            if hasattr(self.dsim, 'sim_time') and hasattr(self.dsim, 'sim_dt'):
                is_valid, errors = validate_simulation_parameters(
                    self.dsim.sim_time, self.dsim.sim_dt
                )
                if not is_valid:
                    self.show_error_dialog("Simulation Parameters Invalid", errors)
                    return False
            
            # Validate block connections
            if hasattr(self.dsim, 'blocks_list') and hasattr(self.dsim, 'line_list'):
                is_valid, errors = ValidationHelper.validate_block_connections(
                    self.dsim.blocks_list, self.dsim.line_list
                )
                if not is_valid:
                    self.show_error_dialog("Block Connections Invalid", errors)
                    return False
                
                # Check for algebraic loops
                no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(
                    self.dsim.blocks_list, self.dsim.line_list
                )
                if not no_loops:
                    self.show_error_dialog("Algebraic Loop Detected", loop_errors)
                    return False
            
            # Check overall simulation state
            is_safe, safety_errors = SafetyChecks.check_simulation_state(self.dsim)
            if not is_safe:
                self.show_error_dialog("Simulation State Unsafe", safety_errors)
                return False
            
            # Log simulation start
            LoggingHelper.log_simulation_start(self.dsim)
            
            # Start performance monitoring
            self.perf_helper.start_timer("total_simulation")
            
            # Use existing DSim initialization
            self.dsim.execution_init()
            
            logger.info("Simulation started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}")
            self.show_error_dialog("Simulation Start Failed", [str(e)])
            return False
    
    def stop_simulation_safely(self):
        """Stop simulation safely with cleanup."""
        try:
            logger.info("Stopping simulation safely...")
            
            # End performance monitoring
            total_duration = self.perf_helper.end_timer("total_simulation")
            
            # Use existing DSim cleanup if available
            if hasattr(self.dsim, 'execution_initialized'):
                self.dsim.execution_initialized = False
            
            # Log simulation end
            if total_duration:
                LoggingHelper.log_simulation_end(self.dsim, total_duration)
            
            # Log performance statistics
            self.perf_helper.log_stats()
            
            logger.info("Simulation stopped safely")
            
        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")
    
    def show_error_dialog(self, title: str, errors: list):
        """Show error dialog to user."""
        error_text = "\\n".join(errors)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle(title)
        msg.setText(error_text)
        msg.exec_()
        logger.error(f"{title}: {error_text}")
    
    def paintEvent(self, event):
        """Paint event with error handling."""
        try:
            self.perf_helper.start_timer("paint_event")
            
            # Create painter for PyQt5
            from PyQt5.QtGui import QPainter
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Use existing DSim display methods with painter
            if hasattr(self.dsim, 'display_blocks'):
                self.dsim.display_blocks(painter)
            
            if hasattr(self.dsim, 'display_lines'):
                self.dsim.display_lines(painter)
            
            if hasattr(self.dsim, 'display_buttons'):
                self.dsim.display_buttons(painter)
            
            if hasattr(self.dsim, 'display_menu_blocks'):
                self.dsim.display_menu_blocks(painter)
            
            # Draw temporary line during connection creation
            if self.line_creation_state == 'start' and self.temp_line:
                painter.setPen(QPen(QColor('black'), 2, Qt.DashLine))
                painter.drawLine(self.temp_line[0], self.temp_line[1])
            
            painter.end()
            
            paint_duration = self.perf_helper.end_timer("paint_event")
            
            # Log slow paint events
            if paint_duration and paint_duration > 0.05:
                logger.warning(f"Slow paint event: {paint_duration:.4f}s")
                
        except Exception as e:
            logger.error(f"Error in paintEvent: {str(e)}")
    
    def mousePressEvent(self, event):
        """Mouse press with safety checks and proper DSim integration."""
        try:
            print(f"DEBUG: Mouse press at ({event.x()}, {event.y()}), button: {event.button()}")
            logger.info(f"Mouse press at ({event.x()}, {event.y()}), button: {event.button()}")
            
            # Handle mouse events by delegating to DSim-compatible methods
            if event.button() == Qt.LeftButton:
                self.handle_left_click_improved(event)
            elif event.button() == Qt.RightButton:
                self.handle_right_click_improved(event)
            
            # Update display
            self.update()
            
        except Exception as e:
            logger.error(f"Error in mousePressEvent: {str(e)}")

    def handle_left_click_improved(self, event):
        """Handle left mouse click with DSim integration."""
        try:
            pos = event.pos()
            
            # Check if clicking on buttons
            if hasattr(self.dsim, 'buttons_list'):
                for button in self.dsim.buttons_list:
                    if hasattr(button, 'collision') and isinstance(button.collision, QRect):
                        if button.collision.contains(pos):
                            self.handle_button_click_improved(button)
                            return
            
            # Check if clicking on menu blocks (for adding blocks)
            if hasattr(self.dsim, 'menu_blocks'):
                logger.info(f"DSim has menu_blocks, count: {len(self.dsim.menu_blocks)}")
                if len(self.dsim.menu_blocks) == 0:
                    logger.warning("menu_blocks list is empty!")
                logger.debug(f"Checking {len(self.dsim.menu_blocks)} menu blocks")
                for menu_block in self.dsim.menu_blocks:
                    logger.debug(f"Menu block: {getattr(menu_block, 'fn_name', 'Unknown')}, has collision: {hasattr(menu_block, 'collision')}")
                    if hasattr(menu_block, 'collision') and menu_block.collision:
                        logger.debug(f"Menu block collision rect: {menu_block.collision}")
                    if hasattr(menu_block, 'collision') and isinstance(menu_block.collision, QRect):
                        logger.debug(f"Menu block collision rect: {menu_block.collision}, contains pos: {menu_block.collision.contains(pos)}")
                        if menu_block.collision.contains(pos):
                            self.handle_menu_block_click_improved(menu_block, pos)
                            return
            
            # Check if clicking on existing blocks
            if hasattr(self.dsim, 'blocks_list'):
                logger.debug(f"Checking {len(self.dsim.blocks_list)} blocks for collision")
                for block in self.dsim.blocks_list:
                    logger.debug(f"Block {getattr(block, 'name', 'Unknown')}: has rectf={hasattr(block, 'rectf')}")
                    if hasattr(block, 'rectf') and isinstance(block.rectf, QRect):
                        logger.debug(f"Block {block.name} rectf: {block.rectf}, contains pos: {block.rectf.contains(pos)}")
                        if block.rectf.contains(pos):
                            logger.info(f"Block clicked: {block.name}")
                            self.handle_block_click_improved(block, event)
                            return
            
            # Check if clicking on lines
            if hasattr(self.dsim, 'line_list'):
                for line in self.dsim.line_list:
                    # Simple line collision detection (would need improvement for complex lines)
                    if hasattr(line, 'x1') and hasattr(line, 'y1') and hasattr(line, 'x2') and hasattr(line, 'y2'):
                        # Check if click is near the line (simplified)
                        if self.point_near_line(pos, line):
                            self.handle_line_click_improved(line, event)
                            return
            
            logger.info("Click on empty area - no button, menu block, or existing block clicked")
            # Cancel any ongoing line creation
            if self.line_creation_state == 'start':
                logger.info("Canceling line creation due to empty area click")
                self.cancel_line_creation()
            # Reset drag state if needed
            if self.state != State.IDLE:
                self.reset_drag_state()
            
        except Exception as e:
            logger.error(f"Error in handle_left_click_improved: {str(e)}")
    
    def handle_right_click_improved(self, event):
        """Handle right mouse click."""
        try:
            pos = event.pos()
            logger.debug(f"Right click at ({pos.x()}, {pos.y()})")
            
            # Check if right-clicking on a block to configure it
            if hasattr(self.dsim, 'blocks_list'):
                for block in self.dsim.blocks_list:
                    if hasattr(block, 'rect') and isinstance(block.rect, QRect):
                        if block.rect.contains(pos):
                            self.configure_block_improved(block)
                            return
            
        except Exception as e:
            logger.error(f"Error in handle_right_click_improved: {str(e)}")
    
    def handle_button_click_improved(self, button):
        """Handle button click."""
        try:
            button_text = getattr(button, 'text', '')
            logger.info(f"Button clicked: {button_text}")
            
            # Handle different button types based on actual button names
            if button_text == 'new':
                if hasattr(self.dsim, 'clear_all'):
                    self.dsim.clear_all()
                    logger.info("New diagram - cleared all blocks and lines")
                    
            elif button_text == 'save':
                if hasattr(self.dsim, 'save'):
                    self.dsim.save()
                    logger.info("Save requested")
                    
            elif button_text == 'load':
                if hasattr(self.dsim, 'open'):
                    self.dsim.open()
                    logger.info("Load requested")
                    
            elif button_text == 'play':
                logger.info("Starting simulation...")
                self.start_simulation_with_validation()
                
            elif button_text == 'pause':
                logger.info("Pause simulation")
                if hasattr(self.dsim, 'execution_pause'):
                    self.dsim.execution_pause = True
                    
            elif button_text == 'stop':
                logger.info("Stop simulation")
                self.stop_simulation_safely()
                
            elif button_text == 'plot':
                logger.info("Plot results")
                if hasattr(self.dsim, 'plot_again'):
                    self.dsim.plot_again()
                    
            elif button_text == 'capture':
                logger.info("Screenshot requested")
                if hasattr(self.dsim, 'screenshot'):
                    self.dsim.screenshot()
                    
            else:
                logger.warning(f"Unknown button: {button_text}")
            
        except Exception as e:
            logger.error(f"Error in handle_button_click_improved: {str(e)}")
    
    def handle_menu_block_click_improved(self, menu_block, pos):
        """Handle menu block click to add new block."""
        try:
            block_name = getattr(menu_block, 'fn_name', 'Unknown')
            logger.info(f"Menu block clicked: {block_name}")
            
            # Add new block to canvas - use the original approach
            if hasattr(self.dsim, 'add_block'):
                # Simply pass the original pos (QPoint) directly like the original code
                new_block = self.dsim.add_block(menu_block, pos)
                logger.info(f"Added {block_name} at ({pos.x()}, {pos.y()})")
                
                # Start dragging the newly created block immediately (like original)
                if new_block:
                    self.start_drag(new_block, pos)
                
                # Trigger a repaint to show the new block
                self.update()
            else:
                logger.error("DSim object does not have add_block method")
            
        except Exception as e:
            logger.error(f"Error in handle_menu_block_click_improved: {str(e)}")
    
    def handle_block_click_improved(self, block, event):
        """Handle click on existing block."""
        try:
            block_name = getattr(block, 'name', 'Unknown')
            logger.debug(f"Block clicked: {block_name}")
            
            # Check if Ctrl is held for selection
            if event.modifiers() & Qt.ControlModifier:
                logger.info(f"Ctrl+Click on {block_name} - toggle selection")
                # Toggle selection logic would go here
                if hasattr(block, 'selected'):
                    block.selected = not block.selected
                    logger.info(f"Block {block_name} selection: {block.selected}")
            else:
                # Check if clicking on a port for connection
                port = block.port_collision(event.pos())
                if port[0] in ["i", "o"]:
                    logger.info(f"Port clicked on {block_name}: {port}")
                    self.handle_port_click(block, port)
                else:
                    # Start dragging the block
                    logger.debug(f"Starting drag on existing block: {block_name}")
                    self.start_drag(block, event.pos())
            
        except Exception as e:
            logger.error(f"Error in handle_block_click_improved: {str(e)}")
    
    def handle_line_click_improved(self, line, event):
        """Handle click on line."""
        try:
            logger.debug("Line clicked")
            
            # Check if Ctrl is held for selection
            if event.modifiers() & Qt.ControlModifier:
                logger.info("Ctrl+Click on line - toggle selection")
                # Toggle line selection logic would go here
            
        except Exception as e:
            logger.error(f"Error in handle_line_click_improved: {str(e)}")
    
    def configure_block_improved(self, block):
        """Configure block properties."""
        try:
            block_name = getattr(block, 'name', 'Unknown')
            logger.info(f"Configure block: {block_name}")
            
            # Show configuration dialog
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Block Configuration")
            msg.setText(f"Configuration for {block_name}\nThis would open the block configuration dialog.")
            msg.exec_()
            
        except Exception as e:
            logger.error(f"Error in configure_block_improved: {str(e)}")
    
    def point_near_line(self, point, line):
        """Check if point is near a line (simplified collision detection)."""
        try:
            # Very simple line collision - just check if point is within bounding box
            if hasattr(line, 'x1') and hasattr(line, 'y1') and hasattr(line, 'x2') and hasattr(line, 'y2'):
                min_x = min(line.x1, line.x2) - 5
                max_x = max(line.x1, line.x2) + 5
                min_y = min(line.y1, line.y2) - 5
                max_y = max(line.y1, line.y2) + 5
                
                return (min_x <= point.x() <= max_x and min_y <= point.y() <= max_y)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in point_near_line: {str(e)}")
            return False
    
    def keyPressEvent(self, event):
        """Key press with enhanced shortcuts."""
        try:
            # Existing shortcuts
            if event.key() == Qt.Key_N and event.modifiers() == Qt.ControlModifier:
                logger.info("New diagram requested")
                self.dsim.clear_all()
                
            elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
                logger.info("Save requested")
                # Use existing DSim save method
                if hasattr(self.dsim, 'save'):
                    self.dsim.save()
                    
            elif event.key() == Qt.Key_O and event.modifiers() == Qt.ControlModifier:
                logger.info("Open requested")
                # Use existing DSim open method
                if hasattr(self.dsim, 'open'):
                    self.dsim.open()
                    
            elif event.key() == Qt.Key_E and event.modifiers() == Qt.ControlModifier:
                logger.info("Simulation start requested")
                self.start_simulation_with_validation()
                
            elif event.key() == Qt.Key_Escape:
                logger.info("Stop simulation requested")
                self.stop_simulation_safely()
                
            # New enhanced shortcuts
            elif event.key() == Qt.Key_V and event.modifiers() == Qt.ControlModifier:
                logger.info("Validation requested")
                self.run_comprehensive_validation()
                
            elif event.key() == Qt.Key_P and event.modifiers() == Qt.ControlModifier:
                logger.info("Performance stats requested")
                self.perf_helper.log_stats()
                
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {str(e)}")
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation and show results."""
        try:
            logger.info("Running comprehensive validation...")
            
            all_errors = []
            all_warnings = []
            
            # Validate blocks and connections
            if hasattr(self.dsim, 'blocks_list') and hasattr(self.dsim, 'line_list'):
                is_valid, messages = ValidationHelper.validate_block_connections(
                    self.dsim.blocks_list, self.dsim.line_list
                )
                if not is_valid:
                    all_errors.extend(messages)
                
                # Check algebraic loops
                no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(
                    self.dsim.blocks_list, self.dsim.line_list
                )
                if not no_loops:
                    all_errors.extend(loop_errors)
            
            # Check simulation state
            is_safe, safety_errors = SafetyChecks.check_simulation_state(self.dsim)
            if not is_safe:
                all_errors.extend(safety_errors)
            
            # Show results
            if all_errors:
                self.show_error_dialog("Validation Errors Found", all_errors)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Validation Complete")
                msg.setText("All validation checks passed!")
                msg.exec_()
                logger.info("All validation checks passed")
                
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
    
    def handle_port_click(self, block, port):
        """Handle port click for line creation."""
        try:
            logger.debug(f"Port clicked on block {block.name}, port: {port}")
            if self.line_creation_state is None:
                if port[0] == 'o':  # Start line from output port
                    self.line_creation_state = 'start'
                    self.line_start_block = block
                    self.line_start_port = port[1]
                    self.temp_line = (block.out_coords[port[1]], QPoint(block.out_coords[port[1]]))
                    logger.info(f"Started line creation from {block.name} output port {port[1]}")
            elif self.line_creation_state == 'start':
                if port[0] == 'i':  # End line at input port
                    logger.info(f"Completing line to {block.name} input port {port[1]}")
                    self.finish_line_creation(block, port[1])
                else:
                    logger.info("Canceling line creation - clicked on output port")
                    self.cancel_line_creation()
            self.update()
        except Exception as e:
            logger.error(f"Error in handle_port_click: {str(e)}")
    
    def finish_line_creation(self, end_block, end_port):
        """Complete line creation between two blocks."""
        try:
            logger.debug(f"Finishing line creation from {self.line_start_block.name} to {end_block.name}")
            if hasattr(self.dsim, 'add_line'):
                new_line = self.dsim.add_line(
                    (self.line_start_block.name, self.line_start_port, self.line_start_block.out_coords[self.line_start_port]),
                    (end_block.name, end_port, end_block.in_coords[end_port])
                )
                if new_line:
                    logger.info(f"Line created: {self.line_start_block.name} -> {end_block.name}")
                else:
                    logger.warning("Failed to create line")
            self.cancel_line_creation()
        except Exception as e:
            logger.error(f"Error in finish_line_creation: {str(e)}")
            self.cancel_line_creation()
    
    def cancel_line_creation(self):
        """Cancel current line creation."""
        try:
            logger.debug("Canceling line creation")
            self.line_creation_state = None
            self.line_start_block = None
            self.line_start_port = None
            self.temp_line = None
            self.update()
        except Exception as e:
            logger.error(f"Error in cancel_line_creation: {str(e)}")

    def start_drag(self, block, pos):
        """Start dragging a block."""
        try:
            logger.debug(f"Starting drag on block: {block.name}")
            self.state = State.DRAGGING
            self.dragging_block = block
            self.drag_offset = pos - QPoint(block.left, block.top)
            self.setCursor(Qt.ClosedHandCursor)
            logger.debug(f"Drag started. State: {self.state}, Dragging block: {self.dragging_block.name}, Offset: {self.drag_offset}")
        except Exception as e:
            logger.error(f"Error in start_drag: {str(e)}")
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for dragging and line creation."""
        try:
            if self.state == State.DRAGGING and self.dragging_block:
                new_pos = event.pos() - self.drag_offset
                self.dragging_block.relocate_Block(new_pos)
                if hasattr(self.dsim, 'update_lines'):
                    self.dsim.update_lines()
                logger.debug(f"Block {self.dragging_block.name} moved to {new_pos}")
                self.update()
            elif self.line_creation_state == 'start':
                # Update temporary line end position
                self.temp_line = (self.temp_line[0], event.pos())
                self.update()
        except Exception as e:
            logger.error(f"Error in mouseMoveEvent: {str(e)}")
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events to stop dragging."""
        try:
            logger.debug(f"Mouse release event, Current state: {self.state}")
            if self.state == State.DRAGGING:
                self.reset_drag_state()
            logger.debug(f"After mouse release, Current state: {self.state}")
        except Exception as e:
            logger.error(f"Error in mouseReleaseEvent: {str(e)}")
    
    def reset_drag_state(self):
        """Reset dragging state."""
        try:
            logger.debug(f"Resetting drag state from {self.state}")
            self.state = State.IDLE
            self.dragging_block = None
            self.drag_offset = None
            self.setCursor(Qt.ArrowCursor)
            logger.debug(f"Drag state reset to: {self.state}")
        except Exception as e:
            logger.error(f"Error in reset_drag_state: {str(e)}")

    def closeEvent(self, event):
        """Clean shutdown."""
        try:
            logger.info("Application closing...")
            
            # Stop simulation if running
            self.stop_simulation_safely()
            
            # Log final performance stats
            self.perf_helper.log_stats()
            
            # Accept the close event
            event.accept()
            
            logger.info("Application closed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            event.accept()


def main():
    """Main application entry point with error handling."""
    try:
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = ImprovedDiaBloSWindow()
        window.show()
        
        logger.info("Application started successfully")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()