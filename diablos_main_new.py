"""
DiaBloS - Modernized main application using the new architecture.

This is the new main entry point that uses the refactored modular architecture
with proper separation of concerns, type hints, and modern Python practices.
"""

import sys
import logging
from typing import Optional
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QPoint
from enum import Enum, auto

from lib.services.simulation_controller import SimulationController
from lib.core.models import Point, SimulationConfig
from lib.core.interfaces import SimulationState

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UIState(Enum):
    """UI interaction states."""
    IDLE = auto()
    DRAGGING_BLOCK = auto()
    CREATING_CONNECTION = auto()
    CONFIGURING_BLOCK = auto()


class ModernDiaBloSWindow(QMainWindow):
    """Modern DiaBloS main window using the new architecture."""
    
    def __init__(self):
        super().__init__()
        logger.info("Initializing Modern DiaBloS Window")
        
        # Initialize the simulation controller
        self.controller = SimulationController()
        
        # Create a new diagram
        self.controller.new_diagram("New Diagram")
        
        # UI state
        self.ui_state = UIState.IDLE
        self.selected_block_id: Optional[str] = None
        self.selected_connection_id: Optional[str] = None
        
        # Connection creation state
        self.connection_source_block: Optional[str] = None
        self.connection_source_port: Optional[int] = None
        self.temp_connection_end = QPoint(0, 0)
        
        # Dragging state
        self.drag_start_pos: Optional[QPoint] = None
        self.dragging_block_id: Optional[str] = None
        
        # Initialize UI
        self.init_ui()
        
        # Setup timer for UI updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(16)  # ~60 FPS
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("DiaBloS - Modern Architecture")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Set white background
        self.setStyleSheet("background-color: white;")
        
        logger.info("UI initialized successfully")
    
    def paintEvent(self, event) -> None:
        """Paint the diagram canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            # Draw blocks
            self.draw_blocks(painter)
            
            # Draw connections
            self.draw_connections(painter)
            
            # Draw temporary connection if creating one
            if self.ui_state == UIState.CREATING_CONNECTION:
                self.draw_temp_connection(painter)
                
        except Exception as e:
            logger.error(f"Error in paintEvent: {e}")
        
        painter.end()
    
    def draw_blocks(self, painter: QPainter) -> None:
        """Draw all blocks on the canvas."""
        diagram = self.controller.get_current_diagram()
        if diagram is None:
            return
        
        for block in diagram.get_all_blocks():
            # Set block appearance based on selection
            if block.id == self.selected_block_id:
                painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red border for selected
                painter.setBrush(QColor(255, 255, 200))     # Light yellow fill
            else:
                painter.setPen(QPen(QColor(0, 0, 0), 2))    # Black border
                painter.setBrush(QColor(200, 200, 255))     # Light blue fill
            
            # Draw block rectangle
            block_rect = self.get_block_rect(block)
            painter.drawRect(block_rect)
            
            # Draw block name
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(block_rect, Qt.AlignCenter, block.name)
            
            # Draw ports
            self.draw_block_ports(painter, block)
    
    def draw_connections(self, painter: QPainter) -> None:
        """Draw all connections between blocks."""
        diagram = self.controller.get_current_diagram()
        if diagram is None:
            return
        
        for connection in diagram.get_connections():
            source_block = diagram.get_block(connection.source_block_id)
            target_block = diagram.get_block(connection.target_block_id)
            
            if source_block is None or target_block is None:
                continue
            
            # Calculate connection endpoints
            source_point = self.get_port_position(source_block, connection.source_port, is_output=True)
            target_point = self.get_port_position(target_block, connection.target_port, is_output=False)
            
            # Set connection appearance
            if connection.id == self.selected_connection_id:
                painter.setPen(QPen(QColor(255, 0, 0), 3))  # Red for selected
            else:
                painter.setPen(QPen(QColor(0, 100, 0), 2))  # Green for normal
            
            # Draw the connection line
            painter.drawLine(source_point, target_point)
    
    def draw_temp_connection(self, painter: QPainter) -> None:
        """Draw temporary connection while creating one."""
        if (self.connection_source_block is None or 
            self.connection_source_port is None):
            return
        
        diagram = self.controller.get_current_diagram()
        if diagram is None:
            return
        
        source_block = diagram.get_block(self.connection_source_block)
        if source_block is None:
            return
        
        source_point = self.get_port_position(
            source_block, self.connection_source_port, is_output=True
        )
        
        painter.setPen(QPen(QColor(100, 100, 100), 2, Qt.DashLine))
        painter.drawLine(source_point, self.temp_connection_end)
    
    def draw_block_ports(self, painter: QPainter, block) -> None:
        """Draw input and output ports for a block."""
        block_rect = self.get_block_rect(block)
        port_size = 8
        
        # Draw input ports (left side)
        for port in range(block.input_ports):
            port_pos = self.get_port_position(block, port, is_output=False)
            port_rect = QPoint(port_pos.x() - port_size//2, port_pos.y() - port_size//2)
            
            painter.setPen(QPen(QColor(0, 0, 255), 2))  # Blue for input ports
            painter.setBrush(QColor(200, 200, 255))
            painter.drawEllipse(port_rect.x(), port_rect.y(), port_size, port_size)
        
        # Draw output ports (right side)
        for port in range(block.output_ports):
            port_pos = self.get_port_position(block, port, is_output=True)
            port_rect = QPoint(port_pos.x() - port_size//2, port_pos.y() - port_size//2)
            
            painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red for output ports
            painter.setBrush(QColor(255, 200, 200))
            painter.drawEllipse(port_rect.x(), port_rect.y(), port_size, port_size)
    
    def get_block_rect(self, block) -> QPoint:
        """Get the rectangle for drawing a block."""
        # Convert our Point to QRect
        x = int(block.position.x)
        y = int(block.position.y)
        width = 120
        height = 60
        return QPoint(x, y).x(), QPoint(x, y).y(), width, height
    
    def get_port_position(self, block, port_number: int, is_output: bool) -> QPoint:
        """Get the position of a specific port on a block."""
        block_rect = self.get_block_rect(block)
        x, y, width, height = block_rect
        
        if is_output:
            # Output ports on the right side
            port_x = x + width
            num_ports = block.output_ports
        else:
            # Input ports on the left side
            port_x = x
            num_ports = block.input_ports
        
        if num_ports == 0:
            port_y = y + height // 2
        else:
            port_spacing = height / (num_ports + 1)
            port_y = y + int(port_spacing * (port_number + 1))
        
        return QPoint(port_x, port_y)
    
    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        try:
            if event.button() == Qt.LeftButton:
                self.handle_left_press(event.pos())
            elif event.button() == Qt.RightButton:
                self.handle_right_press(event.pos())
        except Exception as e:
            logger.error(f"Error in mousePressEvent: {e}")
    
    def handle_left_press(self, pos: QPoint) -> None:
        """Handle left mouse button press."""
        # Check if clicking on a block
        clicked_block = self.get_block_at_position(pos)
        if clicked_block:
            # Check if clicking on a port
            port_info = self.get_port_at_position(clicked_block, pos)
            if port_info:
                self.handle_port_click(clicked_block.id, port_info['port'], port_info['is_output'])
            else:
                # Start dragging the block
                self.start_block_drag(clicked_block.id, pos)
            return
        
        # Check if clicking on a connection
        clicked_connection = self.get_connection_at_position(pos)
        if clicked_connection:
            self.selected_connection_id = clicked_connection.id
            self.selected_block_id = None
            return
        
        # Click on empty space - clear selection
        self.clear_selection()
    
    def handle_right_press(self, pos: QPoint) -> None:
        """Handle right mouse button press (context menu)."""
        clicked_block = self.get_block_at_position(pos)
        if clicked_block:
            self.show_block_context_menu(clicked_block.id, pos)
    
    def handle_port_click(self, block_id: str, port: int, is_output: bool) -> None:
        """Handle clicking on a block port."""
        if self.ui_state == UIState.CREATING_CONNECTION:
            # Complete the connection
            if not is_output:  # Can only connect to input ports
                if self.controller.can_connect_blocks(
                    self.connection_source_block, self.connection_source_port,
                    block_id, port
                ):
                    connection_id = self.controller.create_connection(
                        self.connection_source_block, self.connection_source_port,
                        block_id, port
                    )
                    if connection_id:
                        logger.info(f"Created connection: {connection_id}")
            
            # Reset connection creation state
            self.ui_state = UIState.IDLE
            self.connection_source_block = None
            self.connection_source_port = None
        
        elif is_output:
            # Start creating a connection from output port
            self.ui_state = UIState.CREATING_CONNECTION
            self.connection_source_block = block_id
            self.connection_source_port = port
    
    def start_block_drag(self, block_id: str, pos: QPoint) -> None:
        """Start dragging a block."""
        self.ui_state = UIState.DRAGGING_BLOCK
        self.selected_block_id = block_id
        self.dragging_block_id = block_id
        self.drag_start_pos = pos
    
    def mouseMoveEvent(self, event) -> None:
        """Handle mouse move events."""
        try:
            if self.ui_state == UIState.DRAGGING_BLOCK:
                self.handle_block_drag(event.pos())
            elif self.ui_state == UIState.CREATING_CONNECTION:
                self.temp_connection_end = event.pos()
                self.update()
        except Exception as e:
            logger.error(f"Error in mouseMoveEvent: {e}")
    
    def handle_block_drag(self, pos: QPoint) -> None:
        """Handle dragging a block."""
        if (self.dragging_block_id is None or 
            self.drag_start_pos is None):
            return
        
        # Calculate new position
        delta = pos - self.drag_start_pos
        diagram = self.controller.get_current_diagram()
        if diagram is None:
            return
        
        block = diagram.get_block(self.dragging_block_id)
        if block is None:
            return
        
        new_position = Point(
            block.position.x + delta.x(),
            block.position.y + delta.y()
        )
        
        # Update block position
        self.controller.update_block_position(self.dragging_block_id, new_position)
        self.drag_start_pos = pos
        self.update()
    
    def mouseReleaseEvent(self, event) -> None:
        """Handle mouse release events."""
        try:
            if self.ui_state == UIState.DRAGGING_BLOCK:
                self.ui_state = UIState.IDLE
                self.dragging_block_id = None
                self.drag_start_pos = None
        except Exception as e:
            logger.error(f"Error in mouseReleaseEvent: {e}")
    
    def keyPressEvent(self, event) -> None:
        """Handle key press events."""
        try:
            if event.key() == Qt.Key_Delete:
                self.delete_selected_items()
            elif event.key() == Qt.Key_Escape:
                self.clear_selection()
                if self.ui_state == UIState.CREATING_CONNECTION:
                    self.ui_state = UIState.IDLE
                    self.connection_source_block = None
                    self.connection_source_port = None
        except Exception as e:
            logger.error(f"Error in keyPressEvent: {e}")
    
    def delete_selected_items(self) -> None:
        """Delete selected blocks or connections."""
        if self.selected_block_id:
            self.controller.remove_block(self.selected_block_id)
            self.selected_block_id = None
        elif self.selected_connection_id:
            self.controller.remove_connection(self.selected_connection_id)
            self.selected_connection_id = None
        
        self.update()
    
    def clear_selection(self) -> None:
        """Clear all selections."""
        self.selected_block_id = None
        self.selected_connection_id = None
        self.update()
    
    def get_block_at_position(self, pos: QPoint):
        """Get the block at the given position."""
        diagram = self.controller.get_current_diagram()
        if diagram is None:
            return None
        
        for block in diagram.get_all_blocks():
            block_rect = self.get_block_rect(block)
            x, y, width, height = block_rect
            if (x <= pos.x() <= x + width and 
                y <= pos.y() <= y + height):
                return block
        
        return None
    
    def get_port_at_position(self, block, pos: QPoint):
        """Get port information at the given position."""
        port_size = 8
        
        # Check input ports
        for port in range(block.input_ports):
            port_pos = self.get_port_position(block, port, is_output=False)
            if (abs(pos.x() - port_pos.x()) <= port_size and 
                abs(pos.y() - port_pos.y()) <= port_size):
                return {'port': port, 'is_output': False}
        
        # Check output ports
        for port in range(block.output_ports):
            port_pos = self.get_port_position(block, port, is_output=True)
            if (abs(pos.x() - port_pos.x()) <= port_size and 
                abs(pos.y() - port_pos.y()) <= port_size):
                return {'port': port, 'is_output': True}
        
        return None
    
    def get_connection_at_position(self, pos: QPoint):
        """Get the connection at the given position."""
        # Implementation would check if position is near a connection line
        # For now, return None (simplified)
        return None
    
    def show_block_context_menu(self, block_id: str, pos: QPoint) -> None:
        """Show context menu for a block."""
        # This would show a context menu for block configuration
        # For now, just log the action
        logger.info(f"Context menu for block {block_id} at {pos}")
    
    def update_display(self) -> None:
        """Update the display (called by timer)."""
        # Check if simulation is running and update accordingly
        sim_state = self.controller.get_simulation_state()
        if sim_state == SimulationState.RUNNING:
            # Perform simulation step
            result = self.controller.step_simulation()
            if not result.success:
                logger.error(f"Simulation step failed: {result.errors}")
        
        # Update the display
        self.update()
    
    def create_test_blocks(self) -> None:
        """Create some test blocks for demonstration."""
        # Create a simple test diagram
        step_id = self.controller.create_block("Step", "Step Block", Point(50, 100))
        gain_id = self.controller.create_block("Gain", "Gain Block", Point(250, 100))
        scope_id = self.controller.create_block("Scope", "Scope Block", Point(450, 100))
        
        if step_id and gain_id and scope_id:
            # Connect the blocks
            self.controller.create_connection(step_id, 0, gain_id, 0)
            self.controller.create_connection(gain_id, 0, scope_id, 0)
            
            logger.info("Created test diagram with Step -> Gain -> Scope")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = ModernDiaBloSWindow()
    
    # Create test blocks for demonstration
    window.create_test_blocks()
    
    window.show()
    
    logger.info("Modern DiaBloS application started")
    
    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()