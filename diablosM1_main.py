"""
diablos_main_m1.py - Module to run the simulator interface
"""

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QFileDialog, QDialog
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QCursor, QMouseEvent
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
import pyqtgraph as pg
from lib.lib import *
from lib.dialogs import ParamDialog
import logging
from enum import Enum, auto



logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)
#logging.getLogger('functions').setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.CRITICAL)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class State(Enum):
    IDLE = auto()
    DRAGGING = auto()
    CONNECTING = auto()
    CONFIGURING = auto()

class DiaBloSWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.debug("Initializing DiaBloSWindow")
        self.sim_init = DSim()
        self.initUI()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(int(1000 / self.sim_init.FPS))

        self.state = State.IDLE
        self.active_block = None
        self.src_port = None
        self.drag_start_pos = None
        self.temp_line_end = None
        self.configuring = False
        self.dragging_block = None
        self.drag_offset = None

        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None

        

        self.setMouseTracking(True)
        self.installEventFilter(self)

        self.log_current_state("Initial state after launching the app")
        logger.debug("DiaBloSWindow initialized")

    def update_simulation(self):
        if self.sim_init.execution_initialized and not self.sim_init.execution_pause:
            self.sim_init.execution_loop()
        self.update()
        
    def initUI(self):
        self.setWindowTitle("DiaBloS - Diagram Block Simulator")
        self.setGeometry(100, 100, self.sim_init.SCREEN_WIDTH, self.sim_init.SCREEN_HEIGHT)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.sim_init.menu_blocks_init()
        self.sim_init.main_buttons_init()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            self.sim_init.display_menu_blocks(painter)
            self.sim_init.display_buttons(painter)
            self.sim_init.display_blocks(painter)
            self.sim_init.display_lines(painter)

            if self.line_creation_state == 'start' and self.temp_line:
                painter.setPen(QPen(QColor('black'), 2, Qt.DashLine))
                painter.drawLine(self.temp_line[0], self.temp_line[1])
        finally:
            painter.end()

    def reset_state(self):
        logger.debug(f"Resetting state from {self.state}")
        self.state = State.IDLE
        self.dragging_block = None
        self.drag_offset = None
        self.setCursor(Qt.ArrowCursor)
        logger.debug(f"State reset to: {self.state}")

    def reset_connection(self):
        self.state = State.IDLE
        self.src_port = None
        self.temp_line_end = None
        self.update()
        logger.debug("Connection reset")
        
    def mousePressEvent(self, event):
        logger.debug(f"Mouse press event, button: {event.button()}, Current state: {self.state}")
        
        if event.button() == Qt.LeftButton:
            if self.handle_button_click(event):
                return
            if self.handle_menu_block_click(event):
                return
            clicked_block = self.get_clicked_block(event.pos())
            if clicked_block:
                if event.modifiers() & Qt.ControlModifier:
                    self.toggle_selection(clicked_block)
                else:
                    port = clicked_block.port_collision(event.pos())
                    if port[0] in ["i", "o"]:
                        self.handle_port_click(clicked_block, port)
                    else:
                        self.start_drag(clicked_block, event.pos())
            else:
                clicked_line = self.get_clicked_line(event.pos())
                if clicked_line:
                    if event.modifiers() & Qt.ControlModifier:
                        self.toggle_selection(clicked_line)
                else:
                    self.cancel_line_creation()
                    self.reset_state()
        elif event.button() == Qt.RightButton:
            clicked_block = self.get_clicked_block(event.pos())
            if clicked_block:
                self.configure_block(clicked_block)
            else:
                self.cancel_line_creation()
                self.reset_state()
        
        self.update()
        logger.debug(f"After mouse press event, Current state: {self.state}")

    def log_current_state(self, message):
        logger.debug(f"{message}")
        logger.debug(f"Current state: {self.state}")
        logger.debug(f"Dragging block: {self.dragging_block.name if self.dragging_block else None}")
        logger.debug(f"Drag offset: {self.drag_offset}")

    def handle_left_click(self, event):
        if self.handle_button_click(event):
            return
        if self.handle_menu_block_click(event):
            return
    
    def handle_config_result(self, result, dialog, block):
        if result == QDialog.Accepted:
            new_params = dialog.get_values()
            block.username = new_params.pop('Name', block.username)
            
            # Update only the original parameters
            for key, value in new_params.items():
                if key in block.original_params:
                    block.params[key] = value
                    block.original_params[key] = value
        
        logger.debug(f"Finished configuring block: {block.name}")
        self.update()



    def handle_right_click(self, pos):
        clicked_block = self.get_clicked_block(pos)
        if clicked_block:
            self.configure_block(clicked_block)
        logger.debug(f"Right-click handled on block: {clicked_block.name if clicked_block else 'None'}")

        
    def handle_left_press(self, event):
        logger.debug(f"Handle left press called. Current state: {self.state}")
        if self.handle_button_click(event):
            logger.debug("Button click handled")
            return
        if self.handle_menu_block_click(event):
            logger.debug("Menu block click handled")
            return

        clicked_block = self.get_clicked_block(event.pos())
        if clicked_block:
            logger.debug(f"Clicked on block: {clicked_block.name}")
            self.start_drag(clicked_block, event.pos())
        else:
            logger.debug("Clicked on empty space")
            self.dragging_block = None
            self.drag_offset = None
            self.state = State.IDLE
        
        logger.debug(f"Left press handled, final state: {self.state}")

    def handle_right_press(self, event):
        clicked_block = self.get_clicked_block(event.pos())
        if clicked_block:
            self.configure_block(clicked_block)
        else:
            self.set_state(State.IDLE)
            self.src_port = None
            self.temp_line_end = None
        self.update()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.state == State.DRAGGING and self.dragging_block:
            new_pos = event.pos() - self.drag_offset
            self.dragging_block.relocate_Block(new_pos)
            self.sim_init.update_lines()
            logger.debug(f"Block {self.dragging_block.name} moved to {new_pos}")
        elif self.line_creation_state == 'start':
            self.temp_line = (self.temp_line[0], event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        logger.debug(f"Mouse release event, Current state: {self.state}")
        if self.state == State.DRAGGING:
            self.reset_state()
        logger.debug(f"After mouse release, Current state: {self.state}")


    def start_drag(self, block, pos):
        logger.debug(f"Starting drag on block: {block.name}")
        self.state = State.DRAGGING
        self.dragging_block = block
        self.drag_offset = pos - QPoint(block.left, block.top)
        self.setCursor(Qt.ClosedHandCursor)
        logger.debug(f"Drag started. State: {self.state}, Dragging block: {self.dragging_block.name}, Offset: {self.drag_offset}")
   
   
   
  

    def handle_port_click(self, block, port):
        logger.debug(f"Port clicked on block {block.name}, port: {port}")
        if self.line_creation_state is None:
            if port[0] == 'o':  # Start line from output port
                self.line_creation_state = 'start'
                self.line_start_block = block
                self.line_start_port = port[1]
                self.temp_line = (block.out_coords[port[1]], QPoint(block.out_coords[port[1]]))
        elif self.line_creation_state == 'start':
            if port[0] == 'i':  # End line at input port
                self.finish_line_creation(block, port[1])
            else:
                self.cancel_line_creation()
        self.update()

    def finish_line_creation(self, end_block, end_port):
        logger.debug(f"Finishing line creation from {self.line_start_block.name} to {end_block.name}")
        new_line = self.sim_init.add_line(
            (self.line_start_block.name, self.line_start_port, self.line_start_block.out_coords[self.line_start_port]),
            (end_block.name, end_port, end_block.in_coords[end_port])
        )
        if new_line:
            logger.debug(f"Line created: {self.line_start_block.name} to {end_block.name}")
        else:
            logger.debug("Failed to create line")
        self.cancel_line_creation()

    def cancel_line_creation(self):
        logger.debug("Canceling line creation")
        self.line_creation_state = None
        self.line_start_block = None
        self.line_start_port = None
        self.temp_line = None
        self.update()

    def finish_connection(self, target):
        if isinstance(target, tuple) and isinstance(target[0], DBlock):
            clicked_block, port = target
            if clicked_block and port and port[0] == "i" and self.src_port:
                src_block, src_port = self.src_port
                new_line = self.sim_init.add_line(
                    (src_block.name, src_port[1], QPoint(src_block.out_coords[src_port[1]].x(), src_block.out_coords[src_port[1]].y())),
                    (clicked_block.name, port[1], QPoint(clicked_block.in_coords[port[1]].x(), clicked_block.in_coords[port[1]].y()))
                )
                if new_line:
                    logger.debug(f"Line created: {src_block.name} to {clicked_block.name}")
                else:
                    logger.debug("Failed to create line")
        self.reset_connection()

    def get_clicked_block(self, pos):
        if isinstance(pos, tuple):
            pos = QPoint(*pos)
        elif isinstance(pos, DBlock):
            return pos
        for block in reversed(self.sim_init.blocks_list):
            if block.rectf.contains(pos):
                return block
        return None
    
    def get_clicked_line(self, pos):
        for line in self.sim_init.line_list:
            if line.collision(pos):
                return line
        return None

    def handle_button_click(self, event):
        for button in self.sim_init.buttons_list:
            if button.collision.contains(event.pos()) and button.active:
                logger.debug(f"Button clicked: {button.name}")
                button.pressed = True
                self.handle_button_press(button)
                return True
        return False
    
    def handle_menu_block_click(self, event):
        for menu_block in self.sim_init.menu_blocks:
            if menu_block.collision.contains(event.pos()):
                logger.debug(f"Menu block clicked: {menu_block.block_fn}")
                new_block = self.sim_init.add_block(menu_block, event.pos())
                if new_block:
                    logger.debug(f"New block added: {new_block.name}")
                    self.start_drag(new_block, event.pos())
                    return True
        return False

    def handle_button_press(self, button):
        if button.name == '_new_':
            self.sim_init.clear_all()
        elif button.name == '_load_':
            self.sim_init.open()
        elif button.name == '_save_':
            self.sim_init.save()
        elif button.name == '_play_':
            self.sim_init.execution_init()
        elif button.name == '_pause_':
            self.sim_init.execution_pause = not self.sim_init.execution_pause
            print("EXECUTION:", "PLAY" if not self.sim_init.execution_pause else "PAUSED")
        elif button.name == '_stop_':
            self.sim_init.execution_stop = True
        elif button.name == '_plot_':
            self.sim_init.plot_again()
        elif button.name == '_capture_':
            self.sim_init.screenshot(self)
        self.update()

    def configure_block(self, block):
        logger.debug(f"Configuring block: {block.name}")
        
        # Only include parameters that were originally defined
        original_params = getattr(block, 'original_params', block.params)
        editable_params = {k: block.params.get(k, v) for k, v in original_params.items() 
                        if not k.startswith('_') and k != 'Name'}
        editable_params['Name'] = block.username

        dialog = ParamDialog(block.name, editable_params)
        dialog.finished.connect(lambda result: self.handle_config_result(result, dialog, block))
        dialog.show()

    def show_config_dialog(self, block):
        editable_params = {k: v for k, v in block.params.items() if not k.startswith('_') and k != 'Name'}
        editable_params['Name'] = block.username
        dialog = ParamDialog(block.name, editable_params)
        if dialog.exec_():
            new_params = dialog.get_values()
            block.username = new_params.pop('Name', block.username)
            block.params.update(new_params)
        logger.debug(f"Finished configuring block: {block.name}")
        self.update()

    def set_state(self, new_state):
        logger.debug(f"Changing state from {self.state} to {new_state}")
        self.state = new_state

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.remove_selected_items()
        elif event.key() == Qt.Key_Control:
            self.sim_init.holding_CTRL = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.sim_init.holding_CTRL = False
        super().keyReleaseEvent(event)

    def toggle_selection(self, item):
        item.selected = not item.selected
        self.update()

    def remove_selected_items(self):
        blocks_to_remove = [block for block in self.sim_init.blocks_list if block.selected]
        lines_to_remove = [line for line in self.sim_init.line_list if line.selected]
        
        for block in blocks_to_remove:
            self.sim_init.remove_block_and_lines(block)
        
        for line in lines_to_remove:
            if line in self.sim_init.line_list:
                self.sim_init.line_list.remove(line)
            else:
                logger.warning(f"Attempted to remove a line that was not in the list: {line.name}")
        self.clear_selections()
        self.update()

    def remove_block_and_lines(self, block):
        self.blocks_list.remove(block)
        self.line_list = [line for line in self.line_list if not self.check_line_block(line, [block.name])]


    def remove_selected_blocks(self):
        blocks_to_remove = [block for block in self.sim_init.blocks_list if block.selected]
        for block in blocks_to_remove:
            self.sim_init.remove_block_and_lines(block)
        self.update()

    def clear_selections(self):
        for block in self.sim_init.blocks_list:
            block.selected = False
        for line in self.sim_init.line_list:
            line.selected = False 

    def closeEvent(self, event):
        event.accept()

    def eventFilter(self, obj, event):
        if isinstance(event, QMouseEvent):
            if event.type() == QEvent.MouseButtonPress:
                self.mousePressEvent(event)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.mouseReleaseEvent(event)
                return True
            elif event.type() == QEvent.MouseMove:
                self.mouseMoveEvent(event)
                return True
        return super().eventFilter(obj, event)

def main():
    app = QApplication(sys.argv)
    window = DiaBloSWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
