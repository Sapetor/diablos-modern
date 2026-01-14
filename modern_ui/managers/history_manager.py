import logging
import copy
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor

# Import DBlock/DSim dependencies for restoration
# Note: In a cleaner architecture we would use factory methods, but for now we follow existing logic
from lib.simulation.block import DBlock

logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Manages the Undo/Redo stack and state capture/restore for the ModernCanvas.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim
        
        # Undo/Redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50

    def push_undo(self, description="Action"):
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

    def _capture_state(self):
        """Capture current diagram state (Snapshot)."""
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
                    'coords': (block.left, block.top, block.width, block.height),
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
                    'selected': line.selected,
                    'routing_mode': line.routing_mode if hasattr(line, 'routing_mode') else 'bezier',
                    'label': line.label if hasattr(line, 'label') else ''
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

            # Clear hover state (old objects are now invalid)
            # Accessing canvas attributes directly
            if hasattr(self.canvas, 'hovered_block'): self.canvas.hovered_block = None
            if hasattr(self.canvas, 'hovered_line'): self.canvas.hovered_line = None
            if hasattr(self.canvas, 'hovered_port'): self.canvas.hovered_port = None

            # Clear validation errors when state is restored (undo/redo)
            if hasattr(self.canvas, 'clear_validation'):
                self.canvas.clear_validation()

            # Restore blocks by directly creating DBlock instances
            for block_data in state['blocks']:
                try:
                    coords = QRect(*block_data['coords'])

                    # Find the block class if it's a known block type
                    block_class = None
                    block_fn = block_data['block_fn']

                    # Try to import the block class
                    # Note: This dynamic import is consistent with legacy code
                    try:
                        module_name = f"blocks.{block_fn.lower()}"
                        class_name = f"{block_fn.capitalize()}Block"
                        module = __import__(module_name, fromlist=[class_name])
                        block_class = getattr(module, class_name, None)
                    except (ImportError, AttributeError):
                        # logger.debug(f"Could not import block class for {block_fn}")
                        block_class = None

                    # Extract sid from name (e.g., "step0" -> 0)
                    name = block_data['name']
                    sid = int(name[len(block_fn):]) if len(name) > len(block_fn) else 0

                    # Create DBlock directly
                    block = DBlock(
                        block_data['block_fn'],
                        sid,
                        coords,
                        QColor(block_data['color']),
                        block_data['in_ports'],
                        block_data['out_ports'],
                        block_data['b_type'],
                        block_data['io_edit'],
                        block_data['fn_name'],
                        block_data['params'],
                        block_data['external'],
                        block_class=block_class
                    )

                    # Restore selection state and original name
                    block.selected = block_data.get('selected', False)
                    block.name = name

                    # Add to blocks list
                    self.dsim.blocks_list.append(block)

                except Exception as e:
                    logger.error(f"Error restoring block {block_data.get('name', 'unknown')}: {str(e)}")
                    continue

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
                        # Restore routing mode and label
                        if 'routing_mode' in line_data:
                            line.routing_mode = line_data['routing_mode']
                        if 'label' in line_data:
                            line.label = line_data['label']
                        # Recalculate path with restored routing mode (if canvas has block list reference)
                        line.update_line(self.dsim.blocks_list)

            if hasattr(self.canvas, 'update'):
                self.canvas.update()
            return True

        except Exception as e:
            logger.error(f"Error restoring state: {str(e)}")
            return False
