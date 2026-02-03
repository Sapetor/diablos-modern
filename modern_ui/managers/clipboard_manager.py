"""
Clipboard Manager for ModernCanvas
Handles copy, paste, cut, and duplicate operations.
"""

import logging
import copy
from PyQt5.QtCore import QRect, QPoint

logger = logging.getLogger(__name__)


class ClipboardManager:
    """
    Manages clipboard operations for blocks and connections.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim
        self.clipboard_blocks = []
        self.clipboard_connections = []

    def copy_selected_blocks(self):
        """Copy selected blocks to clipboard."""
        try:
            # Find all selected blocks
            selected_blocks = [block for block in self.dsim.blocks_list if block.selected]

            if not selected_blocks:
                logger.info("No blocks selected to copy")
                return

            # Deep copy the block data (not the actual block objects)
            self.clipboard_blocks = []
            selected_indices = {}  # Map original block object to its index in clipboard

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
            if hasattr(self.canvas, 'history_manager'):
                self.canvas.history_manager.push_undo("Paste")

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

                # SPECIAL HANDLING FOR SUBSYSTEM
                if block_fn == 'Subsystem':
                    from blocks.subsystem import Subsystem
                    new_block = Subsystem(
                        block_name=f"Subsystem{sid}",  # Default name will be corrected by DBlock init logic or manually set below
                        sid=sid,
                        coords=new_coords,
                        color=block_data['color']
                    )
                    # Restore other attributes
                    new_block.io_edit = block_data['io_edit']
                    new_block.fn_name = block_data['fn_name']
                    new_block.params = block_data['params'].copy()
                    new_block.params['_name_'] = new_block.name  # Ensure params name matches
                    new_block.external = block_data['external']

                    # Restore internal structure if available
                    if 'sub_blocks' in block_data:
                        try:
                            new_block.sub_blocks = copy.deepcopy(block_data['sub_blocks'])
                            new_block.sub_lines = copy.deepcopy(block_data['sub_lines'])
                            new_block.ports = copy.deepcopy(block_data.get('ports', {}))
                            new_block.ports_map = copy.deepcopy(block_data.get('ports_map', {}))

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
            self.canvas.update()

            logger.info(f"Pasted {len(pasted_blocks)} block(s)")

            # Emit signal for first pasted block if any
            if pasted_blocks:
                self.canvas.block_selected.emit(pasted_blocks[0])

        except Exception as e:
            logger.error(f"Error pasting blocks: {str(e)}")

    def cut_selected_blocks(self):
        """Cut selected blocks to clipboard."""
        self.copy_selected_blocks()
        if hasattr(self.canvas, 'selection_manager'):
            self.canvas.selection_manager.remove_selected_items()

    def _duplicate_block(self, block):
        """Duplicate a single block.

        Args:
            block: The block to duplicate
        """
        try:
            from lib.simulation.menu_block import MenuBlocks

            # Push undo state before duplication
            if hasattr(self.canvas, 'history_manager'):
                self.canvas.history_manager.push_undo("Duplicate")

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
                if hasattr(self.canvas, 'selection_manager'):
                    self.canvas.selection_manager.clear_selections()
                new_block.selected = True
                self.canvas.update()

        except Exception as e:
            logger.error(f"Error duplicating block: {str(e)}")
