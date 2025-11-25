"""
SimulationModel - Data layer for DiaBloS simulation.
Manages blocks, lines, and diagram state.
"""

import logging
import copy
import sys
import os
from typing import List, Dict, Optional, Tuple, Any
from PyQt5.QtGui import QColor
from PyQt5.QtCore import QRect, QPoint
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.block_loader import load_blocks
from lib.simulation.menu_block import MenuBlocks

# Import block size configuration
from config.block_sizes import get_block_size

logger = logging.getLogger(__name__)


class SimulationModel:
    """
    Data model for simulation diagrams.
    Manages blocks, connections, and diagram state without UI dependencies.

    Attributes:
        colors: Color palette for blocks
        menu_blocks: Available block types from blocks/ directory
        blocks_list: Instantiated blocks in the current diagram
        line_list: Connections between blocks
        dirty: Flag indicating if diagram has unsaved changes
    """

    def __init__(self) -> None:
        """Initialize the simulation model with empty state."""
        logger.debug("Initializing SimulationModel")

        # Color palette for blocks
        self.colors: Dict[str, QColor] = {
            'black': QColor(0, 0, 0),
            'red': QColor(255, 0, 0),
            'green': QColor(0, 255, 0),
            'blue': QColor(0, 0, 255),
            'yellow': QColor(255, 255, 0),
            'magenta': QColor(255, 0, 255),
            'cyan': QColor(0, 255, 255),
            'purple': QColor(128, 0, 255),
            'orange': QColor(255, 128, 0),
            'aqua': QColor(0, 255, 128),
            'pink': QColor(255, 0, 128),
            'lime_green': QColor(128, 255, 0),
            'light_blue': QColor(0, 128, 255),
            'dark_red': QColor(128, 0, 0),
            'dark_green': QColor(0, 128, 0),
            'dark_blue': QColor(0, 0, 128),
            'dark_gray': QColor(64, 64, 64),
            'gray': QColor(128, 128, 128),
            'light_gray': QColor(192, 192, 192),
            'white': QColor(255, 255, 255)
        }

        # Data containers
        self.menu_blocks: List[MenuBlocks] = []      # Available block types
        self.blocks_list: List[DBlock] = []          # Instantiated blocks in diagram
        self.line_list: List[DLine] = []             # Connections between blocks

        # State flags
        self.dirty: bool = False                      # Has diagram been modified?

        # Load available block types
        self.load_all_blocks()

    def _get_category_color(self, category: str) -> QColor:
        """
        Get theme-aware color for a block category.

        Args:
            category: Block category (Sources, Math, Control, Sinks, Other)

        Returns:
            QColor from theme manager for the category
        """
        from modern_ui.themes.theme_manager import theme_manager

        category_lower = category.lower() if isinstance(category, str) else str(category).lower()

        if 'source' in category_lower:
            return theme_manager.get_color('block_source')
        elif 'math' in category_lower:
            return theme_manager.get_color('block_process')
        elif 'control' in category_lower:
            return theme_manager.get_color('block_control')
        elif 'sink' in category_lower:
            return theme_manager.get_color('block_sink')
        else:
            return theme_manager.get_color('block_other')

    def load_all_blocks(self) -> None:
        """
        Load all available block types from the blocks/ directory with theme-aware colors.
        Creates MenuBlock instances for each available block type.
        """
        self.menu_blocks = []
        block_classes = load_blocks()

        for block_class in block_classes:
            block = block_class()

            # Determine I/O editability
            io_edit = 'none'
            if hasattr(block, 'io_editable'):
                io_edit = block.io_editable
            elif len(block.inputs) > 0 and len(block.outputs) > 0:
                io_edit = 'both'
            elif len(block.inputs) > 0:
                io_edit = 'input'
            elif len(block.outputs) > 0:
                io_edit = 'output'

            # Get block type
            b_type = getattr(block, 'b_type', 2)

            # Process parameters
            ex_params = {}
            if hasattr(block, 'params') and block.params:
                for param_name, param_info in block.params.items():
                    if isinstance(param_info, dict) and 'default' in param_info:
                        ex_params[param_name] = param_info['default']
                    else:
                        ex_params[param_name] = param_info

            # Determine function name
            if hasattr(block, 'fn_name'):
                fn_name = block.fn_name
            else:
                fn_name = block.block_name.lower()

            # Get category and determine theme-aware color
            category = getattr(block, 'category', 'Other')
            block_color = self._get_category_color(category)

            # Get block-specific size from configuration
            block_size = get_block_size(block.block_name)

            menu_block = MenuBlocks(
                block_fn=block.block_name,
                fn_name=fn_name,
                io_params={
                    'inputs': len(block.inputs),
                    'outputs': len(block.outputs),
                    'b_type': b_type,
                    'io_edit': io_edit
                },
                ex_params=ex_params,
                b_color=block_color,
                coords=block_size,  # Use configured block size
                external=getattr(block, 'external', False),
                block_class=block_class,
                colors=self.colors
            )

            # Store category on menu block for later reference
            menu_block.category = category
            self.menu_blocks.append(menu_block)

    def add_block(self, block: MenuBlocks, m_pos: QPoint) -> DBlock:
        """
        Add a new block instance to the diagram.

        Args:
            block: MenuBlock template containing block type and parameters
            m_pos: QPoint position for the block's center

        Returns:
            The newly created and added block instance

        Raises:
            None - errors are logged but not raised
        """
        logger.debug(f"Adding new block of type {block.block_fn} at position {m_pos}")

        # Find next available ID for this block type
        id_list = [int(b_elem.name[len(b_elem.block_fn):])
                   for b_elem in self.blocks_list
                   if b_elem.block_fn == block.block_fn]
        sid = max(id_list) + 1 if id_list else 0

        try:
            # Calculate block position (centered on mouse)
            mouse_x = int(m_pos.x() - block.side_length[0] // 2)
            mouse_y = int(m_pos.y() - block.side_length[1] // 2)
            width = int(block.size[0])
            height = int(block.size[1])

            block_collision = QRect(mouse_x, mouse_y, width, height)
        except Exception as e:
            logger.error(f"Error creating QRect: {str(e)}")
            # Fallback with explicit integer conversion
            mouse_x = int(float(m_pos.x()) - float(block.side_length[0]) // 2)
            mouse_y = int(float(m_pos.y()) - float(block.side_length[1]) // 2)
            width = int(float(block.size[0]))
            height = int(float(block.size[1]))
            block_collision = QRect(mouse_x, mouse_y, width, height)

        # Create the block instance with category information
        category = getattr(block, 'category', 'Other')
        new_block = DBlock(
            block.block_fn, sid, block_collision, block.b_color,
            block.ins, block.outs, block.b_type, block.io_edit,
            block.fn_name, copy.deepcopy(block.params), block.external,
            block_class=block.block_class, colors=self.colors, category=category
        )

        self.blocks_list.append(new_block)
        self.dirty = True
        logger.debug(f"New block created: {new_block.name} (category: {category})")
        return new_block

    def add_line(self, srcData: Optional[Tuple[str, int, QPoint]],
                 dstData: Optional[Tuple[str, int, QPoint]]) -> Optional[DLine]:
        """
        Add a connection line between two blocks.

        Args:
            srcData: Tuple (block_name, port_num, coordinates) for source port
            dstData: Tuple (block_name, port_num, coordinates) for destination port

        Returns:
            The newly created line, or None if srcData/dstData is invalid

        Raises:
            None - errors are logged but not raised
        """
        if srcData is None or dstData is None:
            logger.debug("Error: Invalid line data")
            return None

        # Find next available ID
        id_list = [int(line.name[4:]) for line in self.line_list]
        sid = max(id_list) + 1 if id_list else 0

        try:
            line = DLine(
                sid,
                srcblock=srcData[0], srcport=srcData[1],
                dstblock=dstData[0], dstport=dstData[1],
                points=(srcData[2], dstData[2])
            )
            line.color = QColor(255, 0, 0)  # Red lines

            self.line_list.append(line)
            self.dirty = True
            logger.debug(f"Line created: {line.name}")
            return line
        except Exception as e:
            logger.error(f"Error creating line: {e}")
            return None

    def remove_block(self, block: DBlock) -> None:
        """
        Remove a block and all associated connections.

        Args:
            block: DBlock instance to remove from the diagram
        """
        self.blocks_list.remove(block)
        # Remove all lines connected to this block
        self.line_list = [
            line for line in self.line_list
            if not self._is_line_connected_to_block(line, block.name)
        ]
        self.dirty = True
        logger.debug(f"Removed block {block.name} and associated lines")

    def remove_line(self, line: DLine) -> None:
        """
        Remove a connection line from the diagram.

        Args:
            line: DLine instance to remove
        """
        if line in self.line_list:
            self.line_list.remove(line)
            self.dirty = True
            logger.debug(f"Removed line {line.name}")

    def clear_all(self) -> None:
        """Clear all blocks and lines from the diagram, reset dirty flag."""
        self.blocks_list.clear()
        self.line_list.clear()
        self.dirty = False
        logger.debug("Cleared all blocks and lines")

    def get_block_by_name(self, name: str) -> Optional[DBlock]:
        """
        Find a block by its name.

        Args:
            name: Block name to search for

        Returns:
            DBlock if found, None otherwise
        """
        for block in self.blocks_list:
            if block.name == name:
                return block
        return None

    def is_port_available(self, dst_line: Tuple[str, int, QPoint]) -> bool:
        """
        Check if an input port is already connected.

        Args:
            dst_line: Tuple (block_name, port_num, coordinates) for destination

        Returns:
            True if port is available (not connected), False if already connected
        """
        for line in self.line_list:
            if line.dstblock == dst_line[0] and line.dstport == dst_line[1]:
                return False
        return True

    def update_lines(self) -> None:
        """Update all line positions based on current block positions."""
        for line in self.line_list:
            line.update_line(self.blocks_list)

    def _is_line_connected_to_block(self, line: DLine, block_name: str) -> bool:
        """
        Check if a line is connected to a specific block.

        Args:
            line: DLine to check
            block_name: Name of block to check connection to

        Returns:
            True if line is connected to the block, False otherwise
        """
        return line.dstblock == block_name or line.srcblock == block_name

    def get_diagram_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current diagram.

        Returns:
            Dictionary with keys: 'blocks', 'lines', 'modified', 'block_types'
        """
        return {
            'blocks': len(self.blocks_list),
            'lines': len(self.line_list),
            'modified': self.dirty,
            'block_types': len(set(b.block_fn for b in self.blocks_list))
        }
