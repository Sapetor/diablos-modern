"""
Alignment Tools for DiaBloS Modern UI
Provides alignment and distribution operations for selected blocks.
"""

import logging
from PyQt5.QtCore import QPoint

logger = logging.getLogger(__name__)


class AlignmentTools:
    """
    Provides alignment and distribution operations for blocks.

    All methods are static and operate on a list of blocks.
    """

    @staticmethod
    def align_left(blocks):
        """Align all blocks to the leftmost block's left edge."""
        if len(blocks) < 2:
            return False

        min_left = min(b.left for b in blocks)
        for block in blocks:
            if block.left != min_left:
                block.relocate_Block(QPoint(min_left, block.top))

        logger.info(f"Aligned {len(blocks)} blocks to left edge at x={min_left}")
        return True

    @staticmethod
    def align_right(blocks):
        """Align all blocks to the rightmost block's right edge."""
        if len(blocks) < 2:
            return False

        max_right = max(b.left + b.width for b in blocks)
        for block in blocks:
            new_left = max_right - block.width
            if block.left != new_left:
                block.relocate_Block(QPoint(new_left, block.top))

        logger.info(f"Aligned {len(blocks)} blocks to right edge at x={max_right}")
        return True

    @staticmethod
    def align_center_horizontal(blocks):
        """Align all blocks to the horizontal center of the selection."""
        if len(blocks) < 2:
            return False

        # Calculate center X of bounding box
        min_left = min(b.left for b in blocks)
        max_right = max(b.left + b.width for b in blocks)
        center_x = (min_left + max_right) // 2

        for block in blocks:
            new_left = center_x - block.width // 2
            if block.left != new_left:
                block.relocate_Block(QPoint(new_left, block.top))

        logger.info(f"Aligned {len(blocks)} blocks to horizontal center at x={center_x}")
        return True

    @staticmethod
    def align_top(blocks):
        """Align all blocks to the topmost block's top edge."""
        if len(blocks) < 2:
            return False

        min_top = min(b.top for b in blocks)
        for block in blocks:
            if block.top != min_top:
                block.relocate_Block(QPoint(block.left, min_top))

        logger.info(f"Aligned {len(blocks)} blocks to top edge at y={min_top}")
        return True

    @staticmethod
    def align_bottom(blocks):
        """Align all blocks to the bottommost block's bottom edge."""
        if len(blocks) < 2:
            return False

        max_bottom = max(b.top + b.height for b in blocks)
        for block in blocks:
            new_top = max_bottom - block.height
            if block.top != new_top:
                block.relocate_Block(QPoint(block.left, new_top))

        logger.info(f"Aligned {len(blocks)} blocks to bottom edge at y={max_bottom}")
        return True

    @staticmethod
    def align_center_vertical(blocks):
        """Align all blocks to the vertical center of the selection."""
        if len(blocks) < 2:
            return False

        # Calculate center Y of bounding box
        min_top = min(b.top for b in blocks)
        max_bottom = max(b.top + b.height for b in blocks)
        center_y = (min_top + max_bottom) // 2

        for block in blocks:
            new_top = center_y - block.height // 2
            if block.top != new_top:
                block.relocate_Block(QPoint(block.left, new_top))

        logger.info(f"Aligned {len(blocks)} blocks to vertical center at y={center_y}")
        return True

    @staticmethod
    def distribute_horizontal(blocks):
        """Distribute blocks evenly horizontally (equal spacing between blocks)."""
        if len(blocks) < 3:
            return False

        # Sort blocks by left position
        sorted_blocks = sorted(blocks, key=lambda b: b.left)

        # Calculate total width of all blocks
        total_block_width = sum(b.width for b in sorted_blocks)

        # Calculate available space
        leftmost = sorted_blocks[0].left
        rightmost = sorted_blocks[-1].left + sorted_blocks[-1].width
        total_space = rightmost - leftmost

        # Calculate spacing between blocks
        available_space = total_space - total_block_width
        spacing = available_space / (len(sorted_blocks) - 1) if len(sorted_blocks) > 1 else 0

        # Position blocks
        current_x = leftmost
        for block in sorted_blocks:
            if block.left != int(current_x):
                block.relocate_Block(QPoint(int(current_x), block.top))
            current_x += block.width + spacing

        logger.info(f"Distributed {len(blocks)} blocks horizontally with spacing={spacing:.1f}")
        return True

    @staticmethod
    def distribute_vertical(blocks):
        """Distribute blocks evenly vertically (equal spacing between blocks)."""
        if len(blocks) < 3:
            return False

        # Sort blocks by top position
        sorted_blocks = sorted(blocks, key=lambda b: b.top)

        # Calculate total height of all blocks
        total_block_height = sum(b.height for b in sorted_blocks)

        # Calculate available space
        topmost = sorted_blocks[0].top
        bottommost = sorted_blocks[-1].top + sorted_blocks[-1].height
        total_space = bottommost - topmost

        # Calculate spacing between blocks
        available_space = total_space - total_block_height
        spacing = available_space / (len(sorted_blocks) - 1) if len(sorted_blocks) > 1 else 0

        # Position blocks
        current_y = topmost
        for block in sorted_blocks:
            if block.top != int(current_y):
                block.relocate_Block(QPoint(block.left, int(current_y)))
            current_y += block.height + spacing

        logger.info(f"Distributed {len(blocks)} blocks vertically with spacing={spacing:.1f}")
        return True
