
import logging
from PyQt5.QtCore import QRect

logger = logging.getLogger(__name__)

class SelectionManager:
    """
    Manages selection of blocks and lines on the canvas.
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.dsim = canvas.dsim
        
        # We don't necessarily need to duplicate the lists if we just iterate dsim lists, 
        # but maintaining a cache might be faster. 
        # For now, we will follow the existing pattern of iterating dsim lists or 
        # using the 'selected' attribute on items.

    def clear_selections(self):
        """Clear all block and line selections."""
        for block in self.dsim.blocks_list:
            block.selected = False
        self.clear_line_selections()
        self.canvas.update()

    def clear_line_selections(self):
        """Clear all line selections."""
        for line in self.dsim.line_list:
            line.selected = False
        self.canvas.update()

    def select_all_blocks(self):
        """Select all blocks on canvas."""
        for block in self.dsim.blocks_list:
            block.selected = True
        self.canvas.update()

    def remove_selected_items(self):
        """Remove all selected blocks and lines."""
        # Delegates back to canvas history/removal logic or handles it here?
        # Since it involves Undo and DSim, it's safer to keep the orchestration here 
        # but maybe call back to canvas for Undo?
        # Actually canvas.remove_selected_items already does Undo.
        # We can implement it here if we depend on HistoryManager.
        
        try:
            blocks_to_remove = [block for block in self.dsim.blocks_list if block.selected]
            lines_to_remove = [line for line in self.dsim.line_list if line.selected]

            # Push undo state before deletion
            if (blocks_to_remove or lines_to_remove) and hasattr(self.canvas, 'history_manager'):
                self.canvas.history_manager.push_undo("Delete")

            for block in blocks_to_remove:
                self.dsim.remove_block_and_lines(block)
            for line in lines_to_remove:
                if line in self.dsim.line_list:
                    self.dsim.line_list.remove(line)

            # Clear validation errors when blocks are removed
            if hasattr(self.canvas, 'clear_validation'):
                self.canvas.clear_validation()

            self.canvas.update()
            logger.info(f"Removed {len(blocks_to_remove)} blocks and {len(lines_to_remove)} lines")
            
        except Exception as e:
            logger.error(f"Error removing selected items: {str(e)}")

    def finalize_rect_selection(self, rect):
        """
        Select blocks within the given rectangle.
        Args:
            rect (QRect): The selection rectangle in world coordinates.
        """
        if not rect:
            return

        # Normalize rect
        rect = rect.normalized()
        
        count = 0
        for block in self.dsim.blocks_list:
            block_rect = QRect(block.left, block.top, block.width, block.height)
            if rect.intersects(block_rect):
                block.selected = True
                count += 1
                
        logger.info(f"Rectangle selection completed: {count} block(s) selected")
        self.canvas.update()

    def get_selected_blocks(self):
        """Return a list of currently selected blocks."""
        return [b for b in self.dsim.blocks_list if b.selected]

    def has_selection(self):
        """Check if anything is selected."""
        if any(b.selected for b in self.dsim.blocks_list):
            return True
        if any(l.selected for l in self.dsim.line_list):
            return True
        return False
