
import unittest
from unittest.mock import MagicMock
from PyQt5.QtCore import QRect
from modern_ui.managers.selection_manager import SelectionManager

class TestSelectionManager(unittest.TestCase):
    def setUp(self):
        self.mock_canvas = MagicMock()
        self.mock_dsim = MagicMock()
        self.mock_canvas.dsim = self.mock_dsim
        self.mock_dsim.blocks_list = []
        self.mock_dsim.line_list = []
        
        # Setup history manager mock for undo calls
        self.mock_canvas.history_manager = MagicMock()
        
        self.manager = SelectionManager(self.mock_canvas)

    def test_select_all_blocks(self):
        block1 = MagicMock()
        block1.selected = False
        block2 = MagicMock()
        block2.selected = False
        self.mock_dsim.blocks_list = [block1, block2]
        
        self.manager.select_all_blocks()
        
        self.assertTrue(block1.selected)
        self.assertTrue(block2.selected)
        self.mock_canvas.update.assert_called()

    def test_clear_selections(self):
        block1 = MagicMock()
        block1.selected = True
        self.mock_dsim.blocks_list = [block1]
        
        line1 = MagicMock()
        line1.selected = True
        self.mock_dsim.line_list = [line1]
        
        self.manager.clear_selections()
        
        self.assertFalse(block1.selected)
        self.assertFalse(line1.selected)
        self.mock_canvas.update.assert_called()

    def test_remove_selected_items(self):
        block1 = MagicMock()
        block1.selected = True
        self.mock_dsim.blocks_list = [block1]
        
        self.manager.remove_selected_items()
        
        # Should call remove on dsim
        self.mock_dsim.remove_block_and_lines.assert_called_with(block1)
        # Should push undo
        self.mock_canvas.history_manager.push_undo.assert_called_with("Delete")

    def test_finalize_rect_selection(self):
        block1 = MagicMock()
        block1.left = 10
        block1.top = 10
        block1.width = 50
        block1.height = 50
        
        self.mock_dsim.blocks_list = [block1]
        
        # Rect covering block1
        rect = QRect(0, 0, 100, 100)
        self.manager.finalize_rect_selection(rect)
        
        self.assertTrue(block1.selected)

if __name__ == '__main__':
    unittest.main()
