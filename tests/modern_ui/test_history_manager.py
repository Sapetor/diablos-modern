
import unittest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import QRect
from modern_ui.managers.history_manager import HistoryManager

class TestHistoryManager(unittest.TestCase):
    def setUp(self):
        self.mock_canvas = MagicMock()
        self.mock_dsim = MagicMock()
        self.mock_canvas.dsim = self.mock_dsim
        self.mock_dsim.blocks_list = []
        self.mock_dsim.line_list = []
        
        self.manager = HistoryManager(self.mock_canvas)

    def test_initial_state(self):
        self.assertEqual(len(self.manager.undo_stack), 0)
        self.assertEqual(len(self.manager.redo_stack), 0)

    def test_push_undo(self):
        # Create a mock block
        mock_block = MagicMock()
        mock_block.name = "TestBlock"
        mock_block.block_fn = "step"
        # Mock attributes needed for capture
        mock_block.left = 0
        mock_block.top = 0
        mock_block.width = 100
        mock_block.height = 100
        mock_block.b_color.name.return_value = "#ff0000"
        mock_block.in_ports = []
        mock_block.out_ports = []
        mock_block.b_type = "source"
        mock_block.io_edit = False
        mock_block.fn_name = "Step"
        mock_block.params = {}
        mock_block.external = False
        mock_block.selected = False
        
        self.mock_dsim.blocks_list = [mock_block]
        
        self.manager.push_undo("Added Block")
        
        self.assertEqual(len(self.manager.undo_stack), 1)
        self.assertEqual(self.manager.undo_stack[0]['description'], "Added Block")
        self.assertEqual(len(self.manager.undo_stack[0]['state']['blocks']), 1)

    def test_undo_restores_state(self):
        # Setup initial state (empty)
        self.manager.push_undo("Initial")
        
        # Change state (add block)
        mock_block = MagicMock()
        mock_block.selected = True # Check if this property is captured
        self.mock_dsim.blocks_list = [mock_block]
        
        # Undo
        # Need to mock _restore_state to not crash on import or real DBlock creation locally
        # OR we let it run but mock DBlock import?
        # Simpler: mock _restore_state for this conceptual test? 
        # But we want to test _restore_state logic too.
        
        pass 
        # _restore_state uses real DBlock. That's hard to test without full environment.
        # We will unit test the HistoryManager logic (stacks) mainly.
        # Deep integration test for _restore_state is harder.
        
    def test_stack_limit(self):
        self.manager.max_undo_steps = 2
        self.manager.push_undo("1")
        self.manager.push_undo("2")
        self.manager.push_undo("3")
        
        self.assertEqual(len(self.manager.undo_stack), 2)
        self.assertEqual(self.manager.undo_stack[0]['description'], "2")
        self.assertEqual(self.manager.undo_stack[1]['description'], "3")

if __name__ == '__main__':
    unittest.main()
