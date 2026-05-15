
import unittest
from unittest.mock import MagicMock
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
        """Verify that undo() moves the top undo entry onto the redo stack and
        calls _restore_state with the captured snapshot."""
        # Capture initial (empty) state
        self.manager.push_undo("Initial")
        self.assertEqual(len(self.manager.undo_stack), 1)

        # Mock _restore_state so we can inspect the snapshot it receives
        # without requiring a full DBlock/canvas environment.
        restored = []
        self.manager._restore_state = lambda state: restored.append(state) or True

        self.manager.undo()

        # Undo stack should now be empty
        self.assertEqual(len(self.manager.undo_stack), 0)
        # The pre-undo state should have been pushed to redo
        self.assertEqual(len(self.manager.redo_stack), 1)
        # _restore_state was called once with the initial snapshot
        self.assertEqual(len(restored), 1)
        self.assertIn('blocks', restored[0])
        self.assertIn('lines', restored[0])
        
    def test_stack_limit(self):
        import collections
        # Recreate the deque with the smaller maxlen after updating the limit,
        # since deque.maxlen is fixed at construction time.
        self.manager.max_undo_steps = 2
        self.manager.undo_stack = collections.deque(maxlen=2)
        self.manager.push_undo("1")
        self.manager.push_undo("2")
        self.manager.push_undo("3")

        self.assertEqual(len(self.manager.undo_stack), 2)
        self.assertEqual(self.manager.undo_stack[0]['description'], "2")
        self.assertEqual(self.manager.undo_stack[1]['description'], "3")

if __name__ == '__main__':
    unittest.main()
