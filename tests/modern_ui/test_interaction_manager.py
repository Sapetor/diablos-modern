
import unittest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import Qt, QPoint, QPointF
from modern_ui.interactions.interaction_manager import InteractionManager, State

class TestInteractionManager(unittest.TestCase):

    def setUp(self):
        # Mock the canvas
        self.mock_canvas = MagicMock()
        self.mock_canvas.state = State.IDLE
        self.mock_canvas.dsim = MagicMock()
        self.mock_canvas.dsim.blocks_list = []
        self.mock_canvas.dsim.blocks_list = []
        self.mock_canvas.screen_to_world.side_effect = lambda p: p # Identity transform for simplicity
        
        # Default boolean flags to False so checks don't return early
        self.mock_canvas.panning = False
        self.mock_canvas.is_rect_selecting = False
        self.mock_canvas.snap_enabled = True # Default to true for safety

        self.manager = InteractionManager(self.mock_canvas)

    def test_initial_state(self):
        self.assertEqual(self.manager.state, State.IDLE)
        self.assertEqual(self.mock_canvas.state, State.IDLE)

    def test_state_delegation(self):
        self.manager.state = State.DRAGGING
        self.assertEqual(self.mock_canvas.state, State.DRAGGING)
        
        self.mock_canvas.state = State.CONNECTING
        self.assertEqual(self.manager.state, State.CONNECTING)

    def test_mouse_press_left_select_block(self):
        # mocking event
        mock_event = MagicMock()
        mock_event.button.return_value = Qt.LeftButton
        mock_event.pos.return_value = QPoint(100, 100)
        mock_event.modifiers.return_value = Qt.NoModifier

        # Mock canvas helpers
        self.mock_canvas._check_port_clicks.return_value = False
        
        mock_block = MagicMock()
        self.mock_canvas._get_clicked_block.return_value = mock_block
        
        # Execute
        self.manager.handle_mouse_press(mock_event)

        # Verify
        self.mock_canvas._check_port_clicks.assert_called()
        self.mock_canvas._get_clicked_block.assert_called()
        self.mock_canvas._handle_block_click.assert_called_with(mock_block, QPoint(100, 100))

    def test_mouse_press_pan(self):
        mock_event = MagicMock()
        mock_event.button.return_value = Qt.MiddleButton
        mock_event.pos.return_value = QPoint(50, 50)
        
        self.manager.handle_mouse_press(mock_event)
        
        self.assertTrue(self.mock_canvas.panning)
        self.assertEqual(self.mock_canvas.last_pan_pos, QPoint(50, 50))

    @patch('modern_ui.interactions.interaction_manager.logger')
    def test_mouse_move_dragging_block(self, mock_logger):
        # Setup state
        self.mock_canvas.state = State.DRAGGING
        self.mock_canvas.dragging_block = MagicMock()
        self.mock_canvas.drag_offset = QPointF(10, 10)
        self.mock_canvas.snap_enabled = False
        
        mock_event = MagicMock()
        mock_event.pos.return_value = QPoint(110, 110) # 110 - 10 = 100
        
        self.manager.handle_mouse_move(mock_event)
        
        # Check if error occurred
        if mock_logger.error.called:
            print(f"Captured Error Log: {mock_logger.error.call_args}")
            
        # Verify relocate called
        self.mock_canvas.dragging_block.relocate_Block.assert_called()
        args = self.mock_canvas.dragging_block.relocate_Block.call_args[0]
        self.assertEqual(args[0], QPoint(100, 100))
        self.mock_canvas.update.assert_called()

if __name__ == '__main__':
    unittest.main()
