"""
Unit tests for ClipboardManager - copy/paste functionality.
"""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QRect, QPoint

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modern_ui.managers.clipboard_manager import ClipboardManager
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_canvas(qapp):
    """Create a mock canvas with dsim."""
    canvas = MagicMock()
    dsim = MagicMock()

    # Create real block objects
    block1 = DBlock(
        block_fn='Constant',
        sid=0,
        coords=QRect(100, 100, 60, 40),
        color='#4CAF50',
        in_ports=0,
        out_ports=1,
        b_type='source',
        io_edit=False,
        fn_name='Constant',
        params={'value': 1.0, '_name_': 'Constant0'},
        external=False,
        username='',
        block_class=None,
        colors=None
    )
    block1.selected = True

    block2 = DBlock(
        block_fn='Gain',
        sid=0,
        coords=QRect(250, 100, 60, 40),
        color='#2196F3',
        in_ports=1,
        out_ports=1,
        b_type='math',
        io_edit=False,
        fn_name='Gain',
        params={'gain': 2.0, '_name_': 'Gain0'},
        external=False,
        username='',
        block_class=None,
        colors=None
    )
    block2.selected = True

    block3 = DBlock(
        block_fn='Scope',
        sid=0,
        coords=QRect(400, 100, 60, 40),
        color='#FF9800',
        in_ports=1,
        out_ports=0,
        b_type='sink',
        io_edit=False,
        fn_name='Scope',
        params={'_name_': 'Scope0'},
        external=False,
        username='',
        block_class=None,
        colors=None
    )
    block3.selected = True

    # Set up block names and coordinates
    block1.name = 'Constant0'
    block2.name = 'Gain0'
    block3.name = 'Scope0'

    # Create connections between blocks
    # Constant0 -> Gain0
    line1 = DLine(
        sid=0,
        srcblock='Constant0',
        srcport=0,
        dstblock='Gain0',
        dstport=0,
        points=[QPoint(160, 120), QPoint(250, 120)]
    )

    # Gain0 -> Scope0
    line2 = DLine(
        sid=1,
        srcblock='Gain0',
        srcport=0,
        dstblock='Scope0',
        dstport=0,
        points=[QPoint(310, 120), QPoint(400, 120)]
    )

    # Set up dsim with blocks and connections
    dsim.blocks_list = [block1, block2, block3]
    dsim.connections_list = [line1, line2]
    dsim.line_list = dsim.connections_list  # Alias
    dsim.menu_blocks = []
    dsim.colors = None
    dsim.dirty = False

    canvas.dsim = dsim
    canvas.history_manager = MagicMock()

    return canvas


class TestClipboardCopy:
    """Tests for copy functionality."""

    def test_copy_captures_connections(self, mock_canvas):
        """Test that copying blocks also captures their connections."""
        clipboard = ClipboardManager(mock_canvas)

        # Verify initial state
        assert len(mock_canvas.dsim.blocks_list) == 3
        assert len(mock_canvas.dsim.connections_list) == 2

        # Copy selected blocks
        clipboard.copy_selected_blocks()

        # Verify blocks were copied
        assert len(clipboard.clipboard_blocks) == 3, \
            f"Expected 3 blocks, got {len(clipboard.clipboard_blocks)}"

        # THIS IS THE KEY TEST - verify connections were copied
        assert len(clipboard.clipboard_connections) == 2, \
            f"Expected 2 connections, got {len(clipboard.clipboard_connections)}"

    def test_copy_connection_indices_are_correct(self, mock_canvas):
        """Test that connection indices correctly reference clipboard blocks."""
        clipboard = ClipboardManager(mock_canvas)
        clipboard.copy_selected_blocks()

        # Get the block order in clipboard
        block_fns = [b['block_fn'] for b in clipboard.clipboard_blocks]

        # Verify connection indices match
        for conn in clipboard.clipboard_connections:
            start_idx = conn['start_index']
            end_idx = conn['end_index']

            assert 0 <= start_idx < len(clipboard.clipboard_blocks), \
                f"start_index {start_idx} out of range"
            assert 0 <= end_idx < len(clipboard.clipboard_blocks), \
                f"end_index {end_idx} out of range"

    def test_copy_only_internal_connections(self, mock_canvas):
        """Test that only connections between selected blocks are copied."""
        # Deselect one block
        mock_canvas.dsim.blocks_list[2].selected = False  # Deselect Scope

        clipboard = ClipboardManager(mock_canvas)
        clipboard.copy_selected_blocks()

        # Should only have 2 blocks and 1 connection (Constant->Gain)
        assert len(clipboard.clipboard_blocks) == 2
        assert len(clipboard.clipboard_connections) == 1

    def test_copy_with_no_connections(self, mock_canvas):
        """Test copying blocks that have no connections between them."""
        # Clear connections
        mock_canvas.dsim.connections_list = []
        mock_canvas.dsim.line_list = []

        clipboard = ClipboardManager(mock_canvas)
        clipboard.copy_selected_blocks()

        assert len(clipboard.clipboard_blocks) == 3
        assert len(clipboard.clipboard_connections) == 0


class TestClipboardPaste:
    """Tests for paste functionality."""

    def test_paste_creates_blocks_and_connections(self, mock_canvas):
        """Test that paste creates both blocks and their connections."""
        clipboard = ClipboardManager(mock_canvas)

        # Copy
        clipboard.copy_selected_blocks()
        initial_block_count = len(mock_canvas.dsim.blocks_list)
        initial_conn_count = len(mock_canvas.dsim.connections_list)

        # Deselect original blocks
        for block in mock_canvas.dsim.blocks_list:
            block.selected = False

        # Paste
        clipboard.paste_blocks()

        # Verify new blocks were created
        assert len(mock_canvas.dsim.blocks_list) == initial_block_count + 3, \
            f"Expected {initial_block_count + 3} blocks, got {len(mock_canvas.dsim.blocks_list)}"

        # Verify new connections were created
        assert len(mock_canvas.dsim.connections_list) == initial_conn_count + 2, \
            f"Expected {initial_conn_count + 2} connections, got {len(mock_canvas.dsim.connections_list)}"


class TestConnectionNameResolution:
    """Tests for the name-to-block resolution in copy."""

    def test_name_to_block_mapping(self, mock_canvas):
        """Debug test to verify name mapping works correctly."""
        clipboard = ClipboardManager(mock_canvas)

        # Build the same mapping as copy_selected_blocks
        name_to_block = {b.name: b for b in clipboard.dsim.blocks_list}

        # Verify all block names are in the mapping
        for block in clipboard.dsim.blocks_list:
            assert block.name in name_to_block, f"Block {block.name} not in name_to_block"

        # Verify all connection sources/destinations resolve
        for line in clipboard.dsim.connections_list:
            src_obj = name_to_block.get(line.srcblock)
            dst_obj = name_to_block.get(line.dstblock)

            assert src_obj is not None, \
                f"Could not resolve srcblock '{line.srcblock}'. Available: {list(name_to_block.keys())}"
            assert dst_obj is not None, \
                f"Could not resolve dstblock '{line.dstblock}'. Available: {list(name_to_block.keys())}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
