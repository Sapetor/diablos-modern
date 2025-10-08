"""
Unit tests for SimulationModel class.
"""

import pytest
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor

from lib.models.simulation_model import SimulationModel


@pytest.mark.unit
@pytest.mark.qt
class TestSimulationModelInitialization:
    """Test SimulationModel initialization."""

    def test_model_initializes_with_empty_state(self, simulation_model):
        """Test that a new model starts with empty block and line lists."""
        assert simulation_model.blocks_list == []
        assert simulation_model.line_list == []
        assert simulation_model.dirty == False

    def test_model_initializes_with_colors(self, simulation_model):
        """Test that model has default color palette."""
        assert 'black' in simulation_model.colors
        assert 'red' in simulation_model.colors
        assert 'blue' in simulation_model.colors
        assert isinstance(simulation_model.colors['black'], QColor)

    def test_model_loads_menu_blocks(self, simulation_model):
        """Test that model loads block types from blocks/ directory."""
        assert len(simulation_model.menu_blocks) > 0
        block_names = [mb.block_fn for mb in simulation_model.menu_blocks]
        # Check for some expected blocks
        assert 'Step' in block_names or 'Sine' in block_names


@pytest.mark.unit
@pytest.mark.qt
class TestAddBlock:
    """Test adding blocks to the model."""

    def test_add_block_creates_block_instance(self, simulation_model):
        """Test that add_block creates and returns a DBlock."""
        # Find a block template
        step_block = None
        for mb in simulation_model.menu_blocks:
            if mb.block_fn == 'Step':
                step_block = mb
                break

        if step_block is None:
            pytest.skip("Step block not found in menu_blocks")

        # Add the block
        pos = QPoint(100, 100)
        new_block = simulation_model.add_block(step_block, pos)

        assert new_block is not None
        assert new_block.block_fn == 'Step'
        assert new_block.name == 'step0'
        assert new_block in simulation_model.blocks_list

    def test_add_block_sets_dirty_flag(self, simulation_model):
        """Test that adding a block sets the dirty flag."""
        # Find any block
        if len(simulation_model.menu_blocks) == 0:
            pytest.skip("No blocks available")

        block_template = simulation_model.menu_blocks[0]
        simulation_model.dirty = False

        simulation_model.add_block(block_template, QPoint(100, 100))
        assert simulation_model.dirty == True

    def test_add_multiple_blocks_increments_id(self, simulation_model):
        """Test that adding multiple blocks of same type increments ID."""
        # Find a block type
        block_template = None
        for mb in simulation_model.menu_blocks:
            if mb.block_fn == 'Step':
                block_template = mb
                break

        if block_template is None:
            pytest.skip("Step block not found")

        block1 = simulation_model.add_block(block_template, QPoint(100, 100))
        block2 = simulation_model.add_block(block_template, QPoint(200, 200))

        assert block1.name == 'step0'
        assert block2.name == 'step1'
        assert block1.sid == 0
        assert block2.sid == 1


@pytest.mark.unit
@pytest.mark.qt
class TestAddLine:
    """Test adding connections between blocks."""

    def test_add_line_creates_connection(self, simulation_model, sample_block):
        """Test that add_line creates a DLine instance."""
        # Add two blocks
        simulation_model.blocks_list.append(sample_block)

        from lib.simulation.block import DBlock
        block2 = DBlock(
            'TestBlock2', 1, QRect(300, 300, 100, 80), 'blue',
            1, 1, 2, 'none', 'testblock2', {'gain': 1.0},
            False, colors=simulation_model.colors
        )
        simulation_model.blocks_list.append(block2)

        # Create line data
        src_data = (sample_block.name, 0, sample_block.out_coords[0])
        dst_data = (block2.name, 0, block2.in_coords[0])

        line = simulation_model.add_line(src_data, dst_data)

        assert line is not None
        assert line.srcblock == sample_block.name
        assert line.dstblock == block2.name
        assert line in simulation_model.line_list

    def test_add_line_with_invalid_data_returns_none(self, simulation_model):
        """Test that add_line returns None for invalid data."""
        line = simulation_model.add_line(None, None)
        assert line is None

    def test_add_line_sets_dirty_flag(self, simulation_model, sample_block):
        """Test that adding a line sets dirty flag."""
        simulation_model.blocks_list.append(sample_block)

        from lib.simulation.block import DBlock
        block2 = DBlock(
            'TestBlock2', 1, QRect(300, 300, 100, 80), 'blue',
            1, 1, 2, 'none', 'testblock2', {}, False,
            colors=simulation_model.colors
        )
        simulation_model.blocks_list.append(block2)

        simulation_model.dirty = False

        src_data = (sample_block.name, 0, sample_block.out_coords[0])
        dst_data = (block2.name, 0, block2.in_coords[0])

        simulation_model.add_line(src_data, dst_data)
        assert simulation_model.dirty == True


@pytest.mark.unit
@pytest.mark.qt
class TestRemoveBlock:
    """Test removing blocks from the model."""

    def test_remove_block_removes_from_list(self, simulation_model, sample_block):
        """Test that remove_block removes block from blocks_list."""
        simulation_model.blocks_list.append(sample_block)
        assert sample_block in simulation_model.blocks_list

        simulation_model.remove_block(sample_block)
        assert sample_block not in simulation_model.blocks_list

    def test_remove_block_removes_connected_lines(self, simulation_model, sample_block):
        """Test that removing a block also removes its connections."""
        simulation_model.blocks_list.append(sample_block)

        from lib.simulation.block import DBlock
        block2 = DBlock(
            'TestBlock2', 1, QRect(300, 300, 100, 80), 'blue',
            1, 1, 2, 'none', 'testblock2', {}, False,
            colors=simulation_model.colors
        )
        simulation_model.blocks_list.append(block2)

        # Add a line
        src_data = (sample_block.name, 0, sample_block.out_coords[0])
        dst_data = (block2.name, 0, block2.in_coords[0])
        line = simulation_model.add_line(src_data, dst_data)

        assert len(simulation_model.line_list) == 1

        # Remove the first block
        simulation_model.remove_block(sample_block)

        # Line should be removed too
        assert len(simulation_model.line_list) == 0

    def test_remove_block_sets_dirty_flag(self, simulation_model, sample_block):
        """Test that removing a block sets dirty flag."""
        simulation_model.blocks_list.append(sample_block)
        simulation_model.dirty = False

        simulation_model.remove_block(sample_block)
        assert simulation_model.dirty == True


@pytest.mark.unit
@pytest.mark.qt
class TestRemoveLine:
    """Test removing connections from the model."""

    def test_remove_line_removes_from_list(self, simulation_model, sample_line):
        """Test that remove_line removes line from line_list."""
        simulation_model.line_list.append(sample_line)
        assert sample_line in simulation_model.line_list

        simulation_model.remove_line(sample_line)
        assert sample_line not in simulation_model.line_list

    def test_remove_line_sets_dirty_flag(self, simulation_model, sample_line):
        """Test that removing a line sets dirty flag."""
        simulation_model.line_list.append(sample_line)
        simulation_model.dirty = False

        simulation_model.remove_line(sample_line)
        assert simulation_model.dirty == True


@pytest.mark.unit
@pytest.mark.qt
class TestClearAll:
    """Test clearing all blocks and lines."""

    def test_clear_all_removes_everything(self, simulation_model, sample_block, sample_line):
        """Test that clear_all removes all blocks and lines."""
        simulation_model.blocks_list.append(sample_block)
        simulation_model.line_list.append(sample_line)

        simulation_model.clear_all()

        assert len(simulation_model.blocks_list) == 0
        assert len(simulation_model.line_list) == 0

    def test_clear_all_resets_dirty_flag(self, simulation_model, sample_block):
        """Test that clear_all resets the dirty flag."""
        simulation_model.blocks_list.append(sample_block)
        simulation_model.dirty = True

        simulation_model.clear_all()
        assert simulation_model.dirty == False


@pytest.mark.unit
@pytest.mark.qt
class TestGetBlockByName:
    """Test finding blocks by name."""

    def test_get_block_by_name_finds_existing_block(self, simulation_model, sample_block):
        """Test that get_block_by_name returns the correct block."""
        simulation_model.blocks_list.append(sample_block)

        found = simulation_model.get_block_by_name(sample_block.name)
        assert found == sample_block

    def test_get_block_by_name_returns_none_for_nonexistent(self, simulation_model):
        """Test that get_block_by_name returns None for non-existent block."""
        found = simulation_model.get_block_by_name('nonexistent')
        assert found is None


@pytest.mark.unit
@pytest.mark.qt
class TestIsPortAvailable:
    """Test checking if input ports are available."""

    def test_is_port_available_returns_true_for_unconnected(self, simulation_model, sample_block):
        """Test that is_port_available returns True for unconnected port."""
        simulation_model.blocks_list.append(sample_block)

        dst_data = (sample_block.name, 0, sample_block.in_coords[0])
        assert simulation_model.is_port_available(dst_data) == True

    def test_is_port_available_returns_false_for_connected(self, simulation_model, sample_block):
        """Test that is_port_available returns False for connected port."""
        simulation_model.blocks_list.append(sample_block)

        from lib.simulation.block import DBlock
        block2 = DBlock(
            'TestBlock2', 1, QRect(300, 300, 100, 80), 'blue',
            1, 1, 2, 'none', 'testblock2', {}, False,
            colors=simulation_model.colors
        )
        simulation_model.blocks_list.append(block2)

        # Connect blocks
        src_data = (block2.name, 0, block2.out_coords[0])
        dst_data = (sample_block.name, 0, sample_block.in_coords[0])
        simulation_model.add_line(src_data, dst_data)

        # Port should not be available now
        assert simulation_model.is_port_available(dst_data) == False


@pytest.mark.unit
@pytest.mark.qt
class TestGetDiagramStats:
    """Test diagram statistics retrieval."""

    def test_get_diagram_stats_returns_correct_counts(self, simulation_model, sample_block, sample_line):
        """Test that get_diagram_stats returns accurate statistics."""
        simulation_model.blocks_list.append(sample_block)
        simulation_model.line_list.append(sample_line)
        simulation_model.dirty = True

        stats = simulation_model.get_diagram_stats()

        assert stats['blocks'] == 1
        assert stats['lines'] == 1
        assert stats['modified'] == True
        assert stats['block_types'] >= 1
