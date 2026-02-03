"""
Unit tests for SimulationEngine class.
"""

import pytest
from PyQt5.QtCore import QRect

from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


@pytest.mark.unit
@pytest.mark.qt
class TestSimulationEngineInitialization:
    """Test SimulationEngine initialization."""

    def test_engine_initializes_with_model(self, simulation_engine, simulation_model):
        """Test that engine initializes with a model reference."""
        assert simulation_engine.model == simulation_model

    def test_engine_has_initial_state(self, simulation_engine):
        """Test that engine has correct initial state."""
        assert simulation_engine.execution_initialized == False
        assert simulation_engine.execution_pause == False
        assert simulation_engine.execution_stop == False
        assert simulation_engine.error_msg == ""

    def test_engine_has_default_simulation_params(self, simulation_engine):
        """Test that engine has default simulation parameters."""
        assert simulation_engine.sim_time == 1.0
        assert simulation_engine.sim_dt == 0.01
        assert simulation_engine.real_time == True


@pytest.mark.unit
@pytest.mark.qt
class TestGetNeighbors:
    """Test getting block neighbors (input/output connections)."""

    def test_get_neighbors_returns_empty_for_unconnected_block(self, simulation_engine, simulation_model, sample_block):
        """Test that get_neighbors returns empty lists for unconnected block."""
        simulation_model.blocks_list.append(sample_block)

        inputs, outputs = simulation_engine.get_neighbors(sample_block.name)

        assert inputs == []
        assert outputs == []

    def test_get_neighbors_returns_inputs(self, simulation_engine, simulation_model):
        """Test that get_neighbors correctly identifies input connections."""
        # Create two blocks
        block1 = DBlock('Block1', 0, QRect(100, 100, 100, 80), 'red',
                       1, 1, 2, 'none', 'block1', {}, False,
                       colors=simulation_model.colors)
        block2 = DBlock('Block2', 0, QRect(300, 300, 100, 80), 'blue',
                       1, 1, 2, 'none', 'block2', {}, False,
                       colors=simulation_model.colors)

        simulation_model.blocks_list.extend([block1, block2])

        # Create connection: block1 -> block2
        line = DLine(0, block1.name, 0, block2.name, 0,
                    [block1.out_coords[0], block2.in_coords[0]])
        simulation_model.line_list.append(line)

        # Check block2's inputs
        inputs, outputs = simulation_engine.get_neighbors(block2.name)

        assert len(inputs) == 1
        assert inputs[0]['srcblock'] == block1.name
        assert inputs[0]['srcport'] == 0
        assert inputs[0]['dstport'] == 0

    def test_get_neighbors_returns_outputs(self, simulation_engine, simulation_model):
        """Test that get_neighbors correctly identifies output connections."""
        # Create two blocks
        block1 = DBlock('Block1', 0, QRect(100, 100, 100, 80), 'red',
                       1, 1, 2, 'none', 'block1', {}, False,
                       colors=simulation_model.colors)
        block2 = DBlock('Block2', 0, QRect(300, 300, 100, 80), 'blue',
                       1, 1, 2, 'none', 'block2', {}, False,
                       colors=simulation_model.colors)

        simulation_model.blocks_list.extend([block1, block2])

        # Create connection: block1 -> block2
        line = DLine(0, block1.name, 0, block2.name, 0,
                    [block1.out_coords[0], block2.in_coords[0]])
        simulation_model.line_list.append(line)

        # Check block1's outputs
        inputs, outputs = simulation_engine.get_neighbors(block1.name)

        assert len(outputs) == 1
        assert outputs[0]['dstblock'] == block2.name
        assert outputs[0]['srcport'] == 0
        assert outputs[0]['dstport'] == 0


@pytest.mark.unit
@pytest.mark.qt
class TestGetOutputs:
    """Test getting block output connections."""

    def test_get_outputs_returns_empty_for_unconnected_block(self, simulation_engine, simulation_model, sample_block):
        """Test that get_outputs returns empty list for unconnected block."""
        simulation_model.blocks_list.append(sample_block)

        outputs = simulation_engine.get_outputs(sample_block.name)
        assert outputs == []

    def test_get_outputs_returns_multiple_connections(self, simulation_engine, simulation_model):
        """Test that get_outputs handles multiple output connections."""
        # Create one source and two destination blocks
        source = DBlock('Source', 0, QRect(100, 100, 100, 80), 'red',
                       0, 2, 2, 'none', 'source', {}, False,
                       colors=simulation_model.colors)
        dest1 = DBlock('Dest1', 0, QRect(300, 100, 100, 80), 'blue',
                      1, 1, 2, 'none', 'dest1', {}, False,
                      colors=simulation_model.colors)
        dest2 = DBlock('Dest2', 0, QRect(300, 300, 100, 80), 'green',
                      1, 1, 2, 'none', 'dest2', {}, False,
                      colors=simulation_model.colors)

        simulation_model.blocks_list.extend([source, dest1, dest2])

        # Create connections from source to both destinations
        line1 = DLine(0, source.name, 0, dest1.name, 0,
                     [source.out_coords[0], dest1.in_coords[0]])
        line2 = DLine(1, source.name, 1, dest2.name, 0,
                     [source.out_coords[1], dest2.in_coords[0]])
        simulation_model.line_list.extend([line1, line2])

        outputs = simulation_engine.get_outputs(source.name)

        assert len(outputs) == 2
        output_blocks = [o['dstblock'] for o in outputs]
        assert dest1.name in output_blocks
        assert dest2.name in output_blocks


@pytest.mark.unit
@pytest.mark.qt
class TestCheckDiagramIntegrity:
    """Test diagram validation."""

    def test_check_diagram_integrity_passes_for_valid_diagram(self, simulation_engine, simulation_model):
        """Test that check_diagram_integrity returns True for valid diagram."""
        # Create a simple valid diagram: source -> sink
        source = DBlock('Source', 0, QRect(100, 100, 100, 80), 'red',
                       0, 1, 2, 'none', 'source', {}, False,
                       colors=simulation_model.colors)
        sink = DBlock('Sink', 0, QRect(300, 300, 100, 80), 'blue',
                     1, 0, 2, 'none', 'sink', {}, False,
                     colors=simulation_model.colors)

        simulation_model.blocks_list.extend([source, sink])

        line = DLine(0, source.name, 0, sink.name, 0,
                    [source.out_coords[0], sink.in_coords[0]])
        simulation_model.line_list.append(line)

        result = simulation_engine.check_diagram_integrity()
        assert result == True

    def test_check_diagram_integrity_fails_for_unconnected_input(self, simulation_engine, simulation_model):
        """Test that check_diagram_integrity detects unconnected inputs."""
        # Create block with unconnected input
        block = DBlock('Block', 0, QRect(100, 100, 100, 80), 'red',
                      1, 1, 2, 'none', 'block', {}, False,
                      colors=simulation_model.colors)
        simulation_model.blocks_list.append(block)

        result = simulation_engine.check_diagram_integrity()
        assert result == False

    def test_check_diagram_integrity_fails_for_unconnected_output(self, simulation_engine, simulation_model):
        """Test that check_diagram_integrity detects unconnected outputs."""
        # Create block with unconnected output
        block = DBlock('Block', 0, QRect(100, 100, 100, 80), 'red',
                      1, 1, 2, 'none', 'block', {}, False,
                      colors=simulation_model.colors)
        simulation_model.blocks_list.append(block)

        # Connect input but not output
        source = DBlock('Source', 0, QRect(50, 100, 100, 80), 'blue',
                       0, 1, 2, 'none', 'source', {}, False,
                       colors=simulation_model.colors)
        simulation_model.blocks_list.append(source)

        line = DLine(0, source.name, 0, block.name, 0,
                    [source.out_coords[0], block.in_coords[0]])
        simulation_model.line_list.append(line)

        result = simulation_engine.check_diagram_integrity()
        assert result == False


@pytest.mark.unit
@pytest.mark.qt
class TestGetMaxHierarchy:
    """Test finding maximum hierarchy level."""

    def test_get_max_hierarchy_returns_minus_one_for_uninitialized(self, simulation_engine, simulation_model, sample_block):
        """Test that get_max_hierarchy returns -1 when blocks haven't been hierarchized."""
        simulation_model.blocks_list.append(sample_block)

        max_h = simulation_engine.get_max_hierarchy()
        assert max_h == -1

    def test_get_max_hierarchy_returns_correct_value(self, simulation_engine, simulation_model):
        """Test that get_max_hierarchy returns the highest hierarchy value."""
        block1 = DBlock('Block1', 0, QRect(100, 100, 100, 80), 'red',
                       1, 1, 2, 'none', 'block1', {}, False,
                       colors=simulation_model.colors)
        block2 = DBlock('Block2', 0, QRect(200, 200, 100, 80), 'blue',
                       1, 1, 2, 'none', 'block2', {}, False,
                       colors=simulation_model.colors)
        block3 = DBlock('Block3', 0, QRect(300, 300, 100, 80), 'green',
                       1, 1, 2, 'none', 'block3', {}, False,
                       colors=simulation_model.colors)

        block1.hierarchy = 0
        block2.hierarchy = 1
        block3.hierarchy = 2

        simulation_model.blocks_list.extend([block1, block2, block3])
        # Also populate active_blocks_list which is what get_max_hierarchy uses
        simulation_engine.active_blocks_list = [block1, block2, block3]

        max_h = simulation_engine.get_max_hierarchy()
        assert max_h == 2


@pytest.mark.unit
@pytest.mark.qt
class TestResetExecutionData:
    """Test resetting execution state."""

    def test_reset_execution_data_clears_state(self, simulation_engine, simulation_model):
        """Test that reset_execution_data clears all execution-related state."""
        block = DBlock('Block', 0, QRect(100, 100, 100, 80), 'red',
                      2, 2, 2, 'none', 'block', {}, False,
                      colors=simulation_model.colors)

        # Set some execution state
        block.computed_data = True
        block.data_recieved = 5
        block.data_sent = 3
        block.hierarchy = 2
        block.input_queue = {0: 1.0, 1: 2.0}

        simulation_model.blocks_list.append(block)
        # Also populate active_blocks_list which is what reset_execution_data uses
        simulation_engine.active_blocks_list = [block]

        simulation_engine.reset_execution_data()

        assert block.computed_data == False
        assert block.data_recieved == 0
        assert block.data_sent == 0
        # When called before global_computed_list is set up, hierarchy is reset to -1
        # and input_queue is set to empty dict (DSim behavior)
        assert block.hierarchy == -1
        assert block.input_queue == {}


@pytest.mark.unit
@pytest.mark.qt
class TestUpdateSimParams:
    """Test updating simulation parameters."""

    def test_update_sim_params_sets_values(self, simulation_engine):
        """Test that update_sim_params sets simulation time and dt."""
        simulation_engine.update_sim_params(10.0, 0.001)

        assert simulation_engine.sim_time == 10.0
        assert simulation_engine.sim_dt == 0.001


@pytest.mark.unit
@pytest.mark.qt
class TestGetExecutionStatus:
    """Test retrieving execution status."""

    def test_get_execution_status_returns_current_state(self, simulation_engine):
        """Test that get_execution_status returns correct state dict."""
        simulation_engine.execution_initialized = True
        simulation_engine.execution_pause = False
        simulation_engine.execution_stop = False
        simulation_engine.error_msg = ""
        simulation_engine.sim_time = 5.0
        simulation_engine.sim_dt = 0.01

        status = simulation_engine.get_execution_status()

        assert status['initialized'] == True
        assert status['paused'] == False
        assert status['stopped'] == False
        assert status['error'] is None
        assert status['sim_time'] == 5.0
        assert status['sim_dt'] == 0.01

    def test_get_execution_status_includes_error_message(self, simulation_engine):
        """Test that get_execution_status includes error messages."""
        simulation_engine.error_msg = "Test error"

        status = simulation_engine.get_execution_status()

        assert status['error'] == "Test error"
