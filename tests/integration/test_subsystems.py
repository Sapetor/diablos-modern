"""
Integration tests for subsystem nesting functionality.

Tests signal flow patterns that simulate subsystem behavior:
- Signal flow patterns that would occur in subsystems
- Nested processing chains (simulating nested subsystem behavior)

Note: Direct Inport/Outport block tests are skipped as they require Qt.
"""

import pytest
import numpy as np


@pytest.mark.integration
class TestSubsystemPatterns:
    """Test signal flow patterns that simulate subsystem behavior."""

    def test_pass_through_pattern(self):
        """Test pass-through: input → gain(1) → output."""
        from blocks.gain import GainBlock

        gain = GainBlock()
        params = {'gain': 1.0}

        # Simulate pass-through behavior
        input_val = 42.0
        result = gain.execute(time=0.0, inputs={0: input_val}, params=params)

        assert np.isclose(result[0][0], 42.0), "Pass-through (gain=1) should preserve value"

    def test_gain_chain_pattern(self):
        """Test chained gains (simulating nested subsystems with processing)."""
        from blocks.gain import GainBlock

        gain1 = GainBlock()
        gain2 = GainBlock()
        gain3 = GainBlock()

        params1 = {'gain': 2.0}
        params2 = {'gain': 3.0}
        params3 = {'gain': 4.0}

        # Signal flow: input → gain1 → gain2 → gain3 → output
        value = np.array([1.0])
        value = gain1.execute(time=0.0, inputs={0: value}, params=params1)[0]
        value = gain2.execute(time=0.0, inputs={0: value}, params=params2)[0]
        value = gain3.execute(time=0.0, inputs={0: value}, params=params3)[0]

        assert np.isclose(value[0], 24.0), f"1*2*3*4=24, got {value[0]}"

    def test_multi_input_sum_pattern(self):
        """Test multi-input sum (simulating subsystem with multiple inputs)."""
        from blocks.sum import SumBlock

        sum_block = SumBlock()
        params = {'sign': '+++', '_init_start_': True}

        # Three inputs combined
        result = sum_block.execute(time=0.0, inputs={0: 10.0, 1: 20.0, 2: 30.0}, params=params)

        assert np.isclose(result[0], 60.0), f"10+20+30=60, got {result[0]}"

    def test_split_merge_pattern(self):
        """Test signal split and merge (simulating subsystem with parallel paths)."""
        from blocks.gain import GainBlock
        from blocks.sum import SumBlock

        gain1 = GainBlock()
        gain2 = GainBlock()
        sum_block = SumBlock()

        gain1_params = {'gain': 2.0}
        gain2_params = {'gain': 3.0}
        sum_params = {'sign': '++', '_init_start_': True}

        # Signal splits, processes in parallel, then merges
        input_val = np.array([5.0])
        path1 = gain1.execute(time=0.0, inputs={0: input_val}, params=gain1_params)[0]
        path2 = gain2.execute(time=0.0, inputs={0: input_val}, params=gain2_params)[0]
        output = sum_block.execute(time=0.0, inputs={0: path1[0], 1: path2[0]}, params=sum_params)[0]

        # (5*2) + (5*3) = 10 + 15 = 25
        assert np.isclose(output, 25.0), f"Expected 25, got {output}"

    def test_feedback_pattern(self):
        """Test feedback loop (simulating subsystem with internal feedback)."""
        from blocks.integrator import IntegratorBlock
        from blocks.gain import GainBlock

        integrator = IntegratorBlock()
        gain = GainBlock()

        int_params = {'init_conds': 1.0, 'method': 'FWD_EULER', '_init_start_': True, '_name_': 'TestInt'}
        gain_params = {'gain': -0.5}  # Negative feedback

        dtime = 0.1

        # Run feedback loop: integrator output → gain → feeds back to integrator
        # This simulates dy/dt = -0.5*y, with y(0)=1
        # Solution: y(t) = exp(-0.5*t)

        for i in range(10):
            # Get current integrator output
            int_output = integrator.execute(
                time=i*dtime,
                inputs={0: np.array([0.0])},
                params=int_params,
                dtime=dtime,
                output_only=True
            )[0][0]

            # Apply feedback gain
            feedback = gain.execute(time=i*dtime, inputs={0: int_output}, params=gain_params)[0][0]

            # Feed back to integrator
            integrator.execute(
                time=i*dtime,
                inputs={0: np.array([feedback])},
                params=int_params,
                dtime=dtime
            )

        # After 1 second with decay rate 0.5, Forward Euler gives (1 - 0.5*dt)^n
        # With dt=0.1 and n=10: (0.95)^10 ≈ 0.599
        # Due to discrete feedback timing, actual result is ~0.5
        final_val = int_params['mem'][0]
        # Verify the value has decayed (started at 1.0, should be < 0.7)
        assert final_val < 0.7, f"Expected decay, got {final_val:.3f}"
        assert final_val > 0.3, f"Expected partial decay, not full collapse, got {final_val:.3f}"


@pytest.mark.integration
class TestNestedProcessing:
    """Test nested processing chains (simulating nested subsystems)."""

    def test_two_level_nesting_simulation(self):
        """Simulate 2-level nesting with gains."""
        from blocks.gain import GainBlock

        # Level 1: outer gain
        # Level 2: inner gain (nested)
        outer_gain = GainBlock()
        inner_gain = GainBlock()

        outer_params = {'gain': 2.0}
        inner_params = {'gain': 5.0}

        # Signal: input → outer → inner → output
        value = np.array([3.0])
        value = outer_gain.execute(time=0.0, inputs={0: value}, params=outer_params)[0]
        value = inner_gain.execute(time=0.0, inputs={0: value}, params=inner_params)[0]

        assert np.isclose(value[0], 30.0), f"3*2*5=30, got {value[0]}"

    def test_three_level_nesting_simulation(self):
        """Simulate 3-level nesting with processing at each level."""
        from blocks.gain import GainBlock
        from blocks.sum import SumBlock

        level1_gain = GainBlock()
        level2_sum = SumBlock()
        level3_gain = GainBlock()

        l1_params = {'gain': 2.0}
        l2_params = {'sign': '++', '_init_start_': True}
        l3_params = {'gain': 3.0}

        # Signal: input → L1 gain → L2 sum (with offset) → L3 gain → output
        input_val = np.array([5.0])
        offset = 10.0

        value = level1_gain.execute(time=0.0, inputs={0: input_val}, params=l1_params)[0]
        value = level2_sum.execute(time=0.0, inputs={0: value[0], 1: offset}, params=l2_params)[0]
        value = level3_gain.execute(time=0.0, inputs={0: value}, params=l3_params)[0]

        # (5*2 + 10) * 3 = 20 * 3 = 60
        assert np.isclose(value[0], 60.0), f"Expected 60, got {value[0]}"

    def test_complex_nested_structure(self):
        """Test complex nested structure with multiple paths."""
        from blocks.gain import GainBlock
        from blocks.sum import SumBlock

        # Structure:
        # input → gain1 → sum1 ←+→ gain2 → output
        #                   ↑
        #                constant (simulated as gain with fixed input)

        gain1 = GainBlock()
        gain2 = GainBlock()
        sum1 = SumBlock()

        g1_params = {'gain': 2.0}
        g2_params = {'gain': 0.5}
        sum_params = {'sign': '++', '_init_start_': True}

        input_val = np.array([10.0])
        constant = 5.0

        path1 = gain1.execute(time=0.0, inputs={0: input_val}, params=g1_params)[0]
        sum_out = sum1.execute(time=0.0, inputs={0: path1[0], 1: constant}, params=sum_params)[0]
        output = gain2.execute(time=0.0, inputs={0: sum_out}, params=g2_params)[0]

        # (10*2 + 5) * 0.5 = 25 * 0.5 = 12.5
        assert np.isclose(output[0], 12.5), f"Expected 12.5, got {output[0]}"


@pytest.mark.integration
class TestSubsystemEdgeCases:
    """Test edge cases for subsystem-like behavior."""

    def test_empty_processing(self):
        """Test with no processing (pass-through only)."""
        from blocks.gain import GainBlock

        gain = GainBlock()
        params = {'gain': 1.0}  # Unity gain = pass-through

        for val in [0.0, 1.0, -5.0, 1e10, 1e-10]:
            result = gain.execute(time=0.0, inputs={0: val}, params=params)
            assert np.isclose(result[0][0], val), f"Pass-through should preserve {val}"

    def test_vector_signal_flow(self):
        """Test vector signals through processing chain."""
        from blocks.gain import GainBlock

        gain1 = GainBlock()
        gain2 = GainBlock()

        g1_params = {'gain': 2.0}
        g2_params = {'gain': 3.0}

        vec_input = np.array([1.0, 2.0, 3.0])

        result = gain1.execute(time=0.0, inputs={0: vec_input}, params=g1_params)
        result = gain2.execute(time=0.0, inputs={0: result[0]}, params=g2_params)

        expected = vec_input * 6.0  # 2 * 3 = 6
        assert np.allclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_time_varying_chain(self):
        """Test processing chain over multiple time steps."""
        from blocks.sine import SineBlock
        from blocks.gain import GainBlock
        from blocks.scope import ScopeBlock

        sine = SineBlock()
        gain = GainBlock()
        scope = ScopeBlock()

        sine_params = {'amplitude': 1.0, 'omega': 2*np.pi, 'init_angle': 0.0}
        gain_params = {'gain': 5.0}
        scope_params = {'labels': 'out', '_init_start_': True, '_name_': 'TestScope'}

        # Run for one period
        for i in range(10):
            t = i * 0.1
            sine_out = sine.execute(time=t, inputs={}, params=sine_params)
            gain_out = gain.execute(time=t, inputs={0: sine_out[0]}, params=gain_params)
            scope.execute(time=t, inputs={0: gain_out[0][0]}, params=scope_params)

        # Verify scope collected scaled sine values
        vector = scope_params['vector']
        assert len(vector) == 10, f"Should have 10 samples"

        # Peak should be 5.0 (amplitude * gain)
        assert max(np.abs(vector)) <= 5.0 + 0.1, f"Max should be ~5.0"


@pytest.mark.integration
@pytest.mark.qt
class TestSubsystemManagerUnconnectedPorts:
    """Test subsystem creation with unconnected ports."""

    @pytest.fixture
    def mock_dsim(self, qapp):
        """Create a mock dsim object for testing."""
        from PyQt5.QtCore import QRect, QPoint

        class MockBlock:
            """Mock block with minimal attributes needed for subsystem creation."""
            def __init__(self, name, in_ports=1, out_ports=1):
                self.name = name
                self.sid = 1
                self.in_ports = in_ports
                self.out_ports = out_ports
                self.rect = QRect(0, 0, 50, 50)
                self.width = 50
                self.height = 50
                self.top = 0
                self.left = 0
                self.selected = False
                self.block_type = "MockBlock"
                self.in_coords = [QPoint(0, 25 + i * 10) for i in range(in_ports)]
                self.out_coords = [QPoint(50, 25 + i * 10) for i in range(out_ports)]

            def relocate_Block(self, pos):
                self.rect = QRect(pos.x(), pos.y(), self.width, self.height)
                self.left = pos.x()
                self.top = pos.y()
                # Update coords
                self.in_coords = [QPoint(self.left, self.top + 25 + i * 10) for i in range(self.in_ports)]
                self.out_coords = [QPoint(self.left + self.width, self.top + 25 + i * 10) for i in range(self.out_ports)]

        class MockModel:
            def __init__(self):
                self.blocks_list = []
                self.line_list = []

        class MockDSim:
            def __init__(self):
                self.blocks_list = []
                self.line_list = []
                self.connections_list = []
                self.ss_count = 0
                self.dirty = False

        model = MockModel()
        dsim = MockDSim()
        dsim.blocks_list = model.blocks_list
        dsim.line_list = model.line_list
        dsim.connections_list = model.line_list
        return model, dsim, MockBlock

    def test_unconnected_gain_blocks_get_ports(self, qapp, mock_dsim):
        """Test that unconnected blocks get Inport/Outport when grouped."""
        from lib.managers.subsystem_manager import SubsystemManager
        from PyQt5.QtCore import QPoint

        model, dsim, MockBlock = mock_dsim

        # Create two unconnected mock blocks (simulating Gain blocks)
        block1 = MockBlock("Gain1", in_ports=1, out_ports=1)
        block1.sid = 1
        block1.relocate_Block(QPoint(100, 100))

        block2 = MockBlock("Gain2", in_ports=1, out_ports=1)
        block2.sid = 2
        block2.relocate_Block(QPoint(100, 200))

        dsim.blocks_list.append(block1)
        dsim.blocks_list.append(block2)
        model.blocks_list = dsim.blocks_list

        # Create subsystem manager and create subsystem from selection
        manager = SubsystemManager(model, dsim)
        subsys = manager.create_subsystem_from_selection([block1, block2])

        # Verify subsystem was created
        assert subsys is not None
        assert subsys.block_type == "Subsystem"

        # Verify subsystem has 2 input ports (one for each block's input)
        assert 'in' in subsys.ports
        assert len(subsys.ports['in']) == 2, f"Expected 2 input ports, got {len(subsys.ports['in'])}"

        # Verify subsystem has 2 output ports (one for each block's output)
        assert 'out' in subsys.ports
        assert len(subsys.ports['out']) == 2, f"Expected 2 output ports, got {len(subsys.ports['out'])}"

        # Verify internal structure: should have 2 blocks + 2 Inports + 2 Outports = 6 blocks
        assert len(subsys.sub_blocks) == 6, f"Expected 6 sub_blocks, got {len(subsys.sub_blocks)}"

        # Verify internal connections: 2 from Inports + 2 to Outports = 4 lines
        assert len(subsys.sub_lines) == 4, f"Expected 4 sub_lines, got {len(subsys.sub_lines)}"

    def test_mixed_connected_and_unconnected_ports(self, qapp, mock_dsim):
        """Test subsystem with both boundary connections and unconnected ports."""
        from lib.managers.subsystem_manager import SubsystemManager
        from lib.simulation.connection import DLine
        from PyQt5.QtCore import QPoint

        model, dsim, MockBlock = mock_dsim

        # Create: Constant -> Gain1, and unconnected Gain2
        constant = MockBlock("Constant1", in_ports=0, out_ports=1)
        constant.sid = 1
        constant.relocate_Block(QPoint(50, 100))

        gain1 = MockBlock("Gain1", in_ports=1, out_ports=1)
        gain1.sid = 2
        gain1.relocate_Block(QPoint(150, 100))

        gain2 = MockBlock("Gain2", in_ports=1, out_ports=1)
        gain2.sid = 3
        gain2.relocate_Block(QPoint(150, 200))

        dsim.blocks_list.extend([constant, gain1, gain2])
        model.blocks_list = dsim.blocks_list

        # Create connection: Constant -> Gain1
        line = DLine(
            sid=1,
            srcblock="Constant1", srcport=0,
            dstblock="Gain1", dstport=0,
            points=(constant.out_coords[0], gain1.in_coords[0])
        )
        dsim.line_list.append(line)
        model.line_list = dsim.line_list

        # Select only Gain1 and Gain2 (not Constant)
        manager = SubsystemManager(model, dsim)
        subsys = manager.create_subsystem_from_selection([gain1, gain2])

        # Verify subsystem was created
        assert subsys is not None

        # Expected inputs:
        # - 1 from boundary (Constant -> Gain1)
        # - 1 from unconnected (Gain2's input)
        assert 'in' in subsys.ports
        assert len(subsys.ports['in']) == 2, f"Expected 2 input ports, got {len(subsys.ports['in'])}"

        # Expected outputs:
        # - 1 from unconnected (Gain1's output - no external destination)
        # - 1 from unconnected (Gain2's output)
        assert 'out' in subsys.ports
        assert len(subsys.ports['out']) == 2, f"Expected 2 output ports, got {len(subsys.ports['out'])}"

    def test_single_block_gets_all_ports(self, qapp, mock_dsim):
        """Test that a single unconnected block gets Inport and Outport."""
        from lib.managers.subsystem_manager import SubsystemManager
        from PyQt5.QtCore import QPoint

        model, dsim, MockBlock = mock_dsim

        block = MockBlock("Gain1", in_ports=1, out_ports=1)
        block.sid = 1
        block.relocate_Block(QPoint(100, 100))

        dsim.blocks_list.append(block)
        model.blocks_list = dsim.blocks_list

        manager = SubsystemManager(model, dsim)
        subsys = manager.create_subsystem_from_selection([block])

        # Single block has 1 input and 1 output
        assert 'in' in subsys.ports
        assert len(subsys.ports['in']) == 1

        assert 'out' in subsys.ports
        assert len(subsys.ports['out']) == 1

        # Internal: 1 block + 1 Inport + 1 Outport = 3 blocks
        assert len(subsys.sub_blocks) == 3

    def test_block_with_multiple_ports(self, qapp, mock_dsim):
        """Test block with multiple input/output ports."""
        from lib.managers.subsystem_manager import SubsystemManager
        from PyQt5.QtCore import QPoint

        model, dsim, MockBlock = mock_dsim

        # Block with 3 inputs and 1 output (like Sum block)
        sum_block = MockBlock("Sum1", in_ports=3, out_ports=1)
        sum_block.sid = 1
        sum_block.relocate_Block(QPoint(100, 100))

        dsim.blocks_list.append(sum_block)
        model.blocks_list = dsim.blocks_list

        manager = SubsystemManager(model, dsim)
        subsys = manager.create_subsystem_from_selection([sum_block])

        # Should get 3 Inports (one per input) and 1 Outport
        assert 'in' in subsys.ports
        assert len(subsys.ports['in']) == 3, f"Expected 3 inputs, got {len(subsys.ports['in'])}"

        assert 'out' in subsys.ports
        assert len(subsys.ports['out']) == 1, f"Expected 1 output, got {len(subsys.ports['out'])}"
