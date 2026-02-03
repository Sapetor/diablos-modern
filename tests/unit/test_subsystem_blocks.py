"""
Unit tests for Subsystem, Inport, and Outport blocks.

These blocks are DBlock subclasses used for hierarchical diagram organization:
- Inport: External input to subsystem internals (0 in, 1 out)
- Outport: Internal subsystem output to external (1 in, 0 out)
- Subsystem: Container block with sub_blocks and sub_lines
"""

import pytest
from unittest.mock import MagicMock, patch
from PyQt5.QtCore import QRect


@pytest.mark.unit
class TestInport:
    """Tests for Inport block."""

    def test_inport_creation(self, qapp):
        """Test basic Inport creation."""
        from blocks.inport import Inport
        block = Inport()
        assert block.block_fn == 'Inport'
        assert block.in_ports == 0
        assert block.out_ports == 1

    def test_inport_default_name(self, qapp):
        """Test Inport uses default username 'In1'."""
        from blocks.inport import Inport
        block = Inport()
        assert block.username == 'In1'
        assert block.name == 'inport1'  # Internal name is lowercased

    def test_inport_custom_name(self, qapp):
        """Test Inport with custom username."""
        from blocks.inport import Inport
        block = Inport(block_name='CustomIn')
        assert block.username == 'CustomIn'
        assert block.name == 'inport1'  # Internal name unchanged

    def test_inport_output_port_position(self, qapp):
        """Test Inport has correct output port structure."""
        from blocks.inport import Inport
        block = Inport()
        assert 'out' in block.ports
        assert len(block.ports['out']) == 1
        assert block.ports['out'][0]['type'] == 'output'
        assert block.ports['out'][0]['name'] == '1'

    def test_inport_no_input_ports(self, qapp):
        """Test Inport has no input ports."""
        from blocks.inport import Inport
        block = Inport()
        assert 'in' not in block.ports or not block.ports.get('in')

    def test_inport_update_does_nothing(self, qapp):
        """Test Inport update is a no-op (flattening handles it)."""
        from blocks.inport import Inport
        block = Inport()
        # Should not raise
        block.update(0.0, 0.01)
        block.update(1.0, 0.1)
        block.update(100.0, 1.0)

    def test_inport_dimensions(self, qapp):
        """Test Inport has expected dimensions."""
        from blocks.inport import Inport
        block = Inport()
        assert block.width == 40
        assert block.height == 30

    def test_inport_color(self, qapp):
        """Test Inport default color is green."""
        from blocks.inport import Inport
        from PyQt5.QtGui import QColor
        block = Inport()
        assert block.b_color == QColor('green')

    def test_inport_custom_color(self, qapp):
        """Test Inport with custom color."""
        from blocks.inport import Inport
        from PyQt5.QtGui import QColor
        block = Inport(color='blue')
        assert block.b_color == QColor('blue')

    def test_inport_block_type(self, qapp):
        """Test Inport has correct block_type attribute."""
        from blocks.inport import Inport
        block = Inport()
        assert block.block_type == 'Inport'

    def test_inport_coords_tuple(self, qapp):
        """Test Inport accepts coords as tuple."""
        from blocks.inport import Inport
        block = Inport(coords=(10, 20, 50, 40))
        assert block.left == 10
        assert block.top == 20

    def test_inport_coords_qrect(self, qapp):
        """Test Inport accepts coords as QRect."""
        from blocks.inport import Inport
        rect = QRect(15, 25, 50, 40)
        block = Inport(coords=rect)
        assert block.left == 15
        assert block.top == 25

    def test_inport_sid(self, qapp):
        """Test Inport accepts custom sid."""
        from blocks.inport import Inport
        block = Inport(sid=42)
        assert block.sid == 42


@pytest.mark.unit
class TestOutport:
    """Tests for Outport block."""

    def test_outport_creation(self, qapp):
        """Test basic Outport creation."""
        from blocks.outport import Outport
        block = Outport()
        assert block.block_fn == 'Outport'
        assert block.in_ports == 1
        assert block.out_ports == 0

    def test_outport_default_name(self, qapp):
        """Test Outport uses default username 'Out1'."""
        from blocks.outport import Outport
        block = Outport()
        assert block.username == 'Out1'
        assert block.name == 'outport1'  # Internal name is lowercased

    def test_outport_custom_name(self, qapp):
        """Test Outport with custom username."""
        from blocks.outport import Outport
        block = Outport(block_name='CustomOut')
        assert block.username == 'CustomOut'
        assert block.name == 'outport1'  # Internal name unchanged

    def test_outport_input_port_position(self, qapp):
        """Test Outport has correct input port structure."""
        from blocks.outport import Outport
        block = Outport()
        assert 'in' in block.ports
        assert len(block.ports['in']) == 1
        assert block.ports['in'][0]['type'] == 'input'
        assert block.ports['in'][0]['name'] == '1'

    def test_outport_no_output_ports(self, qapp):
        """Test Outport has no output ports."""
        from blocks.outport import Outport
        block = Outport()
        assert 'out' not in block.ports or not block.ports.get('out')

    def test_outport_update_does_nothing(self, qapp):
        """Test Outport update is a no-op (flattening handles it)."""
        from blocks.outport import Outport
        block = Outport()
        # Should not raise
        block.update(0.0, 0.01)
        block.update(1.0, 0.1)
        block.update(100.0, 1.0)

    def test_outport_dimensions(self, qapp):
        """Test Outport has expected dimensions."""
        from blocks.outport import Outport
        block = Outport()
        assert block.width == 40
        assert block.height == 30

    def test_outport_color(self, qapp):
        """Test Outport default color is red."""
        from blocks.outport import Outport
        from PyQt5.QtGui import QColor
        block = Outport()
        assert block.b_color == QColor('red')

    def test_outport_custom_color(self, qapp):
        """Test Outport with custom color."""
        from blocks.outport import Outport
        from PyQt5.QtGui import QColor
        block = Outport(color='yellow')
        assert block.b_color == QColor('yellow')

    def test_outport_block_type(self, qapp):
        """Test Outport has correct block_type attribute."""
        from blocks.outport import Outport
        block = Outport()
        assert block.block_type == 'Outport'

    def test_outport_coords_tuple(self, qapp):
        """Test Outport accepts coords as tuple."""
        from blocks.outport import Outport
        block = Outport(coords=(30, 40, 60, 50))
        assert block.left == 30
        assert block.top == 40

    def test_outport_coords_qrect(self, qapp):
        """Test Outport accepts coords as QRect."""
        from blocks.outport import Outport
        rect = QRect(35, 45, 60, 50)
        block = Outport(coords=rect)
        assert block.left == 35
        assert block.top == 45

    def test_outport_sid(self, qapp):
        """Test Outport accepts custom sid."""
        from blocks.outport import Outport
        block = Outport(sid=99)
        assert block.sid == 99


@pytest.mark.unit
class TestSubsystem:
    """Tests for Subsystem block."""

    def test_subsystem_creation(self, qapp):
        """Test basic Subsystem creation."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.block_fn == 'Subsystem'
        assert block.sub_blocks == []
        assert block.sub_lines == []

    def test_subsystem_default_ports(self, qapp):
        """Test Subsystem starts with 0 input and 0 output ports."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.in_ports == 0
        assert block.out_ports == 0

    def test_subsystem_update_does_nothing(self, qapp):
        """Test Subsystem update is a no-op (flattening handles execution)."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Should not raise
        block.update(0.0, 0.01)
        block.update(1.0, 0.1)
        block.update(100.0, 1.0)

    def test_subsystem_default_name(self, qapp):
        """Test Subsystem uses default username 'Subsystem'."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.username == 'Subsystem'
        assert block.name == 'subsystem1'  # Internal name is lowercased

    def test_subsystem_custom_name(self, qapp):
        """Test Subsystem with custom username."""
        from blocks.subsystem import Subsystem
        block = Subsystem(block_name='MySubsys')
        assert block.username == 'MySubsys'
        assert block.name == 'subsystem1'  # Internal name unchanged

    def test_subsystem_block_type(self, qapp):
        """Test Subsystem has correct block_type attribute."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.block_type == 'Subsystem'

    def test_subsystem_color(self, qapp):
        """Test Subsystem default color is lightgray."""
        from blocks.subsystem import Subsystem
        from PyQt5.QtGui import QColor
        block = Subsystem()
        assert block.b_color == QColor('lightgray')

    def test_subsystem_custom_color(self, qapp):
        """Test Subsystem with custom color."""
        from blocks.subsystem import Subsystem
        from PyQt5.QtGui import QColor
        block = Subsystem(color='cyan')
        assert block.b_color == QColor('cyan')

    def test_subsystem_ports_map(self, qapp):
        """Test Subsystem has ports_map for port synchronization."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert hasattr(block, 'ports_map')
        assert isinstance(block.ports_map, dict)
        assert block.ports_map == {}

    def test_subsystem_params(self, qapp):
        """Test Subsystem has params dict."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert hasattr(block, 'params')
        assert isinstance(block.params, dict)
        assert block.params == {}

    def test_subsystem_io_edit(self, qapp):
        """Test Subsystem has io_edit enabled (for dynamic port config)."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.io_edit == True

    def test_subsystem_coords_tuple(self, qapp):
        """Test Subsystem accepts coords as tuple."""
        from blocks.subsystem import Subsystem
        block = Subsystem(coords=(50, 60, 150, 120))
        assert block.left == 50
        assert block.top == 60

    def test_subsystem_coords_qrect(self, qapp):
        """Test Subsystem accepts coords as QRect."""
        from blocks.subsystem import Subsystem
        rect = QRect(55, 65, 150, 120)
        block = Subsystem(coords=rect)
        assert block.left == 55
        assert block.top == 65

    def test_subsystem_sid(self, qapp):
        """Test Subsystem accepts custom sid."""
        from blocks.subsystem import Subsystem
        block = Subsystem(sid=123)
        assert block.sid == 123

    def test_subsystem_default_dimensions(self, qapp):
        """Test Subsystem has larger default size than Inport/Outport."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Default coords are (0,0,100,80)
        assert block.width == 100
        assert block.height == 80

    def test_subsystem_sub_blocks_list(self, qapp):
        """Test Subsystem sub_blocks is a mutable list."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Should be able to append
        mock_block = MagicMock()
        block.sub_blocks.append(mock_block)
        assert len(block.sub_blocks) == 1
        assert block.sub_blocks[0] == mock_block

    def test_subsystem_sub_lines_list(self, qapp):
        """Test Subsystem sub_lines is a mutable list."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Should be able to append
        mock_line = MagicMock()
        block.sub_lines.append(mock_line)
        assert len(block.sub_lines) == 1
        assert block.sub_lines[0] == mock_line

    def test_subsystem_b_type(self, qapp):
        """Test Subsystem has b_type=2 (container type)."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        assert block.b_type == 2

    def test_subsystem_update_block_no_ports(self, qapp):
        """Test Subsystem.update_Block with no custom ports uses default behavior."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Should not raise
        block.update_Block()

    def test_subsystem_update_block_with_input_ports(self, qapp):
        """Test Subsystem.update_Block with input ports distributes them evenly."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Add custom input ports
        block.ports = {
            'in': [
                {'pos': (0, 20), 'type': 'input', 'name': '1'},
                {'pos': (0, 60), 'type': 'input', 'name': '2'}
            ]
        }
        block.update_Block()

        assert len(block.in_coords) == 2
        assert block.in_ports == 2

    def test_subsystem_update_block_with_output_ports(self, qapp):
        """Test Subsystem.update_Block with output ports distributes them evenly."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Add custom output ports
        block.ports = {
            'out': [
                {'pos': (100, 20), 'type': 'output', 'name': '1'},
                {'pos': (100, 60), 'type': 'output', 'name': '2'}
            ]
        }
        block.update_Block()

        assert len(block.out_coords) == 2
        assert block.out_ports == 2

    def test_subsystem_update_block_with_mixed_ports(self, qapp):
        """Test Subsystem.update_Block with both input and output ports."""
        from blocks.subsystem import Subsystem
        block = Subsystem()
        # Add custom ports
        block.ports = {
            'in': [
                {'pos': (0, 40), 'type': 'input', 'name': '1'}
            ],
            'out': [
                {'pos': (100, 40), 'type': 'output', 'name': '1'}
            ]
        }
        block.update_Block()

        assert len(block.in_coords) == 1
        assert len(block.out_coords) == 1
        assert block.in_ports == 1
        assert block.out_ports == 1


@pytest.mark.unit
class TestSubsystemBlockIntegration:
    """Integration tests for Inport/Outport within Subsystems."""

    def test_inport_outport_within_subsystem(self, qapp):
        """Test typical pattern: Inport and Outport as children of Subsystem."""
        from blocks.subsystem import Subsystem
        from blocks.inport import Inport
        from blocks.outport import Outport

        subsys = Subsystem(block_name='TestSys')
        inport = Inport(block_name='In1')
        outport = Outport(block_name='Out1')

        # Simulate adding to subsystem
        subsys.sub_blocks.append(inport)
        subsys.sub_blocks.append(outport)

        assert len(subsys.sub_blocks) == 2
        assert subsys.sub_blocks[0].block_type == 'Inport'
        assert subsys.sub_blocks[1].block_type == 'Outport'

    def test_multiple_inports_in_subsystem(self, qapp):
        """Test subsystem with multiple Inports."""
        from blocks.subsystem import Subsystem
        from blocks.inport import Inport

        subsys = Subsystem()
        in1 = Inport(block_name='In1')
        in2 = Inport(block_name='In2', sid=2)
        in3 = Inport(block_name='In3', sid=3)

        subsys.sub_blocks.extend([in1, in2, in3])

        assert len(subsys.sub_blocks) == 3
        assert all(b.block_type == 'Inport' for b in subsys.sub_blocks)
        assert subsys.sub_blocks[0].username == 'In1'
        assert subsys.sub_blocks[1].username == 'In2'
        assert subsys.sub_blocks[2].username == 'In3'

    def test_multiple_outports_in_subsystem(self, qapp):
        """Test subsystem with multiple Outports."""
        from blocks.subsystem import Subsystem
        from blocks.outport import Outport

        subsys = Subsystem()
        out1 = Outport(block_name='Out1')
        out2 = Outport(block_name='Out2', sid=2)

        subsys.sub_blocks.extend([out1, out2])

        assert len(subsys.sub_blocks) == 2
        assert all(b.block_type == 'Outport' for b in subsys.sub_blocks)
        assert subsys.sub_blocks[0].username == 'Out1'
        assert subsys.sub_blocks[1].username == 'Out2'

    def test_subsystem_with_internal_connections(self, qapp):
        """Test subsystem with sub_lines (internal connections)."""
        from blocks.subsystem import Subsystem

        subsys = Subsystem()
        mock_line1 = MagicMock()
        mock_line2 = MagicMock()

        subsys.sub_lines.extend([mock_line1, mock_line2])

        assert len(subsys.sub_lines) == 2

    def test_nested_subsystems_structure(self, qapp):
        """Test that subsystems can contain other subsystems."""
        from blocks.subsystem import Subsystem

        parent = Subsystem(block_name='Parent')
        child = Subsystem(block_name='Child', sid=2)

        parent.sub_blocks.append(child)

        assert len(parent.sub_blocks) == 1
        assert parent.sub_blocks[0].block_type == 'Subsystem'
        assert parent.sub_blocks[0].username == 'Child'
