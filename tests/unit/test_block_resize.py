"""
Unit tests for block resize behavior.

Tests port alignment and dimension consistency when resizing blocks,
particularly for blocks with multiple input/output ports.
"""

import pytest
from PyQt5.QtCore import QRect, QPoint

from lib.simulation.block import DBlock
from blocks.subsystem import Subsystem


@pytest.fixture
def sample_colors():
    """Provide sample color palette for tests."""
    from PyQt5.QtGui import QColor
    return {
        'black': QColor(0, 0, 0),
        'red': QColor(255, 0, 0),
        'blue': QColor(0, 0, 255),
    }


@pytest.fixture
def multi_port_block(qapp, sample_colors):
    """Create a block with multiple input and output ports."""
    block = DBlock(
        block_fn='MultiPort',
        sid=0,
        coords=QRect(100, 100, 100, 100),
        color='red',
        in_ports=3,
        out_ports=2,
        b_type=2,
        io_edit='none',
        fn_name='multiport',
        params={},
        external=False,
        colors=sample_colors
    )
    return block


@pytest.fixture
def subsystem_block(qapp):
    """Create a subsystem with multiple ports."""
    subsys = Subsystem(
        block_name="TestSubsystem",
        sid=1,
        coords=(100, 100, 120, 100),
        color="lightgray"
    )
    # Add port definitions like when created from selection
    subsys.ports = {
        'in': [
            {'pos': (0, 33), 'type': 'input', 'name': '1'},
            {'pos': (0, 66), 'type': 'input', 'name': '2'},
        ],
        'out': [
            {'pos': (120, 50), 'type': 'output', 'name': '1'},
        ]
    }
    subsys.in_ports = 2
    subsys.out_ports = 1
    subsys.update_Block()
    return subsys


class TestBlockRectConsistency:
    """Test that block.rect stays synchronized with block dimensions."""

    @pytest.mark.unit
    def test_rect_matches_dimensions_after_resize(self, multi_port_block):
        """After resize_Block, rect should match width/height."""
        block = multi_port_block

        block.resize_Block(150, 120)

        assert block.rect.width() == block.width
        assert block.rect.height() == block.height
        assert block.width == 150
        assert block.height == 120

    @pytest.mark.unit
    def test_rect_matches_after_minimum_height_enforcement(self, multi_port_block):
        """When minimum height is enforced, rect should still match."""
        block = multi_port_block

        # Try to resize below minimum (multi-port blocks have port-based minimum)
        block.resize_Block(100, 20)  # Very small height

        # Height should be enforced to minimum, and rect should match
        assert block.rect.height() == block.height
        assert block.height >= block.calculate_min_size()

    @pytest.mark.unit
    def test_rectf_includes_port_radius(self, multi_port_block):
        """rectf should extend beyond rect to include port drawing area."""
        block = multi_port_block

        block.resize_Block(100, 100)

        assert block.rectf.left() == block.left - block.port_radius
        assert block.rectf.width() == block.width + 2 * block.port_radius
        assert block.rectf.height() == block.height


class TestMultiPortMinimumHeight:
    """Test minimum height calculation for multi-port blocks."""

    @pytest.mark.unit
    def test_single_port_no_special_minimum(self, qapp, sample_colors):
        """Single-port blocks should use base height as minimum."""
        block = DBlock(
            block_fn='SinglePort',
            sid=0,
            coords=QRect(0, 0, 100, 80),
            color='red',
            in_ports=1,
            out_ports=1,
            b_type=2,
            io_edit='none',
            fn_name='singleport',
            params={},
            external=False,
            colors=sample_colors
        )

        min_height = block.calculate_min_size()
        assert min_height == block.height_base

    @pytest.mark.unit
    def test_multi_port_has_calculated_minimum(self, multi_port_block):
        """Multi-port blocks should have minimum based on port count."""
        block = multi_port_block

        min_height = block.calculate_min_size()

        # With 3 inputs (max ports), minimum should be > base height
        # Formula: (max_ports * PORT_SPACING) + (PORT_MARGIN * 2)
        # = (3 * 20) + (12 * 2) = 84
        assert min_height >= 84


class TestPortPositionScaling:
    """Test that port positions scale correctly during resize."""

    @pytest.mark.unit
    def test_ports_evenly_distributed_after_resize(self, multi_port_block):
        """Ports should be evenly distributed along block height after resize."""
        block = multi_port_block

        block.resize_Block(100, 200)

        # With 3 input ports and height 200:
        # Port 0: y = 100 + 200 * 1/4 = 150
        # Port 1: y = 100 + 200 * 2/4 = 200
        # Port 2: y = 100 + 200 * 3/4 = 250
        expected_y = [150, 200, 250]

        for i, coord in enumerate(block.in_coords):
            assert coord.y() == expected_y[i], f"Port {i} Y position mismatch"

    @pytest.mark.unit
    def test_ports_maintain_proportional_positions(self, multi_port_block):
        """Port positions should maintain proportions when resizing."""
        block = multi_port_block

        # Resize to different height
        block.resize_Block(100, 150)
        positions_150 = [c.y() - block.top for c in block.in_coords]

        block.resize_Block(100, 300)
        positions_300 = [c.y() - block.top for c in block.in_coords]

        # Ratios should be the same (1/4, 2/4, 3/4 of height)
        for i in range(len(positions_150)):
            ratio_150 = positions_150[i] / 150
            ratio_300 = positions_300[i] / 300
            assert abs(ratio_150 - ratio_300) < 0.01, f"Port {i} ratio changed"


class TestSubsystemPortScaling:
    """Test that subsystem ports scale correctly during resize."""

    @pytest.mark.unit
    def test_subsystem_ports_scale_with_height(self, subsystem_block):
        """Subsystem ports should scale proportionally with height."""
        subsys = subsystem_block
        original_height = subsys.height

        # Get original port proportions
        original_ratios = [
            (c.y() - subsys.top) / original_height
            for c in subsys.in_coords
        ]

        # Resize to new height
        subsys.resize_Block(120, 200)

        # Port proportions should be maintained (evenly distributed)
        new_ratios = [
            (c.y() - subsys.top) / subsys.height
            for c in subsys.in_coords
        ]

        # With 2 ports, should be at 1/3 and 2/3
        assert abs(new_ratios[0] - 1/3) < 0.01
        assert abs(new_ratios[1] - 2/3) < 0.01

    @pytest.mark.unit
    def test_subsystem_stored_positions_updated(self, subsystem_block):
        """Stored port positions in subsystem.ports should be updated on resize."""
        subsys = subsystem_block

        subsys.resize_Block(120, 150)

        # Check that stored positions match calculated positions
        for i, port_def in enumerate(subsys.ports['in']):
            stored_y = port_def['pos'][1]
            expected_y = subsys.height * (i + 1) / (len(subsys.ports['in']) + 1)
            assert abs(stored_y - expected_y) < 0.01

    @pytest.mark.unit
    def test_subsystem_rect_updated_on_resize(self, subsystem_block):
        """Subsystem rect should be updated correctly on resize."""
        subsys = subsystem_block

        subsys.resize_Block(150, 120)

        assert subsys.rect.width() == subsys.width == 150
        assert subsys.rect.height() == subsys.height == 120


class TestPortGridSnapping:
    """Test port grid snapping behavior."""

    @pytest.mark.unit
    def test_grid_snapping_disabled_by_default(self, multi_port_block):
        """Grid snapping should be disabled by default for smooth resize."""
        block = multi_port_block

        # Resize to height that would cause grid snap jumps if snapping was enabled
        block.resize_Block(100, 97)  # 97/4 = 24.25 - would snap to 20 or 30

        # Port position should be exact (with int truncation), not grid-snapped
        # With 3 input ports: first port at height * 1/4 = 97 * 0.25 = 24.25 -> 24
        expected_relative_y = int(97 * 1 / 4)  # = 24
        actual_relative_y = block.in_coords[0].y() - block.top

        # If grid snapping was on, this would be 20 or 30, not 24
        assert actual_relative_y == expected_relative_y, \
            f"Expected {expected_relative_y}, got {actual_relative_y} (grid snapping may be on)"
