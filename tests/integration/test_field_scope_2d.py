"""
Integration tests for FieldScope2D verification.

Tests that 2D field visualization blocks correctly:
- Store field snapshots during simulation
- Map time slider to correct time values
- Handle 2D data shapes properly
"""

import pytest
import numpy as np


@pytest.mark.integration
class TestFieldScope2D:
    """Tests for FieldScope2D block."""

    def test_field_scope_2d_initialization(self):
        """Test FieldScope2D initializes storage correctly."""
        from blocks.pde.field_processing_2d import FieldScope2DBlock

        block = FieldScope2DBlock()
        params = {
            'Lx': 1.0,
            'Ly': 1.0,
            'colormap': 'viridis',
            'sample_interval': 1,
            '_init_start_': True
        }

        # Create a test field
        field = np.random.rand(10, 10)

        block.execute(time=0.0, inputs={0: field}, params=params)

        assert '_field_history_2d_' in params, "Should initialize field history"
        assert '_time_history_' in params, "Should initialize time history"
        assert params['_init_start_'] == False, "Init flag should be cleared"

    def test_field_scope_2d_collects_snapshots(self):
        """Test FieldScope2D collects field snapshots over time."""
        from blocks.pde.field_processing_2d import FieldScope2DBlock

        block = FieldScope2DBlock()
        params = {
            'sample_interval': 1,  # Store every frame
            '_init_start_': True
        }

        # Simulate 5 timesteps with different fields
        for i in range(5):
            field = np.full((10, 10), float(i))
            block.execute(time=i * 0.1, inputs={0: field}, params=params)

        history = params['_field_history_2d_']
        times = params['_time_history_']

        assert len(history) == 5, f"Should have 5 snapshots, got {len(history)}"
        assert len(times) == 5, f"Should have 5 time values, got {len(times)}"
        assert np.allclose(times, [0.0, 0.1, 0.2, 0.3, 0.4]), "Times should match simulation"

    def test_field_scope_2d_sample_interval(self):
        """Test FieldScope2D respects sample_interval parameter."""
        from blocks.pde.field_processing_2d import FieldScope2DBlock

        block = FieldScope2DBlock()
        params = {
            'sample_interval': 3,  # Store every 3rd frame
            '_init_start_': True
        }

        # Simulate 10 timesteps
        for i in range(10):
            field = np.full((5, 5), float(i))
            block.execute(time=i * 0.1, inputs={0: field}, params=params)

        history = params['_field_history_2d_']

        # With sample_interval=3, we get snapshots at frames 3, 6, 9 (0-indexed)
        # Actually the counter increments first, so: frames 2, 5, 8 stored
        assert len(history) == 3, f"Should have 3 snapshots with interval=3, got {len(history)}"

    def test_field_scope_2d_preserves_shape(self):
        """Test FieldScope2D preserves field shape correctly."""
        from blocks.pde.field_processing_2d import FieldScope2DBlock

        block = FieldScope2DBlock()
        params = {
            'sample_interval': 1,
            '_init_start_': True
        }

        # Use non-square field
        field = np.random.rand(15, 20)  # Ny=15, Nx=20
        block.execute(time=0.0, inputs={0: field}, params=params)

        stored_field = params['_field_history_2d_'][0]
        assert stored_field.shape == (15, 20), f"Should preserve shape (15, 20), got {stored_field.shape}"

    def test_field_scope_2d_time_mapping(self):
        """Test time values are correctly stored for slider mapping."""
        from blocks.pde.field_processing_2d import FieldScope2DBlock

        block = FieldScope2DBlock()
        params = {
            'sample_interval': 1,
            '_init_start_': True
        }

        # Simulate with irregular time steps
        times = [0.0, 0.1, 0.25, 0.5, 1.0]
        for t in times:
            field = np.ones((5, 5)) * t
            block.execute(time=t, inputs={0: field}, params=params)

        stored_times = params['_time_history_']
        assert stored_times == times, f"Stored times should match input times"

        # Verify field values match times
        for i, t in enumerate(times):
            stored_field = params['_field_history_2d_'][i]
            assert np.allclose(stored_field, t), f"Field at t={t} should have value {t}"


@pytest.mark.integration
class TestFieldProbe2D:
    """Tests for FieldProbe2D block."""

    def test_field_probe_2d_center(self):
        """Test probing center of a field."""
        from blocks.pde.field_processing_2d import FieldProbe2DBlock

        block = FieldProbe2DBlock()
        params = {
            'x_position': 0.5,
            'y_position': 0.5,
            'position_mode': 'normalized'
        }

        # Create field with known value at center
        field = np.zeros((11, 11))
        field[5, 5] = 10.0  # Center value

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        assert np.isclose(result[0], 10.0), f"Should get center value 10.0, got {result[0]}"

    def test_field_probe_2d_corner(self):
        """Test probing corner of a field."""
        from blocks.pde.field_processing_2d import FieldProbe2DBlock

        block = FieldProbe2DBlock()
        params = {
            'x_position': 0.0,
            'y_position': 0.0,
            'position_mode': 'normalized'
        }

        field = np.zeros((10, 10))
        field[0, 0] = 5.0  # Bottom-left corner

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        assert np.isclose(result[0], 5.0), f"Should get corner value 5.0, got {result[0]}"

    def test_field_probe_2d_interpolation(self):
        """Test bilinear interpolation between nodes."""
        from blocks.pde.field_processing_2d import FieldProbe2DBlock

        block = FieldProbe2DBlock()
        params = {
            'x_position': 0.5,
            'y_position': 0.5,
            'position_mode': 'normalized'
        }

        # Create 2x2 field with different corner values
        field = np.array([
            [0.0, 2.0],
            [2.0, 4.0]
        ])

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        # At center (0.5, 0.5), bilinear interpolation gives average
        expected = (0.0 + 2.0 + 2.0 + 4.0) / 4
        assert np.isclose(result[0], expected), f"Expected {expected}, got {result[0]}"

    def test_field_probe_2d_absolute_mode(self):
        """Test absolute position mode."""
        from blocks.pde.field_processing_2d import FieldProbe2DBlock

        block = FieldProbe2DBlock()
        params = {
            'x_position': 0.5,  # Absolute meters
            'y_position': 0.5,
            'position_mode': 'absolute',
            'Lx': 1.0,
            'Ly': 1.0
        }

        field = np.zeros((11, 11))
        field[5, 5] = 7.0  # Center

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        assert np.isclose(result[0], 7.0), f"Absolute mode should find center, got {result[0]}"


@pytest.mark.integration
class TestFieldSlice:
    """Tests for FieldSlice block."""

    def test_field_slice_horizontal(self):
        """Test horizontal slice (constant y)."""
        from blocks.pde.field_processing_2d import FieldSliceBlock

        block = FieldSliceBlock()
        params = {
            'slice_direction': 'x',
            'slice_position': 0.5  # Middle row
        }

        # Create field where each row has unique values
        field = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [4, 4, 4, 4]
        ])

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        # At position 0.5 with Ny=4, j = int(0.5 * 3) = 1, so row index 1
        expected = np.array([2, 2, 2, 2])
        assert np.allclose(result[0], expected), f"Expected row [2,2,2,2], got {result[0]}"

    def test_field_slice_vertical(self):
        """Test vertical slice (constant x)."""
        from blocks.pde.field_processing_2d import FieldSliceBlock

        block = FieldSliceBlock()
        params = {
            'slice_direction': 'y',
            'slice_position': 0.5  # Middle column
        }

        # Create field where each column has unique values
        field = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4]
        ])

        result = block.execute(time=0.0, inputs={0: field}, params=params)

        # At position 0.5 with Nx=4, i = int(0.5 * 3) = 1, so column index 1
        expected = np.array([2, 2, 2, 2])
        assert np.allclose(result[0], expected), f"Expected column [2,2,2,2], got {result[0]}"

    def test_field_slice_edge(self):
        """Test slicing at edge positions."""
        from blocks.pde.field_processing_2d import FieldSliceBlock

        block = FieldSliceBlock()

        field = np.arange(16).reshape(4, 4)

        # First row (position 0)
        params = {'slice_direction': 'x', 'slice_position': 0.0}
        result = block.execute(time=0.0, inputs={0: field}, params=params)
        assert np.allclose(result[0], field[0, :]), "Should get first row"

        # Last row (position 1)
        params = {'slice_direction': 'x', 'slice_position': 1.0}
        result = block.execute(time=0.0, inputs={0: field}, params=params)
        assert np.allclose(result[0], field[3, :]), "Should get last row"
