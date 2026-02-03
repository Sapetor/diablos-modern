"""
Unit tests for Sink block implementations.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestScopeBlock:
    """Tests for Scope block."""

    def test_scope_initialization(self):
        """Test scope initializes correctly."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'signal1', '_init_start_': True}

        result = block.execute(time=0.0, inputs={0: 5.0}, params=params)

        assert 'vector' in params, "Scope should create vector storage"
        assert params['_init_start_'] == False, "Init flag should be cleared"
        assert 'vec_dim' in params, "Should detect vector dimension"
        assert 'vec_labels' in params, "Should create labels"

    def test_scope_collects_data(self):
        """Test scope collects input data over time."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'test', '_init_start_': True}

        # Collect several data points
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            block.execute(time=i*0.1, inputs={0: val}, params=params)

        vector = params['vector']
        assert len(vector) == 5, f"Should have 5 data points, got {len(vector)}"
        assert np.allclose(vector, [1.0, 2.0, 3.0, 4.0, 5.0]), "Should collect values in order"

    def test_scope_vector_input(self):
        """Test scope handles vector inputs."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'default', '_init_start_': True, '_name_': 'TestScope'}

        vec = np.array([1.0, 2.0, 3.0])
        block.execute(time=0.0, inputs={0: vec}, params=params)

        assert params['vec_dim'] == 3, "Should detect vector dimension"
        assert len(params['vec_labels']) == 3, "Should create 3 labels"

    def test_scope_default_labels(self):
        """Test scope creates default labels when not specified."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'default', '_init_start_': True, '_name_': 'MyScope'}

        block.execute(time=0.0, inputs={0: 1.0}, params=params)

        assert params['vec_labels'] == ['MyScope-0'], "Should use block name for default label"

    def test_scope_custom_labels(self):
        """Test scope uses custom comma-separated labels."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'x,y,z', '_init_start_': True}

        vec = np.array([1.0, 2.0, 3.0])
        block.execute(time=0.0, inputs={0: vec}, params=params)

        assert params['vec_labels'] == ['x', 'y', 'z'], "Should parse custom labels"

    def test_scope_skip_flag(self):
        """Test scope respects _skip_ flag for intermediate RK45 steps."""
        from blocks.scope import ScopeBlock
        block = ScopeBlock()
        params = {'labels': 'test', '_init_start_': True, '_skip_': True}

        result = block.execute(time=0.0, inputs={0: 5.0}, params=params)

        assert params['_skip_'] == False, "Skip flag should be cleared"
        assert 'vector' not in params, "Should not store data when skip flag is set"


@pytest.mark.unit
class TestXYGraphBlock:
    """Tests for XYGraph block."""

    def test_xygraph_initialization(self):
        """Test XYGraph initializes data storage."""
        from blocks.xygraph import XYGraphBlock
        block = XYGraphBlock()
        params = {'x_label': 'X', 'y_label': 'Y', '_init_start_': True}

        block.execute(time=0.0, inputs={0: 1.0, 1: 2.0}, params=params)

        assert '_x_data_' in params, "Should have x data storage"
        assert '_y_data_' in params, "Should have y data storage"
        assert params['_init_start_'] == False, "Init flag should be cleared"
        assert isinstance(params['_x_data_'], list), "x_data should be a list"
        assert isinstance(params['_y_data_'], list), "y_data should be a list"

    def test_xygraph_collects_points(self):
        """Test XYGraph collects X,Y pairs."""
        from blocks.xygraph import XYGraphBlock
        block = XYGraphBlock()
        params = {'_init_start_': True}

        # Add several points
        points = [(0, 0), (1, 1), (2, 4), (3, 9)]
        for x, y in points:
            block.execute(time=0.0, inputs={0: float(x), 1: float(y)}, params=params)

        assert len(params['_x_data_']) == 4, "Should have 4 x values"
        assert len(params['_y_data_']) == 4, "Should have 4 y values"
        assert params['_x_data_'] == [0.0, 1.0, 2.0, 3.0], "x values should match input"
        assert params['_y_data_'] == [0.0, 1.0, 4.0, 9.0], "y values should match input"

    def test_xygraph_handles_arrays(self):
        """Test XYGraph extracts first element from array inputs."""
        from blocks.xygraph import XYGraphBlock
        block = XYGraphBlock()
        params = {'_init_start_': True}

        # Pass arrays - should take first element
        block.execute(time=0.0, inputs={0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}, params=params)

        assert params['_x_data_'] == [1.0], "Should extract first x element"
        assert params['_y_data_'] == [3.0], "Should extract first y element"

    def test_xygraph_parametric_curve(self):
        """Test XYGraph for parametric curve (circle)."""
        from blocks.xygraph import XYGraphBlock
        block = XYGraphBlock()
        params = {'_init_start_': True}

        # Generate circle: x = cos(t), y = sin(t)
        for t in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = np.cos(t)
            y = np.sin(t)
            block.execute(time=t, inputs={0: x, 1: y}, params=params)

        x_data = np.array(params['_x_data_'])
        y_data = np.array(params['_y_data_'])

        # Points should be on unit circle
        radii = np.sqrt(x_data**2 + y_data**2)
        assert np.allclose(radii, 1.0, atol=1e-10), "Points should lie on unit circle"


@pytest.mark.unit
class TestDisplayBlock:
    """Tests for Display block."""

    def test_display_formats_value(self):
        """Test Display formats input value."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.2f', 'label': '', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: 3.14159}, params=params)

        assert params['_display_value_'] == '3.14', f"Got {params['_display_value_']}"

    def test_display_with_label(self):
        """Test Display adds label prefix."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.1f', 'label': 'Value', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: 42.5}, params=params)

        assert params['_display_value_'] == 'Value: 42.5', f"Got {params['_display_value_']}"

    def test_display_array_input(self):
        """Test Display handles array input."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.1f', 'label': '', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: np.array([1.0, 2.0])}, params=params)

        # Should show array format
        assert '[' in params['_display_value_'], "Should format as array"
        assert '1.0' in params['_display_value_'], "Should show first value"
        assert '2.0' in params['_display_value_'], "Should show second value"

    def test_display_long_array(self):
        """Test Display truncates long arrays."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.0f', 'label': '', '_display_value_': '---'}

        long_array = np.arange(10)
        block.execute(time=0.0, inputs={0: long_array}, params=params)

        # Should truncate to first 3 elements
        assert '...' in params['_display_value_'], "Should indicate truncation"
        assert '0' in params['_display_value_'], "Should show first element"

    def test_display_negative_value(self):
        """Test Display handles negative values."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.2f', 'label': '', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: -7.5}, params=params)

        assert params['_display_value_'] == '-7.50', f"Got {params['_display_value_']}"

    def test_display_scientific_notation(self):
        """Test Display with scientific notation format."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.2e', 'label': '', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: 1234.5}, params=params)

        assert 'e' in params['_display_value_'].lower(), "Should use scientific notation"
        assert '1.23' in params['_display_value_'], "Should format mantissa correctly"

    def test_display_integer_format(self):
        """Test Display with integer format."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%d', 'label': '', '_display_value_': '---'}

        block.execute(time=0.0, inputs={0: 42.9}, params=params)

        assert params['_display_value_'] == '42', f"Got {params['_display_value_']}"

    def test_display_updates_over_time(self):
        """Test Display updates value across multiple executions."""
        from blocks.display import DisplayBlock
        block = DisplayBlock()
        params = {'format': '%.1f', 'label': 't', '_display_value_': '---'}

        for t in [0.0, 1.0, 2.0, 3.0]:
            block.execute(time=t, inputs={0: t*10}, params=params)
            expected = f"t: {t*10:.1f}"
            assert params['_display_value_'] == expected, f"At t={t}, expected {expected}"
