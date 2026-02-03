"""
Unit tests for AnimationExporter.

Tests export functionality for 1D and 2D field animations.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.mark.unit
class TestAnimationExporter:
    """Tests for AnimationExporter class."""

    def test_init_2d_field(self):
        """Test initialization with 2D field data."""
        from lib.plotting.animation_exporter import AnimationExporter

        # Create synthetic 2D field data: (n_times, Ny, Nx)
        n_times, Ny, Nx = 50, 20, 20
        field_data = np.random.rand(n_times, Ny, Nx)
        time_data = np.linspace(0, 5, n_times)
        params = {'Lx': 1.0, 'Ly': 1.0, 'colormap': 'viridis', 'title': 'Test'}

        exporter = AnimationExporter(field_data, time_data, params, dimension='2d')

        assert exporter.n_frames == n_times
        assert exporter.duration == pytest.approx(5.0)
        assert exporter.grid_size == '20x20'
        assert exporter.dimension == '2d'

    def test_init_1d_field(self):
        """Test initialization with 1D field data."""
        from lib.plotting.animation_exporter import AnimationExporter

        # Create synthetic 1D field data: (n_times, n_nodes)
        n_times, n_nodes = 100, 50
        field_data = np.random.rand(n_times, n_nodes)
        time_data = np.linspace(0, 10, n_times)
        params = {'L': 1.0, 'colormap': 'viridis', 'title': 'Test 1D'}

        exporter = AnimationExporter(field_data, time_data, params, dimension='1d')

        assert exporter.n_frames == n_times
        assert exporter.duration == pytest.approx(10.0)
        assert exporter.grid_size == '50'
        assert exporter.dimension == '1d'

    def test_init_invalid_dimension_2d(self):
        """Test that 2D dimension with wrong shape raises error."""
        from lib.plotting.animation_exporter import AnimationExporter

        # 2D array but dimension='2d' expects 3D
        field_data = np.random.rand(50, 100)
        time_data = np.linspace(0, 5, 50)
        params = {}

        with pytest.raises(ValueError, match="2D field data must be 3D array"):
            AnimationExporter(field_data, time_data, params, dimension='2d')

    def test_init_invalid_dimension_1d(self):
        """Test that 1D dimension with wrong shape raises error."""
        from lib.plotting.animation_exporter import AnimationExporter

        # 3D array but dimension='1d' expects 2D
        field_data = np.random.rand(50, 20, 20)
        time_data = np.linspace(0, 5, 50)
        params = {}

        with pytest.raises(ValueError, match="1D field data must be 2D array"):
            AnimationExporter(field_data, time_data, params, dimension='1d')

    def test_recommended_fps(self):
        """Test FPS recommendation for target duration."""
        from lib.plotting.animation_exporter import AnimationExporter

        field_data = np.random.rand(100, 20, 20)
        time_data = np.linspace(0, 10, 100)
        params = {}

        exporter = AnimationExporter(field_data, time_data, params, dimension='2d')

        # 100 frames / 5 second target = 20 fps
        assert exporter.get_recommended_fps(5.0) == 20

        # 100 frames / 10 second target = 10 fps
        assert exporter.get_recommended_fps(10.0) == 10

        # Minimum is 1 fps
        assert exporter.get_recommended_fps(1000.0) >= 1

    def test_playback_duration(self):
        """Test playback duration calculation."""
        from lib.plotting.animation_exporter import AnimationExporter

        field_data = np.random.rand(60, 20, 20)
        time_data = np.linspace(0, 6, 60)
        params = {}

        exporter = AnimationExporter(field_data, time_data, params, dimension='2d')

        # 60 frames at 15 fps = 4 seconds
        assert exporter.get_playback_duration(15) == pytest.approx(4.0)

        # 60 frames at 30 fps = 2 seconds
        assert exporter.get_playback_duration(30) == pytest.approx(2.0)

    def test_check_writers(self):
        """Test writer availability check returns dict."""
        from lib.plotting.animation_exporter import AnimationExporter

        available = AnimationExporter.check_writers()

        assert isinstance(available, dict)
        assert 'gif' in available
        assert 'mp4' in available
        assert isinstance(available['gif'], bool)
        assert isinstance(available['mp4'], bool)

    def test_single_frame(self):
        """Test edge case: single frame animation."""
        from lib.plotting.animation_exporter import AnimationExporter

        field_data = np.random.rand(1, 20, 20)
        time_data = np.array([0.0])
        params = {'Lx': 1.0, 'Ly': 1.0}

        exporter = AnimationExporter(field_data, time_data, params, dimension='2d')

        assert exporter.n_frames == 1
        assert exporter.duration == 0.0
        assert exporter.get_recommended_fps(5.0) >= 1

    def test_no_time_data(self):
        """Test initialization without time data."""
        from lib.plotting.animation_exporter import AnimationExporter

        field_data = np.random.rand(50, 20, 20)
        params = {'Lx': 1.0, 'Ly': 1.0}

        exporter = AnimationExporter(field_data, None, params, dimension='2d')

        assert exporter.n_frames == 50
        # Time data should default to frame indices
        assert len(exporter.time_data) == 50


@pytest.mark.unit
class TestAnimationExport:
    """Tests for actual export functionality."""

    @pytest.fixture
    def sample_2d_exporter(self):
        """Create sample 2D exporter for tests."""
        from lib.plotting.animation_exporter import AnimationExporter

        n_times, Ny, Nx = 10, 15, 15
        # Create a simple decaying pattern
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)
        X, Y = np.meshgrid(x, y)

        field_data = np.zeros((n_times, Ny, Nx))
        for i in range(n_times):
            decay = np.exp(-i * 0.2)
            field_data[i] = decay * np.sin(np.pi * X) * np.sin(np.pi * Y)

        time_data = np.linspace(0, 1, n_times)
        params = {
            'Lx': 1.0,
            'Ly': 1.0,
            'colormap': 'viridis',
            'title': 'Test Field'
        }

        return AnimationExporter(field_data, time_data, params, dimension='2d')

    @pytest.fixture
    def sample_1d_exporter(self):
        """Create sample 1D exporter for tests."""
        from lib.plotting.animation_exporter import AnimationExporter

        n_times, n_nodes = 10, 50
        x = np.linspace(0, 1, n_nodes)

        field_data = np.zeros((n_times, n_nodes))
        for i in range(n_times):
            decay = np.exp(-i * 0.2)
            field_data[i] = decay * np.sin(np.pi * x)

        time_data = np.linspace(0, 1, n_times)
        params = {
            'L': 1.0,
            'colormap': 'viridis',
            'title': 'Test 1D Field'
        }

        return AnimationExporter(field_data, time_data, params, dimension='1d')

    def test_export_2d_gif(self, sample_2d_exporter):
        """Test exporting 2D animation as GIF."""
        from lib.plotting.animation_exporter import AnimationExporter
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing

        available = AnimationExporter.check_writers()
        if not available.get('gif', False):
            pytest.skip("Pillow not available for GIF export")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_2d.gif')

            success = sample_2d_exporter.export(filepath, format='gif', fps=5, dpi=72)

            assert success, "GIF export should succeed"
            assert os.path.exists(filepath), "GIF file should be created"
            assert os.path.getsize(filepath) > 0, "GIF file should not be empty"

    def test_export_1d_gif(self, sample_1d_exporter):
        """Test exporting 1D animation as GIF."""
        from lib.plotting.animation_exporter import AnimationExporter
        import matplotlib
        matplotlib.use('Agg')

        available = AnimationExporter.check_writers()
        if not available.get('gif', False):
            pytest.skip("Pillow not available for GIF export")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_1d.gif')

            success = sample_1d_exporter.export(filepath, format='gif', fps=5, dpi=72)

            assert success, "GIF export should succeed"
            assert os.path.exists(filepath), "GIF file should be created"
            assert os.path.getsize(filepath) > 0, "GIF file should not be empty"

    def test_export_invalid_format(self, sample_2d_exporter):
        """Test that invalid format returns False."""
        import matplotlib
        matplotlib.use('Agg')

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.avi')

            success = sample_2d_exporter.export(filepath, format='avi', fps=5, dpi=72)

            assert not success, "Invalid format should return False"

    def test_export_creates_parent_dirs(self, sample_2d_exporter):
        """Test that export creates parent directories if needed."""
        from lib.plotting.animation_exporter import AnimationExporter
        import matplotlib
        matplotlib.use('Agg')

        available = AnimationExporter.check_writers()
        if not available.get('gif', False):
            pytest.skip("Pillow not available for GIF export")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir', 'nested', 'test.gif')

            success = sample_2d_exporter.export(filepath, format='gif', fps=5, dpi=72)

            assert success, "Export should succeed"
            assert os.path.exists(filepath), "File should be created in nested directory"

    def test_create_animation_2d(self, sample_2d_exporter):
        """Test creating 2D animation object."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, anim = sample_2d_exporter.create_animation(fps=10, dpi=72)

        assert fig is not None
        assert anim is not None
        assert hasattr(anim, 'save')

        plt.close(fig)

    def test_create_animation_1d(self, sample_1d_exporter):
        """Test creating 1D animation object."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, anim = sample_1d_exporter.create_animation(fps=10, dpi=72)

        assert fig is not None
        assert anim is not None
        assert hasattr(anim, 'save')

        plt.close(fig)
