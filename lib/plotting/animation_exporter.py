"""
Animation exporter for FieldScope visualizations.

Exports 1D and 2D field time-series data as animated GIF or MP4 files
using matplotlib's FuncAnimation.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class AnimationExporter:
    """Export field animations as GIF or MP4."""

    WRITERS = {'gif': 'pillow', 'mp4': 'ffmpeg'}
    QUALITY_PRESETS = {
        'low': 72,
        'medium': 100,
        'high': 150
    }

    def __init__(self, field_data, time_data, params, dimension='2d'):
        """
        Initialize the animation exporter.

        Args:
            field_data: numpy array
                - For 1D: shape (n_times, n_nodes)
                - For 2D: shape (n_times, Ny, Nx)
            time_data: array of simulation times
            params: dict with visualization parameters
                - Lx, Ly (for 2D) or L (for 1D): domain size
                - colormap: matplotlib colormap name
                - title: plot title
                - clim_min, clim_max: color limits (optional)
            dimension: '1d' or '2d'
        """
        self.field_data = np.asarray(field_data)
        self.time_data = np.asarray(time_data) if time_data is not None else np.arange(len(field_data))
        self.params = params
        self.dimension = dimension.lower()

        # Validate data shape
        if self.dimension == '2d' and self.field_data.ndim != 3:
            raise ValueError(f"2D field data must be 3D array, got {self.field_data.ndim}D")
        if self.dimension == '1d' and self.field_data.ndim != 2:
            raise ValueError(f"1D field data must be 2D array, got {self.field_data.ndim}D")

    @property
    def n_frames(self):
        """Number of frames in the animation."""
        return len(self.field_data)

    @property
    def duration(self):
        """Total simulation duration in seconds."""
        if len(self.time_data) > 1:
            return float(self.time_data[-1] - self.time_data[0])
        return 0.0

    @property
    def grid_size(self):
        """Grid dimensions as string (e.g., '50x50' for 2D or '100' for 1D)."""
        if self.dimension == '2d':
            _, Ny, Nx = self.field_data.shape
            return f"{Nx}x{Ny}"
        else:
            _, n_nodes = self.field_data.shape
            return str(n_nodes)

    def get_recommended_fps(self, target_duration=5.0):
        """
        Calculate recommended FPS for a target playback duration.

        Args:
            target_duration: Desired playback duration in seconds

        Returns:
            Recommended frames per second (minimum 1)
        """
        return max(1, int(self.n_frames / target_duration))

    def get_playback_duration(self, fps):
        """
        Calculate playback duration for given FPS.

        Args:
            fps: Frames per second

        Returns:
            Playback duration in seconds
        """
        if fps <= 0:
            return float('inf')
        return self.n_frames / fps

    def create_animation(self, fps=15, figsize=(8, 6), dpi=100):
        """
        Create matplotlib FuncAnimation object.

        Args:
            fps: Frames per second
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for rendering

        Returns:
            tuple: (fig, animation) - matplotlib figure and FuncAnimation object
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        if self.dimension == '2d':
            return self._create_2d_animation(fps, figsize, dpi)
        else:
            return self._create_1d_animation(fps, figsize, dpi)

    def _create_2d_animation(self, fps, figsize, dpi):
        """Create 2D heatmap animation."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        n_times, Ny, Nx = self.field_data.shape

        # Get parameters
        Lx = float(self.params.get('Lx', 1.0))
        Ly = float(self.params.get('Ly', 1.0))
        colormap = self.params.get('colormap', 'viridis')
        title = self.params.get('title', '2D Field')
        clim_min = self.params.get('clim_min', np.min(self.field_data))
        clim_max = self.params.get('clim_max', np.max(self.field_data))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Initial frame
        extent = [0, Lx, 0, Ly]
        im = ax.imshow(
            self.field_data[0],
            aspect='equal',
            origin='lower',
            extent=extent,
            cmap=colormap,
            vmin=clim_min,
            vmax=clim_max,
            interpolation='bilinear'
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value')

        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        title_text = ax.set_title(f"{title} at t={self.time_data[0]:.3f}")

        plt.tight_layout()

        def update(frame):
            im.set_data(self.field_data[frame])
            title_text.set_text(f"{title} at t={self.time_data[frame]:.3f}")
            return [im, title_text]

        anim = FuncAnimation(
            fig, update,
            frames=n_times,
            interval=1000 / fps,
            blit=True,
            repeat=True
        )

        return fig, anim

    def _create_1d_animation(self, fps, figsize, dpi):
        """Create 1D line plot animation."""
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        n_times, n_nodes = self.field_data.shape

        # Get parameters
        L = float(self.params.get('L', 1.0))
        title = self.params.get('title', 'Field Evolution')

        # Spatial positions
        x_positions = np.linspace(0, L, n_nodes)

        # Y-axis limits
        y_min = np.min(self.field_data)
        y_max = np.max(self.field_data)
        y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        y_min -= y_margin
        y_max += y_margin

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Initial frame
        line, = ax.plot(x_positions, self.field_data[0], 'b-', linewidth=2)

        ax.set_xlim(0, L)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Field Value')
        ax.grid(True, alpha=0.3)
        title_text = ax.set_title(f"{title} at t={self.time_data[0]:.3f}")

        plt.tight_layout()

        def update(frame):
            line.set_ydata(self.field_data[frame])
            title_text.set_text(f"{title} at t={self.time_data[frame]:.3f}")
            return [line, title_text]

        anim = FuncAnimation(
            fig, update,
            frames=n_times,
            interval=1000 / fps,
            blit=True,
            repeat=True
        )

        return fig, anim

    def export(self, filepath, format='gif', fps=15, dpi=100, progress_callback=None):
        """
        Export animation to file.

        Args:
            filepath: Output file path
            format: 'gif' or 'mp4'
            fps: Frames per second
            dpi: Resolution (dots per inch)
            progress_callback: Optional callback function(frame, total) for progress

        Returns:
            bool: True on success, False on failure
        """
        import matplotlib.pyplot as plt

        format = format.lower()
        if format not in self.WRITERS:
            logger.error(f"Unsupported format: {format}. Use 'gif' or 'mp4'")
            return False

        writer_name = self.WRITERS[format]

        # Check writer availability
        available = self.check_writers()
        if not available.get(format, False):
            logger.error(f"Writer '{writer_name}' not available for {format} export")
            return False

        try:
            # Create animation
            fig, anim = self.create_animation(fps=fps, dpi=dpi)

            # Set up writer
            if format == 'gif':
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=fps)
            else:  # mp4
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=fps, bitrate=1800)

            # Ensure parent directory exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save animation
            logger.info(f"Exporting {self.n_frames} frames to {filepath}")

            if progress_callback:
                # Custom save with progress
                anim.save(
                    str(filepath),
                    writer=writer,
                    progress_callback=lambda i, n: progress_callback(i, n)
                )
            else:
                anim.save(str(filepath), writer=writer)

            plt.close(fig)

            logger.info(f"Animation exported successfully to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export animation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @staticmethod
    def check_writers():
        """
        Check which animation writers are available.

        Returns:
            dict: {'gif': bool, 'mp4': bool}
        """
        available = {'gif': False, 'mp4': False}

        # Check Pillow for GIF
        try:
            from PIL import Image
            available['gif'] = True
        except ImportError:
            pass

        # Check ffmpeg for MP4
        try:
            import subprocess
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                timeout=5
            )
            available['mp4'] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        return available
