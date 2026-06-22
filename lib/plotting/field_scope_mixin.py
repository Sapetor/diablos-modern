"""
Field / PDE / multi-agent scope rendering for :class:`ScopePlotter`.

Extracted (behavior-preserving) as a mixin so the ~1.2k-line ScopePlotter is
split across files without changing any call site: ``plot_again`` dispatches to
``self._plot_field_scope`` / ``self._plot_field_scope_2d`` / ``self._plot_agent_scope``
exactly as before, and these methods reach the core helpers (``self._register_figure``)
through the shared instance.

Covers FieldScope (1D heatmap / slider), FieldScope2D (animated heatmap), and
AgentScope (multi-agent 2D scatter animation), plus the animation Export dialog.
"""
import logging

import numpy as np

from lib.plotting.animation_exporter import AnimationExporter

logger = logging.getLogger(__name__)


class _FieldScopeRenderMixin:
    """Field/PDE/agent rendering methods for :class:`ScopePlotter`."""

    def _plot_field_scope(self, block):
        """
        Plot spatiotemporal field data as a 2D heatmap.

        Displays field evolution over time:
        - X-axis: spatial position (0 to L)
        - Y-axis: time
        - Color: field value

        Used for visualizing PDE simulation results.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        # Try to set Qt5 backend if not already set
        try:
            if matplotlib.get_backend() != 'Qt5Agg':
                matplotlib.use('Qt5Agg')
        except Exception:
            pass  # Already using a backend

        params = getattr(block, 'exec_params', block.params)

        # Get stored field history
        field_history = params.get('_field_history_', None)
        time_history = params.get('_time_history_', None)

        logger.info(f"FieldScope {block.name}: Starting plot, field_history type={type(field_history)}, time_history type={type(time_history)}")

        # Check for valid data (handle both list and numpy array)
        has_field_data = field_history is not None and (
            (hasattr(field_history, '__len__') and len(field_history) >= 2) or
            (hasattr(field_history, 'shape') and field_history.shape[0] >= 2)
        )
        if not has_field_data:
            logger.warning(f"FieldScope {block.name}: No field data to plot")
            return

        # Convert to numpy arrays
        try:
            field_data = np.array(field_history)  # Shape: (n_times, n_nodes)
            time_data = np.array(time_history) if time_history is not None else None
            logger.info(f"FieldScope {block.name}: field_data shape={field_data.shape}, time_data shape={time_data.shape if time_data is not None else 'None'}")
        except Exception as e:
            logger.error(f"FieldScope {block.name}: Error converting data: {e}")
            return

        if field_data.ndim != 2:
            logger.warning(f"FieldScope {block.name}: Expected 2D field data, got {field_data.ndim}D")
            return

        n_times, n_nodes = field_data.shape

        # Get spatial parameters
        L = float(params.get('L', 1.0))
        x_positions = np.linspace(0, L, n_nodes)

        # Get plot configuration
        title = params.get('title', 'Field Evolution')
        colormap = params.get('colormap', 'viridis')
        display_mode = params.get('display_mode', 'heatmap')
        clim_min = params.get('clim_min', None)
        clim_max = params.get('clim_max', None)

        # Check display mode
        if display_mode == 'slider':
            # Animated line plot with time slider
            self._plot_field_scope_slider(block, field_data, time_data, x_positions, L, title, colormap)
            return

        # Default: heatmap mode
        # Build time extent
        if time_data is not None and len(time_data) > 0:
            t_min = float(time_data[0])
            t_max = float(time_data[-1])
        else:
            t_min = 0.0
            t_max = 1.0

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        self._register_figure(fig)

        # Create 2D heatmap using imshow
        # Extent: [x_min, x_max, t_min, t_max]
        extent = [0, L, t_min, t_max]

        # Set color limits
        if clim_min is None:
            clim_min = np.min(field_data)
        if clim_max is None:
            clim_max = np.max(field_data)

        logger.info(f"FieldScope {block.name}: Plotting extent={extent}, clim=[{clim_min}, {clim_max}]")

        # Plot the field data
        # Note: imshow expects data as [rows, cols] where rows=y, cols=x
        # Our data is [time, space], so we need origin='lower' to have time increase upward
        im = ax.imshow(field_data,
                       aspect='auto',
                       origin='lower',
                       extent=extent,
                       cmap=colormap,
                       vmin=clim_min,
                       vmax=clim_max,
                       interpolation='bilinear')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value')

        # Labels and title
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Time (t)')
        ax.set_title(f"{title} ({block.name})")

        plt.tight_layout()

        # Use non-blocking show for Qt compatibility
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        logger.info(f"FieldScope {block.name}: Plot displayed successfully")

    def _plot_field_scope_slider(self, block, field_data, time_data, x_positions, L, title, colormap):
        """
        Plot 1D field evolution as animated line plot with time slider.

        Shows the field profile at each time step with interactive slider.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        n_times, n_nodes = field_data.shape

        # Set y-axis limits from full data range
        y_min = np.min(field_data)
        y_max = np.max(field_data)
        y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 0.1
        y_min -= y_margin
        y_max += y_margin

        # Create figure with slider space
        fig, ax = plt.subplots(figsize=(10, 7))
        self._register_figure(fig)
        plt.subplots_adjust(bottom=0.15)

        # Initial time
        initial_time = time_data[0] if time_data is not None else 0.0

        # Plot initial field profile
        line, = ax.plot(x_positions, field_data[0], 'b-', linewidth=2)

        ax.set_xlim(0, L)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Position x')
        ax.set_ylabel('Field Value')
        ax.grid(True, alpha=0.3)
        title_text = ax.set_title(f"{title} at t={initial_time:.3f}s ({block.name})")

        # Add time slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        time_slider = Slider(
            ax=ax_slider,
            label='Time',
            valmin=0,
            valmax=n_times - 1,
            valinit=0,
            valstep=1
        )

        def update(val):
            frame_idx = int(time_slider.val)
            line.set_ydata(field_data[frame_idx])
            t = time_data[frame_idx] if time_data is not None else frame_idx
            title_text.set_text(f"{title} at t={t:.3f}s ({block.name})")
            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        # Store slider reference to prevent garbage collection
        fig._time_slider = time_slider

        # Add Export button
        from matplotlib.widgets import Button
        ax_export = plt.axes([0.85, 0.02, 0.1, 0.03])
        export_btn = Button(ax_export, 'Export')

        def on_export_clicked(event):
            self._show_export_dialog(
                block,
                field_data,
                time_data,
                params={
                    'L': L,
                    'colormap': colormap,
                    'title': title
                },
                dimension='1d'
            )

        export_btn.on_clicked(on_export_clicked)
        fig._export_btn = export_btn  # Prevent GC

        # Use non-blocking show for Qt compatibility
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        logger.info(f"FieldScope {block.name}: Slider plot displayed successfully")

    def _plot_field_scope_2d(self, block):
        """
        Plot 2D field evolution as animated heatmap with time slider.

        Shows the 2D field with interactive time slider to explore evolution.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        try:
            if matplotlib.get_backend() != 'Qt5Agg':
                matplotlib.use('Qt5Agg')
        except Exception:
            logger.debug("Could not switch matplotlib backend to Qt5Agg", exc_info=True)

        params = getattr(block, 'exec_params', block.params)

        # Get stored field history
        field_history = params.get('_field_history_2d_', None)
        time_history = params.get('_time_history_', None)

        logger.info(f"FieldScope2D {block.name}: Starting plot")

        # Check for valid data
        has_field_data = field_history is not None and (
            (hasattr(field_history, '__len__') and len(field_history) >= 1) or
            (hasattr(field_history, 'shape') and field_history.shape[0] >= 1)
        )
        if not has_field_data:
            logger.warning(f"FieldScope2D {block.name}: No field data to plot")
            return

        # Convert to numpy array
        try:
            field_data = np.array(field_history)  # Shape: (n_times, Ny, Nx)
            time_data = np.array(time_history) if time_history is not None else None
            logger.info(f"FieldScope2D {block.name}: field_data shape={field_data.shape}")
        except Exception as e:
            logger.error(f"FieldScope2D {block.name}: Error converting data: {e}")
            return

        if field_data.ndim != 3:
            logger.warning(f"FieldScope2D {block.name}: Expected 3D field data, got {field_data.ndim}D")
            return

        n_times, Ny, Nx = field_data.shape

        # Get spatial parameters
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))

        # Get plot configuration
        title = params.get('title', '2D Field')
        colormap = params.get('colormap', 'viridis')
        clim_min = params.get('clim_min', None)
        clim_max = params.get('clim_max', None)

        # Set color limits from full data range for consistent colorbar
        if clim_min is None:
            clim_min = np.min(field_data)
        if clim_max is None:
            clim_max = np.max(field_data)

        # Create figure with slider space
        fig, ax = plt.subplots(figsize=(10, 9))
        self._register_figure(fig)
        plt.subplots_adjust(bottom=0.15)

        # Start with initial frame (t=0) to show the initial condition
        initial_field = field_data[0]
        initial_time = time_data[0] if time_data is not None and len(time_data) > 0 else 0.0

        # Plot the field data
        extent = [0, Lx, 0, Ly]
        im = ax.imshow(initial_field,
                       aspect='equal',
                       origin='lower',
                       extent=extent,
                       cmap=colormap,
                       vmin=clim_min,
                       vmax=clim_max,
                       interpolation='bilinear')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        title_text = ax.set_title(f"{title} at t={initial_time:.3f} ({block.name})")

        # Add time slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        time_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=n_times - 1,
            valinit=0,
            valstep=1
        )

        def update(val):
            frame_idx = int(time_slider.val)
            im.set_data(field_data[frame_idx])
            t = time_data[frame_idx] if time_data is not None else frame_idx
            title_text.set_text(f"{title} at t={t:.3f} ({block.name})")
            fig.canvas.draw_idle()

        time_slider.on_changed(update)

        # Store slider reference to prevent garbage collection
        fig._time_slider = time_slider

        # Add Export button
        from matplotlib.widgets import Button
        ax_export = plt.axes([0.85, 0.02, 0.1, 0.03])
        export_btn = Button(ax_export, 'Export')

        def on_export_clicked(event):
            self._show_export_dialog(
                block,
                field_data,
                time_data,
                params={
                    'Lx': Lx,
                    'Ly': Ly,
                    'colormap': colormap,
                    'title': title,
                    'clim_min': clim_min,
                    'clim_max': clim_max
                },
                dimension='2d'
            )

        export_btn.on_clicked(on_export_clicked)
        fig._export_btn = export_btn  # Prevent GC

        # Use non-blocking show for Qt compatibility
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

        logger.info(f"FieldScope2D {block.name}: Plot displayed with {n_times} frames")

    def _plot_agent_scope(self, block):
        """
        Plot multi-agent 2D trajectories as a scatter animation with time slider
        and Export button (GIF / MP4).

        Block input: flat positions vector [x1, y1, x2, y2, ...] per time step.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button

        params = getattr(block, 'exec_params', block.params)
        history = params.get('_pos_history_', None)
        time_history = params.get('_time_history_', None)

        if history is None or len(history) < 2:
            logger.warning(f"AgentScope {block.name}: no data")
            return

        try:
            n_agents = int(params.get('n_agents', 0))
            history_arr = np.asarray(history, dtype=float)
            if n_agents <= 0:
                n_agents = history_arr.shape[1] // 2
            # Reshape to (T, N, 2)
            positions = history_arr.reshape(-1, n_agents, 2)
        except (ValueError, TypeError) as e:
            logger.error(f"AgentScope {block.name}: cannot reshape history -> {e}")
            return

        n_times = positions.shape[0]
        time_data = np.asarray(time_history, dtype=float) if time_history is not None else np.arange(n_times)

        title = params.get('title', 'Agent trajectories')
        show_trails = bool(params.get('show_trails', True))
        trail_length = int(params.get('trail_length', 0) or 0)

        x_min, x_max = float(np.min(positions[..., 0])), float(np.max(positions[..., 0]))
        y_min, y_max = float(np.min(positions[..., 1])), float(np.max(positions[..., 1]))
        pad_x = 0.1 * (x_max - x_min) if x_max > x_min else 1.0
        pad_y = 0.1 * (y_max - y_min) if y_max > y_min else 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        self._register_figure(fig)
        plt.subplots_adjust(bottom=0.18)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        title_text = ax.set_title(f"{title}  t={time_data[0]:.3f}s ({block.name})")

        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(n_agents)]

        trail_lines = []
        if show_trails:
            for i in range(n_agents):
                line, = ax.plot([], [], '-', color=colors[i], linewidth=1.5, alpha=0.6)
                trail_lines.append(line)

        scatter = ax.scatter(positions[0, :, 0], positions[0, :, 1],
                             c=colors, s=80, edgecolors='black', zorder=3)
        labels = [ax.annotate(str(i + 1),
                              (positions[0, i, 0], positions[0, i, 1]),
                              fontsize=9, ha='center', va='center', zorder=4)
                  for i in range(n_agents)]

        ax_slider = plt.axes([0.2, 0.05, 0.55, 0.03])
        time_slider = Slider(
            ax=ax_slider, label='Time',
            valmin=0, valmax=n_times - 1, valinit=0, valstep=1,
        )

        def update(_val):
            frame = int(time_slider.val)
            scatter.set_offsets(positions[frame])
            for i in range(n_agents):
                labels[i].set_position((positions[frame, i, 0], positions[frame, i, 1]))
            if show_trails:
                start = 0 if trail_length <= 0 else max(0, frame - trail_length)
                for i in range(n_agents):
                    trail_lines[i].set_data(positions[start:frame + 1, i, 0],
                                            positions[start:frame + 1, i, 1])
            title_text.set_text(f"{title}  t={time_data[frame]:.3f}s ({block.name})")
            fig.canvas.draw_idle()

        time_slider.on_changed(update)
        fig._time_slider = time_slider

        ax_export = plt.axes([0.82, 0.05, 0.12, 0.03])
        export_btn = Button(ax_export, 'Export')

        def on_export_clicked(_event):
            self._show_export_dialog(
                block, positions, time_data,
                params={
                    'title': title,
                    'show_trails': show_trails,
                    'trail_length': trail_length,
                },
                dimension='agents',
            )

        export_btn.on_clicked(on_export_clicked)
        fig._export_btn = export_btn

        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.flush_events()
        logger.info(f"AgentScope {block.name}: animation displayed ({n_times} frames, {n_agents} agents)")

    def _show_export_dialog(self, block, field_data, time_data, params, dimension):
        """
        Show the animation export dialog.

        Args:
            block: The FieldScope or FieldScope2D block
            field_data: numpy array of field values over time
            time_data: numpy array of time values
            params: dict with visualization parameters (L/Lx/Ly, colormap, title, etc.)
            dimension: '1d' or '2d'
        """
        try:
            from PyQt5.QtWidgets import QApplication
            from modern_ui.widgets.animation_export_dialog import AnimationExportDialog

            # Create exporter
            exporter = AnimationExporter(
                field_data=field_data,
                time_data=time_data,
                params=params,
                dimension=dimension
            )

            # Get or create QApplication (needed for dialog)
            app = QApplication.instance()
            if app is None:
                logger.warning("No QApplication instance found for export dialog")
                return

            # Show dialog
            dialog = AnimationExportDialog(
                exporter=exporter,
                block_name=block.name
            )
            dialog.exec_()

        except ImportError as e:
            logger.error(f"Failed to import export dialog: {e}")
        except Exception as e:
            logger.error(f"Failed to show export dialog: {e}")
            import traceback
            logger.error(traceback.format_exc())
