
import logging
import numpy as np
from lib.plotting.signal_plot import SignalPlot
from lib.plotting.animation_exporter import AnimationExporter

logger = logging.getLogger(__name__)

class ScopePlotter:
    """
    Handles plotting logic for Scope, XYGraph, and FFT blocks.
    extracted from DSim to separate concerns.
    """
    def __init__(self, dsim):
        self.dsim = dsim
        self.plotty = None
        # Sync to dsim for backward compatibility
        if hasattr(self.dsim, 'plotty'):
            self.dsim.plotty = None

    def plot_again(self):
        """
        :purpose: Plots the data saved in Scope and XYGraph blocks without needing to execute the simulation again.
        """
        if self.dsim.dirty:
            logger.error("ERROR: The diagram has been modified. Please run the simulation again.")
            return
        try:
            # Use active blocks from engine if available (flattened execution), otherwise model list
            has_engine = hasattr(self.dsim, 'engine')
            use_active = has_engine and self.dsim.engine.active_blocks_list
            source_blocks = self.dsim.engine.active_blocks_list if use_active else self.dsim.blocks_list
            
            logger.info(f"PLOT DEBUG: Engine={has_engine}, UseActive={use_active}, Blocks={len(source_blocks)}")
            
            # Plot Scopes - scope stores data in exec_params during execution
            vectors = []
            for x in source_blocks:
                if x.block_fn == 'Scope':
                    # Check exec_params first (used during execution), fallback to params
                    params = getattr(x, 'exec_params', x.params)
                    vec = params.get('vector')
                    if vec is None:
                        vec = x.params.get('vector')
                    
                    vec_len = len(vec) if vec is not None else 'None'
                    logger.info(f"PLOT DEBUG: Found Scope {x.name}. Vector len: {vec_len}")
                    vectors.append(vec)
            valid_vectors = [v for v in vectors if v is not None and hasattr(v, '__len__') and len(v) > 0]
            if valid_vectors:
                self.pyqtPlotScope()
                # Print verification metrics for comparison scopes
                self._print_verification_summary(source_blocks)
            else:
                logger.info("PLOT: No scope data available to plot.")
            
            # Plot XYGraphs
            for block in source_blocks:
                if block.block_fn == 'XYGraph':
                    params = getattr(block, 'exec_params', block.params)
                    x_data = params.get('_x_data_', [])
                    y_data = params.get('_y_data_', [])
                    if x_data and y_data:
                        self._plot_xygraph(block)
            
            # Plot FFT spectrums
            for block in source_blocks:
                if block.block_fn == 'FFT':
                    params = getattr(block, 'exec_params', block.params)
                    buffer = params.get('_fft_buffer_', [])
                    if buffer and len(buffer) > 1:
                        self._plot_fft(block)

            # Plot FieldScope (spatiotemporal field visualization)
            for block in source_blocks:
                if block.block_fn == 'FieldScope':
                    params = getattr(block, 'exec_params', block.params)
                    field_history = params.get('_field_history_', None)
                    # Handle both list and numpy array
                    has_data = field_history is not None and (
                        (hasattr(field_history, '__len__') and len(field_history) > 1) or
                        (hasattr(field_history, 'shape') and field_history.shape[0] > 1)
                    )
                    logger.info(f"PLOT DEBUG: FieldScope {block.name} has_data={has_data}")
                    if has_data:
                        try:
                            self._plot_field_scope(block)
                        except Exception as e:
                            logger.error(f"PLOT ERROR: FieldScope {block.name} plotting failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())

            # Plot FieldScope2D (animated 2D field visualization)
            for block in source_blocks:
                if block.block_fn == 'FieldScope2D':
                    params = getattr(block, 'exec_params', block.params)
                    field_history = params.get('_field_history_2d_', None)
                    # Handle both list and numpy array
                    has_data = field_history is not None and (
                        (hasattr(field_history, '__len__') and len(field_history) > 1) or
                        (hasattr(field_history, 'shape') and field_history.shape[0] > 1)
                    )
                    logger.info(f"PLOT DEBUG: FieldScope2D {block.name} has_data={has_data}")
                    if has_data:
                        try:
                            self._plot_field_scope_2d(block)
                        except Exception as e:
                            logger.error(f"PLOT ERROR: FieldScope2D {block.name} plotting failed: {e}")
                            import traceback
                            logger.error(traceback.format_exc())

        except Exception as e:
            logger.error(f"PLOT: Error during plotting ({str(e)})")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_xygraph(self, block):
        """Plot XY graph data for a single XYGraph block."""
        import matplotlib.pyplot as plt
        
        # Data is stored in exec_params during execution
        params = getattr(block, 'exec_params', block.params)
        x_data = params.get('_x_data_', [])
        y_data = params.get('_y_data_', [])
        
        if not x_data or not y_data:
            return
        
        title = block.params.get('title', 'XY Plot')
        x_label = block.params.get('x_label', 'X')
        y_label = block.params.get('y_label', 'Y')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_data, y_data, 'b-', linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} ({block.name})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()

    def _plot_fft(self, block):
        """Plot FFT spectrum for a single FFT block."""
        import matplotlib.pyplot as plt

        params = getattr(block, 'exec_params', block.params)
        buffer = params.get('_fft_buffer_', [])
        time_data = params.get('_fft_time_', [])

        if not buffer or len(buffer) < 2:
            return

        # Convert to numpy array
        signal = np.array(buffer)
        if signal.ndim > 1:
            signal = signal[:, 0]  # Take first channel

        # Calculate sample rate
        if len(time_data) > 1:
            dt = np.mean(np.diff(time_data))
            fs = 1.0 / dt if dt > 0 else 1.0
        else:
            fs = 1.0 / self.dsim.sim_dt if hasattr(self.dsim, 'sim_dt') else 1.0

        # Apply window function
        window_type = params.get('window', 'hann')
        n = len(signal)
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:
            window = np.ones(n)

        windowed_signal = signal * window

        # Compute FFT
        fft_result = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        magnitude = np.abs(fft_result)

        # Normalize
        if params.get('normalize', True):
            max_mag = np.max(magnitude)
            if max_mag > 0:
                magnitude = magnitude / max_mag

        # Convert to dB if requested
        if params.get('log_scale', False):
            magnitude = 20 * np.log10(magnitude + 1e-12)

        # Plot
        title = params.get('title', 'FFT Spectrum')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, magnitude, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)' if params.get('log_scale', False) else 'Magnitude')
        ax.set_title(f"{title} ({block.name})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, fs/2])

        plt.tight_layout()
        plt.show()

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
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

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
            pass

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

    def _plot_field_snapshot(self, block):
        """
        Plot the current field snapshot as a 1D profile.

        Useful for viewing instantaneous field distribution.
        """
        import matplotlib.pyplot as plt

        params = getattr(block, 'exec_params', block.params)

        # Get the latest field
        field_history = params.get('_field_history_', [])
        time_history = params.get('_time_history_', [])

        if not field_history:
            logger.warning(f"FieldScope {block.name}: No field data for snapshot")
            return

        # Get the last field
        field = np.array(field_history[-1])
        current_time = time_history[-1] if time_history else 0.0

        L = float(params.get('L', 1.0))
        n_nodes = len(field)
        x = np.linspace(0, L, n_nodes)

        title = params.get('title', 'Field Profile')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, field, 'b-', linewidth=2)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Field Value')
        ax.set_title(f"{title} at t={current_time:.4f} ({block.name})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_field_animation(self, block, interval=50):
        """
        Create an animated plot of field evolution.

        Args:
            block: FieldScope block
            interval: Time between frames in milliseconds
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        params = getattr(block, 'exec_params', block.params)

        field_history = params.get('_field_history_', [])
        time_history = params.get('_time_history_', [])

        if not field_history or len(field_history) < 2:
            logger.warning(f"FieldScope {block.name}: Not enough data for animation")
            return

        field_data = np.array(field_history)
        time_data = np.array(time_history)
        n_times, n_nodes = field_data.shape

        L = float(params.get('L', 1.0))
        x = np.linspace(0, L, n_nodes)

        title = params.get('title', 'Field Evolution')

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot(x, field_data[0], 'b-', linewidth=2)

        ax.set_xlim(0, L)
        ax.set_ylim(np.min(field_data) * 1.1, np.max(field_data) * 1.1)
        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Field Value')
        time_text = ax.set_title(f"{title} t=0.000 ({block.name})")
        ax.grid(True, alpha=0.3)

        def animate(frame):
            line.set_ydata(field_data[frame])
            time_text.set_text(f"{title} t={time_data[frame]:.3f} ({block.name})")
            return line, time_text

        anim = FuncAnimation(fig, animate, frames=n_times,
                            interval=interval, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()

        return anim

    def _is_discrete_upstream(self, block_name, visited=None):
        """
        Determine if a block (or any of its ancestors) is discrete-time/ZOH.
        """
        if visited is None:
            visited = set()

        if block_name in visited:
            return False
        visited.add(block_name)

        # Lookup the block in the model
        block = None
        try:
            block = self.dsim.model.get_block_by_name(block_name)
        except Exception:
            # Fallback in case model is not available for some reason
            block = next((b for b in self.dsim.blocks_list if b.name == block_name), None)

        if block is None:
            return False

        discrete_blocks = {'DiscreteTranFn', 'DiscreteStateSpace', 'ZeroOrderHold'}
        continuous_blocks = {'Integrator', 'TranFn', 'StateSpace', 'Derivative'}

        if block.block_fn in discrete_blocks:
            return True
        if block.block_fn in continuous_blocks:
            return False

        inputs, _ = self.dsim.get_neighbors(block.name)
        for conn in inputs:
            src_block = conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None)
            if src_block and self._is_discrete_upstream(src_block, visited):
                return True
        return False

    def _scope_step_modes(self):
        """
        Build a list of step-mode flags (one per Scope block) for plotting.
        """
        modes = []
        # Use active blocks from engine if available
        source_blocks = self.dsim.engine.active_blocks_list if hasattr(self.dsim, 'engine') and self.dsim.engine.active_blocks_list else self.dsim.blocks_list
        
        for block in source_blocks:
            if block.block_fn == 'Scope':
                # Note: get_neighbors uses active list if execution initialized, 
                # but we need to ensure dsim helper knows which list or engine helper is used.
                # dsim.get_neighbors delegates to engine.get_neighbors? No, DSim has its own.
                # We should use engine.get_neighbors if possible.
                inputs, _ = self.dsim.engine.get_neighbors(block.name) if hasattr(self.dsim, 'engine') else self.dsim.get_neighbors(block.name)
                if not inputs:
                    modes.append(False)
                    continue
                step_mode = any(
                    self._is_discrete_upstream(
                        conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None)
                    )
                    for conn in inputs
                    if (conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None))
                )
                modes.append(step_mode)
        return modes

    def get_scope_traces(self):
        """
        Collect scope data as a flat list of traces for the waveform inspector.
        Returns (timeline, traces) where traces is a list of dicts:
        {'name': str, 'y': np.ndarray, 'step': bool}
        """
        if not hasattr(self.dsim, 'timeline') or self.dsim.timeline is None or len(self.dsim.timeline) == 0:
            return None, []

        step_modes = self._scope_step_modes()
        traces = []
        step_idx = 0
        
        source_blocks = self.dsim.engine.active_blocks_list if hasattr(self.dsim, 'engine') and self.dsim.engine.active_blocks_list else self.dsim.blocks_list
        for block in source_blocks:
            if block.block_fn != 'Scope':
                continue
            labels = block.exec_params.get('vec_labels', block.params.get('vec_labels'))
            vec = block.exec_params.get('vector', block.params.get('vector'))
            if vec is None:
                continue
            arr = np.array(vec)
            step_flag = step_modes[step_idx] if step_idx < len(step_modes) else False
            step_idx += 1

            if arr.ndim == 1:
                name = labels if isinstance(labels, str) else block.name
                traces.append({'name': name, 'y': arr, 'step': step_flag})
            elif arr.ndim == 2:
                # Multiple channels
                for i in range(arr.shape[1]):
                    if isinstance(labels, (list, tuple)) and i < len(labels):
                        name = labels[i]
                    else:
                        name = f"{block.name}[{i}]"
                    traces.append({'name': name, 'y': arr[:, i], 'step': step_flag})
            else:
                continue

        return self.dsim.timeline, traces

    def _print_verification_summary(self, source_blocks):
        """
        Print verification metrics for scopes that appear to be comparisons.

        Detects scopes with 2 signals and computes error metrics between them.
        """
        for block in source_blocks:
            if block.block_fn != 'Scope':
                continue

            params = getattr(block, 'exec_params', block.params)
            labels = params.get('vec_labels', params.get('labels', ''))
            vector = params.get('vector', [])

            if vector is None or len(vector) == 0:
                continue

            arr = np.array(vector)
            logger.info(f"Verification check {block.name}: arr.shape={arr.shape}, ndim={arr.ndim}")

            if arr.ndim != 2:
                # Skip single-signal scopes
                continue

            n_signals = arr.shape[1]

            # Check if this looks like a comparison (2 signals, or labels suggest comparison)
            is_comparison = n_signals == 2 or 'comparison' in block.name.lower() or 'error' in block.name.lower()

            if is_comparison and n_signals >= 2:
                # Compute error metrics between first two signals
                signal1 = arr[:, 0]
                signal2 = arr[:, 1]

                error = signal1 - signal2
                max_error = np.max(np.abs(error))
                rms_error = np.sqrt(np.mean(error**2))
                max_value = max(np.max(np.abs(signal1)), np.max(np.abs(signal2)))

                # Relative error (avoid division by zero)
                if max_value > 1e-10:
                    rel_max_error = max_error / max_value * 100
                    rel_rms_error = rms_error / max_value * 100
                else:
                    rel_max_error = 0.0
                    rel_rms_error = 0.0

                # Print verification summary
                print("\n" + "="*60)
                print(f"VERIFICATION SUMMARY: {block.name}")
                print("="*60)
                if labels:
                    # Handle labels as either list or comma-separated string
                    if isinstance(labels, list):
                        label_parts = [l.strip() for l in labels]
                    else:
                        label_parts = [l.strip() for l in labels.split(',')]
                    if len(label_parts) >= 2:
                        print(f"  Signal 1: {label_parts[0]}")
                        print(f"  Signal 2: {label_parts[1]}")
                print(f"  Max Absolute Error:  {max_error:.6e}")
                print(f"  RMS Error:           {rms_error:.6e}")
                print(f"  Max Relative Error:  {rel_max_error:.4f}%")
                print(f"  RMS Relative Error:  {rel_rms_error:.4f}%")

                # Sample values at key times
                timeline = self.dsim.timeline
                if len(timeline) > 0:
                    print(f"\n  Sample Values:")
                    print(f"  {'Time':>8}  {'Signal 1':>12}  {'Signal 2':>12}  {'Error':>12}")
                    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

                    # Show t=0, t=1, t=2, and final time
                    sample_times = [0.0, 1.0, 2.0, timeline[-1]]
                    for t in sample_times:
                        idx = np.argmin(np.abs(timeline - t))
                        if idx < len(signal1):
                            print(f"  {timeline[idx]:8.3f}  {signal1[idx]:12.6f}  {signal2[idx]:12.6f}  {error[idx]:12.6e}")

                # Verdict
                if rel_max_error < 1.0:
                    verdict = "EXCELLENT - Error < 1%"
                elif rel_max_error < 5.0:
                    verdict = "GOOD - Error < 5%"
                elif rel_max_error < 10.0:
                    verdict = "ACCEPTABLE - Error < 10%"
                else:
                    verdict = "NEEDS REVIEW - Error >= 10%"

                print(f"\n  Verdict: {verdict}")
                print("="*60 + "\n")

                logger.info(f"Verification {block.name}: Max Error={max_error:.6e}, RMS={rms_error:.6e}, Rel={rel_max_error:.4f}%")

    def pyqtPlotScope(self):
        """
        :purpose: Plots the data saved in Scope blocks using pyqtgraph.
        """
        logger.debug("Attempting to plot...")
        labels_list = []
        vector_list = []
        # Use active blocks from engine if available
        source_blocks = self.dsim.engine.active_blocks_list if hasattr(self.dsim, 'engine') and self.dsim.engine.active_blocks_list else self.dsim.blocks_list
        
        for block in source_blocks:
            if block.block_fn == 'Scope':
                logger.debug(f"Found Scope block: {block.name}")
                # Use exec_params as that's where simulation data is stored
                b_labels = block.exec_params.get('vec_labels', block.params.get('vec_labels'))
                labels_list.append(b_labels)
                b_vectors = block.exec_params.get('vector', block.params.get('vector'))
                if b_vectors is None:
                    b_vectors = []
                vector_list.append(b_vectors)
                logger.debug(f"Full vector for {block.name}: shape {np.shape(b_vectors)}")
                logger.debug(f"Labels: {b_labels}")
                logger.debug(f"Vector length: {len(b_vectors)}")

        step_modes = self._scope_step_modes()

        if labels_list and vector_list:
            logger.debug("Creating SignalPlot...")

            # Expand multi-signal scopes into individual 1D signals
            # This is needed because SignalPlot expects each vector to be 1D
            flat_labels = []
            flat_vectors = []
            flat_step_modes = []

            for idx, (labels, vec) in enumerate(zip(labels_list, vector_list)):
                arr = np.array(vec).astype(float)
                step_flag = step_modes[idx] if idx < len(step_modes) else False

                # Check if this is an interleaved multi-signal vector that needs reshaping
                # The Scope block stores vec_dim for multi-dimensional inputs
                scope_block = [b for b in source_blocks if b.block_fn == 'Scope'][idx] if idx < len([b for b in source_blocks if b.block_fn == 'Scope']) else None
                if scope_block and arr.ndim == 1:
                    vec_dim = scope_block.exec_params.get('vec_dim', scope_block.params.get('vec_dim', 1))
                    if vec_dim > 1 and len(arr) >= vec_dim:
                        # Reshape interleaved flat array to 2D: (num_samples, vec_dim)
                        num_samples = len(arr) // vec_dim
                        arr = arr[:num_samples * vec_dim].reshape(num_samples, vec_dim)
                        logger.debug(f"Reshaped interleaved vector to {arr.shape}")

                if arr.ndim == 2 and arr.shape[1] > 1:
                    # Multi-signal scope: split into individual signals
                    if isinstance(labels, list):
                        signal_labels = labels
                    elif isinstance(labels, str):
                        signal_labels = [l.strip() for l in labels.split(',')]
                    else:
                        signal_labels = [f"Signal {i}" for i in range(arr.shape[1])]

                    for col in range(arr.shape[1]):
                        if col < len(signal_labels):
                            flat_labels.append(signal_labels[col])
                        else:
                            flat_labels.append(f"Signal {col}")
                        flat_vectors.append(arr[:, col])
                        flat_step_modes.append(step_flag)
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    # Single signal stored as 2D - flatten it
                    if isinstance(labels, list) and len(labels) > 0:
                        flat_labels.append(labels[0])
                    elif isinstance(labels, str):
                        flat_labels.append(labels)
                    else:
                        flat_labels.append(f"Signal {idx}")
                    flat_vectors.append(arr.flatten())
                    flat_step_modes.append(step_flag)
                else:
                    # Already 1D
                    if isinstance(labels, list) and len(labels) > 0:
                        flat_labels.append(labels[0])
                    elif isinstance(labels, str):
                        flat_labels.append(labels)
                    else:
                        flat_labels.append(f"Signal {idx}")
                    flat_vectors.append(arr)
                    flat_step_modes.append(step_flag)

            # Log data summary for diagnostics
            t_arr = self.dsim.timeline.astype(float)
            for si, (lbl, vec) in enumerate(zip(flat_labels, flat_vectors)):
                v = np.array(vec)
                logger.info(f"Scope plot [{si}] '{lbl}': len={len(v)}, min={v.min():.4f}, max={v.max():.4f}, "
                           f"first={v[0]:.4f}, last={v[-1]:.4f}, step_mode={flat_step_modes[si] if si < len(flat_step_modes) else False}")

            # Close previous plot window if it exists (prevents stale windows lingering)
            if self.plotty is not None:
                try:
                    self.plotty.close()
                    self.plotty.deleteLater()
                except Exception:
                    pass

            # Use step mode for discrete/ZOH signals to keep values constant between samples
            self.plotty = SignalPlot(self.dsim.sim_dt, flat_labels, len(self.dsim.timeline), step_mode=flat_step_modes)
            # Sync to dsim for backward compatibility
            self.dsim.plotty = self.plotty
            try:
                self.plotty.loop(new_t=t_arr, new_y=flat_vectors)

                # Force visibility
                self.plotty.show()
                self.plotty.raise_()
                self.plotty.activateWindow()

                logger.debug("SignalPlot should be visible now.")
            except Exception as e:
                logger.error(f"Error in plotting: {e}")
        else:
            logger.debug("No data to plot.")

    def dynamic_pyqtPlotScope(self, step):
        """
        :purpose: Plots the data saved in Scope blocks dynamically with pyqtgraph.
        """
        if not self.dsim.dynamic_plot:
            return

        if step == 0:  # init
            labels_list = []
            for block in self.dsim.blocks_list:
                if block.block_fn == 'Scope':
                    b_labels = block.params['vec_labels']
                    labels_list.append(b_labels)

            if labels_list != []:
                # Close previous plot window if it exists
                if self.plotty is not None:
                    try:
                        self.plotty.close()
                        self.plotty.deleteLater()
                    except Exception:
                        pass

                step_modes = self._scope_step_modes()
                self.plotty = SignalPlot(self.dsim.sim_dt, labels_list, self.dsim.plot_trange, step_mode=step_modes)
                # Sync to dsim for backward compatibility
                self.dsim.plotty = self.plotty

        elif step == 1: # loop
            vector_list = []
            for block in self.dsim.blocks_list:
                if block.block_fn == 'Scope':
                    b_vectors = block.params['vector']
                    vector_list.append(b_vectors)
            if len(vector_list) > 0:
                self.plotty.loop(new_t=self.dsim.timeline, new_y=vector_list)
            else:
                self.dsim.dynamic_plot = False
                logger.info("DYNAMIC PLOT: OFF")
