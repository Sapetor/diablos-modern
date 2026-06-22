import logging

import numpy as np

from lib.plotting.field_scope_mixin import _FieldScopeRenderMixin
from lib.plotting.pyqtgraph_scope_mixin import _PyQtGraphScopeMixin

logger = logging.getLogger(__name__)

class ScopePlotter(_FieldScopeRenderMixin, _PyQtGraphScopeMixin):
    """
    Handles plotting logic for Scope, XYGraph, and FFT blocks.
    extracted from DSim to separate concerns.
    """
    def __init__(self, dsim):
        self.dsim = dsim
        self.plotty = None
        # Matplotlib figures created by the various _plot_* helpers. Tracked so
        # they can be closed on the next plot_again() instead of accumulating
        # (and retaining their Qt windows, widgets and captured data arrays)
        # across simulation runs.
        self._open_figs = []
        # Sync to dsim for backward compatibility
        if hasattr(self.dsim, 'plotty'):
            self.dsim.plotty = None

    def _close_open_figures(self):
        """Close any matplotlib figures created by previous plot calls.

        pyplot keeps a global registry of figures, so without this each re-plot
        would leak the old figure (and its Qt window, Slider/Button widgets and
        the large data arrays captured in their update/export closures).
        """
        import matplotlib.pyplot as plt
        for fig in self._open_figs:
            try:
                plt.close(fig)
            except Exception:
                logger.debug("Failed to close a tracked matplotlib figure", exc_info=True)
        self._open_figs = []

    def _register_figure(self, fig):
        """Track a freshly-created figure so it can be closed on the next plot.

        Also closes the figure (releasing its captured closures) when the user
        dismisses its window.
        """
        self._open_figs.append(fig)
        try:
            fig.canvas.mpl_connect(
                'close_event',
                lambda event, f=fig: self._on_figure_closed(f)
            )
        except Exception:
            logger.debug("Failed to connect close_event handler for figure", exc_info=True)

    def _on_figure_closed(self, fig):
        """Drop a figure from the tracking list once its window is closed."""
        try:
            self._open_figs.remove(fig)
        except ValueError:
            logger.debug("Closed figure was not in the tracking list", exc_info=True)

    def _close_plotty(self):
        """Close the previous SignalPlot window if one exists.

        Disconnect destroyed BEFORE deleteLater -- otherwise the OLD widget's
        destroyed signal fires asynchronously and runs _on_plotty_destroyed
        AFTER self.plotty has been reassigned to the new window, which nulls
        the new reference and lets Qt garbage-collect the new (visible) window.
        """
        if self.plotty is not None:
            try:
                self.plotty.destroyed.disconnect()
            except (TypeError, RuntimeError):
                logger.debug("No destroyed signal to disconnect on previous plot window", exc_info=True)
            try:
                self.plotty.close()
                self.plotty.deleteLater()
            except Exception:
                logger.debug("Failed to close/delete previous plot window", exc_info=True)
            self.plotty = None
            if hasattr(self.dsim, 'plotty'):
                self.dsim.plotty = None

    def plot_again(self):
        """
        :purpose: Plots the data saved in Scope and XYGraph blocks without needing to execute the simulation again.
        """
        if self.dsim.dirty:
            logger.error("ERROR: The diagram has been modified. Please run the simulation again.")
            return
        # Close any matplotlib figures left over from a previous plot so they
        # don't accumulate across runs.
        self._close_open_figures()
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
                # No scope data this run -- close any stale SignalPlot window
                # from a previous run so it doesn't linger showing old data.
                self._close_plotty()
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

            # Plot AgentScope (multi-agent 2D scatter animation)
            for block in source_blocks:
                if block.block_fn == 'AgentScope':
                    params = getattr(block, 'exec_params', block.params)
                    history = params.get('_pos_history_', None)
                    has_data = history is not None and len(history) > 1
                    logger.info(f"PLOT DEBUG: AgentScope {block.name} has_data={has_data}")
                    if has_data:
                        try:
                            self._plot_agent_scope(block)
                        except Exception as e:
                            logger.error(f"PLOT ERROR: AgentScope {block.name} plotting failed: {e}")
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
        self._register_figure(fig)
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
        self._register_figure(fig)
        ax.plot(freqs, magnitude, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)' if params.get('log_scale', False) else 'Magnitude')
        ax.set_title(f"{title} ({block.name})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, fs/2])

        plt.tight_layout()
        plt.show()

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
        Print verification metrics for scopes with verify_mode set explicitly.

        Only prints when the scope's verify_mode parameter is set to
        "comparison" or "objective". Skips "auto" and "none" (the default)
        so regular user diagrams don't get unsolicited terminal output.
        """
        for block in source_blocks:
            if block.block_fn != 'Scope':
                continue

            params = getattr(block, 'exec_params', block.params)

            # Only print verification when explicitly requested
            verify_mode = params.get('verify_mode', 'none')
            if verify_mode in ('none', 'auto'):
                continue

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
                    print("\n  Sample Values:")
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
