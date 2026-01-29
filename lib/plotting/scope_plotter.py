
import logging
import numpy as np
from lib.plotting.signal_plot import SignalPlot

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
                        
        except Exception as e:
            logger.info(f"PLOT: Skipping plot; scope data not available yet ({str(e)})")

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
            # Use step mode for discrete/ZOH signals to keep values constant between samples
            self.plotty = SignalPlot(self.dsim.sim_dt, labels_list, len(self.dsim.timeline), step_mode=step_modes)
            # Sync to dsim for backward compatibility
            self.dsim.plotty = self.plotty
            try:
                # Prepare data: ensure flattening if necessary
                clean_vectors = []
                for v in vector_list:
                    arr = np.array(v).astype(float)
                    # if shape is (N, 1), flatten to (N,)
                    if arr.ndim == 2 and arr.shape[1] == 1:
                        arr = arr.flatten()
                    clean_vectors.append(arr)
                    
                self.plotty.loop(new_t=self.dsim.timeline.astype(float), new_y=clean_vectors)
                
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
