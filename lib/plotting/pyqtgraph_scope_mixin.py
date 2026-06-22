"""
pyqtgraph live-plot path for :class:`ScopePlotter`.

Extracted (behavior-preserving) as a mixin so the ~1.2k-line ScopePlotter is
split across files without changing any call site: ``self.pyqtPlotScope`` and
``self.dynamic_pyqtPlotScope`` are still reached through the shared instance
(e.g. from ``plot_again`` and the tuning controller). These build a
:class:`SignalPlot` window and reuse the core helpers (``self._scope_step_modes``,
``self._close_plotty``) via ``self``.
"""
import logging

import numpy as np

from lib.plotting.signal_plot import SignalPlot

logger = logging.getLogger(__name__)


class _PyQtGraphScopeMixin:
    """pyqtgraph SignalPlot rendering methods for :class:`ScopePlotter`."""

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
                v = np.asarray(vec).flatten()
                logger.info(f"Scope plot [{si}] '{lbl}': len={len(v)}, min={float(v.min()):.4f}, max={float(v.max()):.4f}, "
                           f"first={float(v[0]):.4f}, last={float(v[-1]):.4f}, step_mode={flat_step_modes[si] if si < len(flat_step_modes) else False}")

            # Close previous plot window if it exists (prevents stale windows
            # lingering). _close_plotty disconnects destroyed BEFORE deleteLater
            # so the old widget's async destroyed signal can't null the new
            # reference (see _close_plotty for the full rationale).
            self._close_plotty()

            # Use step mode for discrete/ZOH signals to keep values constant between samples
            self.plotty = SignalPlot(self.dsim.sim_dt, flat_labels, len(self.dsim.timeline), step_mode=flat_step_modes)
            # Null out both references when the C++ QWidget is destroyed to avoid
            # RuntimeError on the next simulation run.
            self.plotty.destroyed.connect(lambda: self._on_plotty_destroyed())
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

    def _on_plotty_destroyed(self):
        """Called when the SignalPlot QWidget is destroyed by the user closing it.
        Clears both references so subsequent simulation runs don't hit a deleted C++ object."""
        self.plotty = None
        self.dsim.plotty = None

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
                # Close previous plot window if it exists. See _close_plotty for
                # why we must disconnect destroyed BEFORE deleteLater.
                self._close_plotty()

                step_modes = self._scope_step_modes()
                self.plotty = SignalPlot(self.dsim.sim_dt, labels_list, self.dsim.plot_trange, step_mode=step_modes)
                # Null out both references when the C++ QWidget is destroyed.
                self.plotty.destroyed.connect(lambda: self._on_plotty_destroyed())
                # Sync to dsim for backward compatibility
                self.dsim.plotty = self.plotty

        elif step == 1: # loop
            vector_list = []
            for block in self.dsim.blocks_list:
                if block.block_fn == 'Scope':
                    b_vectors = block.params['vector']
                    vector_list.append(b_vectors)
            if len(vector_list) > 0 and self.plotty is not None:
                self.plotty.loop(new_t=self.dsim.timeline, new_y=vector_list)
            else:
                self.dsim.dynamic_plot = False
                logger.info("DYNAMIC PLOT: OFF")
