"""
Tuning Controller - Orchestrates live parameter tuning re-simulation.

Connects the TuningPanel slider signals to headless re-simulation via DSim,
then updates existing SignalPlot windows in-place.
"""

import logging
import numpy as np
from PyQt5.QtCore import QObject, QTimer, Qt

logger = logging.getLogger(__name__)


class TuningController(QObject):
    """
    Orchestrates the re-simulation cycle for live parameter tuning.

    On each slider change:
    1. Accumulates param changes (debounced at 50ms)
    2. Applies pending changes to block.params
    3. Calls dsim.run_tuning_simulation() headlessly
    4. Updates existing SignalPlot via plotty.loop()
    """

    def __init__(self, dsim, scope_plotter, parent=None):
        super().__init__(parent)
        self.dsim = dsim
        self.scope_plotter = scope_plotter
        self._sim_time = None
        self._sim_dt = None
        self._active = False
        self._pending_changes = {}  # (block_name, param_name) -> value
        self._status_callback = None  # Optional: fn(str) for status messages

        # Debounce timer: accumulate changes, fire once user pauses
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(50)
        self._debounce.timeout.connect(self._execute_tuning)

    def set_status_callback(self, callback):
        """Set a callback for status messages (e.g. statusbar update)."""
        self._status_callback = callback

    def store_sim_params(self, sim_time, sim_dt):
        """Store simulation parameters from the last normal run."""
        self._sim_time = sim_time
        self._sim_dt = sim_dt
        self._active = True
        self._set_plotty_on_top(True)
        logger.info(f"Tuning controller armed: T={sim_time}s, dt={sim_dt}s")

    def deactivate(self):
        """Deactivate tuning (e.g. when diagram is modified)."""
        self._active = False
        self._pending_changes.clear()
        self._debounce.stop()
        self._set_plotty_on_top(False)

    def _set_plotty_on_top(self, on_top):
        """Toggle WindowStaysOnTopHint on the scope window."""
        plotty = self.scope_plotter.plotty
        if plotty is None:
            return
        try:
            was_visible = plotty.isVisible()
            if on_top:
                plotty.setWindowFlags(plotty.windowFlags() | Qt.WindowStaysOnTopHint)
            else:
                plotty.setWindowFlags(plotty.windowFlags() & ~Qt.WindowStaysOnTopHint)
            # setWindowFlags hides the widget, so re-show if it was visible
            if was_visible:
                plotty.show()
        except Exception as e:
            logger.debug(f"Could not set plotty on-top: {e}")

    @property
    def is_active(self):
        return self._active and self._sim_time is not None

    def on_param_changed(self, block_name, param_name, value):
        """
        Called by TuningPanel on every slider move.
        Accumulates changes and restarts debounce timer.
        """
        if not self.is_active:
            self._set_status("Run simulation first (F5) before tuning")
            return

        self._pending_changes[(block_name, param_name)] = value
        self._debounce.start()  # Restart the 50ms timer

    def _execute_tuning(self):
        """Apply pending param changes, re-simulate, update plots."""
        if not self._pending_changes:
            return

        # 1. Apply pending param changes to block.params
        changes = dict(self._pending_changes)
        self._pending_changes.clear()

        import re
        blocks_list = self.dsim.blocks_list
        for (block_name, param_name), value in changes.items():
            for block in blocks_list:
                if block.name == block_name:
                    # Handle indexed list params: "denominator[1]" -> update list element
                    match = re.match(r'^(.+)\[(\d+)\]$', param_name)
                    if match:
                        base_name, idx = match.group(1), int(match.group(2))
                        base_val = block.params.get(base_name)
                        if isinstance(base_val, (list, tuple)) and idx < len(base_val):
                            base_val = list(base_val)
                            base_val[idx] = value
                            block.params[base_name] = base_val
                    else:
                        block.params[param_name] = value
                    break

        # 2. Run headless re-simulation
        self._set_status("Re-simulating...")
        success, error_msg = self.dsim.run_tuning_simulation(
            self._sim_time, self._sim_dt
        )

        if not success:
            self._set_status(f"Tuning re-sim failed: {error_msg}")
            logger.warning(f"Tuning re-simulation failed: {error_msg}")
            return

        # 3. Collect scope data and update existing plots
        self._update_plots()
        self._set_status("Tuning: parameters updated")

    def _update_plots(self):
        """Read scope data from blocks and update the existing SignalPlot."""
        plotty = self.scope_plotter.plotty
        if plotty is None or not plotty.isVisible():
            # Plot window was closed — recreate it
            self.scope_plotter.pyqtPlotScope()
            self._set_plotty_on_top(True)
            return

        # Collect scope vectors (same logic as pyqtPlotScope)
        source_blocks = (
            self.dsim.engine.active_blocks_list
            if hasattr(self.dsim, 'engine') and self.dsim.engine.active_blocks_list
            else self.dsim.blocks_list
        )

        flat_vectors = []
        for block in source_blocks:
            if block.block_fn == 'Scope':
                params = getattr(block, 'exec_params', block.params)
                vec = params.get('vector', block.params.get('vector'))
                if vec is None:
                    vec = []
                arr = np.array(vec).astype(float)

                # Handle multi-dimensional scope data
                vec_dim = params.get('vec_dim', block.params.get('vec_dim', 1))
                if arr.ndim == 1 and vec_dim > 1 and len(arr) >= vec_dim:
                    num_samples = len(arr) // vec_dim
                    arr = arr[:num_samples * vec_dim].reshape(num_samples, vec_dim)

                if arr.ndim == 2 and arr.shape[1] > 1:
                    for col in range(arr.shape[1]):
                        flat_vectors.append(arr[:, col])
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    flat_vectors.append(arr.flatten())
                else:
                    flat_vectors.append(arr)

        if flat_vectors and self.dsim.timeline is not None:
            try:
                plotty.loop(
                    new_t=self.dsim.timeline.astype(float),
                    new_y=flat_vectors
                )
            except Exception as e:
                logger.error(f"Error updating tuning plots: {e}")
                # Fallback: recreate plot
                self.scope_plotter.pyqtPlotScope()

    def _set_status(self, msg):
        """Send status message via callback if available."""
        if self._status_callback:
            self._status_callback(msg)
        logger.debug(f"Tuning status: {msg}")
