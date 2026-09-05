"""Background worker for batch ("as fast as possible") simulation runs.

Runs :meth:`lib.lib.DSim.execution_batch` on a ``QThread`` so a long run no
longer freezes the window. Before this, ``SimulationController.run_batch``
called ``execution_batch()`` synchronously on the GUI thread behind a single
``QApplication.processEvents()`` and a wait cursor: the window stopped
repainting, the menu bar stopped responding, and there was no way to cancel.

Signals:
  * ``progress(t_now, t_end)`` -- emitted at most every ``_MIN_EMIT_INTERVAL``
    seconds while the interpreter loop advances (the compiled solver runs
    inside SciPy and has no step hook, so it reports nothing until it returns).
  * ``finished_ok()``          -- the run completed (or was cancelled cleanly).
  * ``failed(message)``        -- the run raised; ``message`` is the error text.

Threading contract: ``execution_batch(defer_plots=True)`` suppresses every Qt
call the interpreter loop would otherwise make (dynamic scope updates and the
end-of-run ``pyqtPlotScope``). Plotting is the caller's job, on the GUI thread,
from the ``finished_ok`` slot. The worker still writes to the diagram's blocks,
so the window must not step the same ``DSim`` while it runs -- see
``modern_ui.controllers.simulation_controller.batch_simulation_active``.
"""

import logging
import time

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

# Progress is emitted at most this often (seconds); a 1 ms-dt run would
# otherwise queue tens of thousands of cross-thread signals per second.
_MIN_EMIT_INTERVAL = 0.05


class BatchSimulationWorker(QThread):
    """Run ``dsim.execution_batch()`` off the UI thread, reporting progress."""

    progress = pyqtSignal(float, float)  # t_now, t_end
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, dsim, parent=None):
        """
        Args:
            dsim: the DSim to run. ``execution_init()`` must already have
                succeeded on it (the controller does that on the GUI thread).
            parent: optional QObject parent.
        """
        super().__init__(parent)
        self.dsim = dsim
        self._cancelled = False
        self._last_emit = 0.0

    def cancel(self):
        """Request cooperative cancellation (polled once per interpreter step)."""
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled

    def _on_progress(self, t_now, t_end):
        now = time.monotonic()
        if now - self._last_emit < _MIN_EMIT_INTERVAL:
            return
        self._last_emit = now
        self.progress.emit(float(t_now), float(t_end or 0.0))

    def run(self):
        """Execute the batch run; emit ``finished_ok`` or ``failed``."""
        try:
            self.dsim.execution_batch(
                progress_cb=self._on_progress,
                cancel_cb=lambda: self._cancelled,
                defer_plots=True,
            )
            self.finished_ok.emit()
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            logger.exception("Batch simulation worker failed")
            self.failed.emit(str(e))
