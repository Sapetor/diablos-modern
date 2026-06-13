"""Background worker for parameter-sweep runs.

Runs :meth:`lib.analysis.parameter_sweep.ParameterSweepRunner.run` on a
``QThread`` so the GUI stays responsive while a sweep grid executes, a progress
dialog can update per grid point, and the user can cancel mid-run. Mirrors
``MonteCarloWorker``.

Signals:
  * ``progress(done, total)`` -- emitted once per completed grid point.
  * ``finished(result)``      -- the sweep-result dict; on cancellation it holds
                                 the partial grid gathered so far.
  * ``failed(message)``       -- the run raised; ``message`` is the error text.
"""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from lib.analysis.parameter_sweep import ParameterSweepRunner

logger = logging.getLogger(__name__)


class ParameterSweepWorker(QThread):
    """Run a parameter sweep off the UI thread, reporting progress."""

    progress = pyqtSignal(int, int)   # done, total
    finished = pyqtSignal(object)     # sweep-result dict
    failed = pyqtSignal(str)          # error message

    def __init__(self, dsim, selection, parent=None):
        """
        Args:
            dsim: DSim instance to sweep (never mutated past restore).
            selection: dict from ``ParameterSweepDialog.get_selection()`` -- keys
                ``axes`` (list of {block, param, values}), ``sim_time``, ``sim_dt``.
            parent: optional QObject parent.
        """
        super().__init__(parent)
        self.dsim = dsim
        self.selection = dict(selection or {})
        self._cancelled = False

    def cancel(self):
        """Request cooperative cancellation (polled before each grid point)."""
        self._cancelled = True

    def run(self):
        """Execute the sweep; emit ``finished`` or ``failed``."""
        sel = self.selection
        try:
            result = ParameterSweepRunner(self.dsim).run(
                axes=sel.get("axes"),
                sim_time=sel.get("sim_time"),
                sim_dt=sel.get("sim_dt"),
                progress_cb=lambda done, total: self.progress.emit(done, total),
                cancel_cb=lambda: self._cancelled,
            )
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            logger.exception("Parameter-sweep worker failed")
            self.failed.emit(str(e))
