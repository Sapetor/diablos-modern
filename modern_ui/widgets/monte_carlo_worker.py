"""Background worker for Monte-Carlo ensemble runs.

Runs :meth:`lib.analysis.monte_carlo.MonteCarloRunner.run` on a ``QThread`` so
the GUI stays responsive while an ensemble executes, a progress dialog can
update per completed run, and the user can cancel mid-run. Mirrors the
``ExportWorker`` pattern in ``animation_export_dialog.py``.

Signals:
  * ``progress(done, total)`` -- emitted once per completed run.
  * ``finished(result)``      -- the ensemble-result dict (see ``MonteCarloRunner``);
                                 on cancellation it holds the partial ensemble.
  * ``failed(message)``       -- the run raised; ``message`` is the error text.

The worker only reads/restores the diagram the way ``MonteCarloRunner`` already
does. Keep the run modal at the UI level (a modal progress dialog) so nothing
else mutates ``dsim`` while the worker is iterating.
"""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from lib.analysis.monte_carlo import MonteCarloRunner

logger = logging.getLogger(__name__)


class MonteCarloWorker(QThread):
    """Run a Monte-Carlo ensemble off the UI thread, reporting progress."""

    progress = pyqtSignal(int, int)   # done, total
    finished = pyqtSignal(object)     # ensemble-result dict
    failed = pyqtSignal(str)          # error message

    def __init__(self, dsim, selection, parent=None):
        """
        Args:
            dsim: DSim instance to ensemble-run (never mutated past restore).
            selection: dict from ``MonteCarloDialog.get_selection()`` -- keys
                ``n_runs``, ``master_seed``, ``sim_time``, ``sim_dt`` and an
                optional ``samplers``.
            parent: optional QObject parent.
        """
        super().__init__(parent)
        self.dsim = dsim
        self.selection = dict(selection or {})
        self._cancelled = False

    def cancel(self):
        """Request cooperative cancellation (polled before each run)."""
        self._cancelled = True

    def run(self):
        """Execute the ensemble; emit ``finished`` or ``failed``."""
        sel = self.selection
        try:
            result = MonteCarloRunner(self.dsim).run(
                n_runs=sel.get("n_runs", 100),
                master_seed=sel.get("master_seed", 12345),
                sim_time=sel.get("sim_time"),
                sim_dt=sel.get("sim_dt"),
                samplers=sel.get("samplers"),
                progress_cb=lambda done, total: self.progress.emit(done, total),
                cancel_cb=lambda: self._cancelled,
            )
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            logger.exception("Monte-Carlo worker failed")
            self.failed.emit(str(e))
