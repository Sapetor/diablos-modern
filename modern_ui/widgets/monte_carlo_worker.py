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

The worker runs against an isolated ``DSim`` copy supplied by
``ExperimentController`` -- the runner rewrites blocks_list/line_list/timeline
while it iterates, which raced the window's repaint timer when it was handed the
live diagram. The modal progress dialog additionally keeps the user from editing
the diagram mid-experiment.
"""

import logging
from concurrent.futures.process import BrokenProcessPool

from PyQt6.QtCore import QThread, pyqtSignal

from lib.analysis.monte_carlo import MonteCarloRunner

logger = logging.getLogger(__name__)


def run_with_pool_fallback(run, workers):
    """``run(workers)``; if the process pool cannot work here, rerun in-process.

    A pool that breaks (``BrokenProcessPool``: a worker failed to start or
    died) falls back to ``workers=1``. Results do not depend on the worker
    count, so the rerun yields the same experiment. Shared with the sweep worker.
    """
    workers = int(workers or 1)
    if workers <= 1:
        return run(1)
    try:
        return run(workers)
    except BrokenProcessPool:
        logger.warning("Experiment process pool failed; rerunning in-process", exc_info=True)
        return run(1)


class MonteCarloWorker(QThread):
    """Run a Monte-Carlo ensemble off the UI thread, reporting progress."""

    progress = pyqtSignal(int, int)  # done, total
    finished = pyqtSignal(object)  # ensemble-result dict
    failed = pyqtSignal(str)  # error message

    def __init__(self, dsim, selection, parent=None):
        """
        Args:
            dsim: DSim to ensemble-run. ExperimentController hands over a
                private copy (``DSim.clone_for_analysis``) so the worker never
                touches the diagram the GUI is painting.
            selection: dict from ``MonteCarloDialog.get_selection()`` -- keys
                ``n_runs``, ``master_seed``, ``sim_time``, ``sim_dt`` and the
                optional ``samplers``, ``workers`` (default 1) and
                ``keep_traces`` (default True).
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

        def _run(workers):
            return MonteCarloRunner(self.dsim).run(
                n_runs=sel.get("n_runs", 100),
                master_seed=sel.get("master_seed", 12345),
                sim_time=sel.get("sim_time"),
                sim_dt=sel.get("sim_dt"),
                samplers=sel.get("samplers"),
                progress_cb=lambda done, total: self.progress.emit(done, total),
                cancel_cb=lambda: self._cancelled,
                keep_traces=sel.get("keep_traces", True),
                workers=workers,
            )

        try:
            result = run_with_pool_fallback(_run, sel.get("workers", 1))
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            logger.exception("Monte-Carlo worker failed")
            self.failed.emit(str(e))
