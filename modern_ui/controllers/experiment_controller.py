"""
ExperimentController -- the window-side orchestration for the Analysis menu's
"experiment" features: linearization, operating-point (trim) solving,
Monte-Carlo ensembles and parameter sweeps.

Extracted (behavior-preserving) from ``ModernDiaBloSWindow`` so the main window
keeps only thin facades. Constructed with the main window and holds it as
``self.window`` (same pattern as ``modern_ui/managers``).

Responsibilities live here; the actual numeric work lives one layer deeper:
linearize/trim delegate to ``AnalysisController`` (``modern_ui/controllers/
analysis_controller.py``) and the ensemble/sweep runs are driven off the UI
thread by ``MonteCarloWorker`` / ``ParameterSweepWorker``
(``modern_ui/widgets/``). This class opens the dialogs, drives the modal
progress dialogs, and shows the result windows.

Worker lifetime: the running ``QThread`` references are stored back on the
window as ``window._mc_worker`` / ``window._sweep_worker`` (one experiment of
each kind at a time), and joined on shutdown via :meth:`cancel_workers` (wired
from ``ModernDiaBloSWindow.closeEvent``). Keeping them on the window preserves
the teardown contract exercised by
``tests/modern_ui/test_main_window_worker_teardown.py``.
"""

import logging

logger = logging.getLogger(__name__)


class ExperimentController:
    """Owns the window-side analysis/experiment action handlers."""

    def __init__(self, main_window):
        self.window = main_window
        # Retain top-level result windows so they are not garbage-collected the
        # moment the handler returns (they have no parent widget).
        self._result_windows = []

    def _retain_window(self, win):
        """Keep a parentless top-level result window alive (prevent GC) and show it."""
        self._result_windows.append(win)
        win.show()

    def linearize_and_analyze(self):
        """Linearize the current diagram at an operating point and show analysis.

        Opens an input/output picker, runs the numerical linearizer on the
        compiled ODE RHS, and shows a pole-zero / Bode / summary window.
        """
        from PyQt5.QtWidgets import QMessageBox, QDialog
        window = self.window
        if not window.dsim.blocks_list:
            QMessageBox.information(window, "Linearize & Analyze", "No blocks to analyze.")
            return

        from modern_ui.widgets.linearize_dialog import LinearizeDialog
        dlg = LinearizeDialog(window.dsim, parent=window)
        if dlg.exec_() != QDialog.Accepted:
            return
        sel = dlg.get_selection()

        from modern_ui.controllers.analysis_controller import AnalysisController
        result = AnalysisController(window.dsim).analyze(
            input_blocks=sel.get("input_blocks") or None,
            output_blocks=sel.get("output_blocks") or None,
            find_trim=sel.get("find_trim", False),
        )

        from modern_ui.widgets.linearization_result_window import LinearizationResultWindow
        win = LinearizationResultWindow(result)  # top-level window
        self._retain_window(win)

    def find_operating_point(self):
        """Solve for an equilibrium (trim point) of the current diagram.

        Runs the operating-point solver on the compiled ODE RHS and shows the
        equilibrium state values in a table. The result's operating point can be
        copied and reused as a starting point for linearization.
        """
        from PyQt5.QtWidgets import QMessageBox
        window = self.window
        if not window.dsim.blocks_list:
            QMessageBox.information(
                window, "Find Operating Point", "No blocks to analyze.")
            return

        from modern_ui.controllers.analysis_controller import AnalysisController
        result = AnalysisController(window.dsim).find_trim()

        from modern_ui.widgets.operating_point_window import OperatingPointWindow
        win = OperatingPointWindow(result)  # top-level window
        self._retain_window(win)

    def run_monte_carlo(self):
        """Run a Monte-Carlo ensemble of the current diagram and show statistics.

        Opens a dialog (N runs, master seed, sim time/dt), runs the seeded
        ensemble on a background thread behind a cancellable progress dialog, and
        shows a mean + percentile-band / outcome-histogram window. The run is off
        the UI thread so the GUI stays responsive; the modal progress dialog
        keeps anything else from mutating the diagram mid-run, and cancelling
        still shows the partial ensemble gathered so far.
        """
        from PyQt5.QtWidgets import QMessageBox, QDialog, QProgressDialog
        from PyQt5.QtCore import Qt
        window = self.window
        if not window.dsim.blocks_list:
            QMessageBox.information(window, "Monte Carlo", "No blocks to simulate.")
            return
        # Re-entrancy guard: one ensemble at a time (it mutates/restores diagram params).
        if getattr(window, '_mc_worker', None) is not None:
            QMessageBox.information(
                window, "Monte Carlo", "A Monte-Carlo run is already running.")
            return

        from modern_ui.widgets.monte_carlo_dialog import MonteCarloDialog
        dlg = MonteCarloDialog(window.dsim, parent=window)
        if dlg.exec_() != QDialog.Accepted:
            return
        sel = dlg.get_selection()
        n_runs = int(sel.get("n_runs", 100))

        progress = QProgressDialog(
            "Running Monte-Carlo ensemble...", "Cancel", 0, n_runs, window)
        progress.setWindowTitle("Monte Carlo")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        from modern_ui.widgets.monte_carlo_worker import MonteCarloWorker
        worker = MonteCarloWorker(window.dsim, sel, parent=window)

        def _on_progress(done, total):
            if progress.maximum() != total:
                progress.setMaximum(total)
            progress.setLabelText(
                f"Running Monte-Carlo ensemble... ({done}/{total})")
            progress.setValue(done)

        def _on_finished(result):
            progress.close()
            window._mc_worker = None
            self._show_ensemble_result(result)

        def _on_failed(msg):
            progress.close()
            window._mc_worker = None
            QMessageBox.critical(
                window, "Monte Carlo", f"Monte-Carlo run failed:\n{msg}")

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.failed.connect(_on_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        progress.canceled.connect(worker.cancel)

        # Retain a reference so the QThread is not garbage-collected mid-run.
        # Show the modal progress dialog before starting the thread.
        window._mc_worker = worker
        progress.show()
        worker.start()

    def _show_ensemble_result(self, result):
        """Open a (retained) results window for a Monte-Carlo ensemble result."""
        from modern_ui.widgets.ensemble_result_window import EnsembleResultWindow
        win = EnsembleResultWindow(result)  # top-level window
        self._retain_window(win)

    def run_parameter_sweep(self):
        """Sweep one/two block parameters across a grid and show the results.

        Opens a dialog (axes, ranges, sim time/dt), runs the grid on a background
        thread behind a cancellable progress dialog, and shows a response-family /
        metric-vs-parameter window (1-D) or an outcome-metric heatmap (2-D). The
        run is off the UI thread; the modal progress dialog keeps anything else
        from mutating the diagram mid-run, and cancelling still shows the partial
        grid gathered so far.
        """
        from PyQt5.QtWidgets import QMessageBox, QDialog, QProgressDialog
        from PyQt5.QtCore import Qt
        window = self.window
        if not window.dsim.blocks_list:
            QMessageBox.information(window, "Parameter Sweep", "No blocks to simulate.")
            return
        # Re-entrancy guard: one sweep at a time (it mutates/restores diagram params).
        if getattr(window, '_sweep_worker', None) is not None:
            QMessageBox.information(
                window, "Parameter Sweep", "A parameter sweep is already running.")
            return

        from modern_ui.widgets.parameter_sweep_dialog import (
            ParameterSweepDialog, sweepable_blocks,
        )
        if not sweepable_blocks(window.dsim):
            QMessageBox.information(
                window, "Parameter Sweep",
                "No block exposes a numeric scalar parameter to sweep.")
            return

        dlg = ParameterSweepDialog(window.dsim, parent=window)
        if dlg.exec_() != QDialog.Accepted:
            return
        sel = dlg.get_selection()
        if any(not ax.get("param") for ax in sel.get("axes", [])):
            QMessageBox.information(
                window, "Parameter Sweep", "Please choose a parameter for each axis.")
            return

        # Total grid points = product of each axis's value count.
        total = 1
        for ax in sel.get("axes", []):
            total *= max(1, len(ax.get("values", [])))

        progress = QProgressDialog(
            "Running parameter sweep...", "Cancel", 0, total, window)
        progress.setWindowTitle("Parameter Sweep")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)

        from modern_ui.widgets.parameter_sweep_worker import ParameterSweepWorker
        worker = ParameterSweepWorker(window.dsim, sel, parent=window)

        def _on_progress(done, total_):
            if progress.maximum() != total_:
                progress.setMaximum(total_)
            progress.setLabelText(f"Running parameter sweep... ({done}/{total_})")
            progress.setValue(done)

        def _on_finished(result):
            progress.close()
            window._sweep_worker = None
            self._show_sweep_result(result)

        def _on_failed(msg):
            progress.close()
            window._sweep_worker = None
            QMessageBox.critical(
                window, "Parameter Sweep", f"Parameter sweep failed:\n{msg}")

        worker.progress.connect(_on_progress)
        worker.finished.connect(_on_finished)
        worker.failed.connect(_on_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        progress.canceled.connect(worker.cancel)

        # Retain a reference so the QThread is not garbage-collected mid-run.
        # Show the modal progress dialog before starting the thread.
        window._sweep_worker = worker
        progress.show()
        worker.start()

    def _show_sweep_result(self, result):
        """Open a (retained) results window for a parameter-sweep result."""
        from modern_ui.widgets.sweep_result_window import SweepResultWindow
        win = SweepResultWindow(result)  # top-level window
        self._retain_window(win)

    def cancel_workers(self):
        """Cancel and join any running Monte-Carlo / parameter-sweep worker
        threads.

        Without this, closing the window while an ensemble or sweep is running
        destroys a live ``QThread`` ("QThread: Destroyed while thread is still
        running"), which aborts the process. Both workers cancel cooperatively
        (the flag is polled before each run), so a bounded ``wait()`` joins them.

        The worker references live on the window (``window._mc_worker`` /
        ``window._sweep_worker``); this reads and clears them there.
        """
        window = self.window
        for attr in ('_mc_worker', '_sweep_worker'):
            worker = getattr(window, attr, None)
            if worker is None:
                continue
            try:
                if worker.isRunning():
                    worker.cancel()  # cooperative stop, polled before each run
                    if not worker.wait(10000):  # bounded join (ms)
                        logger.warning(
                            "%s did not stop within timeout on close.", attr)
            except RuntimeError:
                # Underlying C++ QThread already deleted; nothing to join.
                pass
            setattr(window, attr, None)
