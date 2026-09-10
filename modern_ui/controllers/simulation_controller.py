"""Simulation Controller - Orchestrates the simulation lifecycle.

Extracted from ModernCanvas to keep the canvas focused on rendering and
interaction. Owns validation, execution start/stop, batch execution, and
logging the post-run verification report (built by
``lib.engine.verification_report``, which the headless CLI shares). Communicates status to the UI via the
``status_changed`` signal (the canvas re-emits it as its own
``simulation_status_changed`` so existing listeners are unaffected).
"""

import logging

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from lib.i18n import tr
from lib.diagram_validator import check_simulation_state, validate_block_connections
from lib.engine.verification_report import build_verification_report, report_blocks

logger = logging.getLogger(__name__)

# Batch runs that are currently executing on a worker thread. The window's
# 60 FPS ``safe_update`` must not call ``dsim.execution_loop()`` while one of
# these is stepping the same DSim, so it consults ``batch_simulation_active()``.
# A set (not a bool) so a stale entry from one controller cannot unblock
# another, and module-level so main_window does not have to reach into the
# canvas's private controller reference.
_ACTIVE_BATCH_WORKERS = set()


def batch_simulation_active() -> bool:
    """True while any batch simulation is running on a worker thread."""
    return bool(_ACTIVE_BATCH_WORKERS)


class SimulationController(QObject):
    """Drives validation, start/stop, and batch execution for a DSim model."""

    status_changed = pyqtSignal(str)  # Emitted when simulation status changes

    # Emitted when a threaded batch run ends (completed, cancelled or failed).
    # The window uses it to re-arm the tuning panel and reset the toolbar.
    batch_finished = pyqtSignal(bool)  # ok

    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        self.dsim = dsim
        self._batch_worker = None

    def start(self):
        """Start simulation with validation."""
        try:
            logger.info("Starting simulation from canvas")

            # Run validation first
            is_valid, errors = validate_block_connections(
                self.dsim.blocks_list, self.dsim.line_list
            )

            if not is_valid:
                error_msg = "\n".join(errors)
                logger.error(f"Simulation validation failed: {error_msg}")
                self.status_changed.emit(tr("Validation failed: {error}", error=error_msg))
                return False

            # Check simulation state safety
            is_safe, safety_errors = check_simulation_state(self.dsim)
            if not is_safe:
                error_msg = "\n".join(safety_errors)
                logger.error(f"Simulation safety check failed: {error_msg}")
                self.status_changed.emit(tr("Safety check failed: {error}", error=error_msg))
                return False

            # Start simulation
            if hasattr(self.dsim, "execution_init"):
                success = self.dsim.execution_init()
                if success:
                    if self.dsim.real_time:
                        self.status_changed.emit(tr("Simulation started"))
                        logger.info("Simulation started successfully")
                        return True
                    else:
                        self.run_batch()
                        return True
                else:
                    error_msg = (
                        self.dsim.error_msg
                        if hasattr(self.dsim, "error_msg") and self.dsim.error_msg
                        else tr("Initialization failed (see logs).")
                    )
                    logger.error(f"Simulation initialization failed. {error_msg}")
                    self.status_changed.emit(
                        tr("Simulation failed to start. {error}", error=error_msg)
                    )
                    # Also pop up a message box, parented to the owning widget so
                    # it stays attached to / centered on the main window and
                    # inherits the application theme.
                    parent_widget = self.parent() if isinstance(self.parent(), QWidget) else None
                    msgBox = QMessageBox(parent_widget)
                    msgBox.setIcon(QMessageBox.Icon.Critical)
                    msgBox.setText(tr("Simulation Failed to Start"))
                    msgBox.setInformativeText(error_msg)
                    msgBox.setWindowTitle(tr("Simulation Error"))
                    msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msgBox.exec()
                    return False
            else:
                logger.error("DSim does not have execution_init method")
                self.status_changed.emit(tr("Simulation start failed"))
                return False

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            self.status_changed.emit(tr("Error: {error}", error=str(e)))
            return False

    def run_batch(self):
        """Run the simulation in batch mode (as fast as possible).

        Runs on a ``BatchSimulationWorker`` thread so the window keeps
        repainting and the run can be cancelled (Stop). It used to run
        synchronously on the GUI thread behind a single ``processEvents()``,
        which froze the window for the whole run with no way out.

        The live-plot case (``dynamic_plot``) stays synchronous: it drives
        pyqtgraph from inside the step loop, and Qt widgets may only be touched
        from the GUI thread.
        """
        if getattr(self.dsim, "dynamic_plot", False):
            self._run_batch_blocking()
            return

        if self._batch_worker is not None:
            logger.warning("A batch simulation is already running.")
            return

        from modern_ui.widgets.batch_simulation_worker import BatchSimulationWorker

        logger.info("Running simulation in batch mode (worker thread).")
        self.status_changed.emit(tr("Running simulation..."))

        worker = BatchSimulationWorker(self.dsim, parent=self)
        # Bound methods of this QObject (GUI-thread affinity), so Qt queues the
        # calls onto the GUI thread -- results and plots are never touched from
        # the worker. QThread.finished (not our finished_ok) drives deleteLater,
        # so the object is only destroyed after run() has returned.
        worker.progress.connect(self._on_batch_progress)
        worker.finished_ok.connect(self._on_batch_ok)
        worker.failed.connect(self._on_batch_failed)
        worker.finished.connect(worker.deleteLater)
        self._batch_worker = worker
        _ACTIVE_BATCH_WORKERS.add(worker)
        worker.start()

    def _run_batch_blocking(self):
        """Synchronous batch run (live-plot path); blocks the GUI thread."""
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        logger.info("Running simulation in batch mode (blocking, dynamic plot).")
        self.status_changed.emit(tr("Running simulation..."))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            self.dsim.execution_batch()
        finally:
            QApplication.restoreOverrideCursor()
        self._finish_batch(True, "")

    def _on_batch_progress(self, t_now, t_end):
        if t_end:
            self.status_changed.emit(
                tr(
                    "Running simulation... t = {t_now:.4g} / {t_end:.4g} s",
                    t_now=t_now,
                    t_end=t_end,
                )
            )

    def _on_batch_ok(self):
        """Worker finished normally (or was cancelled). GUI thread."""
        self._on_batch_done(True, "")

    def _on_batch_failed(self, message):
        """Worker raised. GUI thread."""
        self._on_batch_done(False, message)

    def _on_batch_done(self, ok, message):
        """Worker-thread completion, delivered on the GUI thread by Qt."""
        worker = self._batch_worker
        self._batch_worker = None
        if worker is not None:
            _ACTIVE_BATCH_WORKERS.discard(worker)
        self._finish_batch(ok, message)

    def _finish_batch(self, ok, message):
        """Post-run work: status, plots, verification report. GUI thread only."""
        if not ok:
            logger.error(f"Batch simulation failed: {message}")
            self.status_changed.emit(f"Simulation failed: {message}")
            self.batch_finished.emit(False)
            return

        solver_type = getattr(self.dsim, "last_solver_type", "Standard")
        self.status_changed.emit(tr("Simulation finished [{solver}]", solver=solver_type))
        logger.info(f"Batch simulation finished. Solver: {solver_type}")
        # Non-modal, and deliberately the *last* status line so it is what the
        # user is left looking at. Same channel the run already reports through
        # (status_changed -> status bar); nothing blocks and nothing is popped.
        self._report_stiffness()
        # Plotting is deliberately done here rather than inside the run: the
        # worker sets defer_plots so no Qt object is created off the GUI thread.
        self.dsim.plot_again()
        self._print_terminal_verification()
        self.batch_finished.emit(True)

    def _report_stiffness(self):
        """Suggest an implicit solver when the last run looked stiff.

        The compiled runner already logged the full diagnosis at warning level;
        this puts the one-line, actionable half in front of the user without
        interrupting them. Silent unless the heuristic actually fired.
        """
        engine = getattr(self.dsim, "engine", None)
        getter = getattr(engine, "get_solver_diagnostics", None)
        if not callable(getter):
            return
        try:
            diagnostics = getter() or {}
        except Exception as e:  # noqa: BLE001 - never break the end of a good run
            logger.debug("Could not read solver diagnostics: %s", e)
            return
        if not diagnostics.get("stiffness_suspected"):
            return
        stiffness = diagnostics.get("stiffness") or {}
        self.status_changed.emit(
            tr(
                "This diagram looks stiff — {method} took {steps} solver steps per output "
                "sample. Try the {suggested} solver in Simulation settings.",
                method=diagnostics.get("method_used", "RK45"),
                steps="{:.0f}".format(stiffness.get("work_ratio", 0.0)),
                suggested=stiffness.get("suggested_method", "LSODA"),
            )
        )

    def is_batch_running(self):
        """True while this controller's batch run is executing on a thread."""
        return self._batch_worker is not None

    def cancel_batch(self, wait_ms=5000):
        """Cancel a threaded batch run and join it (bounded)."""
        worker = self._batch_worker
        if worker is None:
            return False
        try:
            if worker.isRunning():
                worker.cancel()
                if not worker.wait(wait_ms):
                    logger.error("Batch simulation worker did not stop within %d ms", wait_ms)
                    return False
        except RuntimeError:
            # Underlying C++ QThread already deleted; nothing to join.
            pass
        _ACTIVE_BATCH_WORKERS.discard(worker)
        self._batch_worker = None
        return True

    def _print_terminal_verification(self):
        """Log the post-run verification report (see lib.engine.verification_report).

        This used to ``print()`` ~20 lines straight to stdout. In a frozen
        windowed build stdout is an ``io.StringIO`` created by
        ``diablos_modern.py``, so the report was invisible *and* accumulated in
        memory for the life of the process. The report is one multi-line
        ``logger.info`` record, which reaches the log file and the console
        handler alike.
        """
        try:
            report = build_verification_report(report_blocks(self.dsim))
        except Exception as e:  # noqa: BLE001 - never break the end of a good run
            logger.warning(f"Could not assemble verification results: {e}", exc_info=True)
            return
        if report.has_data:
            logger.info(report.text)
        else:
            logger.info("Simulation completed - no verification data")

    def stop(self):
        """Stop simulation safely."""
        try:
            # A threaded batch run owns the step loop; ask it to stop and join
            # before clearing the flag, or it would keep stepping.
            self.cancel_batch()

            if hasattr(self.dsim, "execution_initialized"):
                self.dsim.execution_initialized = False

            self.status_changed.emit(tr("Simulation stopped"))
            logger.info("Simulation stopped")

        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")

    def current_time(self):
        """Get current simulation time."""
        if hasattr(self.dsim, "t"):
            return getattr(self.dsim, "t", 0.0)
        return 0.0

    def is_running(self):
        """Check if simulation is running."""
        return getattr(self.dsim, "execution_initialized", False)
