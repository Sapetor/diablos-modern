"""Simulation Controller - Orchestrates the simulation lifecycle.

Extracted from ModernCanvas to keep the canvas focused on rendering and
interaction. Owns validation, execution start/stop, batch execution, and
logging the post-run verification report (built by
``lib.engine.verification_report``, which the headless CLI shares).

Two signals face the UI, and the canvas re-emits both under its own names:

* ``status_changed(str)`` -- a human-readable, translated status line.
* ``state_changed(str, str)`` -- the run *state* (``idle`` / ``running`` /
  ``paused`` / ``stopping`` / ``error``) plus the message that explains an
  error.  ``stopping`` lasts from Stop until the batch worker has actually
  returned; only then does the controller announce "stopped".  Widgets
  take their state from this signal only; a status line is never parsed for
  keywords (that broke in translated UIs and on any file name containing
  "run").  The state is emitted *before* the matching status line, so a pill
  reset to "Ready" by the idle transition still receives the closing message.
"""

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from lib.i18n import tr
from lib.diagram_validator import check_simulation_state, validate_block_connections
from lib.engine.verification_report import build_verification_report, report_blocks

logger = logging.getLogger(__name__)

#: Blocks that show or save a run's results; a diagram with none of them
#: finishes silently, so the controller says so (``_report_nothing_to_show``).
RESULT_VIEWER_BLOCKS = frozenset(
    {"Scope", "XYGraph", "FFT", "Display", "AgentScope", "FieldScope", "FieldScope2D", "Export"}
)

# Batch runs that are currently executing on a worker thread. The window's
# 60 FPS ``safe_update`` must not call ``dsim.execution_loop()`` while one of
# these is stepping the same DSim, so it consults ``batch_simulation_active()``.
# A set (not a bool) so a stale entry from one controller cannot unblock
# another, and module-level so main_window does not have to reach into the
# canvas's private controller reference.
_ACTIVE_BATCH_WORKERS = set()


SIM_STATES = ("idle", "running", "paused", "stopping", "error")


def batch_simulation_active() -> bool:
    """True while any batch simulation is running on a worker thread."""
    return bool(_ACTIVE_BATCH_WORKERS)


class SimulationController(QObject):
    """Drives validation, start/stop, and batch execution for a DSim model."""

    status_changed = pyqtSignal(str)  # Emitted when simulation status changes

    # (state, message): state is one of SIM_STATES; message is the text that
    # explains an "error" state and "" otherwise.
    state_changed = pyqtSignal(str, str)

    # Emitted when a threaded batch run ends (completed, cancelled or failed).
    # The window uses it to re-arm the tuning panel and reset the toolbar.
    batch_finished = pyqtSignal(bool)  # ok
    # A run could not start or stopped on an error: (raw message, flattened name
    # of the offending block or ""). The window turns it into error-panel
    # entries, canvas highlights and a toast -- never a modal dialog.
    errors_reported = pyqtSignal(str, str)

    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        self.dsim = dsim
        self._batch_worker = None
        # Stop was pressed during the current batch run; its end is then
        # announced as "stopped", not "finished".
        self._stop_requested = False

    def _set_state(self, state, message=""):
        if state not in SIM_STATES:
            raise ValueError(f"unknown simulation state {state!r}")
        self.state_changed.emit(state, message)

    def _report_error(self, message, raw=None, block=""):
        """Put the UI in the error state and show ``message``.

        ``raw`` is the bare error text (without the status-line wrapper) that
        goes to the error panel; ``block`` is the engine's offending block.
        """
        self._set_state("error", message)
        self.status_changed.emit(message)
        self.errors_reported.emit(raw if raw is not None else message, block or "")

    def start(self):
        """Start simulation with validation."""
        try:
            logger.info("Starting simulation from canvas")

            if self._batch_worker is not None:
                # A batch run (possibly one that is stopping) still owns the
                # DSim; execution_init() below would re-arm it under the worker.
                logger.warning("Start ignored: the previous batch run has not finished.")
                self.status_changed.emit(tr("The previous run is still stopping."))
                return False

            # Run validation first
            is_valid, errors = validate_block_connections(
                self.dsim.blocks_list, self.dsim.line_list
            )

            if not is_valid:
                error_msg = "\n".join(errors)
                logger.error(f"Simulation validation failed: {error_msg}")
                self._report_error(tr("Validation failed: {error}", error=error_msg), raw=error_msg)
                return False

            # Check simulation state safety
            is_safe, safety_errors = check_simulation_state(self.dsim)
            if not is_safe:
                error_msg = "\n".join(safety_errors)
                logger.error(f"Simulation safety check failed: {error_msg}")
                self._report_error(
                    tr("Safety check failed: {error}", error=error_msg), raw=error_msg
                )
                return False

            # Start simulation
            if hasattr(self.dsim, "execution_init"):
                success = self.dsim.execution_init()
                if success:
                    if self.dsim.real_time:
                        self._set_state("running")
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
                    # Non-modal: the error panel + toast carry the message and the
                    # offending block (see ModernDiaBloSWindow._on_simulation_errors).
                    self._report_error(
                        tr("Simulation failed to start. {error}", error=error_msg),
                        raw=error_msg,
                        block=getattr(self.dsim, "error_block", ""),
                    )
                    return False
            else:
                logger.error("DSim does not have execution_init method")
                self._report_error(tr("Simulation start failed"))
                return False

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            self._report_error(tr("Error: {error}", error=str(e)))
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
        self._set_state("running")
        self.status_changed.emit(tr("Running simulation..."))

        self._stop_requested = False
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
        self._set_state("running")
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
        self._on_batch_done(True, "", self.sender())

    def _on_batch_failed(self, message):
        """Worker raised. GUI thread."""
        self._on_batch_done(False, message, self.sender())

    def _on_batch_done(self, ok, message, worker=None):
        """Worker-thread completion, delivered on the GUI thread by Qt.

        ``worker`` is the signal's sender. A worker that ``cancel_batch`` has
        already joined and dropped is stale: its queued signal arrives after
        the fact and must not end (or plot over) a newer run.
        """
        current = self._batch_worker
        if worker is not None and worker is not current:
            _ACTIVE_BATCH_WORKERS.discard(worker)
            return
        self._batch_worker = None
        if current is not None:
            _ACTIVE_BATCH_WORKERS.discard(current)
        stopped = self._stop_requested
        self._stop_requested = False
        if stopped and ok and not getattr(self.dsim, "error_msg", ""):
            self._finish_stopped()
        else:
            self._finish_batch(ok, message)

    def _finish_stopped(self):
        """The batch worker has returned after Stop. GUI thread only.

        Whatever the run recorded before it stopped is plotted (the
        interpreter keeps its partial traces; a compiled run cancelled inside
        the solve has none, which closes a stale plot window).
        """
        self._set_state("idle")
        self.status_changed.emit(tr("Simulation stopped"))
        logger.info("Batch simulation stopped by the user.")
        self.dsim.plot_again()
        self.batch_finished.emit(False)

    def _finish_batch(self, ok, message):
        """Post-run work: status, plots, verification report. GUI thread only."""
        if ok and getattr(self.dsim, "error_msg", ""):
            # A block failed mid-run: the engine stops the loop and records the
            # error, so the worker still ends "normally".
            ok, message = False, self.dsim.error_msg
        if not ok:
            logger.error(f"Batch simulation failed: {message}")
            self._report_error(
                tr("Simulation failed: {error}", error=message),
                raw=message,
                block=getattr(self.dsim, "error_block", ""),
            )
            self.batch_finished.emit(False)
            return

        solver_type = getattr(self.dsim, "last_solver_type", "Standard")
        finished = tr("Simulation finished [{solver}]", solver=solver_type)
        summary = self._solver_diagnostics_summary()
        if summary:
            finished = f"{finished}  |  {summary}"
        self._set_state("idle")
        self.status_changed.emit(finished)
        logger.info(f"Batch simulation finished. Solver: {solver_type}")
        # Non-modal, and deliberately the *last* status line so it is what the
        # user is left looking at. Same channel the run already reports through
        # (status_changed -> status bar); nothing blocks and nothing is popped.
        self._report_stiffness()
        self._report_nothing_to_show()
        # Plotting is deliberately done here rather than inside the run: the
        # worker sets defer_plots so no Qt object is created off the GUI thread.
        self.dsim.plot_again()
        self._print_terminal_verification()
        self.batch_finished.emit(True)

    def _solver_diagnostics_summary(self):
        """The compiled solver's one-line diagnostics for the run that just ended.

        Empty when the interpreter ran (it records none), so the status bar
        keeps the plain finished message.
        """
        try:
            summary = self.dsim.last_solver_diagnostics_summary
        except Exception:  # noqa: BLE001 - never break the end of a good run
            logger.debug("Could not read solver diagnostics", exc_info=True)
            return ""
        if summary:
            logger.info("Solver diagnostics: %s", summary)
        return summary or ""

    def _report_nothing_to_show(self):
        """Say so when the run had no block that shows or saves a result.

        Without one a successful run ends with no window and no visible
        change, which reads as "Play did nothing". Same non-modal status
        channel as the stiffness hint.
        """

        def walk(blocks):
            for b in blocks:
                yield b
                yield from walk(getattr(b, "sub_blocks", None) or [])

        if any(
            getattr(b, "block_fn", "") in RESULT_VIEWER_BLOCKS for b in walk(self.dsim.blocks_list)
        ):
            return
        self.status_changed.emit(
            tr(
                "Simulation finished, but there is nothing to show: add a Scope "
                "(or Display / XYGraph) block to see the results."
            )
        )

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
        """Cancel a threaded batch run and join it (bounded). Blocks.

        For teardown (window close, tests). Stop goes through :meth:`stop`,
        which does not block the GUI thread.
        """
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
        """Stop simulation safely.

        A threaded batch run is only *asked* to stop: the state goes to
        ``stopping`` and "Simulation stopped" is announced when the worker's
        finished signal arrives (``_on_batch_done``), never before -- the GUI
        thread is not blocked waiting for it.
        """
        try:
            if self._batch_worker is not None:
                # The worker owns the step loop and the DSim until it returns;
                # it clears execution_initialized itself on the way out.
                if not self._stop_requested:
                    self._stop_requested = True
                    self._batch_worker.cancel()
                    self._set_state("stopping")
                    self.status_changed.emit(tr("Stopping simulation..."))
                    logger.info("Stop requested; waiting for the batch run to finish.")
                return

            if hasattr(self.dsim, "execution_initialized"):
                self.dsim.execution_initialized = False

            self._set_state("idle")
            self.status_changed.emit(tr("Simulation stopped"))
            logger.info("Simulation stopped")

        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")

    def shutdown(self, wait_ms=5000):
        """Stop and join any batch worker before the window is torn down.

        Unlike :meth:`stop` this blocks (bounded): a QThread destroyed while
        still running aborts the process.
        """
        self.stop()
        return self.cancel_batch(wait_ms)

    def pause(self):
        """Pause the interactive step loop (Play resumes it via ``start``)."""
        if hasattr(self.dsim, "execution_pause"):
            self.dsim.execution_pause = True
        self._set_state("paused")

    def current_time(self):
        """Get current simulation time."""
        if hasattr(self.dsim, "t"):
            return getattr(self.dsim, "t", 0.0)
        return 0.0

    def is_running(self):
        """Check if simulation is running."""
        return getattr(self.dsim, "execution_initialized", False)
