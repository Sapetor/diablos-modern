"""Simulation Controller - Orchestrates the simulation lifecycle.

Extracted from ModernCanvas to keep the canvas focused on rendering and
interaction. Owns validation, execution start/stop, batch execution, and the
post-run terminal verification report. Communicates status to the UI via the
``status_changed`` signal (the canvas re-emits it as its own
``simulation_status_changed`` so existing listeners are unaffected).
"""

import logging

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QWidget

from lib.improvements import SafetyChecks, ValidationHelper

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
            is_valid, errors = ValidationHelper.validate_block_connections(
                self.dsim.blocks_list, self.dsim.line_list
            )

            if not is_valid:
                error_msg = "\n".join(errors)
                logger.error(f"Simulation validation failed: {error_msg}")
                self.status_changed.emit(f"Validation failed: {error_msg}")
                return False

            # Check simulation state safety
            is_safe, safety_errors = SafetyChecks.check_simulation_state(self.dsim)
            if not is_safe:
                error_msg = "\n".join(safety_errors)
                logger.error(f"Simulation safety check failed: {error_msg}")
                self.status_changed.emit(f"Safety check failed: {error_msg}")
                return False

            # Start simulation
            if hasattr(self.dsim, "execution_init"):
                success = self.dsim.execution_init()
                if success:
                    if self.dsim.real_time:
                        self.status_changed.emit("Simulation started")
                        logger.info("Simulation started successfully")
                        return True
                    else:
                        self.run_batch()
                        return True
                else:
                    error_msg = (
                        self.dsim.error_msg
                        if hasattr(self.dsim, "error_msg") and self.dsim.error_msg
                        else "Initialization failed (see logs)."
                    )
                    logger.error(f"Simulation initialization failed. {error_msg}")
                    self.status_changed.emit(f"Simulation failed to start. {error_msg}")
                    # Also pop up a message box, parented to the owning widget so
                    # it stays attached to / centered on the main window and
                    # inherits the application theme.
                    parent_widget = self.parent() if isinstance(self.parent(), QWidget) else None
                    msgBox = QMessageBox(parent_widget)
                    msgBox.setIcon(QMessageBox.Critical)
                    msgBox.setText("Simulation Failed to Start")
                    msgBox.setInformativeText(error_msg)
                    msgBox.setWindowTitle("Simulation Error")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.exec_()
                    return False
            else:
                logger.error("DSim does not have execution_init method")
                self.status_changed.emit("Simulation start failed")
                return False

        except Exception as e:
            logger.error(f"Error starting simulation: {str(e)}", exc_info=True)
            self.status_changed.emit(f"Error: {str(e)}")
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
        self.status_changed.emit("Running simulation...")

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
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt

        logger.info("Running simulation in batch mode (blocking, dynamic plot).")
        self.status_changed.emit("Running simulation...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.dsim.execution_batch()
        finally:
            QApplication.restoreOverrideCursor()
        self._finish_batch(True, "")

    def _on_batch_progress(self, t_now, t_end):
        if t_end:
            self.status_changed.emit(f"Running simulation... t = {t_now:.4g} / {t_end:.4g} s")

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
        self.status_changed.emit(f"Simulation finished [{solver_type}]")
        logger.info(f"Batch simulation finished. Solver: {solver_type}")
        # Plotting is deliberately done here rather than inside the run: the
        # worker sets defer_plots so no Qt object is created off the GUI thread.
        self.dsim.plot_again()
        self._print_terminal_verification()
        self.batch_finished.emit(True)

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
        """Log the post-run verification report.

        This used to ``print()`` ~20 lines straight to stdout. In a frozen
        windowed build stdout is an ``io.StringIO`` created by
        ``diablos_modern.py``, so the report was invisible *and* accumulated in
        memory for the life of the process. The report is now assembled into
        one multi-line ``logger.info`` record, which reaches the log file and
        the console handler alike.
        """
        import numpy as np

        report = []

        def emit(line=""):
            report.append(line)

        try:
            # Use active blocks from engine if available, otherwise fall back to blocks_list
            has_engine = hasattr(self.dsim, "engine") and self.dsim.engine is not None
            use_active = has_engine and len(self.dsim.engine.active_blocks_list) > 0
            blocks_source = (
                self.dsim.engine.active_blocks_list if use_active else self.dsim.blocks_list
            )

            # Collect Display block values
            display_values = {}
            for block in blocks_source:
                if block.block_fn == "Display":
                    params = block.params or {}
                    display_val = params.get("_display_value_", "---")
                    label = params.get("label", "")
                    block_name = label if label else block.username
                    display_values[block_name] = display_val

            # Collect StateVariable final states (optimization convergence)
            state_values = {}
            for block in blocks_source:
                if block.block_fn == "StateVariable":
                    exec_params = getattr(block, "exec_params", {}) or {}
                    state = exec_params.get("_state_")
                    initial = exec_params.get("initial_value")
                    if state is not None:
                        state_arr = np.atleast_1d(state)
                        initial_arr = np.atleast_1d(initial) if initial is not None else None
                        block_name = block.username if block.username else block.name
                        state_values[block_name] = {"final": state_arr, "initial": initial_arr}

            # Collect Scope convergence info (first/last values)
            scope_convergence = {}
            for block in blocks_source:
                if block.block_fn == "Scope":
                    exec_params = getattr(block, "exec_params", {}) or {}
                    vec = exec_params.get("vector")
                    if vec is not None and hasattr(vec, "__len__") and len(vec) > 0:
                        arr = np.array(vec)
                        vec_dim = exec_params.get("vec_dim", 1)
                        labels = exec_params.get("vec_labels", block.username)

                        # Reshape if interleaved multi-dimensional
                        if arr.ndim == 1 and vec_dim > 1 and len(arr) >= vec_dim:
                            num_samples = len(arr) // vec_dim
                            arr = arr[: num_samples * vec_dim].reshape(num_samples, vec_dim)

                        block_name = block.username if block.username else block.name
                        if arr.ndim == 2:
                            first_val = arr[0, :]
                            last_val = arr[-1, :]
                        else:
                            first_val = arr[0] if len(arr) > 0 else None
                            last_val = arr[-1] if len(arr) > 0 else None
                        scope_convergence[block_name] = {
                            "labels": labels,
                            "first": first_val,
                            "last": last_val,
                            "samples": len(arr),
                            "data": arr,
                            "verify_mode": exec_params.get("verify_mode", "auto"),
                        }

            # Build output with verification checks
            has_output = display_values or state_values or scope_convergence
            all_checks_passed = True

            if has_output:
                emit("\n" + "=" * 60)
                emit("VERIFICATION RESULTS")
                emit("=" * 60)

                # Display block values
                if display_values:
                    emit("\n📊 Display Values:")
                    for name, value in display_values.items():
                        emit(f"   {name}: {value}")

                # StateVariable convergence check
                if state_values:
                    emit("\n🎯 Optimization Convergence:")
                    for name, info in state_values.items():
                        final = info["final"]
                        initial = info["initial"]

                        # Check if converged to near zero (common for quadratic minimization)
                        final_norm = np.linalg.norm(final)
                        converged_to_zero = final_norm < 1e-3

                        # Check if state changed from initial
                        if initial is not None:
                            initial_norm = np.linalg.norm(initial)
                            state_changed = not np.allclose(final, initial, rtol=1e-2)
                            reduction = (
                                (initial_norm - final_norm) / initial_norm
                                if initial_norm > 0
                                else 0
                            )
                        else:
                            state_changed = True
                            reduction = None

                        # Format output
                        if len(final) <= 4:
                            final_str = np.array2string(final, precision=6, suppress_small=True)
                        else:
                            final_str = f"[{final[0]:.4g}, ..., {final[-1]:.4g}]"

                        status = "✓" if (converged_to_zero or state_changed) else "✗"
                        if not (converged_to_zero or state_changed):
                            all_checks_passed = False

                        emit(f"   {status} {name}: {final_str}")
                        if reduction is not None and reduction > 0:
                            emit(f"      ‖x‖ reduced by {reduction * 100:.1f}%")
                        if converged_to_zero:
                            emit(f"      Converged to ‖x‖ = {final_norm:.2e}")

                # Scope convergence verification
                if scope_convergence:
                    emit("\n📈 Signal Convergence:")

                    def format_val(v):
                        if v is None:
                            return "N/A"
                        v = np.atleast_1d(v)
                        if len(v) == 1:
                            return f"{float(v[0]):.6g}"
                        elif len(v) <= 3:
                            return np.array2string(v, precision=4, suppress_small=True)
                        else:
                            return f"[{v[0]:.4g}, {v[1]:.4g}, ...]"

                    for name, info in scope_convergence.items():
                        first = info["first"]
                        last = info["last"]
                        samples = info["samples"]

                        # Check convergence criteria
                        first_norm = np.linalg.norm(np.atleast_1d(first))
                        last_norm = np.linalg.norm(np.atleast_1d(last))

                        # Get explicit verification mode or fall back to heuristics
                        verify_mode = info.get("verify_mode", "auto")

                        if verify_mode == "none":
                            # Skip this scope entirely
                            continue

                        if verify_mode == "auto":
                            # Fall back to name-based heuristics (current behavior)
                            # Note: removed 'error' from is_objective keywords to avoid false positives
                            is_objective = any(
                                kw in name.lower() for kw in ["f_", "cost", "obj", "norm", "value"]
                            )
                            is_state = any(
                                kw in name.lower() for kw in ["x_", "state", "traj", "position"]
                            )
                        elif verify_mode == "objective":
                            is_objective = True
                            is_state = False
                        elif verify_mode == "trajectory":
                            is_objective = False
                            is_state = True
                        else:  # "comparison" or unknown
                            is_objective = False
                            is_state = False

                        if is_objective and first_norm > 0:
                            # Objective should decrease significantly
                            reduction = (first_norm - last_norm) / first_norm
                            converged = reduction > 0.9 or last_norm < 1e-6
                            status = "✓" if converged else "✗"
                            if not converged:
                                all_checks_passed = False
                            emit(f"   {status} {name}: {format_val(first)} → {format_val(last)}")
                            if reduction > 0:
                                emit(f"      Reduced by {reduction * 100:.1f}%")
                        elif is_state:
                            # State should change and ideally converge
                            changed = not np.allclose(first, last, rtol=0.01)
                            status = "✓" if changed else "✗"
                            if not changed:
                                all_checks_passed = False
                            emit(f"   {status} {name}: {format_val(first)} → {format_val(last)}")
                        else:
                            # Generic scope or comparison mode - just show values (no pass/fail)
                            emit(
                                f"   • {name} ({samples} pts): "
                                f"{format_val(first)} → {format_val(last)}"
                            )

                # Final verdict
                emit("\n" + "-" * 60)
                if all_checks_passed:
                    emit("✓ VERIFICATION PASSED")
                else:
                    emit("✗ VERIFICATION FAILED - Check values above")
                emit("=" * 60)
                logger.info("\n".join(report))
            else:
                logger.info("Simulation completed - no verification data")

        except Exception as e:
            # Emit whatever was collected before the failure, then the reason.
            if report:
                logger.info("\n".join(report))
            logger.warning(f"Could not assemble verification results: {e}", exc_info=True)

    def stop(self):
        """Stop simulation safely."""
        try:
            # A threaded batch run owns the step loop; ask it to stop and join
            # before clearing the flag, or it would keep stepping.
            self.cancel_batch()

            if hasattr(self.dsim, "execution_initialized"):
                self.dsim.execution_initialized = False

            self.status_changed.emit("Simulation stopped")
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
