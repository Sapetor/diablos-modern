"""Simulation Controller - Orchestrates the simulation lifecycle.

Extracted from ModernCanvas to keep the canvas focused on rendering and
interaction. Owns validation, execution start/stop, batch execution, and the
post-run terminal verification report. Communicates status to the UI via the
``status_changed`` signal (the canvas re-emits it as its own
``simulation_status_changed`` so existing listeners are unaffected).
"""

import logging
import sys

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QWidget

from lib.improvements import SafetyChecks, ValidationHelper

logger = logging.getLogger(__name__)


class SimulationController(QObject):
    """Drives validation, start/stop, and batch execution for a DSim model."""

    status_changed = pyqtSignal(str)  # Emitted when simulation status changes

    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        self.dsim = dsim

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
            if hasattr(self.dsim, 'execution_init'):
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
                    error_msg = self.dsim.error_msg if hasattr(self.dsim, 'error_msg') and self.dsim.error_msg else "Initialization failed (see logs)."
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
        """Run the simulation in batch mode (as fast as possible)."""
        logger.info("Running simulation in batch mode.")
        self.status_changed.emit("Running simulation...")

        # This will block the UI. In a real application, this should be run in a separate thread.
        self.dsim.execution_batch()

        solver_type = getattr(self.dsim, 'last_solver_type', 'Standard')
        self.status_changed.emit(f"Simulation finished [{solver_type}]")
        logger.info(f"Batch simulation finished. Solver: {solver_type}")
        self.dsim.plot_again()

        # Print verification results to terminal
        self._print_terminal_verification()

    def _print_terminal_verification(self):
        """Print verification results to terminal after simulation completes."""
        import numpy as np
        try:
            # Use active blocks from engine if available, otherwise fall back to blocks_list
            has_engine = hasattr(self.dsim, 'engine') and self.dsim.engine is not None
            use_active = has_engine and len(self.dsim.engine.active_blocks_list) > 0
            blocks_source = self.dsim.engine.active_blocks_list if use_active else self.dsim.blocks_list

            # Collect Display block values
            display_values = {}
            for block in blocks_source:
                if block.block_fn == 'Display':
                    params = block.params or {}
                    display_val = params.get('_display_value_', '---')
                    label = params.get('label', '')
                    block_name = label if label else block.username
                    display_values[block_name] = display_val

            # Collect StateVariable final states (optimization convergence)
            state_values = {}
            for block in blocks_source:
                if block.block_fn == 'StateVariable':
                    exec_params = getattr(block, 'exec_params', {}) or {}
                    state = exec_params.get('_state_')
                    initial = exec_params.get('initial_value')
                    if state is not None:
                        state_arr = np.atleast_1d(state)
                        initial_arr = np.atleast_1d(initial) if initial is not None else None
                        block_name = block.username if block.username else block.name
                        state_values[block_name] = {'final': state_arr, 'initial': initial_arr}

            # Collect Scope convergence info (first/last values)
            scope_convergence = {}
            for block in blocks_source:
                if block.block_fn == 'Scope':
                    exec_params = getattr(block, 'exec_params', {}) or {}
                    vec = exec_params.get('vector')
                    if vec is not None and hasattr(vec, '__len__') and len(vec) > 0:
                        arr = np.array(vec)
                        vec_dim = exec_params.get('vec_dim', 1)
                        labels = exec_params.get('vec_labels', block.username)

                        # Reshape if interleaved multi-dimensional
                        if arr.ndim == 1 and vec_dim > 1 and len(arr) >= vec_dim:
                            num_samples = len(arr) // vec_dim
                            arr = arr[:num_samples * vec_dim].reshape(num_samples, vec_dim)

                        block_name = block.username if block.username else block.name
                        if arr.ndim == 2:
                            first_val = arr[0, :]
                            last_val = arr[-1, :]
                        else:
                            first_val = arr[0] if len(arr) > 0 else None
                            last_val = arr[-1] if len(arr) > 0 else None
                        scope_convergence[block_name] = {
                            'labels': labels,
                            'first': first_val,
                            'last': last_val,
                            'samples': len(arr),
                            'data': arr,
                            'verify_mode': exec_params.get('verify_mode', 'auto'),
                        }

            # Build output with verification checks
            has_output = display_values or state_values or scope_convergence
            all_checks_passed = True
            check_results = []

            if has_output:
                print("\n" + "=" * 60, flush=True)
                print("VERIFICATION RESULTS", flush=True)
                print("=" * 60, flush=True)

                # Display block values
                if display_values:
                    print("\n📊 Display Values:", flush=True)
                    for name, value in display_values.items():
                        print(f"   {name}: {value}", flush=True)

                # StateVariable convergence check
                if state_values:
                    print("\n🎯 Optimization Convergence:", flush=True)
                    for name, info in state_values.items():
                        final = info['final']
                        initial = info['initial']

                        # Check if converged to near zero (common for quadratic minimization)
                        final_norm = np.linalg.norm(final)
                        converged_to_zero = final_norm < 1e-3

                        # Check if state changed from initial
                        if initial is not None:
                            initial_norm = np.linalg.norm(initial)
                            state_changed = not np.allclose(final, initial, rtol=1e-2)
                            reduction = (initial_norm - final_norm) / initial_norm if initial_norm > 0 else 0
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

                        print(f"   {status} {name}: {final_str}", flush=True)
                        if reduction is not None and reduction > 0:
                            print(f"      ‖x‖ reduced by {reduction*100:.1f}%", flush=True)
                        if converged_to_zero:
                            print(f"      Converged to ‖x‖ = {final_norm:.2e}", flush=True)

                # Scope convergence verification
                if scope_convergence:
                    print("\n📈 Signal Convergence:", flush=True)

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
                        first = info['first']
                        last = info['last']
                        samples = info['samples']
                        data = info['data']

                        # Check convergence criteria
                        first_norm = np.linalg.norm(np.atleast_1d(first))
                        last_norm = np.linalg.norm(np.atleast_1d(last))

                        # Get explicit verification mode or fall back to heuristics
                        verify_mode = info.get('verify_mode', 'auto')

                        if verify_mode == "none":
                            # Skip this scope entirely
                            continue

                        if verify_mode == "auto":
                            # Fall back to name-based heuristics (current behavior)
                            # Note: removed 'error' from is_objective keywords to avoid false positives
                            is_objective = any(kw in name.lower() for kw in ['f_', 'cost', 'obj', 'norm', 'value'])
                            is_state = any(kw in name.lower() for kw in ['x_', 'state', 'traj', 'position'])
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
                            print(f"   {status} {name}: {format_val(first)} → {format_val(last)}", flush=True)
                            if reduction > 0:
                                print(f"      Reduced by {reduction*100:.1f}%", flush=True)
                        elif is_state:
                            # State should change and ideally converge
                            changed = not np.allclose(first, last, rtol=0.01)
                            status = "✓" if changed else "✗"
                            if not changed:
                                all_checks_passed = False
                            print(f"   {status} {name}: {format_val(first)} → {format_val(last)}", flush=True)
                        else:
                            # Generic scope or comparison mode - just show values (no pass/fail)
                            print(f"   • {name} ({samples} pts): {format_val(first)} → {format_val(last)}", flush=True)

                # Final verdict
                print("\n" + "-" * 60, flush=True)
                if all_checks_passed:
                    print("✓ VERIFICATION PASSED", flush=True)
                else:
                    print("✗ VERIFICATION FAILED - Check values above", flush=True)
                print("=" * 60 + "\n", flush=True)
                sys.stdout.flush()
            else:
                print("\n[Simulation completed - no verification data]", flush=True)
                sys.stdout.flush()

        except Exception as e:
            # Log the actual error for debugging
            print(f"\n[Could not print verification results: {e}]", file=sys.stderr, flush=True)
            logger.warning(f"Could not print verification results: {e}")

    def stop(self):
        """Stop simulation safely."""
        try:
            if hasattr(self.dsim, 'execution_initialized'):
                self.dsim.execution_initialized = False

            self.status_changed.emit("Simulation stopped")
            logger.info("Simulation stopped")

        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")

    def current_time(self):
        """Get current simulation time."""
        if hasattr(self.dsim, 't'):
            return getattr(self.dsim, 't', 0.0)
        return 0.0

    def is_running(self):
        """Check if simulation is running."""
        return getattr(self.dsim, 'execution_initialized', False)
