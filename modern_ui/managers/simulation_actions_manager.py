"""
SimulationActionsManager -- the window-side simulation handlers: start (with
pre-run validation), stop, pause, single-step, and the fast-solver toggle.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

Note: this is the *window-side* orchestration (validation, error panel, toolbar
state, status messages, tuning-controller arming). The actual run loop lives in
the canvas's own SimulationController; these handlers call ``window.canvas``.
"""

import logging

logger = logging.getLogger(__name__)


class SimulationActionsManager:
    """Owns the window-side simulation control handlers."""

    def __init__(self, main_window):
        self.window = main_window

    def start(self) -> None:
        """Start simulation with validation."""
        window = self.window
        if not hasattr(window, 'canvas'):
            window.status_message.setText("Canvas not available")
            return

        # Run diagram validation first
        from lib.diagram_validator import ErrorSeverity

        logger.info("Running pre-simulation validation...")
        errors = window.canvas.run_validation()

        # Check for critical errors that block simulation
        has_errors = any(e.severity == ErrorSeverity.ERROR for e in errors)

        if errors:
            # Show error panel with results
            window.error_panel.set_errors(errors)

            if has_errors:
                # Critical errors found - don't start simulation
                error_count = sum(1 for e in errors if e.severity == ErrorSeverity.ERROR)
                window.status_message.setText(f"Cannot start simulation: {error_count} error(s) found")
                logger.warning(f"Simulation blocked by {error_count} validation error(s)")

                # Show a message box for critical errors
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    window,
                    "Validation Errors",
                    f"Cannot start simulation due to {error_count} validation error(s).\n\n"
                    f"Please fix the errors shown in the error panel before running."
                )
                return
            else:
                # Only warnings - allow simulation but notify user
                warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)
                logger.info(f"Starting simulation with {warning_count} warning(s)")
                window.status_message.setText(f"Starting simulation with {warning_count} warning(s)...")
        else:
            # No errors or warnings - clear error panel
            window.error_panel.clear()
            logger.info("Validation passed - no errors or warnings")
            window.status_message.setText("Starting simulation...")

        # Clear validation indicators from canvas before starting
        # (errors will be shown in panel, don't need red borders during simulation)
        window.canvas.clear_validation()

        # Start the simulation
        # Check fast solver preference
        if hasattr(window, 'use_fast_solver'):
            window.dsim.use_fast_solver = window.use_fast_solver

        window.canvas.start_simulation()

        # Arm tuning controller after batch simulation completes
        # (safe_update timer can't detect batch completion since it runs synchronously)
        if not window.canvas.is_simulation_running():
            sim_time = getattr(window.dsim, 'sim_time', None)
            sim_dt = getattr(window.dsim, 'sim_dt', None)
            if sim_time and sim_dt:
                window.tuning_controller.store_sim_params(sim_time, sim_dt)

    def stop(self):
        """Stop simulation."""
        window = self.window
        if hasattr(window, 'canvas'):
            window.canvas.stop_simulation()
        window.toolbar.set_simulation_state(False, False)
        window.status_message.setText("Simulation stopped")

    def pause(self):
        """Pause simulation."""
        window = self.window
        if hasattr(window.dsim, 'execution_pause'):
            window.dsim.execution_pause = True
        window.toolbar.set_simulation_state(True, True)

    def step(self):
        """Execute a single timestep of the simulation.

        If simulation is not running, it will be initialized first,
        allowing step-by-step execution from t=0.
        """
        window = self.window
        if not hasattr(window.dsim, 'single_step'):
            window.status_message.setText("Single-step not available")
            return

        # Check if this is the first step (will initialize)
        was_initialized = window.dsim.execution_initialized

        success = window.dsim.single_step()
        if success:
            if not was_initialized:
                window.status_message.setText(f"Started stepping at t={window.dsim.time_step:.4f}s")
            else:
                window.status_message.setText(f"Stepped to t={window.dsim.time_step:.4f}s")
            window.canvas.update()
            # Keep toolbar in paused state (step always pauses)
            window.toolbar.set_simulation_state(True, True)
        else:
            # Check if simulation ended or failed to start
            if not window.dsim.execution_initialized:
                window.toolbar.set_simulation_state(False, False)
                if was_initialized:
                    window.status_message.setText("Simulation finished")
                else:
                    window.status_message.setText("Failed to initialize simulation")
            else:
                window.status_message.setText("Step failed")

    def toggle_fast_solver(self, checked):
        """Toggle fast solver mode."""
        window = self.window
        window.use_fast_solver = checked
        if hasattr(window, 'dsim'):
            window.dsim.use_fast_solver = checked
        logger.info(f"Fast Solver enabled: {checked}")
