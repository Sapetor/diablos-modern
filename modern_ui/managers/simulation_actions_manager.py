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

from lib.i18n import tr

logger = logging.getLogger(__name__)


class SimulationActionsManager:
    """Owns the window-side simulation control handlers."""

    def __init__(self, main_window):
        self.window = main_window

    def _toast_errors(self, text: str) -> None:
        """Show an error toast that opens the error panel when clicked."""
        window = self.window
        toast = getattr(window, "toast", None)
        if toast is not None:
            toast.show_message(
                text, duration=6000, is_error=True, on_click=window.error_panel.reveal
            )
        else:  # pragma: no cover - the toast is built with the canvas
            window.error_panel.reveal()

    def on_run_errors(self, message: str, block_name: str = "") -> None:
        """A run failed: list the errors, mark the blocks, toast -- no modal.

        ``message`` is the engine/validator text (one error per line);
        ``block_name`` the engine's flattened name of the offending block.
        """
        from lib.error_locator import build_run_errors

        window = self.window
        root_blocks = window.canvas.dsim.get_root_context()[0]
        errors = build_run_errors(root_blocks, message, block_name)
        if not errors:
            return
        window.error_panel.set_errors(errors)
        window.canvas.clear_validation()
        window.canvas.mark_runtime_errors([b for e in errors for b in e.blocks])
        n = len(errors)
        self._toast_errors(
            tr("Simulation failed — {n} error (click to view)", n=n)
            if n == 1
            else tr("Simulation failed — {n} errors (click to view)", n=n)
        )

    def jump_to_error(self, error) -> bool:
        """Select the error's block and centre the canvas on it (entering its subsystem)."""
        window = self.window
        canvas = window.canvas
        candidates = list(getattr(error, "blocks", None) or [])
        if not candidates and getattr(error, "block_name", ""):
            from lib.error_locator import find_block_by_path

            found = find_block_by_path(canvas.dsim.get_root_context()[0], error.block_name)
            candidates = [found[0]] if found else []
        for block in candidates:
            if canvas.reveal_block(block):
                for other in candidates:  # keep every offender selected
                    if other is not block and any(b is other for b in canvas.dsim.blocks_list):
                        other.selected = True
                window.status_message.setText(tr("Showing error: {message}", message=error.message))
                return True
        return False

    def start(self) -> None:
        """Start simulation with validation."""
        window = self.window
        if not hasattr(window, "canvas"):
            window.status_message.setText(tr("Canvas not available"))
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
                window.status_message.setText(
                    tr("Cannot start simulation: {count} error(s) found", count=error_count)
                )
                logger.warning(f"Simulation blocked by {error_count} validation error(s)")

                # Non-modal: the panel already lists them; the toast opens it.
                self._toast_errors(
                    tr("Cannot start simulation — {n} error (click to view)", n=error_count)
                    if error_count == 1
                    else tr("Cannot start simulation — {n} errors (click to view)", n=error_count)
                )
                return
            else:
                # Only warnings - allow simulation but notify user
                warning_count = sum(1 for e in errors if e.severity == ErrorSeverity.WARNING)
                logger.info(f"Starting simulation with {warning_count} warning(s)")
                window.status_message.setText(
                    tr("Starting simulation with {count} warning(s)...", count=warning_count)
                )
        else:
            # No errors or warnings - clear error panel
            window.error_panel.clear()
            logger.info("Validation passed - no errors or warnings")
            window.status_message.setText(tr("Starting simulation..."))

        # Clear validation indicators from canvas before starting
        # (errors will be shown in panel, don't need red borders during simulation)
        window.canvas.clear_validation()

        # Start the simulation
        # Check fast solver preference
        if hasattr(window, "use_fast_solver"):
            window.dsim.use_fast_solver = window.use_fast_solver

        window.canvas.start_simulation()

        # Arm tuning controller after batch simulation completes
        # (safe_update timer can't detect batch completion since it runs synchronously)
        if not window.canvas.is_simulation_running():
            sim_time = getattr(window.dsim, "sim_time", None)
            sim_dt = getattr(window.dsim, "sim_dt", None)
            if sim_time and sim_dt:
                window.tuning_controller.store_sim_params(sim_time, sim_dt)

    def open_settings(self) -> bool:
        """Open Simulation > Simulation Settings... and apply what is accepted.

        Play no longer pops this dialog (it runs with the stored settings), so
        this action is the way in. The dialog is pre-filled from the live DSim
        values; accepting it marks the diagram dirty whenever one of the
        settings that the ``.diablos`` file stores actually changed.

        Returns True if the dialog was accepted.
        """
        window = self.window
        dsim = window.dsim

        values = dsim.open_simulation_dialog(parent=window)
        if values is None:
            return False

        if dsim.apply_sim_settings(values):
            dsim.dirty = True
            window.status_message.setText(tr("Simulation settings updated"))
        else:
            window.status_message.setText(tr("Simulation settings unchanged"))

        # The transport's t-readout shows the new horizon straight away (but
        # not mid-run, where it would rewind the live readout to t=0).
        if hasattr(window, "toolbar") and not window.canvas.is_simulation_running():
            window.toolbar.set_simulation_time(0.0, dsim.sim_time)
        return True

    def stop(self):
        """Stop simulation (the controller emits the idle state and message)."""
        window = self.window
        if hasattr(window, "canvas"):
            window.canvas.stop_simulation()
        else:
            window.status_message.setText(tr("Simulation stopped"))

    def pause(self):
        """Pause simulation (the controller emits the paused state)."""
        self.window.canvas.pause_simulation()

    def step(self):
        """Execute a single timestep of the simulation.

        If simulation is not running, it will be initialized first,
        allowing step-by-step execution from t=0.
        """
        window = self.window
        if not hasattr(window.dsim, "single_step"):
            window.status_message.setText(tr("Single-step not available"))
            return

        # Check if this is the first step (will initialize)
        was_initialized = window.dsim.execution_initialized

        success = window.dsim.single_step()
        if success:
            if not was_initialized:
                window.status_message.setText(
                    tr("Started stepping at t={time:.4f}s", time=window.dsim.time_step)
                )
            else:
                window.status_message.setText(
                    tr("Stepped to t={time:.4f}s", time=window.dsim.time_step)
                )
            window.canvas.update()
            # Stepping is a paused run: the loop is armed but not free-running.
            window._on_simulation_state_changed("paused")
        else:
            # Check if simulation ended or failed to start
            if not window.dsim.execution_initialized:
                window._on_simulation_state_changed("idle")
                if was_initialized:
                    window.status_message.setText(tr("Simulation finished"))
                else:
                    window.status_message.setText(tr("Failed to initialize simulation"))
            else:
                window.status_message.setText(tr("Step failed"))

    def toggle_fast_solver(self, checked):
        """Toggle fast solver mode."""
        window = self.window
        window.use_fast_solver = checked
        if hasattr(window, "dsim"):
            window.dsim.use_fast_solver = checked
        logger.info(f"Fast Solver enabled: {checked}")
