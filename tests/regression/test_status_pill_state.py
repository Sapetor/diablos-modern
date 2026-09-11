"""The toolbar status pill stayed on "Simulating…" after a run ended.

``ModernToolbar.set_status()`` used to derive the pill *state* from the message
text, and every end-of-run message ("Simulation finished", "Simulation finished
[Fast (Compiled)]") contains the substring "simulat", so the closing status line
of a completed run put the pill straight back into the running state -- after the
explicit ``set_simulation_state(False, ...)`` had already cleared it, with
nothing following to clear it again.  The keywords never matched a translated UI
either.  State is now owned by ``set_simulation_state()`` / ``set_error_state()``
alone, and a message can only change the label.
"""

import pytest

pytestmark = pytest.mark.regression


@pytest.mark.qt
class TestStatusPillClearsAfterRun:
    def _pill_state(self, window):
        return window.toolbar.status_pill.property("state")

    @pytest.mark.parametrize(
        "final_message",
        [
            "Simulation finished",  # interactive path (safe_update)
            "Simulation finished [Fast (Compiled)]",  # batch path (_finish_batch)
            # _report_solver_diagnostics appends to the finished line *after*
            # the state was cleared, so it gets its own case.
            "Simulation finished [Fast (Compiled)]  |  RK45, 412 steps",
        ],
    )
    def test_pill_is_idle_after_the_final_message(self, window, final_message):
        window.toolbar.set_simulation_state(True, False)
        assert self._pill_state(window) == "running"

        # End of run, in the order the real code does it: clear the state, then
        # write the closing status line.
        window.toolbar.set_simulation_state(False, False)
        window.status_message.setText(final_message)

        assert self._pill_state(window) == "idle"

    def test_running_state_survives_a_progress_message(self, window):
        """A per-step progress line must not wipe the "Simulating…" label."""
        window.toolbar.set_simulation_state(True, False)
        window.status_message.setText("Running simulation... t = 3.5 / 20 s")
        assert self._pill_state(window) == "running"

    def test_idle_message_is_still_displayed(self, window):
        window.toolbar.set_simulation_state(False, False)
        window.status_message.setText("Diagram opened")
        assert window.toolbar.status_pill._label.text() == "Diagram opened"


@pytest.mark.qt
class TestFailedRunFeedback:
    """A failed run must show red *and* release the transport buttons."""

    def test_failure_turns_the_pill_red(self, window):
        window.toolbar.set_simulation_state(True, False)
        window._on_simulation_status_changed("Simulation failed to start. Bad D matrix")
        assert window.toolbar.status_pill.property("state") == "error"

    def test_failure_still_resets_the_transport(self, window):
        """The transport has no state accessor; assert on the buttons it drives."""
        window.toolbar.set_simulation_state(True, False)
        window._on_simulation_status_changed("Simulation failed to start. Bad D matrix")
        transport = window.toolbar.transport
        assert transport.play_btn.isEnabled() is True
        assert transport.stop_btn.isEnabled() is False
