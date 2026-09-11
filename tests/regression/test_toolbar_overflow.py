"""Long status messages used to push the toolbar tools into the overflow menu.

The status pill doubles as the status-message display, and its QLabel was
uncapped, so the pill grew to whatever the message needed -- 726 px for the
stiffness warning, against 70 px idle. That pushed the toolbar's size hint past
the window width, and QToolBar responded by silently moving the trailing items
(Plot, Capture, Auto-route, the theme toggle) into the ">>" overflow menu. To
the user the buttons had simply disappeared, and they stayed gone because the
pill never shrank back on its own.

The pill is now width-capped and elides, with the full text in its tooltip.
The invariant worth pinning is the one that makes eviction impossible: the
toolbar's size hint fits inside the window's own minimum width no matter what
is written to the status bar.
"""

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.qt]

# Longest real message in the codebase (simulation_controller._report_stiffness).
STIFFNESS = (
    "This diagram looks stiff — RK45 took 1284 solver steps per output sample. "
    "Try the LSODA solver in Simulation settings."
)


@pytest.fixture(autouse=True)
def _clear_status(window):
    yield
    window.status_message.setText("")


@pytest.mark.parametrize(
    "message",
    [
        STIFFNESS,
        "Screenshot saved: /Users/someone/Desktop/diablos_capture_2026_09_11.png",
        "x" * 400,
    ],
)
def test_toolbar_still_fits_after_a_long_message(window, message):
    window.status_message.setText(message)
    assert window.toolbar.sizeHint().width() <= window.minimumWidth()


def test_pill_is_width_capped(window):
    window.status_message.setText(STIFFNESS)
    pill = window.toolbar.status_pill
    assert pill.sizeHint().width() <= pill.maximumWidth()


def test_elided_text_keeps_the_full_message_in_the_tooltip(window):
    window.status_message.setText(STIFFNESS)
    pill = window.toolbar.status_pill
    assert pill._label.text() != STIFFNESS, "expected the label to be elided"
    assert pill.toolTip() == STIFFNESS


def test_short_message_is_not_elided_and_gets_no_tooltip(window):
    window.status_message.setText("Diagram opened")
    pill = window.toolbar.status_pill
    assert pill._label.text() == "Diagram opened"
    assert pill.toolTip() == ""


def test_pill_shrinks_back_after_a_long_message(window):
    window.status_message.setText(STIFFNESS)
    wide = window.toolbar.status_pill.sizeHint().width()
    window.status_message.setText("Ready")
    assert window.toolbar.status_pill.sizeHint().width() < wide


class TestTooltipOwnership:
    """Eliding borrows the tooltip slot; it must hand it back.

    The pill's tooltip has an owner -- status_bar_manager sets "Simulation
    state" and re-sets it on every language change. Writing the elided full
    text there unconditionally wiped that, and the invariant in
    tests/modern_ui/test_toolbar_statusbar_tooltips.py ("core pills have
    non-empty tooltips") became false as soon as any status message arrived.
    """

    def test_base_tooltip_survives_a_short_message(self, window):
        window.status_message.setText("Diagram opened")
        assert window.status_pill.toolTip() == "Simulation state"

    def test_long_message_borrows_the_slot_then_returns_it(self, window):
        pill = window.status_pill
        window.status_message.setText(STIFFNESS)
        assert pill.toolTip() == STIFFNESS, "full text should be reachable while elided"
        window.status_message.setText("Ready")
        assert pill.toolTip() == "Simulation state"
