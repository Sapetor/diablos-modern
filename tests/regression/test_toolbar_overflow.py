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


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1280, 800)
    yield w
    w.close()


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
