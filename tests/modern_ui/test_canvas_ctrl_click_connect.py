"""Regression tests for the Ctrl+Click connection path in ModernCanvas.

``_handle_block_click`` completes a connection by calling ``_finish_line_creation``,
which always ends in ``cancel_line_creation() -> end_connection()`` and therefore
clears ``ConnectionState.source_block`` to ``None``. The handler used to reach for
``conn_state.source_block.selected`` *after* that call, raising
``AttributeError: 'NoneType' object has no attribute 'selected'`` on every single
Ctrl+Click connection. The error was swallowed by the handler's ``except``, so the
connection still formed but the source block was never deselected and the state was
never re-armed for the next Ctrl+Click.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PyQt5.QtCore import Qt, QPoint

from modern_ui.widgets.canvas_state import ConnectionState
from modern_ui.widgets.modern_canvas import ModernCanvas


def _make_block(name, in_ports=1, out_ports=1):
    return SimpleNamespace(
        name=name,
        fn_name=name,
        in_ports=in_ports,
        out_ports=out_ports,
        selected=False,
    )


@pytest.fixture
def canvas_stub():
    """A minimal stand-in exposing only what _handle_block_click touches.

    Built as a stub rather than a real widget so the test exercises the handler's
    logic without needing a live canvas; _finish_line_creation clears the shared
    ConnectionState exactly as the real ConnectionManager does.
    """
    conn_state = ConnectionState()
    stub = MagicMock()
    stub.connection_manager.connection_state = conn_state
    stub.dsim.line_list = []

    # Mirror the real call chain: finish -> cancel -> end_connection -> state cleared.
    stub._finish_line_creation.side_effect = lambda *_args, **_kw: conn_state.end_connection()

    stub.conn_state = conn_state
    return stub


@pytest.mark.qt
@pytest.mark.unit
def test_ctrl_click_connect_does_not_raise(canvas_stub):
    """The Ctrl+Click connect path must complete without logging an error."""
    source = _make_block("step0")
    target = _make_block("tranfn0")
    canvas_stub.connection_manager.connection_state.source_block = source

    with (
        patch(
            "modern_ui.widgets.modern_canvas.QApplication.keyboardModifiers",
            return_value=Qt.ControlModifier,
        ),
        patch("modern_ui.widgets.modern_canvas.logger") as mock_logger,
    ):
        ModernCanvas._handle_block_click(canvas_stub, target, QPoint(10, 10))

    mock_logger.error.assert_not_called()
    canvas_stub._finish_line_creation.assert_called_once_with(target, 0)


@pytest.mark.qt
@pytest.mark.unit
def test_ctrl_click_connect_rearms_state_for_next_connection(canvas_stub):
    """After connecting, the source is deselected and the target becomes the new source."""
    source = _make_block("step0")
    target = _make_block("tranfn0")
    source.selected = True
    conn_state = canvas_stub.connection_manager.connection_state
    conn_state.source_block = source

    with (
        patch(
            "modern_ui.widgets.modern_canvas.QApplication.keyboardModifiers",
            return_value=Qt.ControlModifier,
        ),
        patch("modern_ui.widgets.modern_canvas.logger"),
    ):
        ModernCanvas._handle_block_click(canvas_stub, target, QPoint(10, 10))

    assert source.selected is False, "source block should be deselected after connecting"
    assert target.selected is True, "target block should become selected"
    assert conn_state.source_block is target, "target should be re-armed as the next source"


@pytest.mark.qt
@pytest.mark.unit
def test_ctrl_click_picks_first_free_target_input_port(canvas_stub):
    """Port selection skips inputs that are already wired."""
    source = _make_block("step0")
    target = _make_block("sum0", in_ports=2)
    canvas_stub.connection_manager.connection_state.source_block = source
    canvas_stub.dsim.line_list = [
        SimpleNamespace(srcblock="other", srcport=0, dstblock="sum0", dstport=0)
    ]

    with (
        patch(
            "modern_ui.widgets.modern_canvas.QApplication.keyboardModifiers",
            return_value=Qt.ControlModifier,
        ),
        patch("modern_ui.widgets.modern_canvas.logger"),
    ):
        ModernCanvas._handle_block_click(canvas_stub, target, QPoint(10, 10))

    canvas_stub._finish_line_creation.assert_called_once_with(target, 1)
