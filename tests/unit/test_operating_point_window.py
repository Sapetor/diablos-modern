"""Tests for the OperatingPointWindow (modern_ui/widgets/operating_point_window.py).

The window CONSUMES the headless trim result-dict from
AnalysisController.find_trim and only renders it. These tests build sample
result dicts and assert the widget constructs, populates the state table, and
copies a usable dict literal to the clipboard.
"""

import pytest

from PyQt6.QtWidgets import QWidget, QTableWidget, QApplication

from modern_ui.widgets.operating_point_window import OperatingPointWindow


def _ok_result():
    return {
        "ok": True,
        "error": "",
        "success": True,
        "message": "The solution converged.",
        "residual_norm": 1.2e-12,
        "states": [
            {"name": "Integrator0", "value": 2.0},
            {"name": "Integrator1", "value": -1.5},
        ],
        "operating_point": {"Integrator0": 2.0, "Integrator1": -1.5},
        "summary": "Operating point found (trim converged).\nStates: 2",
    }


@pytest.fixture
def make_window(qapp):
    """Build result windows that are torn down *deterministically*.

    The window holds pyqtgraph plots, which queue deferred events on their
    children. Letting Python garbage-collect the widget whenever a test's
    local went out of scope destroyed the C++ side while Qt was still
    dispatching those events -- a "Pure virtual function called" abort that
    surfaced (flakily) in whichever test happened to process events next.
    """
    windows = []

    def _make(*args, **kwargs):
        win = OperatingPointWindow(*args, **kwargs)
        windows.append(win)
        return win

    yield _make
    for win in windows:
        win.close()
        win.deleteLater()
    qapp.processEvents()


@pytest.mark.unit
class TestOperatingPointWindow:
    def test_builds_as_qwidget(self, qapp, make_window):
        win = make_window(_ok_result())
        assert isinstance(win, QWidget)
        assert "Operating Point" in win.windowTitle()

    def test_table_populated(self, qapp, make_window):
        win = make_window(_ok_result())
        table = win.findChild(QTableWidget)
        assert table is not None
        assert table.rowCount() == 2
        assert table.item(0, 0).text() == "Integrator0"
        assert "2" in table.item(0, 1).text()

    def test_copy_to_clipboard(self, qapp, make_window):
        win = make_window(_ok_result())
        win._copy_to_clipboard()
        text = QApplication.clipboard().text()
        assert "Integrator0" in text
        assert text.strip().startswith("{") and text.strip().endswith("}")

    def test_error_result_has_no_table(self, qapp, make_window):
        result = {
            "ok": False,
            "error": "Diagram is not compilable.",
            "success": False,
            "message": "",
            "residual_norm": None,
            "states": [],
            "operating_point": {},
            "summary": "Operating-point search unavailable.",
        }
        win = make_window(result)
        assert isinstance(win, QWidget)
        assert win.findChild(QTableWidget) is None

    def test_empty_states_builds(self, qapp, make_window):
        result = _ok_result()
        result["states"] = []
        result["operating_point"] = {}
        win = make_window(result)
        assert isinstance(win, QWidget)
