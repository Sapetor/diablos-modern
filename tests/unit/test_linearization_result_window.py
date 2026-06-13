"""
Tests for the Linearize & Analyze results window
(modern_ui/widgets/linearization_result_window.py).

The window CONSUMES the shared linearization result-dict contract and only
plots the arrays already present in it -- it performs no linearization of its
own. These tests build hand-crafted sample result dicts and assert the widget
constructs without error and exposes the expected tabbed layout.

Run offscreen:
    $env:QT_QPA_PLATFORM="offscreen"
"""

import numpy as np
import pytest

from PyQt5.QtWidgets import QWidget, QTabWidget

from modern_ui.widgets.linearization_result_window import LinearizationResultWindow


def _sample_ok_result(with_bode=True):
    """A fully populated, healthy result dict (2-state SISO system)."""
    # Bode arrays over a log frequency sweep.
    if with_bode:
        w = np.logspace(-1, 2, 50)
        bode = {
            "w": w.tolist(),
            "mag_db": (-20.0 * np.log10(w)).tolist(),
            "phase_deg": (-90.0 - 10.0 * np.log10(w)).tolist(),
        }
        t = np.linspace(0, 5, 60)
        step_response = {"t": t.tolist(), "y": (1.0 - np.exp(-t)).tolist()}
        impulse_response = {"t": t.tolist(), "y": np.exp(-t).tolist()}
    else:
        bode = None
        step_response = None
        impulse_response = None

    return {
        "ok": True,
        "error": "",
        "n_states": 2,
        "state_names": ["x0", "x1"],
        "input_names": ["u0"],
        "output_names": ["y0"],
        "A": [[0.0, 1.0], [-2.0, -3.0]],
        "B": [[0.0], [1.0]],
        "C": [[1.0, 0.0]],
        "D": [[0.0]],
        "poles": [[-1.0, 0.0], [-2.0, 0.0]],
        "zeros": [[-0.5, 0.0]],
        "is_stable": True,
        "time_constants": [1.0, 0.5],
        "oscillatory_modes": [{"omega_n": 1.41, "zeta": 0.7, "period": 4.4}],
        "gain_margin_db": 12.0,
        "phase_margin_deg": 45.0,
        "gain_crossover": 3.2,
        "phase_crossover": 1.1,
        "tf_num": [1.0],
        "tf_den": [1.0, 3.0, 2.0],
        "bode": bode,
        "step_response": step_response,
        "impulse_response": impulse_response,
        "controllable": True,
        "observable": True,
        "operating_point": {"Integrator0": 0.0, "Integrator1": 0.0},
        "summary": "Linearized 2-state system.\nStable: yes\nPoles: -1, -2",
    }


@pytest.mark.unit
class TestLinearizationResultWindow:
    def test_builds_as_qwidget(self, qapp):
        win = LinearizationResultWindow(_sample_ok_result())
        assert isinstance(win, QWidget)

    def test_window_title_set(self, qapp):
        win = LinearizationResultWindow(_sample_ok_result())
        assert "Linearized" in win.windowTitle()

    def test_has_all_tabs(self, qapp):
        win = LinearizationResultWindow(_sample_ok_result())
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        assert "Pole-Zero" in labels
        assert "Bode" in labels
        assert "Step" in labels
        assert "Impulse" in labels
        assert "Summary" in labels

    def test_bode_none_builds_without_error(self, qapp):
        """A result without Bode (no I/O designated) still builds with all tabs.

        Step/Impulse tabs are present too, showing their 'designate I/O' hint.
        """
        win = LinearizationResultWindow(_sample_ok_result(with_bode=False))
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5

    def test_error_result_builds_without_tabs(self, qapp):
        """When ok is False the window shows the error, not the tab widget."""
        result = {
            "ok": False,
            "error": "Diagram is not compilable.",
            "n_states": 0,
            "state_names": [],
            "input_names": [],
            "output_names": [],
            "A": [],
            "B": [],
            "C": [],
            "D": [],
            "poles": [],
            "zeros": [],
            "is_stable": False,
            "time_constants": [],
            "oscillatory_modes": [],
            "gain_margin_db": None,
            "phase_margin_deg": None,
            "gain_crossover": None,
            "phase_crossover": None,
            "tf_num": None,
            "tf_den": None,
            "bode": None,
            "controllable": None,
            "observable": None,
            "operating_point": {},
            "summary": "",
        }
        win = LinearizationResultWindow(result)
        assert isinstance(win, QWidget)
        # No tab widget in the error path.
        assert win.findChild(QTabWidget) is None

    def test_minimal_ok_result_builds(self, qapp):
        """A-only result (no B/C/D, no poles/zeros, no bode) still builds."""
        result = {
            "ok": True,
            "error": "",
            "n_states": 1,
            "state_names": ["x0"],
            "input_names": [],
            "output_names": [],
            "A": [[-3.0]],
            "B": [],
            "C": [],
            "D": [],
            "poles": [[-3.0, 0.0]],
            "zeros": [],
            "is_stable": True,
            "time_constants": [1.0 / 3.0],
            "oscillatory_modes": [],
            "gain_margin_db": None,
            "phase_margin_deg": None,
            "gain_crossover": None,
            "phase_crossover": None,
            "tf_num": None,
            "tf_den": None,
            "bode": None,
            "controllable": None,
            "observable": None,
            "operating_point": {"Integrator0": 0.0},
            "summary": "x' = -3x",
        }
        win = LinearizationResultWindow(result)
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5
