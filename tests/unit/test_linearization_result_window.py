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

import gc

import numpy as np
import pyqtgraph as pg
import pytest

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
from PyQt6.QtWidgets import QWidget, QTabWidget, QPushButton, QApplication

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


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


@pytest.fixture
def make_window(qapp):
    """Build result windows and close them at the end, as a user would.

    The window tears its plots down in closeEvent and deletes itself on close
    (it used to rely on this fixture's deleteLater + processEvents to avoid a
    pyqtgraph teardown abort; see test_closing_tears_the_plots_down).
    """
    windows = []

    def _make(result):
        win = LinearizationResultWindow(result)
        windows.append(win)
        return win

    yield _make
    for win in windows:
        if not sip.isdeleted(win):
            win.close()
    _flush_deletes()


@pytest.mark.unit
class TestLinearizationResultWindow:
    def test_builds_as_qwidget(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        assert isinstance(win, QWidget)

    def test_window_title_set(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        assert "Linearized" in win.windowTitle()

    def test_has_all_tabs(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        assert "Pole-Zero" in labels
        assert "Bode" in labels
        assert "Step" in labels
        assert "Impulse" in labels
        assert "Summary" in labels

    def test_bode_none_builds_without_error(self, qapp, make_window):
        """A result without Bode (no I/O designated) still builds with all tabs.

        Step/Impulse tabs are present too, showing their 'designate I/O' hint.
        """
        win = make_window(_sample_ok_result(with_bode=False))
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5

    def test_error_result_builds_without_tabs(self, qapp, make_window):
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
        win = make_window(result)
        assert isinstance(win, QWidget)
        # No tab widget in the error path.
        assert win.findChild(QTabWidget) is None

    def test_minimal_ok_result_builds(self, qapp, make_window):
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
        win = make_window(result)
        tabs = win.findChild(QTabWidget)
        assert tabs is not None
        assert tabs.count() == 5


def _export_buttons(win):
    return {b.text(): b for b in win.findChildren(QPushButton)}


@pytest.mark.unit
class TestLinearizationExportBar:
    def test_export_bar_present_when_ok(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        btns = _export_buttons(win)
        assert "Copy as Python" in btns
        assert "Copy as MATLAB" in btns
        assert "Save Data..." in btns

    def test_export_bar_absent_when_not_ok(self, qapp, make_window):
        win = make_window({"ok": False, "error": "nope"})
        btns = _export_buttons(win)
        assert "Copy as Python" not in btns
        assert "Save Data..." not in btns

    def test_copy_python_puts_code_on_clipboard(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        _export_buttons(win)["Copy as Python"].click()
        text = QApplication.clipboard().text()
        assert text.startswith("# Linearized state-space model")
        assert "import numpy as np" in text
        assert "control.ss(A, B, C, D)" in text

    def test_copy_matlab_puts_code_on_clipboard(self, qapp, make_window):
        win = make_window(_sample_ok_result())
        _export_buttons(win)["Copy as MATLAB"].click()
        text = QApplication.clipboard().text()
        assert text.lstrip().startswith("%")
        assert "sys = ss(A, B, C, D);" in text

    def test_save_data_writes_loadable_file(self, qapp, make_window, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QFileDialog

        out = tmp_path / "exported.mat"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "MAT-file (*.mat)")),
        )
        win = make_window(_sample_ok_result())
        _export_buttons(win)["Save Data..."].click()

        assert out.exists()
        from scipy.io import loadmat

        loaded = loadmat(str(out))
        assert np.array_equal(loaded["A"], np.array(_sample_ok_result()["A"]))


@pytest.mark.unit
class TestTeardown:
    def test_closing_tears_the_plots_down(self, qapp):
        """Every PlotWidget is closed (ViewBox and axes out of the scene)
        before Qt deletes the window, which it does on close."""
        win = LinearizationResultWindow(_sample_ok_result())
        win.show()
        qapp.processEvents()
        plots = win.findChildren(pg.PlotWidget)
        assert len(plots) == 5  # pole-zero, magnitude, phase, step, impulse
        win.close()
        assert all(p.plotItem is None for p in plots)
        _flush_deletes()
        assert sip.isdeleted(win)

    def test_the_flash_timer_dies_with_the_window(self, qapp):
        win = LinearizationResultWindow(_sample_ok_result())
        win._flash("copied")
        timers = [t for t in win.findChildren(QTimer) if t.isActive()]
        assert timers and all(t.parent() is win for t in timers)
        win.close()
        _flush_deletes()
        assert all(sip.isdeleted(t) for t in timers)

    def test_windows_closed_while_others_are_built(self, qapp):
        """The pattern that used to abort the full suite: windows destroyed and
        collected while new ones are under construction."""
        for _ in range(8):
            old = LinearizationResultWindow(_sample_ok_result())
            old.show()
            qapp.processEvents()
            new = LinearizationResultWindow(_sample_ok_result())
            old.close()
            del old
            gc.collect()
            new.show()
            qapp.processEvents()
            new.close()
            del new
            _flush_deletes()

    def test_the_controller_forgets_closed_windows_and_closes_the_rest(self, qapp):
        from types import SimpleNamespace

        from modern_ui.controllers.experiment_controller import ExperimentController

        ctl = ExperimentController(SimpleNamespace())
        first = LinearizationResultWindow(_sample_ok_result())
        second = LinearizationResultWindow(_sample_ok_result())
        ctl._retain_window(first)
        ctl._retain_window(second)
        first.close()
        _flush_deletes()
        assert ctl._result_windows == [second]
        ctl.close_result_windows()
        _flush_deletes()
        assert ctl._result_windows == [] and sip.isdeleted(second)
