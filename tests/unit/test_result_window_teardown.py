"""Result windows free themselves, and their pyqtgraph plots, on close.

A result window left alive until an arbitrary garbage collection (or exit)
lets Qt delete its scene with pyqtgraph items still in it, which can abort the
interpreter ("Pure virtual function called"). Every result window therefore
deletes itself on close after closing its plots
(``modern_ui/widgets/plot_teardown.py``).
"""

import pyqtgraph as pg
import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication


def _windows():
    from modern_ui.widgets.ensemble_result_window import EnsembleResultWindow
    from modern_ui.widgets.linearization_result_window import LinearizationResultWindow
    from modern_ui.widgets.operating_point_window import OperatingPointWindow
    from modern_ui.widgets.sweep_result_window import SweepResultWindow
    from tests.unit.test_ensemble_result_window import _sample_result as ensemble
    from tests.unit.test_linearization_result_window import _sample_ok_result as linearized
    from tests.unit.test_operating_point_window import _ok_result as trim
    from tests.unit.test_sweep_result_window import _sweep_1d, _sweep_2d

    return {
        # name: (factory, number of PlotWidgets)
        "ensemble": (lambda: EnsembleResultWindow(ensemble()), 2),
        "sweep_1d": (lambda: SweepResultWindow(_sweep_1d()), 2),
        "sweep_2d": (lambda: SweepResultWindow(_sweep_2d()), 1),
        "linearization": (lambda: LinearizationResultWindow(linearized()), 5),
        "operating_point": (lambda: OperatingPointWindow(trim()), 0),
    }


CASES = ["ensemble", "sweep_1d", "sweep_2d", "linearization", "operating_point"]


@pytest.mark.unit
@pytest.mark.parametrize("name", CASES)
def test_closing_frees_the_window_and_closes_its_plots(qapp, name):
    factory, n_plots = _windows()[name]
    win = factory()
    win.show()
    qapp.processEvents()
    plots = win.findChildren(pg.PlotWidget)
    assert len(plots) == n_plots
    win.close()
    assert all(p.plotItem is None for p in plots)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    assert sip.isdeleted(win)
