"""Deterministic teardown for top-level result windows that hold pyqtgraph plots.

A pyqtgraph item keeps Python references to its ViewBox. If Qt deletes a
scene with the items still in it -- at an arbitrary garbage collection, or at
exit -- an item can repaint against a ViewBox that is already gone ("Pure
virtual function called"). A result window therefore deletes itself on close
and closes its plots first: ``PlotWidget.close()`` takes the ViewBox and axes
out of the scene as pyqtgraph intends. Whoever retains the window drops it on
``destroyed`` (``ExperimentController._retain_window``).
"""

import pyqtgraph as pg
from PyQt6.QtCore import Qt


def delete_on_close(window) -> None:
    """Free ``window`` as soon as it is closed, not at a later collection."""
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)


def close_plots(window) -> None:
    """Close every pyqtgraph PlotWidget under ``window`` (idempotent)."""
    for plot in window.findChildren(pg.PlotWidget):
        if plot.plotItem is not None:
            plot.close()
