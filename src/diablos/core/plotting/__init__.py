"""
Plotting module for DiaBloS.
Contains SignalPlot and other plotting utilities.

``SignalPlot`` (pyqtgraph, PyQt6) is imported on first use so the engine,
which imports ``diablos.gui.plotting.scope_plotter``, does not require PyQt6.
"""

__all__ = ["SignalPlot"]


def __getattr__(name):
    if name == "SignalPlot":
        from diablos.gui.plotting.signal_plot import SignalPlot

        return SignalPlot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
