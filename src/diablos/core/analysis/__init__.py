"""
Analysis package - linearization, trim, Monte Carlo and parameter sweeps.

The Bode / Nyquist / root-locus / LQR *windows* (``ControlSystemAnalyzer`` and
``analyzers/``) are GUI code and live in ``diablos.gui.analysis``;
``diablos.core.analysis.ControlSystemAnalyzer`` still resolves there, lazily.
"""

from diablos.core.analysis.linearizer import Linearizer

__all__ = ["Linearizer"]


def __getattr__(name):
    if name == "ControlSystemAnalyzer":
        from diablos.gui.analysis.control_system_analyzer import ControlSystemAnalyzer

        return ControlSystemAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
