"""
Analysis package - Control system analysis and linearization tools.
"""

from diablos.core.analysis.linearizer import Linearizer

# ControlSystemAnalyzer may have additional dependencies
try:
    from diablos.core.analysis.control_system_analyzer import ControlSystemAnalyzer

    __all__ = ["Linearizer", "ControlSystemAnalyzer"]
except ImportError:
    ControlSystemAnalyzer = None
    __all__ = ["Linearizer"]
