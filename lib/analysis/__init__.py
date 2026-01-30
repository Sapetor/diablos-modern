"""
Analysis package - Control system analysis and linearization tools.
"""

from lib.analysis.linearizer import Linearizer

# ControlSystemAnalyzer may have additional dependencies
try:
    from lib.analysis.control_system_analyzer import ControlSystemAnalyzer
    __all__ = ['Linearizer', 'ControlSystemAnalyzer']
except ImportError:
    ControlSystemAnalyzer = None
    __all__ = ['Linearizer']
