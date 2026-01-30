"""
Engine package - Business logic for DiaBloS.
Contains simulation execution and analysis logic.
"""

from lib.engine.simulation_engine import SimulationEngine
from lib.engine.optimization_engine import OptimizationEngine

# SymbolicEngine requires SymPy (optional dependency)
try:
    from lib.engine.symbolic_engine import SymbolicEngine
    __all__ = ['SimulationEngine', 'OptimizationEngine', 'SymbolicEngine']
except ImportError:
    SymbolicEngine = None
    __all__ = ['SimulationEngine', 'OptimizationEngine']
