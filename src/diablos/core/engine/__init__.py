"""
Engine package - Business logic for DiaBloS.
Contains simulation execution and analysis logic.
"""

from diablos.core.engine.simulation_engine import SimulationEngine
from diablos.core.engine.optimization_engine import OptimizationEngine

# SymbolicEngine requires SymPy (optional dependency)
try:
    from diablos.core.engine.symbolic_engine import SymbolicEngine

    __all__ = ["SimulationEngine", "OptimizationEngine", "SymbolicEngine"]
except ImportError:
    SymbolicEngine = None
    __all__ = ["SimulationEngine", "OptimizationEngine"]
