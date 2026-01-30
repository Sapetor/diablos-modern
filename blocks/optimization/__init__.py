"""
Optimization Blocks Package

This package provides blocks for parameter optimization and model calibration.
These blocks work with the OptimizationEngine to run optimization loops
over simulations.

Optimization Workflow:
1. User places Parameter blocks for tunable values
2. User connects signals to CostFunction blocks
3. User adds an Optimizer block
4. Run Optimization:
   a. OptimizationEngine finds all Parameter/Cost blocks
   b. scipy.optimize.minimize calls objective function
   c. Each objective evaluation runs full simulation
   d. Final optimal parameters written back to Parameter blocks

Available blocks:
- Parameter: Tunable parameter with bounds and scaling
- CostFunction: Objective function accumulator (ISE, IAE, ITAE, terminal)
- Constraint: Inequality/equality constraint for constrained optimization
- Optimizer: Meta-block that triggers optimization
- DataFit: Model calibration block for fitting to experimental data
"""

from blocks.optimization.parameter import ParameterBlock
from blocks.optimization.cost_function import CostFunctionBlock
from blocks.optimization.constraint import ConstraintBlock
from blocks.optimization.optimizer import OptimizerBlock
from blocks.optimization.data_fit import DataFitBlock

__all__ = [
    'ParameterBlock',
    'CostFunctionBlock',
    'ConstraintBlock',
    'OptimizerBlock',
    'DataFitBlock',
]
