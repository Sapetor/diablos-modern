"""
Optimization Primitives Package

This package provides low-level building blocks for constructing optimization
algorithms visually using feedback loops, rather than just configuring an
external optimizer.

Example: Gradient descent as a block diagram:
    X_{k+1} = X_k - α * ∇f(X_k)

Available blocks:
- ObjectiveFunction: Evaluate f(x) from Python expression
- NumericalGradient: Compute ∇f from finite difference inputs
- VectorPerturb: Perturb x[index] by ε for finite difference
- StateVariable: Hold state x(k), output current, accept next
- VectorGain: Scale vector y = α * x
- VectorSum: Vector add/subtract y = ±x1 ± x2 ...
- LinearSystemSolver: Solve Ax = b
- RootFinder: One Newton step for F(x) = 0
- ResidualNorm: Compute ‖F‖ for convergence
- Momentum: Momentum-accelerated gradient descent
- Adam: Adam optimizer with adaptive learning rates
"""

from diablos.blocks.optimization_primitives.objective_function import ObjectiveFunctionBlock
from diablos.blocks.optimization_primitives.numerical_gradient import NumericalGradientBlock
from diablos.blocks.optimization_primitives.vector_perturb import VectorPerturbBlock
from diablos.blocks.optimization_primitives.state_variable import StateVariableBlock
from diablos.blocks.optimization_primitives.vector_gain import VectorGainBlock
from diablos.blocks.optimization_primitives.vector_sum import VectorSumBlock
from diablos.blocks.optimization_primitives.linear_system_solver import LinearSystemSolverBlock
from diablos.blocks.optimization_primitives.root_finder import RootFinderBlock
from diablos.blocks.optimization_primitives.residual_norm import ResidualNormBlock
from diablos.blocks.optimization_primitives.momentum import MomentumBlock
from diablos.blocks.optimization_primitives.adam import AdamBlock

__all__ = [
    "ObjectiveFunctionBlock",
    "NumericalGradientBlock",
    "VectorPerturbBlock",
    "StateVariableBlock",
    "VectorGainBlock",
    "VectorSumBlock",
    "LinearSystemSolverBlock",
    "RootFinderBlock",
    "ResidualNormBlock",
    "MomentumBlock",
    "AdamBlock",
]
