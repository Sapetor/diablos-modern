"""
Optimizer Block - Meta-block that triggers optimization

This block defines the optimization settings and triggers the optimization
process. It's a meta-block that doesn't participate in normal simulation
but controls the OptimizationEngine.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class OptimizerBlock(BaseBlock):
    """
    Optimizer Meta-Block.

    This block configures the optimization process. When optimization
    is triggered, the OptimizationEngine uses this block's settings
    to run the optimization loop.

    Supported optimization methods (via scipy.optimize):
    - L-BFGS-B: Bounded quasi-Newton (default)
    - SLSQP: Sequential Least Squares Programming
    - Nelder-Mead: Derivative-free simplex
    - differential_evolution: Global search
    - Powell: Derivative-free conjugate direction
    - COBYLA: Constrained optimization by linear approximation
    """

    @property
    def block_name(self):
        return "Optimizer"

    @property
    def category(self):
        return "Optimization"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Optimizer - Triggers and configures optimization"
            "\n\nMeta-block that controls the optimization process."
            "\nDoes not participate in normal simulation."
            "\n\nOptimization methods:"
            "\n- L-BFGS-B: Bounded quasi-Newton (default)"
            "\n- SLSQP: Sequential Least Squares (supports constraints)"
            "\n- Nelder-Mead: Derivative-free simplex"
            "\n- differential_evolution: Global optimizer"
            "\n- Powell: Conjugate direction method"
            "\n\nParameters:"
            "\n- method: Optimization algorithm"
            "\n- max_iter: Maximum iterations"
            "\n- tol: Convergence tolerance"
            "\n- use_constraints: Enable constraint handling"
            "\n- verbose: Print progress"
        )

    @property
    def params(self):
        return {
            "method": {
                "type": "string",
                "default": "L-BFGS-B",
                "doc": "Optimization method"
            },
            "max_iter": {
                "type": "int",
                "default": 100,
                "doc": "Maximum number of iterations"
            },
            "tol": {
                "type": "float",
                "default": 1e-6,
                "doc": "Convergence tolerance"
            },
            "use_constraints": {
                "type": "bool",
                "default": True,
                "doc": "Enable constraint handling"
            },
            "use_penalty": {
                "type": "bool",
                "default": False,
                "doc": "Use penalty method for constraints"
            },
            "penalty_factor": {
                "type": "float",
                "default": 1000.0,
                "doc": "Penalty factor for constraint violations"
            },
            "verbose": {
                "type": "bool",
                "default": True,
                "doc": "Print optimization progress"
            },
            "multistart": {
                "type": "int",
                "default": 1,
                "doc": "Number of random restarts for global search"
            },
            # Differential evolution specific
            "popsize": {
                "type": "int",
                "default": 15,
                "doc": "Population size for differential_evolution"
            },
            "mutation": {
                "type": "float",
                "default": 0.8,
                "doc": "Mutation factor for differential_evolution"
            },
            "recombination": {
                "type": "float",
                "default": 0.7,
                "doc": "Recombination rate for differential_evolution"
            },
            # Results storage
            "_optimal_cost_": {
                "type": "float",
                "default": np.inf,
                "doc": "Internal: Best cost found"
            },
            "_n_iterations_": {
                "type": "int",
                "default": 0,
                "doc": "Internal: Number of function evaluations"
            },
            "_converged_": {
                "type": "bool",
                "default": False,
                "doc": "Internal: Did optimizer converge"
            },
        }

    @property
    def inputs(self):
        return []  # Meta-block, no inputs

    @property
    def outputs(self):
        return []  # Meta-block, no outputs

    @property
    def requires_inputs(self):
        return False

    @property
    def requires_outputs(self):
        return False

    def draw_icon(self, block_rect):
        """Draw optimizer icon - converging arrows to minimum."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw converging search pattern
        # Left arrow
        path.moveTo(0.2, 0.3)
        path.lineTo(0.45, 0.5)
        path.moveTo(0.25, 0.38)
        path.lineTo(0.2, 0.3)
        path.lineTo(0.28, 0.33)
        # Right arrow
        path.moveTo(0.8, 0.3)
        path.lineTo(0.55, 0.5)
        path.moveTo(0.75, 0.38)
        path.lineTo(0.8, 0.3)
        path.lineTo(0.72, 0.33)
        # Bottom arrow
        path.moveTo(0.5, 0.85)
        path.lineTo(0.5, 0.6)
        path.moveTo(0.45, 0.75)
        path.lineTo(0.5, 0.85)
        path.lineTo(0.55, 0.75)
        # Star at minimum
        path.addEllipse(0.45, 0.5, 0.1, 0.1)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Optimizer doesn't execute during normal simulation."""
        return {'E': False}

    def get_optimizer_config(self, params):
        """
        Get optimization configuration for the OptimizationEngine.

        Returns:
            dict with all optimization settings
        """
        return {
            'method': params.get('method', 'L-BFGS-B'),
            'max_iter': int(params.get('max_iter', 100)),
            'tol': float(params.get('tol', 1e-6)),
            'use_constraints': params.get('use_constraints', True),
            'use_penalty': params.get('use_penalty', False),
            'penalty_factor': float(params.get('penalty_factor', 1000.0)),
            'verbose': params.get('verbose', True),
            'multistart': int(params.get('multistart', 1)),
            'popsize': int(params.get('popsize', 15)),
            'mutation': float(params.get('mutation', 0.8)),
            'recombination': float(params.get('recombination', 0.7)),
        }

    def store_results(self, params, result):
        """
        Store optimization results.

        Args:
            params: Block params dict
            result: scipy.optimize result object
        """
        params['_optimal_cost_'] = float(result.fun) if hasattr(result, 'fun') else np.inf
        params['_n_iterations_'] = int(result.nfev) if hasattr(result, 'nfev') else 0
        params['_converged_'] = bool(result.success) if hasattr(result, 'success') else False

    def get_results(self, params):
        """Get stored optimization results."""
        return {
            'optimal_cost': params.get('_optimal_cost_', np.inf),
            'n_evaluations': params.get('_n_iterations_', 0),
            'converged': params.get('_converged_', False),
        }
