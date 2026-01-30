"""
Optimization Engine

Orchestrates parameter optimization by running simulations in a loop.
Works with Parameter, CostFunction, Constraint, and Optimizer blocks
to perform optimization on block diagrams.

Workflow:
1. Find all optimization-related blocks in the diagram
2. Extract tunable parameters from Parameter blocks
3. Create objective function that runs simulation and returns cost
4. Call scipy.optimize with the configured method
5. Write optimal parameters back to Parameter blocks
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
from scipy import optimize

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """
    Optimization engine that runs simulation-based optimization.

    Uses scipy.optimize to minimize cost functions computed from
    block diagram simulations.
    """

    def __init__(self, dsim=None):
        """
        Initialize the optimization engine.

        Args:
            dsim: Reference to DSim instance for running simulations
        """
        self.dsim = dsim
        self.parameter_blocks = []
        self.cost_function_blocks = []
        self.constraint_blocks = []
        self.data_fit_blocks = []
        self.optimizer_block = None
        self.n_evaluations = 0
        self.best_cost = np.inf
        self.history = []

    def find_optimization_blocks(self, blocks: List) -> bool:
        """
        Find all optimization-related blocks in the diagram.

        Args:
            blocks: List of blocks in the diagram

        Returns:
            True if optimizer and at least one parameter found
        """
        self.parameter_blocks = []
        self.cost_function_blocks = []
        self.constraint_blocks = []
        self.data_fit_blocks = []
        self.optimizer_block = None

        for block in blocks:
            block_type = getattr(block, 'block_fn', '')

            if block_type == 'Parameter':
                self.parameter_blocks.append(block)
            elif block_type == 'CostFunction':
                self.cost_function_blocks.append(block)
            elif block_type == 'Constraint':
                self.constraint_blocks.append(block)
            elif block_type == 'DataFit':
                self.data_fit_blocks.append(block)
            elif block_type == 'Optimizer':
                self.optimizer_block = block

        if self.optimizer_block is None:
            logger.warning("No Optimizer block found in diagram")
            return False

        if len(self.parameter_blocks) == 0:
            logger.warning("No Parameter blocks found in diagram")
            return False

        logger.info(f"Found optimization setup: "
                   f"{len(self.parameter_blocks)} parameters, "
                   f"{len(self.cost_function_blocks)} cost functions, "
                   f"{len(self.constraint_blocks)} constraints, "
                   f"{len(self.data_fit_blocks)} data fits")

        return True

    def get_parameter_info(self) -> List[Dict]:
        """
        Get information about all tunable parameters.

        Returns:
            List of parameter info dicts
        """
        params_info = []

        for block in self.parameter_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                info = block.block_instance.get_optimization_info(block.params)
            else:
                info = {
                    'name': block.params.get('name', block.name),
                    'value': float(block.params.get('value', 1.0)),
                    'lower': float(block.params.get('lower', 0.0)),
                    'upper': float(block.params.get('upper', 10.0)),
                    'scale': block.params.get('scale', 'linear'),
                    'fixed': block.params.get('fixed', False),
                }

            if not info['fixed']:
                info['block'] = block
                params_info.append(info)

        return params_info

    def set_parameters(self, x: np.ndarray, params_info: List[Dict]):
        """
        Set parameter values in the diagram.

        Args:
            x: Optimization variable vector
            params_info: Parameter info list
        """
        for i, info in enumerate(params_info):
            block = info['block']

            # Transform from optimizer space to physical space
            value = self._transform_from_optimizer(x[i], info)

            # Set the value
            block.params['value'] = float(value)
            if hasattr(block, 'exec_params'):
                block.exec_params['value'] = float(value)

    def _transform_to_optimizer(self, value: float, info: Dict) -> float:
        """Transform physical value to optimizer space."""
        scale = info.get('scale', 'linear')
        lower = info.get('lower', 0.0)
        upper = info.get('upper', 10.0)

        if scale == 'log':
            return np.log(max(value, 1e-10))
        elif scale == 'normalized':
            return (value - lower) / (upper - lower) if upper > lower else 0.5
        else:
            return value

    def _transform_from_optimizer(self, opt_value: float, info: Dict) -> float:
        """Transform optimizer value to physical space."""
        scale = info.get('scale', 'linear')
        lower = info.get('lower', 0.0)
        upper = info.get('upper', 10.0)

        if scale == 'log':
            value = np.exp(opt_value)
        elif scale == 'normalized':
            value = lower + opt_value * (upper - lower)
        else:
            value = opt_value

        return np.clip(value, lower, upper)

    def reset_blocks(self):
        """Reset all optimization blocks for a new simulation."""
        for block in self.cost_function_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                block.block_instance.reset(block.params)
            else:
                block.params['_accumulated_cost_'] = 0.0
                block.params['_init_start_'] = True

        for block in self.constraint_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                block.block_instance.reset(block.params)
            else:
                block.params['_init_start_'] = True

        for block in self.data_fit_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                block.block_instance.reset(block.params)
            else:
                block.params['_init_start_'] = True

    def compute_cost(self) -> float:
        """
        Compute total cost from all cost function and data fit blocks.

        Returns:
            Total weighted cost
        """
        total_cost = 0.0

        for block in self.cost_function_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                cost = block.block_instance.get_final_cost(block.params)
            else:
                weight = float(block.params.get('weight', 1.0))
                accumulated = block.params.get('_accumulated_cost_', 0.0)
                cost = accumulated * weight

            total_cost += cost

        for block in self.data_fit_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                cost = block.block_instance.get_final_error(block.params)
            else:
                cost = 0.0

            total_cost += cost

        return total_cost

    def compute_constraints(self) -> List[Tuple[str, float]]:
        """
        Compute constraint values.

        Returns:
            List of (type, value) tuples for scipy
        """
        constraints = []

        for block in self.constraint_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                ctype, value = block.block_instance.get_constraint_value(block.params)
            else:
                # Default constraint computation
                ctype = 'ineq'
                value = 0.0

            constraints.append((ctype, value))

        return constraints

    def compute_penalty(self) -> float:
        """
        Compute penalty for constraint violations.

        Returns:
            Total penalty value
        """
        penalty = 0.0

        for block in self.constraint_blocks:
            if hasattr(block, 'block_instance') and block.block_instance:
                penalty += block.block_instance.get_penalty(block.params)
            else:
                penalty_weight = float(block.params.get('penalty_weight', 1000.0))
                violation = block.params.get('_violation_', 0.0)
                penalty += penalty_weight * violation ** 2

        return penalty

    def create_objective(self, params_info: List[Dict], config: Dict) -> Callable:
        """
        Create the objective function for scipy.optimize.

        Args:
            params_info: Parameter information
            config: Optimizer configuration

        Returns:
            Objective function f(x) -> cost
        """
        verbose = config.get('verbose', True)
        use_penalty = config.get('use_penalty', False)

        def objective(x):
            self.n_evaluations += 1

            # Set parameters
            self.set_parameters(x, params_info)

            # Reset blocks
            self.reset_blocks()

            # Run simulation
            if self.dsim is not None:
                try:
                    # Reinitialize and run simulation
                    self.dsim.engine.execution_initialized = False
                    success = self.dsim.execution_init()

                    if not success:
                        logger.warning("Simulation initialization failed")
                        return 1e10

                    # Run batch simulation
                    self.dsim.execution_batch()

                except Exception as e:
                    logger.error(f"Simulation failed: {e}")
                    return 1e10

            # Compute cost
            cost = self.compute_cost()

            # Add penalty if enabled
            if use_penalty:
                cost += self.compute_penalty()

            # Update best
            if cost < self.best_cost:
                self.best_cost = cost

            # Log progress
            if verbose and self.n_evaluations % 10 == 0:
                param_str = ", ".join([f"{info['name']}={self._transform_from_optimizer(x[i], info):.4g}"
                                      for i, info in enumerate(params_info)])
                logger.info(f"Eval {self.n_evaluations}: cost={cost:.6g} ({param_str})")

            self.history.append({
                'n': self.n_evaluations,
                'x': x.copy(),
                'cost': cost,
            })

            return cost

        return objective

    def create_constraints_scipy(self, params_info: List[Dict]) -> List[Dict]:
        """
        Create constraint dictionaries for scipy.optimize.

        Returns:
            List of constraint dicts for scipy
        """
        scipy_constraints = []

        for block in self.constraint_blocks:
            constraint_type = block.params.get('type', '<=')

            if constraint_type == '==':
                scipy_type = 'eq'
            else:
                scipy_type = 'ineq'

            def make_constraint_func(block):
                def constraint_func(x):
                    # Parameters are set by objective function
                    constraints = self.compute_constraints()
                    for ctype, value in constraints:
                        # Return value for this specific constraint
                        pass
                    return 0.0  # Placeholder
                return constraint_func

            scipy_constraints.append({
                'type': scipy_type,
                'fun': make_constraint_func(block),
            })

        return scipy_constraints

    def run_optimization(self, blocks: List = None) -> Dict:
        """
        Run the optimization process.

        Args:
            blocks: List of blocks (uses dsim if not provided)

        Returns:
            Dict with optimization results
        """
        if blocks is None and self.dsim is not None:
            blocks = self.dsim.blocks_list

        if not self.find_optimization_blocks(blocks):
            return {'success': False, 'message': 'Missing optimization blocks'}

        # Get parameter info
        params_info = self.get_parameter_info()

        if len(params_info) == 0:
            return {'success': False, 'message': 'No tunable parameters'}

        # Get optimizer configuration
        if hasattr(self.optimizer_block, 'block_instance') and self.optimizer_block.block_instance:
            config = self.optimizer_block.block_instance.get_optimizer_config(self.optimizer_block.params)
        else:
            config = {
                'method': self.optimizer_block.params.get('method', 'L-BFGS-B'),
                'max_iter': int(self.optimizer_block.params.get('max_iter', 100)),
                'tol': float(self.optimizer_block.params.get('tol', 1e-6)),
                'verbose': self.optimizer_block.params.get('verbose', True),
            }

        # Initialize
        self.n_evaluations = 0
        self.best_cost = np.inf
        self.history = []

        # Initial guess
        x0 = np.array([self._transform_to_optimizer(info['value'], info)
                      for info in params_info])

        # Bounds
        bounds = []
        for info in params_info:
            if info['scale'] == 'normalized':
                bounds.append((0.0, 1.0))
            elif info['scale'] == 'log':
                bounds.append((np.log(max(info['lower'], 1e-10)),
                              np.log(info['upper'])))
            else:
                bounds.append((info['lower'], info['upper']))

        # Create objective
        objective = self.create_objective(params_info, config)

        # Run optimization
        method = config.get('method', 'L-BFGS-B')

        logger.info(f"Starting optimization with {method}")
        logger.info(f"Parameters: {[info['name'] for info in params_info]}")
        logger.info(f"Initial values: {[info['value'] for info in params_info]}")

        try:
            if method.lower() == 'differential_evolution':
                result = optimize.differential_evolution(
                    objective,
                    bounds,
                    maxiter=config.get('max_iter', 100),
                    tol=config.get('tol', 1e-6),
                    popsize=config.get('popsize', 15),
                    mutation=config.get('mutation', 0.8),
                    recombination=config.get('recombination', 0.7),
                    disp=config.get('verbose', True),
                )
            else:
                result = optimize.minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={
                        'maxiter': config.get('max_iter', 100),
                        'disp': config.get('verbose', True),
                    },
                    tol=config.get('tol', 1e-6),
                )

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {'success': False, 'message': str(e)}

        # Extract optimal parameters
        optimal_params = {}
        for i, info in enumerate(params_info):
            value = self._transform_from_optimizer(result.x[i], info)
            optimal_params[info['name']] = value

            # Write back to block
            info['block'].params['value'] = float(value)
            if hasattr(info['block'], 'exec_params'):
                info['block'].exec_params['value'] = float(value)

        # Store results in optimizer block
        if hasattr(self.optimizer_block, 'block_instance') and self.optimizer_block.block_instance:
            self.optimizer_block.block_instance.store_results(self.optimizer_block.params, result)
        else:
            self.optimizer_block.params['_optimal_cost_'] = float(result.fun)
            self.optimizer_block.params['_n_iterations_'] = self.n_evaluations
            self.optimizer_block.params['_converged_'] = bool(result.success)

        logger.info(f"Optimization complete!")
        logger.info(f"Optimal cost: {result.fun:.6g}")
        logger.info(f"Optimal parameters: {optimal_params}")
        logger.info(f"Function evaluations: {self.n_evaluations}")
        logger.info(f"Converged: {result.success}")

        return {
            'success': result.success,
            'optimal_cost': float(result.fun),
            'optimal_params': optimal_params,
            'n_evaluations': self.n_evaluations,
            'history': self.history,
            'message': result.message if hasattr(result, 'message') else '',
        }
