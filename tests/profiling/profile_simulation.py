"""
Profiling script for DiaBloS simulation engine.

Identifies hot paths and provides optimization recommendations.
Run with: python tests/profiling/profile_simulation.py
"""

import cProfile
import pstats
import io
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_test_simulation():
    """Create a medium-complexity test simulation."""
    from blocks.step import StepBlock
    from blocks.gain import GainBlock
    from blocks.sum import SumBlock
    from blocks.integrator import IntegratorBlock
    from blocks.scope import ScopeBlock

    blocks = []

    # Create blocks
    step = StepBlock()
    step_params = {'value': 1.0, 'delay': 0.1, 'type': 'up',
                   'pulse_start_up': True, '_init_start_': True, '_name_': 'step'}

    gain = GainBlock()
    gain_params = {'gain': 2.0}

    sum_block = SumBlock()
    sum_params = {'sign': '+-', '_init_start_': True}

    integrator = IntegratorBlock()
    int_params = {'init_conds': 0.0, 'method': 'FWD_EULER',
                  '_init_start_': True, '_name_': 'int'}

    scope = ScopeBlock()
    scope_params = {'labels': 'out', '_init_start_': True, '_name_': 'scope'}

    return {
        'step': (step, step_params),
        'gain': (gain, gain_params),
        'sum': (sum_block, sum_params),
        'integrator': (integrator, int_params),
        'scope': (scope, scope_params)
    }


def run_block_level_simulation(num_steps=1000, dt=0.01):
    """Run block-level simulation manually (bypasses engine overhead)."""
    blocks = create_test_simulation()

    step, step_params = blocks['step']
    gain, gain_params = blocks['gain']
    sum_block, sum_params = blocks['sum']
    integrator, int_params = blocks['integrator']
    scope, scope_params = blocks['scope']

    y = 0.0  # integrator output (feedback)

    for i in range(num_steps):
        t = i * dt

        # Step input
        step_out = step.execute(t, {}, step_params)
        step_val = step_out[0][0] if hasattr(step_out[0], '__len__') else step_out[0]

        # Gain on step
        gain_out = gain.execute(t, {0: step_val}, gain_params)

        # Sum (gain - feedback)
        sum_out = sum_block.execute(t, {0: gain_out[0], 1: y}, sum_params)

        # Integrator
        int_out = integrator.execute(t, {0: sum_out[0]}, int_params, dtime=dt)
        y = int_out[0][0] if hasattr(int_out[0], '__len__') else int_out[0]

        # Scope
        scope.execute(t, {0: y}, scope_params)

    return scope_params.get('vector', [])


def run_pde_simulation(num_steps=100, dt=0.001):
    """Run PDE block simulation (more computationally intensive)."""
    from blocks.pde.heat_equation_1d import HeatEquation1DBlock

    block = HeatEquation1DBlock()
    params = {
        'length': 1.0,
        'nx': 100,
        'alpha': 0.01,
        'bc_type': 'dirichlet',
        'left_bc_value': 0.0,
        'right_bc_value': 0.0,
        'initial_condition': 'sine',
        '_init_start_': True,
        'dtime': dt
    }

    for i in range(num_steps):
        result = block.execute(i * dt, {}, params)

    return result


def run_optimization_simulation(num_iterations=100):
    """Run optimization primitives simulation."""
    from blocks.optimization_primitives.state_variable import StateVariableBlock
    from blocks.optimization_primitives.objective_function import ObjectiveFunctionBlock
    from blocks.optimization_primitives.numerical_gradient import NumericalGradientBlock
    from blocks.optimization_primitives.vector_gain import VectorGainBlock
    from blocks.optimization_primitives.vector_sum import VectorSumBlock

    state = StateVariableBlock()
    obj = ObjectiveFunctionBlock()
    grad = NumericalGradientBlock()
    gain = VectorGainBlock()
    sum_block = VectorSumBlock()

    state_params = {'initial_value': '[5.0, 5.0]', 'n_variables': 2, '_init_start_': True}
    obj_params = {'expression': 'x1**2 + x2**2', 'n_variables': 2}
    grad_params = {'expression': 'x1**2 + x2**2', 'n_variables': 2, 'epsilon': 1e-6, '_init_start_': True}
    gain_params = {'gain': 0.1}
    sum_params = {'signs': '+-'}

    x = np.array([5.0, 5.0])

    for i in range(num_iterations):
        # Get current state
        state_out = state.execute(i, {0: x}, state_params)
        x_current = state_out[0]

        # Compute gradient (this is expensive)
        grad_out = grad.execute(i, {0: x_current}, grad_params)

        # Scale gradient
        gain_out = gain.execute(i, {0: grad_out[0]}, gain_params)

        # Update state
        sum_out = sum_block.execute(i, {0: x_current, 1: gain_out[0]}, sum_params)
        x = sum_out[0]

    return x


def profile_function(func, *args, **kwargs):
    """Profile a function and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start

    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    return result, elapsed, stream.getvalue()


def main():
    print("=" * 60)
    print("DiaBloS Simulation Engine Profiling")
    print("=" * 60)

    # 1. Block-level simulation (control system)
    print("\n1. Block-level Control System Simulation (1000 steps)")
    print("-" * 40)
    result, elapsed, stats = profile_function(run_block_level_simulation, 1000, 0.01)
    print(f"Elapsed time: {elapsed:.4f}s")
    print(f"Final value: {result[-1] if len(result) > 0 else 'N/A'}")
    print("\nTop functions by cumulative time:")
    print(stats[:2000])

    # 2. PDE simulation
    print("\n2. PDE Heat Equation Simulation (100 steps, 100 nodes)")
    print("-" * 40)
    result, elapsed, stats = profile_function(run_pde_simulation, 100, 0.001)
    print(f"Elapsed time: {elapsed:.4f}s")
    print("\nTop functions by cumulative time:")
    print(stats[:2000])

    # 3. Optimization primitives
    print("\n3. Optimization Primitives (100 iterations)")
    print("-" * 40)
    result, elapsed, stats = profile_function(run_optimization_simulation, 100)
    print(f"Elapsed time: {elapsed:.4f}s")
    print(f"Final x: {result}")
    print("\nTop functions by cumulative time:")
    print(stats[:2000])

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    print("""
Based on typical DiaBloS simulation profiles:

1. BLOCK EXECUTION (execute() methods):
   - Each block.execute() call has Python overhead
   - Optimize: Cache numpy arrays, avoid repeated atleast_1d()

2. PARAMETER ACCESS (params.get()):
   - Called many times per step
   - Optimize: Resolve params once at init, not every step

3. NUMPY OPERATIONS:
   - Array creation overhead (np.array, np.atleast_1d)
   - Optimize: Reuse arrays, preallocate outputs

4. LOOP OVERHEAD:
   - Python for-loops are slow
   - Optimize: Vectorize where possible, use compiled solver

5. PDE BLOCKS (compute_derivatives):
   - Laplacian computation is expensive
   - Optimize: Use sparse matrices, numba JIT compilation

6. NUMERICAL GRADIENT:
   - Multiple function evaluations per step
   - Optimize: Use analytical gradients when available
""")


if __name__ == '__main__':
    main()
