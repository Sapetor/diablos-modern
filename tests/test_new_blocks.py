"""
Test script for PDE and Optimization blocks.

Run with: python tests/test_new_blocks.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

def test_block_loading():
    """Test that all new blocks can be loaded and instantiated."""
    print("=== Test 1: Block Loading ===")
    from lib.block_loader import load_blocks

    blocks = load_blocks()
    print(f"Total blocks loaded: {len(blocks)}")

    pde_blocks = []
    opt_blocks = []
    errors = []

    for b in blocks:
        try:
            inst = b()
            cat = getattr(inst, 'category', 'Unknown')
            name = inst.block_name
            if cat == 'PDE':
                pde_blocks.append(name)
            elif cat == 'Optimization':
                opt_blocks.append(name)
        except Exception as e:
            errors.append(f'{b.__name__}: {e}')

    print(f"\nPDE blocks ({len(pde_blocks)}): {pde_blocks}")
    print(f"Optimization blocks ({len(opt_blocks)}): {opt_blocks}")

    if errors:
        print(f"\nErrors: {errors}")
        return False

    print("\n[PASS] All blocks instantiated successfully!")
    return True


def test_heat_equation():
    """Test HeatEquation1D block execution."""
    print("\n=== Test 2: HeatEquation1D ===")
    from blocks.pde.heat_equation_1d import HeatEquation1DBlock

    block = HeatEquation1DBlock()
    params = {
        'alpha': 0.1,
        'L': 1.0,
        'N': 10,
        'bc_type_left': 'Dirichlet',
        'bc_type_right': 'Dirichlet',
        'init_conds': [0.0],
        '_init_start_': True,
        'dtime': 0.01,
    }

    # Execute with boundary conditions
    inputs = {0: 0.0, 1: 100.0, 2: 0.0}  # q_src=0, bc_left=100, bc_right=0

    result = block.execute(0.0, inputs, params)
    T_field = result[0]
    T_avg = result[1]

    print(f"Initial T_field shape: {T_field.shape}")
    print(f"Initial T_avg: {T_avg}")
    print(f"T_field[0] (left BC): {T_field[0]}")
    print(f"T_field[-1] (right BC): {T_field[-1]}")

    # Run a few steps
    for i in range(10):
        result = block.execute(0.01 * (i+1), inputs, params)

    T_field = result[0]
    T_avg = result[1]
    print(f"\nAfter 10 steps:")
    print(f"T_avg: {T_avg:.4f}")
    print(f"T_field: {np.round(T_field, 2)}")

    # Check that heat is diffusing from left BC
    if T_field[0] == 100.0 and T_field[-1] == 0.0 and T_avg > 0:
        print("\n[PASS] HeatEquation1D working correctly!")
        return True
    else:
        print("\n[FAIL] Unexpected behavior")
        return False


def test_wave_equation():
    """Test WaveEquation1D block execution."""
    print("\n=== Test 3: WaveEquation1D ===")
    from blocks.pde.wave_equation_1d import WaveEquation1DBlock

    block = WaveEquation1DBlock()
    params = {
        'c': 1.0,
        'damping': 0.0,
        'L': 1.0,
        'N': 20,
        'bc_type_left': 'Dirichlet',
        'bc_type_right': 'Dirichlet',
        'init_displacement': 'gaussian',
        'init_velocity': [0.0],
        '_init_start_': True,
        'dtime': 0.01,
    }

    inputs = {0: 0.0, 1: 0.0, 2: 0.0}  # force=0, bc_left=0, bc_right=0

    result = block.execute(0.0, inputs, params)
    u_field = result[0]
    energy = result[2]

    print(f"Initial u_field shape: {u_field.shape}")
    print(f"Initial energy: {energy:.4f}")
    print(f"Max displacement: {np.max(np.abs(u_field)):.4f}")

    # Run a few steps
    for i in range(10):
        result = block.execute(0.01 * (i+1), inputs, params)

    u_field = result[0]
    energy_final = result[2]
    print(f"\nAfter 10 steps:")
    print(f"Energy: {energy_final:.4f}")
    print(f"Max displacement: {np.max(np.abs(u_field)):.4f}")

    print("\n[PASS] WaveEquation1D executes without errors!")
    return True


def test_field_probe():
    """Test FieldProbe block."""
    print("\n=== Test 4: FieldProbe ===")
    from blocks.pde.field_processing import FieldProbeBlock

    block = FieldProbeBlock()
    params = {
        'position': 0.5,
        'position_mode': 'normalized',
        'L': 1.0,
    }

    # Create a linear field: f(x) = x
    field = np.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]
    inputs = {0: field}

    result = block.execute(0.0, inputs, params)
    value = result[0]

    print(f"Field: {field}")
    print(f"Probe at position 0.5: {value}")

    if abs(value - 5.0) < 0.01:
        print("\n[PASS] FieldProbe interpolation correct!")
        return True
    else:
        print(f"\n[FAIL] Expected 5.0, got {value}")
        return False


def test_field_integral():
    """Test FieldIntegral block."""
    print("\n=== Test 5: FieldIntegral ===")
    from blocks.pde.field_processing import FieldIntegralBlock

    block = FieldIntegralBlock()
    params = {
        'L': 1.0,
        'normalize': False,
    }

    # Constant field f(x) = 2, integral should be 2*L = 2
    field = np.ones(11) * 2.0
    inputs = {0: field}

    result = block.execute(0.0, inputs, params)
    integral = result[0]

    print(f"Field: constant 2.0 over [0, 1]")
    print(f"Integral: {integral}")

    if abs(integral - 2.0) < 0.1:
        print("\n[PASS] FieldIntegral correct!")
        return True
    else:
        print(f"\n[FAIL] Expected ~2.0, got {integral}")
        return False


def test_parameter_block():
    """Test Parameter block."""
    print("\n=== Test 6: Parameter Block ===")
    from blocks.optimization.parameter import ParameterBlock

    block = ParameterBlock()
    params = {
        'name': 'Kp',
        'value': 2.5,
        'lower': 0.0,
        'upper': 10.0,
        'scale': 'linear',
        'fixed': False,
    }

    result = block.execute(0.0, {}, params)
    value = result[0]

    print(f"Parameter 'Kp' value: {value}")

    # Test get_optimization_info
    info = block.get_optimization_info(params)
    print(f"Optimization info: {info}")

    if value == 2.5 and info['name'] == 'Kp':
        print("\n[PASS] Parameter block works!")
        return True
    else:
        print("\n[FAIL] Unexpected output")
        return False


def test_cost_function():
    """Test CostFunction block."""
    print("\n=== Test 7: CostFunction Block ===")
    from blocks.optimization.cost_function import CostFunctionBlock

    block = CostFunctionBlock()
    params = {
        'cost_type': 'ISE',
        'target': 0.0,
        'weight': 1.0,
        '_init_start_': True,
    }

    # Simulate error signal over time
    errors = [1.0, 0.8, 0.5, 0.3, 0.1]
    dt = 0.1
    params['dtime'] = dt

    for i, err in enumerate(errors):
        inputs = {0: err}
        result = block.execute(i * dt, inputs, params)

    final_cost = params.get('_accumulated_cost_', 0)
    print(f"Errors: {errors}")
    print(f"Accumulated ISE cost: {final_cost:.4f}")

    if final_cost > 0:
        print("\n[PASS] CostFunction accumulates cost!")
        return True
    else:
        print("\n[FAIL] Cost should be > 0")
        return False


def test_draw_icons():
    """Test that all new blocks have draw_icon methods that return valid paths."""
    print("\n=== Test 8: Block Icons ===")

    # We need PyQt5 for this test
    try:
        from PyQt5.QtGui import QPainterPath
    except ImportError:
        print("PyQt5 not available, skipping icon test")
        return True

    from lib.block_loader import load_blocks
    blocks = load_blocks()

    missing_icons = []
    icon_errors = []

    for b in blocks:
        try:
            inst = b()
            cat = getattr(inst, 'category', 'Unknown')
            if cat in ['PDE', 'Optimization']:
                if hasattr(inst, 'draw_icon'):
                    try:
                        path = inst.draw_icon(None)
                        if path is not None and not isinstance(path, QPainterPath):
                            icon_errors.append(f"{inst.block_name}: returned {type(path)}")
                    except Exception as e:
                        icon_errors.append(f"{inst.block_name}: {e}")
                else:
                    missing_icons.append(inst.block_name)
        except Exception as e:
            pass

    if missing_icons:
        print(f"Blocks without draw_icon: {missing_icons}")
    if icon_errors:
        print(f"Icon errors: {icon_errors}")

    if not missing_icons and not icon_errors:
        print("\n[PASS] All PDE/Optimization blocks have valid icons!")
        return True
    else:
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing PDE and Optimization Blocks")
    print("=" * 60)

    results = []

    results.append(("Block Loading", test_block_loading()))
    results.append(("HeatEquation1D", test_heat_equation()))
    results.append(("WaveEquation1D", test_wave_equation()))
    results.append(("FieldProbe", test_field_probe()))
    results.append(("FieldIntegral", test_field_integral()))
    results.append(("Parameter Block", test_parameter_block()))
    results.append(("CostFunction Block", test_cost_function()))
    results.append(("Block Icons", test_draw_icons()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
