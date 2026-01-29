"""
Control System Block Tests for DiaBloS

Tests for:
- Integrator (continuous integration)
- Derivative (finite difference)
- PID (proportional-integral-derivative controller)
- TransferFunction (dynamic systems)
- Mux/Demux (signal routing)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())


def test_integrator_block():
    """Test IntegratorBlock integrates signal correctly."""
    from blocks.integrator import IntegratorBlock
    
    block = IntegratorBlock()
    
    # Test integration of constant 1.0 over time
    params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True}
    
    # Run 100 steps at dt=0.01 (total 1 second)
    for i in range(100):
        result = block.execute(i * 0.01, {0: np.array([1.0])}, params, dtime=0.01)
    
    # After integrating 1.0 for 1 second, should be ~1.0
    final_value = params.get('mem', [0])[0]
    assert np.isclose(final_value, 1.0, rtol=0.05), f"Integrator: expected ~1.0, got {final_value}"
    print("[PASS] IntegratorBlock FWD_EULER (constant=1.0 for 1s -> 1.0)")
    
    # Test integration of ramp (t) - integral should be t^2/2
    params = {'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True}
    
    for i in range(100):
        t = i * 0.01
        result = block.execute(t, {0: np.array([t])}, params, dtime=0.01)
    
    # Integral of t from 0 to 1 = t^2/2 = 0.5
    final_value = params.get('mem', [0])[0]
    assert np.isclose(final_value, 0.5, rtol=0.1), f"Integrator ramp: expected ~0.5, got {final_value}"
    print("[PASS] IntegratorBlock FWD_EULER (ramp -> 0.5)")
    
    # Test initial condition
    params = {'init_conds': 5.0, 'method': 'FWD_EULER', '_init_start_': True}
    result = block.execute(0, {0: np.array([0.0])}, params, output_only=True, dtime=0.01)
    assert np.isclose(result[0], 5.0), f"Integrator init_cond: expected 5.0, got {result[0]}"
    print("[PASS] IntegratorBlock initial condition")



def test_derivative_block():
    """Test DerivativeBlock calculates derivatives correctly."""
    from blocks.derivative import DerivativeBlock
    
    block = DerivativeBlock()
    
    # Initial call - derivative should be 0
    params = {'_init_start_': True}
    result = block.execute(0.0, {0: np.array([0.0])}, params)
    assert np.isclose(result[0], 0.0), f"Derivative init: expected 0, got {result[0]}"
    
    # Constant signal - derivative should be 0
    result = block.execute(0.1, {0: np.array([5.0])}, params)
    # First step after change shows the jump
    result = block.execute(0.2, {0: np.array([5.0])}, params)
    assert np.isclose(result[0], 0.0), f"Derivative constant: expected 0, got {result[0]}"
    print("[PASS] DerivativeBlock constant signal")
    
    # Ramp signal (slope = 2) - derivative should be 2
    block2 = DerivativeBlock()
    params2 = {'_init_start_': True}
    
    for i in range(10):
        t = i * 0.1
        value = 2.0 * t  # ramp with slope 2
        result = block2.execute(t, {0: np.array([value])}, params2)
    
    # After a few steps, derivative should stabilize to slope
    assert np.isclose(result[0], 2.0, rtol=0.1), f"Derivative ramp: expected ~2.0, got {result[0]}"
    print("[PASS] DerivativeBlock ramp signal (slope=2)")


def test_pid_block():
    """Test PIDBlock controller behavior."""
    from blocks.pid import PIDBlock
    
    block = PIDBlock()
    
    # P-only controller: output = Kp * error
    params = {'Kp': 2.0, 'Ki': 0.0, 'Kd': 0.0, 'dtime': 0.01, '_init_start_': True}
    
    # Setpoint=10, Measurement=3 -> error=7, output=14
    inputs = {0: np.array([10.0]), 1: np.array([3.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 14.0), f"PID P-only: expected 14.0, got {result[0]}"
    print("[PASS] PIDBlock P-only")
    
    # PI controller: test that integral accumulates
    block2 = PIDBlock()
    params2 = {'Kp': 1.0, 'Ki': 1.0, 'Kd': 0.0, 'dtime': 0.01, '_init_start_': True}
    
    # Constant error over time should increase integral
    for i in range(100):
        inputs = {0: np.array([10.0]), 1: np.array([0.0])}  # error = 10
        result = block2.execute(i * 0.01, inputs, params2)
    
    # After 1 second: P term = 10, I term = 10 * 1s = 10, total ~20
    assert result[0] > 15.0, f"PID PI: expected >15, got {result[0]}"
    print("[PASS] PIDBlock PI accumulation")
    
    # PID with output limits (anti-windup)
    block3 = PIDBlock()
    params3 = {'Kp': 10.0, 'Ki': 0.0, 'Kd': 0.0, 'u_min': -5.0, 'u_max': 5.0, 'dtime': 0.01, '_init_start_': True}
    
    # Large error, but output should be clamped
    inputs = {0: np.array([100.0]), 1: np.array([0.0])}
    result = block3.execute(0, inputs, params3)
    assert np.isclose(result[0], 5.0), f"PID saturated: expected 5.0, got {result[0]}"
    print("[PASS] PIDBlock output saturation")


def test_mux_block():
    """Test MuxBlock combines multiple signals into vector."""
    from blocks.mux import MuxBlock
    
    block = MuxBlock()
    
    # Mux returns a list of inputs
    params = {}
    inputs = {0: np.array([1.0]), 1: np.array([2.0])}
    result = block.execute(0, inputs, params)
    
    # Result is a list of arrays
    assert len(result[0]) == 2, f"Mux: expected 2 elements, got {len(result[0])}"
    print("[PASS] MuxBlock 2 inputs")


def test_demux_block():
    """Test DemuxBlock splits vector into separate signals."""
    from blocks.demux import DemuxBlock
    
    block = DemuxBlock()
    
    # Split a vector into separate outputs - needs output_shape param
    params = {'output_shape': 1, '_name_': 'demux_test'}
    inputs = {0: np.array([10.0, 20.0])}  # 2 elements for 2 outputs
    result = block.execute(0, inputs, params)
    
    assert np.isclose(result[0], 10.0), f"Demux[0]: expected 10.0, got {result[0]}"
    assert np.isclose(result[1], 20.0), f"Demux[1]: expected 20.0, got {result[1]}"
    print("[PASS] DemuxBlock 2 outputs")


def test_transfer_function_block():
    """Test DiscreteTransferFunctionBlock implements z-transform correctly."""
    from blocks.discrete_transfer_function import DiscreteTransferFunctionBlock
    
    block = DiscreteTransferFunctionBlock()
    
    # Simple integrator: H(z) = Ts / (z - 1) = [Ts] / [1, -1]
    # With Ts=0.01, this should integrate the input
    params = {
        'numerator': [0.01],
        'denominator': [1.0, -1.0],
        '_init_start_': True,
        'dtime': 0.01
    }
    
    # Step input - should integrate
    for i in range(100):
        result = block.execute(i * 0.01, {0: np.array([1.0])}, params)
    
    # After 100 steps of integrating 1.0 with Ts=0.01, should be ~1.0
    assert np.isclose(result[0], 1.0, rtol=0.1), f"DTF integrator: expected ~1.0, got {result[0]}"
    print("[PASS] DiscreteTransferFunctionBlock integrator")


def test_zero_order_hold():
    """Test ZeroOrderHoldBlock samples and holds signal."""
    from blocks.zero_order_hold import ZeroOrderHoldBlock
    
    block = ZeroOrderHoldBlock()
    
    # Parameter is 'sampling_time', not 'sample_time'
    params = {'sampling_time': 0.1, '_init_start_': True}
    
    # First sample
    result = block.execute(0.0, {0: np.array([5.0])}, params)
    assert np.isclose(result[0], 5.0), f"ZOH first: expected 5.0, got {result[0]}"
    
    # Before next sample time - should hold previous value
    result = block.execute(0.05, {0: np.array([10.0])}, params)
    assert np.isclose(result[0], 5.0), f"ZOH hold: expected 5.0, got {result[0]}"
    
    # After sample time - should update
    result = block.execute(0.15, {0: np.array([10.0])}, params)
    assert np.isclose(result[0], 10.0), f"ZOH update: expected 10.0, got {result[0]}"
    print("[PASS] ZeroOrderHoldBlock")


def test_exponential_block():
    """Test ExponentialBlock computes a * exp(b * x)."""
    from blocks.exponential import ExponentialBlock
    
    block = ExponentialBlock()
    
    # a * exp(b * 0) = a * 1 = a
    params = {'a': 1.0, 'b': 1.0}
    result = block.execute(0, {0: np.array([0.0])}, params)
    assert np.isclose(result[0], 1.0), f"Exp(0): expected 1.0, got {result[0]}"
    
    # 1 * exp(1 * 1) = e ~ 2.718
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert np.isclose(result[0], np.e, rtol=0.01), f"Exp(1): expected {np.e}, got {result[0]}"
    
    # 2 * exp(1 * 1) = 2e
    params = {'a': 2.0, 'b': 1.0}
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert np.isclose(result[0], 2 * np.e, rtol=0.01), f"2*Exp(1): expected {2*np.e}, got {result[0]}"
    
    print("[PASS] ExponentialBlock")


def test_selector_block():
    """Test SelectorBlock selects elements from vector."""
    from blocks.selector import SelectorBlock
    
    block = SelectorBlock()
    
    # Selector uses string indices like "0,2"
    params = {'indices': '0,2'}
    inputs = {0: np.array([10.0, 20.0, 30.0, 40.0])}
    result = block.execute(0, inputs, params)
    
    expected = np.array([10.0, 30.0])
    assert np.allclose(result[0], expected), f"Selector: expected {expected}, got {result[0]}"
    print("[PASS] SelectorBlock")


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Control System Block Tests")
    print("=" * 50)
    
    tests = [
        ("IntegratorBlock", test_integrator_block),
        ("DerivativeBlock", test_derivative_block),
        ("PIDBlock", test_pid_block),
        ("MuxBlock", test_mux_block),
        ("DemuxBlock", test_demux_block),
        ("DiscreteTransferFunction", test_transfer_function_block),
        ("ZeroOrderHoldBlock", test_zero_order_hold),
        ("ExponentialBlock", test_exponential_block),
        ("SelectorBlock", test_selector_block),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    sys.exit(1 if failed > 0 else 0)
