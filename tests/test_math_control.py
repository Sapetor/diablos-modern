"""
Math, Control, and Nonlinear Block Tests for DiaBloS

Tests:
- Math: Gain, Sum, Abs, SigProduct (multiplication)
- Control: Saturation, Deadband, RateLimiter, Delay
- Nonlinear: Hysteresis, Switch
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())


# ==================== MATH BLOCKS ====================

def test_gain_block():
    """Test GainBlock with scalar, vector, and matrix gains."""
    from blocks.gain import GainBlock
    
    block = GainBlock()
    
    # Test scalar gain
    params = {'gain': 3.0}
    result = block.execute(0, {0: np.array([2.0])}, params)
    assert np.isclose(result[0], 6.0), f"Scalar gain: expected 6.0, got {result[0]}"
    print("[PASS] GainBlock scalar")
    
    # Test vector input with scalar gain
    params = {'gain': 2.0}
    result = block.execute(0, {0: np.array([1.0, 2.0, 3.0])}, params)
    expected = np.array([2.0, 4.0, 6.0])
    assert np.allclose(result[0], expected), f"Vector with scalar gain: expected {expected}, got {result[0]}"
    print("[PASS] GainBlock vector with scalar gain")
    
    # Test matrix gain (2x2 @ 2x1 = 2x1)
    params = {'gain': [[1.0, 0.0], [0.0, 2.0]]}  # Scale y by 2
    result = block.execute(0, {0: np.array([3.0, 4.0])}, params)
    expected = np.array([3.0, 8.0])
    assert np.allclose(result[0], expected), f"Matrix gain: expected {expected}, got {result[0]}"
    print("[PASS] GainBlock matrix")


def test_sum_block():
    """Test SumBlock with different sign configurations."""
    from blocks.sum import SumBlock
    
    block = SumBlock()
    
    # Test ++ (add two inputs)
    params = {'sign': '++'}
    inputs = {0: np.array([3.0]), 1: np.array([4.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 7.0), f"Sum ++: expected 7.0, got {result[0]}"
    print("[PASS] SumBlock ++")
    
    # Test +- (subtract)
    params = {'sign': '+-'}
    inputs = {0: np.array([10.0]), 1: np.array([3.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 7.0), f"Sum +-: expected 7.0, got {result[0]}"
    print("[PASS] SumBlock +-")
    
    # Test ++- (three inputs)
    params = {'sign': '++-'}
    inputs = {0: np.array([5.0]), 1: np.array([3.0]), 2: np.array([2.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 6.0), f"Sum ++-: expected 6.0, got {result[0]}"
    print("[PASS] SumBlock ++-")
    
    # Test with vectors
    params = {'sign': '+-'}
    inputs = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([0.5, 0.5, 0.5])}
    result = block.execute(0, inputs, params)
    expected = np.array([0.5, 1.5, 2.5])
    assert np.allclose(result[0], expected), f"Sum vector: expected {expected}, got {result[0]}"
    print("[PASS] SumBlock vector")


def test_abs_block():
    """Test AbsBlock takes absolute value."""
    from blocks.abs_block import AbsBlock
    
    block = AbsBlock()
    params = {}
    
    # Positive input stays positive
    result = block.execute(0, {0: np.array([5.0])}, params)
    assert np.isclose(result[0], 5.0), f"Abs positive: expected 5.0, got {result[0]}"
    
    # Negative becomes positive
    result = block.execute(0, {0: np.array([-5.0])}, params)
    assert np.isclose(result[0], 5.0), f"Abs negative: expected 5.0, got {result[0]}"
    
    # Vector
    result = block.execute(0, {0: np.array([-1.0, 2.0, -3.0])}, params)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result[0], expected), f"Abs vector: expected {expected}, got {result[0]}"
    
    print("[PASS] AbsBlock")


def test_sigproduct_block():
    """Test SigProductBlock (signal multiplication)."""
    from blocks.sigproduct import SigProductBlock
    
    block = SigProductBlock()
    params = {}
    
    # Multiply two scalars
    inputs = {0: 3.0, 1: 4.0}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 12.0), f"Product: expected 12.0, got {result[0]}"
    
    # Multiply three scalars
    inputs = {0: 2.0, 1: 3.0, 2: 4.0}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 24.0), f"Product 3 inputs: expected 24.0, got {result[0]}"
    
    print("[PASS] SigProductBlock")


# ==================== CONTROL BLOCKS ====================

def test_saturation_block():
    """Test SaturationBlock clips values."""
    from blocks.saturation import SaturationBlock
    
    block = SaturationBlock()
    
    # Within limits - unchanged
    params = {'min': -1.0, 'max': 1.0}
    result = block.execute(0, {0: np.array([0.5])}, params)
    assert np.isclose(result[0], 0.5), f"Sat within: expected 0.5, got {result[0]}"
    
    # Above max - clips to max
    result = block.execute(0, {0: np.array([2.0])}, params)
    assert np.isclose(result[0], 1.0), f"Sat over max: expected 1.0, got {result[0]}"
    
    # Below min - clips to min
    result = block.execute(0, {0: np.array([-2.0])}, params)
    assert np.isclose(result[0], -1.0), f"Sat under min: expected -1.0, got {result[0]}"
    
    print("[PASS] SaturationBlock")


def test_deadband_block():
    """Test DeadbandBlock matches Simulink Dead Zone behavior."""
    from blocks.deadband import DeadbandBlock
    
    block = DeadbandBlock()
    
    # Dead zone from -0.5 to +0.5
    params = {'start': -0.5, 'end': 0.5}
    
    # Inside deadzone - output is 0
    result = block.execute(0, {0: np.array([0.3])}, params)
    assert np.isclose(result[0], 0.0), f"Deadband inside: expected 0, got {result[0]}"
    
    # Above end - output = input - end (1.0 - 0.5 = 0.5)
    result = block.execute(0, {0: np.array([1.0])}, params)
    assert np.isclose(result[0], 0.5), f"Deadband above: expected 0.5, got {result[0]}"
    
    # Below start - output = input - start (-1.0 - (-0.5) = -0.5)
    result = block.execute(0, {0: np.array([-1.0])}, params)
    assert np.isclose(result[0], -0.5), f"Deadband below: expected -0.5, got {result[0]}"
    
    # Exactly at boundary (end) - output is 0
    result = block.execute(0, {0: np.array([0.5])}, params)
    assert np.isclose(result[0], 0.0), f"Deadband at end: expected 0, got {result[0]}"
    
    print("[PASS] DeadbandBlock (Simulink behavior)")


def test_rate_limiter_block():
    """Test RateLimiterBlock limits rate of change."""
    from blocks.rate_limiter import RateLimiterBlock
    
    block = RateLimiterBlock()
    
    # Initialize with first value
    params = {'rising_slew': 10.0, 'falling_slew': -10.0, '_init_start_': True, 'dtime': 0.1}
    result = block.execute(0, {0: np.array([0.0])}, params)
    
    # Small change within rate limit - should pass through
    result = block.execute(0.1, {0: np.array([0.5])}, params)
    # Rate is 0.5/0.1 = 5, which is < 10.0, so should pass
    
    # Large change exceeds rate limit - should be limited
    result = block.execute(0.2, {0: np.array([50.0])}, params)
    # Can only rise by 10.0 * 0.1 = 1.0 per step from previous
    assert result[0] < 50.0, f"Rate limit: expected < 50.0, got {result[0]}"
    
    print("[PASS] RateLimiterBlock")


def test_delay_block():
    """Test DelayBlock delays signal."""
    from blocks.delay import DelayBlock
    
    block = DelayBlock()
    
    # Initialize
    params = {'delay_time': 0.1, 'init_value': 0.0, '_init_start_': True}
    
    # First output should be initial value
    result = block.execute(0, {0: np.array([5.0])}, params, dtime=0.01)
    assert np.isclose(result[0], 0.0), f"Delay init: expected 0.0, got {result[0]}"
    
    # After delay time, should output the delayed input
    for i in range(15):  # 15 * 0.01 = 0.15 > 0.1 delay
        result = block.execute(i * 0.01, {0: np.array([5.0])}, params, dtime=0.01)
    
    # Should now be outputting the delayed 5.0
    assert np.isclose(result[0], 5.0), f"Delay output: expected 5.0, got {result[0]}"
    
    print("[PASS] DelayBlock")


# ==================== NONLINEAR BLOCKS ====================

def test_hysteresis_block():
    """Test HysteresisBlock with switching thresholds."""
    from blocks.hysteresis import HysteresisBlock
    
    block = HysteresisBlock()
    
    # upper = threshold to switch high, lower = threshold to switch low
    params = {'upper': 1.0, 'lower': 0.5, 'high': 1.0, 'low': 0.0, '_init_start_': True}
    
    # Start below lower - output low
    result = block.execute(0, {0: np.array([0.3])}, params)
    assert np.isclose(result[0], 0.0), f"Hysteresis off: expected 0.0, got {result[0]}"
    
    # Rise above upper - output high
    result = block.execute(0.1, {0: np.array([1.5])}, params)
    assert np.isclose(result[0], 1.0), f"Hysteresis on: expected 1.0, got {result[0]}"
    
    # Fall but still above lower - stay high
    result = block.execute(0.2, {0: np.array([0.7])}, params)
    assert np.isclose(result[0], 1.0), f"Hysteresis stay on: expected 1.0, got {result[0]}"
    
    # Fall below lower - turn low
    result = block.execute(0.3, {0: np.array([0.3])}, params)
    assert np.isclose(result[0], 0.0), f"Hysteresis turn off: expected 0.0, got {result[0]}"
    
    print("[PASS] HysteresisBlock")


def test_switch_block():
    """Test SwitchBlock selects between two inputs."""
    from blocks.switch import SwitchBlock
    
    block = SwitchBlock()
    
    # Port 0 = ctrl, Port 1 = in0, Port 2 = in1
    # ctrl >= threshold selects in0 (port 1), ctrl < threshold selects in1 (port 2)
    params = {'threshold': 0.5, 'n_inputs': 2, 'mode': 'threshold'}
    
    # Control signal high (>= threshold) - select in0 (port 1)
    inputs = {0: 1.0, 1: np.array([10.0]), 2: np.array([20.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 10.0), f"Switch high: expected 10.0, got {result[0]}"
    
    # Control signal low (< threshold) - select in1 (port 2)
    inputs = {0: 0.0, 1: np.array([10.0]), 2: np.array([20.0])}
    result = block.execute(0, inputs, params)
    assert np.isclose(result[0], 20.0), f"Switch low: expected 20.0, got {result[0]}"
    
    print("[PASS] SwitchBlock")


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Math/Control/Nonlinear Block Tests")
    print("=" * 50)
    
    tests = [
        # Math blocks
        ("GainBlock", test_gain_block),
        ("SumBlock", test_sum_block),
        ("AbsBlock", test_abs_block),
        ("SigProductBlock", test_sigproduct_block),
        # Control blocks
        ("SaturationBlock", test_saturation_block),
        ("DeadbandBlock", test_deadband_block),
        ("RateLimiterBlock", test_rate_limiter_block),
        ("DelayBlock", test_delay_block),
        # Nonlinear blocks
        ("HysteresisBlock", test_hysteresis_block),
        ("SwitchBlock", test_switch_block),
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
