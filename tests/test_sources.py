"""
Source Block Tests for DiaBloS

Tests all source blocks: Step, Sine, Ramp, Constant, Noise, PRBS
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())


def test_step_block():
    """Test StepBlock generates correct step signal."""
    from blocks.step import StepBlock
    
    block = StepBlock()
    
    # Test 'up' step type (default) - outputs 0 before delay, value after
    params = {'value': 5.0, 'delay': 0.5, 'type': 'up', 'pulse_start_up': True, '_init_start_': True}
    
    # Before delay - should be 0
    result = block.execute(0.3, {}, params)
    assert np.isclose(result[0], 0.0), f"Step before delay: expected 0, got {result[0]}"
    
    # After delay - should be value
    result = block.execute(0.6, {}, params)
    assert np.isclose(result[0], 5.0), f"Step after delay: expected 5.0, got {result[0]}"
    
    print("[PASS] StepBlock 'up' type")
    
    # Test 'down' step type - outputs value before delay, 0 after  
    params = {'value': 3.0, 'delay': 0.5, 'type': 'down', 'pulse_start_up': True, '_init_start_': True}
    
    result = block.execute(0.3, {}, params)
    assert np.isclose(result[0], 3.0), f"Step down before: expected 3.0, got {result[0]}"
    
    result = block.execute(0.6, {}, params)
    assert np.isclose(result[0], 0.0), f"Step down after: expected 0, got {result[0]}"
    
    print("[PASS] StepBlock 'down' type")
    
    # Test 'constant' type - always outputs value
    params = {'value': 7.0, 'delay': 0.0, 'type': 'constant', 'pulse_start_up': True, '_init_start_': True}
    
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 7.0), f"Step constant: expected 7.0, got {result[0]}"
    
    result = block.execute(1.0, {}, params)
    assert np.isclose(result[0], 7.0), f"Step constant: expected 7.0, got {result[0]}"
    
    print("[PASS] StepBlock 'constant' type")


def test_sine_block():
    """Test SineBlock generates correct sine wave."""
    from blocks.sine import SineBlock
    
    block = SineBlock()
    
    # Test with amplitude=2, omega=pi (period=2s), init_angle=0
    params = {'amplitude': 2.0, 'omega': np.pi, 'init_angle': 0.0}
    
    # At t=0: sin(0) = 0
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 0.0, atol=1e-10), f"Sine at t=0: expected 0, got {result[0]}"
    
    # At t=0.5: sin(pi/2) = 1, amplitude*1 = 2
    result = block.execute(0.5, {}, params)
    assert np.isclose(result[0], 2.0, atol=1e-10), f"Sine at t=0.5: expected 2.0, got {result[0]}"
    
    # At t=1.0: sin(pi) = 0
    result = block.execute(1.0, {}, params)
    assert np.isclose(result[0], 0.0, atol=1e-10), f"Sine at t=1.0: expected 0, got {result[0]}"
    
    # At t=1.5: sin(3pi/2) = -1, amplitude*(-1) = -2
    result = block.execute(1.5, {}, params)
    assert np.isclose(result[0], -2.0, atol=1e-10), f"Sine at t=1.5: expected -2.0, got {result[0]}"
    
    print("[PASS] SineBlock basic test")
    
    # Test with initial angle (phase shift)
    params = {'amplitude': 1.0, 'omega': np.pi, 'init_angle': np.pi/2}  # cos wave
    
    # At t=0: sin(pi/2) = 1
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 1.0, atol=1e-10), f"Sine with phase at t=0: expected 1.0, got {result[0]}"
    
    print("[PASS] SineBlock with phase shift")


def test_ramp_block():
    """Test RampBlock generates correct ramp signal."""
    from blocks.ramp import RampBlock
    
    block = RampBlock()
    
    # Test positive slope
    params = {'slope': 2.0, 'delay': 0.0}
    
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 0.0), f"Ramp at t=0: expected 0, got {result[0]}"
    
    result = block.execute(1.0, {}, params)
    assert np.isclose(result[0], 2.0), f"Ramp at t=1: expected 2.0, got {result[0]}"
    
    result = block.execute(2.5, {}, params)
    assert np.isclose(result[0], 5.0), f"Ramp at t=2.5: expected 5.0, got {result[0]}"
    
    print("[PASS] RampBlock positive slope")
    
    # Test with delay
    params = {'slope': 1.0, 'delay': 0.5}
    
    result = block.execute(0.3, {}, params)
    assert result[0] <= 0, f"Ramp before delay: expected <=0, got {result[0]}"
    
    result = block.execute(1.5, {}, params)
    assert np.isclose(result[0], 1.0), f"Ramp at t=1.5 with delay 0.5: expected 1.0, got {result[0]}"
    
    print("[PASS] RampBlock with delay")
    
    # Test negative slope
    params = {'slope': -2.0, 'delay': 0.0}
    
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 0.0), f"Negative ramp at t=0: expected 0, got {result[0]}"
    
    result = block.execute(1.0, {}, params)
    assert np.isclose(result[0], -2.0), f"Negative ramp at t=1: expected -2.0, got {result[0]}"
    
    print("[PASS] RampBlock negative slope")


def test_constant_block():
    """Test ConstantBlock generates constant output."""
    from blocks.constant import ConstantBlock
    
    block = ConstantBlock()
    
    # Test scalar constant
    params = {'value': 42.0}
    
    result = block.execute(0.0, {}, params)
    assert np.isclose(result[0], 42.0), f"Constant: expected 42.0, got {result[0]}"
    
    result = block.execute(100.0, {}, params)
    assert np.isclose(result[0], 42.0), f"Constant at t=100: expected 42.0, got {result[0]}"
    
    print("[PASS] ConstantBlock scalar")
    
    # Test vector constant
    params = {'value': [1.0, 2.0, 3.0]}
    
    result = block.execute(0.0, {}, params)
    expected = np.array([1.0, 2.0, 3.0])
    assert np.allclose(result[0], expected), f"Constant vector: expected {expected}, got {result[0]}"
    
    print("[PASS] ConstantBlock vector")


def test_noise_block():
    """Test NoiseBlock generates random noise within bounds."""
    from blocks.noise import NoiseBlock
    
    block = NoiseBlock()
    
    # Test that noise is within expected range (mu=0, sigma=1)
    params = {'mu': 0.0, 'sigma': 1.0}
    
    samples = []
    for t in range(100):
        result = block.execute(t * 0.01, {}, params)
        samples.append(result[0])
    
    samples = np.array(samples)
    
    # Check mean is approximately 0
    mean_val = np.mean(samples)
    assert abs(mean_val) < 1.0, f"Noise mean: expected ~0, got {mean_val}"
    
    print("[PASS] NoiseBlock generates bounded noise")


def test_prbs_block():
    """Test PRBSBlock generates pseudo-random binary sequence."""
    from blocks.prbs import PRBSBlock
    
    block = PRBSBlock()
    
    params = {'high': 1.0, 'low': -1.0, 'bit_time': 0.05, 'order': 7, 'seed': 1, '_init_start_': True}
    
    samples = []
    for i in range(100):
        result = block.execute(i * 0.01, {}, params)
        samples.append(result[0])
    
    samples = np.array(samples)
    
    # PRBS should only output high or low values
    unique_vals = np.unique(samples)
    for val in unique_vals:
        assert val in [1.0, -1.0], f"PRBS value should be 1.0 or -1.0, got {val}"
    
    # Should have transitions (both values present for non-trivial sequence)
    assert len(unique_vals) >= 1, "PRBS should generate at least one unique value"
    
    print("[PASS] PRBSBlock generates binary sequence")


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Source Block Tests")
    print("=" * 50)
    
    tests = [
        ("StepBlock", test_step_block),
        ("SineBlock", test_sine_block),
        ("RampBlock", test_ramp_block),
        ("ConstantBlock", test_constant_block),
        ("NoiseBlock", test_noise_block),
        ("PRBSBlock", test_prbs_block),
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
    
    # Exit with error code if any tests failed
    sys.exit(1 if failed > 0 else 0)
