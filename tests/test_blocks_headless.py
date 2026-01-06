"""
Headless Block Testing for DiaBloS

Tests block functions directly without GUI.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())

from lib.diagram_builder import DiagramBuilder


def test_diagram_builder():
    """Test that DiagramBuilder creates valid diagram files."""
    builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    
    step = builder.add_block("Step", x=50, y=100, name="step",
                             params={"start_time": 0.0, "h_start": 0.0, "h_final": 1.0})
    gain = builder.add_block("Gain", x=150, y=100, name="gain",
                             params={"gain": 2.0})
    scope = builder.add_block("Scope", x=250, y=100, name="scope")
    
    builder.connect(step, 0, gain, 0)
    builder.connect(gain, 0, scope, 0)
    
    assert len(builder.blocks) == 3, f"Expected 3 blocks, got {len(builder.blocks)}"
    assert len(builder.lines) == 2, f"Expected 2 lines, got {len(builder.lines)}"
    
    # Check block properties
    step_block = builder.get_block("step")
    assert step_block["block_fn"] == "Step"
    assert step_block["params"]["h_final"] == 1.0
    
    gain_block = builder.get_block("gain")
    assert gain_block["params"]["gain"] == 2.0
    
    # Check line connections
    line1 = builder.lines[0]
    assert line1["srcblock"] == "step"
    assert line1["dstblock"] == "gain"
    
    print("[PASS] DiagramBuilder test")


def test_block_functions():
    """Test block functions directly without simulation."""
    from lib import functions
    
    # Test Gain
    params = {"gain": 3.0}
    inputs = {0: np.array([2.0])}
    result = functions.gain(0, inputs, params)
    assert np.isclose(result[0], 6.0), f"Gain: expected 6.0, got {result[0]}"
    print("[PASS] Gain function test")
    
    # Test Sum with +-
    params = {"sign": "+-"}
    inputs = {0: np.array([5.0]), 1: np.array([3.0])}
    result = functions.sum_fn(0, inputs, params)
    assert np.isclose(result[0], 2.0), f"Sum: expected 2.0, got {result[0]}"
    print("[PASS] Sum function test")
    
    # Test Step
    params = {"start_time": 0.5, "h_start": 0.0, "h_final": 1.0}
    result_before = functions.step(0.4, {}, params.copy())
    result_after = functions.step(0.6, {}, params.copy())
    assert np.isclose(result_before[0], 0.0), f"Step before: expected 0.0, got {result_before[0]}"
    assert np.isclose(result_after[0], 1.0), f"Step after: expected 1.0, got {result_after[0]}"
    print("[PASS] Step function test")
    
    # Test Integrator initialization
    params = {"init_conds": 0.0, "method": "FWD_EULER", "_init_start_": True}
    result = functions.integrator(0, {0: np.array([1.0])}, params, dtime=0.01)
    assert "mem" in params, "Integrator should initialize 'mem'"
    print("[PASS] Integrator initialization test")


def test_integrator_integration():
    """Test integrator over multiple steps."""
    from lib import functions
    
    # Integrate constant 1.0 for 100 steps with dt=0.01
    params = {"init_conds": 0.0, "method": "FWD_EULER", "_init_start_": True}
    input_val = {0: np.array([1.0])}
    
    for i in range(100):
        result = functions.integrator(i * 0.01, input_val, params, dtime=0.01)
    
    # After integrating 1.0 for 1 second, should be close to 1.0
    final_value = params.get("mem", [0])[0]
    assert np.isclose(final_value, 1.0, rtol=0.05), f"Integrator: expected ~1.0, got {final_value}"
    print("[PASS] Integrator integration test")


def test_saturation():
    """Test saturation block."""
    from lib import functions
    
    params = {"upper_limit": 1.0, "lower_limit": -1.0}
    
    # Under limit
    result = functions.saturation(0, {0: np.array([0.5])}, params.copy())
    assert np.isclose(result[0], 0.5), f"Saturation: expected 0.5, got {result[0]}"
    
    # Over upper limit
    result = functions.saturation(0, {0: np.array([2.0])}, params.copy())
    assert np.isclose(result[0], 1.0), f"Saturation: expected 1.0, got {result[0]}"
    
    # Under lower limit
    result = functions.saturation(0, {0: np.array([-2.0])}, params.copy())
    assert np.isclose(result[0], -1.0), f"Saturation: expected -1.0, got {result[0]}"
    
    print("[PASS] Saturation test")


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Headless Block Tests")
    print("=" * 50)
    
    tests = [
        ("DiagramBuilder", test_diagram_builder),
        ("Block Functions", test_block_functions),
        ("Integrator Integration", test_integrator_integration),
        ("Saturation", test_saturation),
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
