"""
Legacy test for Sine block with omega parameter.

NOTE: This test is skipped because it uses legacy DBlock API that requires
GUI initialization. The Sine block is properly tested in:
- tests/unit/test_source_blocks.py::TestSineBlock
"""

import sys
import os
import numpy as np
import logging
import pytest

# Skip this legacy test - Sine block is tested in tests/unit/test_source_blocks.py
pytestmark = pytest.mark.skip(
    reason="Legacy DBlock API test. Sine block is tested in tests/unit/test_source_blocks.py"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from PyQt5.QtCore import QRect
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine as Line
    from lib.engine.simulation_engine import SimulationEngine
except ImportError:
    print("Imports failed.")
    sys.exit(1)

logging.basicConfig(level=logging.ERROR)

class MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}

def test_sine_params():
    print("Testing Sine Params (Omega=2)...")
    model = MockModel()
    engine = SimulationEngine(model)
    
    T = 10.0
    dt = 0.01
    
    # create lowercase 'sine' block with 'omega' param
    sine = DBlock("Sine", 1, QRect(0,0,50,50), "blue")
    sine.block_fn = "sine" # Lowercase to test fix
    sine.params = {'amplitude': 1.0, 'omega': 2.0, 'init_angle': 0.0} # 'omega' instead of 'frequency'
    sine.hierarchy = 0
    
    scope = DBlock("Scope", 2, QRect(100,0,50,50), "black")
    scope.block_fn = "Scope"
    scope.hierarchy = 1
    
    line = Line(1, sine.name, 0, scope.name, 0, [])
    
    blocks = [sine, scope]
    lines = [line]
    
    # Check compilability
    if not engine.check_compilability(blocks):
        print("Error: System reported as NOT compilable (Case sensitivity fix failed).")
        return
        
    print("System is compilable.")
    
    # Run
    t_span = (0.0, T)
    success = engine.run_compiled_simulation(blocks, lines, t_span, dt)
    
    if not success:
        print("Simulation failed.")
        return
        
    data = np.array(scope.exec_params['vector'])
    t = np.arange(0, T+dt, dt)[:len(data)]
    
    # Expected: sin(2t)
    expected = np.sin(2.0 * t)
    
    # Check correlations/error
    # Note: t array might not perfectly align if solver steps differ from dt
    # But scope repays using dense t maybe?
    # Scope in replay loop uses 'val'. 
    # Since we didn't force t_eval in run_compiled_simulation to exactly match our reconstruction here if adaptive steps are used?
    # Wait, run_compiled_simulation uses t_eval = np.arange(...)
    # So output size should match.
    
    if len(data) != len(expected):
        # Resize to match smaller
        n = min(len(data), len(expected))
        data = data[:n]
        expected = expected[:n]
        
    error = np.abs(data - expected)
    print(f"Max Error: {np.max(error):.4e}")
    
    if np.max(error) < 1e-2:
        print("Test PASSED (Sine matches sin(2t)).")
    else:
        print("Test FAILED (Data does not match sin(2t)).")
        print(f"First 10 data: {data[:10]}")
        print(f"First 10 exp: {expected[:10]}")

if __name__ == "__main__":
    test_sine_params()
