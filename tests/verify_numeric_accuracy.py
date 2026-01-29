
import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Starting imports...")
try:
    print("Importing QRect...")
    from PyQt5.QtCore import QRect
    print("Importing DBlock...")
    from lib.simulation.block import DBlock
    print("Importing Line...")
    from lib.simulation.connection import DLine as Line
    print("Importing SimulationEngine...")
    from lib.engine.simulation_engine import SimulationEngine
    print("Imports success.")
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"General Import Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.ERROR)

class MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}

def run_verification():
    print("Initializing Verification System...")
    model = MockModel()
    engine = SimulationEngine(model)
    
    T_Sim = 10000.0
    DT = 0.01
    
    engine.sim_dt = DT
    engine.sim_time = T_Sim
    engine.execution_time = T_Sim
    
    blocks = []
    lines = []
    
    # 1. Sine Block (Source) - Freq=1.0
    # DBlock(block_fn, sid, coords, color)
    sine = DBlock("Sine", 1, QRect(0,0,50,50), "blue")
    sine.block_fn = "Sine" 
    sine.params = {'amplitude': 1.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 0.0}
    sine.hierarchy = 0
    blocks.append(sine)
    
    # 2. Integrator Block (Target)
    integ = DBlock("Integrator", 2, QRect(100,0,50,50), "green")
    integ.block_fn = "Integrator"
    integ.params = {'init_conds': 0.0}
    integ.hierarchy = 1
    blocks.append(integ)
    
    # Connection
    line = Line(1, sine.name, 0, integ.name, 0, [])
    lines.append(line)
    
    # 3. Scope (Observer)
    scope = DBlock("Scope", 3, QRect(200,0,50,50), "black")
    scope.block_fn = "Scope"
    scope.hierarchy = 2
    blocks.append(scope)
    
    line2 = Line(2, integ.name, 0, scope.name, 0, [])
    lines.append(line2)
    
    model.blocks_list = blocks
    model.line_list = lines
    
    print(f"System: Sine(1.0) -> Integrator. Running for {T_Sim}s (dt={DT})...")
    
    # Check compilability
    if not engine.check_compilability(blocks):
        print("Error: System not compilable.")
        return
        
    # Run Fast Solver
    t_span = (0.0, T_Sim)
    
    # Important: SimulationEngine init might need display scaling or something
    # handled by mocking? DBlock uses QRect which works headless.
    
    try:
        success = engine.run_compiled_simulation(blocks, lines, t_span, DT)
    except Exception as e:
        print(f"Simulation Exception: {e}")
        import traceback
        traceback.print_exc()
        return

    if not success:
        print("Error: Simulation failed.")
        return
        
    print("Simulation Completed.")
    
    # Analyze Scope Data
    if 'vector' in scope.exec_params:
        data = np.array(scope.exec_params['vector'])
        print(f"Data Points: {len(data)}")
        if len(data) == 0:
            print("Error: Empty data vector.")
            return

        print(f"Stats: Min={np.min(data):.4f}, Max={np.max(data):.4f}, Mean={np.mean(data):.4f}")
        
        # Expected: 1 - cos(t).
        # Max = 2, Min = 0.
        
        expected_min = 0.0
        expected_max = 2.0
        
        tolerance = 0.1 # generous tolerance for 10000s integration
        
        if abs(np.min(data) - expected_min) < tolerance and abs(np.max(data) - expected_max) < tolerance:
            print("Result VALID (matches 1-cos(t)).")
        else:
            print(f"Result INVALID. Expected [0, 2], got [{np.min(data):.4f}, {np.max(data):.4f}]")
            
    else:
        print("Error: No data in Scope.")

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        import traceback
        traceback.print_exc()
