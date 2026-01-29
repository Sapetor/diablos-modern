
import sys
import os
import logging
import numpy as np

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt5.QtCore import QRect
from lib.engine.simulation_engine import SimulationEngine
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine as Line

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}
        
    def link_goto_from(self):
        pass

def create_simple_system():
    model = MockModel()
    engine = SimulationEngine(model)
    
    # 1. Sine Wave
    sine = DBlock("Sine", "Sine1", coords=QRect(0,0,50,50), color="blue")
    sine.params['amplitude'] = 1.0
    sine.params['frequency'] = 1.0
    sine.hierarchy = 0
    sine.block_fn = 'Sine'
    
    # 2. Integrator
    integ = DBlock("Integrator", "Integ1", coords=QRect(100,0,50,50), color="green")
    integ.params['init_conds'] = 0.0
    integ.block_fn = 'Integrator' 
    integ.hierarchy = 1
    
    # 3. Scope
    scope = DBlock("Scope", "Scope1", coords=QRect(200,0,50,50), color="black")
    scope.block_fn = 'Scope' # Not in compilable list? Check SystemCompiler
    scope.get_id() # Init buffers
    scope.hierarchy = 2

    # Connections
    # Sine -> Integrator
    line1 = Line(sine, 0, integ, 0)
    
    # Integrator -> Scope
    line2 = Line(integ, 0, scope, 0)
    
    model.blocks_list = [sine, integ] 
    # NOTE: Scope is NOT in COMPILABLE_BLOCKS allowlist yet? 
    # Let's check SystemCompiler content.
    # If not, check_compilability will fail.
    
    model.line_list = [line1, line2]
    
    return engine, model, [sine, integ, scope]

def test_fast_solver():
    engine, model, blocks = create_simple_system()
    
    # Add Scope to allowed blocks logic in SystemCompiler?
    # Or just test with Sine->Integrator (system is valid even without Scope for solver purpose, 
    # but practically we start simulation with all blocks).
    
    # Let's inspect SystemCompiler allowed blocks.
    # Compilable: Integrator, Gain, Sum, Constant, Sine, Step, TransferFcn, Mux, Demux.
    # Scope is NOT in list.
    # So `check_compilability` should return False if Scope is present.
    
    logger.info("--- TEST 1: Check Compilability (Expect False due to Scope) ---")
    allowed = engine.check_compilability(blocks)
    logger.info(f"Compilable with Scope? {allowed}")
    
    # Remove Scope for test
    blocks_no_scope = [b for b in blocks if b.name != "Scope1"]
    
    logger.info("--- TEST 2: Check Compilability (Expect True) ---")
    allowed = engine.check_compilability(blocks_no_scope)
    logger.info(f"Compilable without Scope? {allowed}")
    
    if allowed:
        logger.info("--- TEST 3: Run Compiled Simulation ---")
        lines = [l for l in model.line_list if l.dst_block.name != "Scope1"]
        t_span = (0.0, 10.0)
        dt = 0.01
        
        success = engine.run_compiled_simulation(blocks_no_scope, lines, t_span, dt)
        if success:
            logger.info("Simulation Successful!")
            logger.info(f"Timeline steps: {len(engine.timeline)}")
            logger.info(f"Output shape: {engine.outs.shape}")
            logger.info(f"Final Value: {engine.outs[0, -1]}")
            # Integral of sin(t) is -cos(t) + C.
            # int_0^10 sin(t) dt = [-cos(10) - (-cos(0))] = -cos(10) + 1
            # cos(10) ~ -0.839. Result ~ 1.839.
            expected = -np.cos(10) + 1
            logger.info(f"Expected: {expected}")
        else:
            logger.error("Simulation Failed")


if __name__ == "__main__":
    try:
        test_fast_solver()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

