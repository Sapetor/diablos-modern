
import sys
import os
import time
import logging
from PyQt5.QtCore import QRect

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.engine.simulation_engine import SimulationEngine
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine as Line

# Setup logging
logging.basicConfig(level=logging.ERROR) # Silence info for benchmark
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

class MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}
        self.dirty = False
        
    def link_goto_from(self):
        pass

def create_benchmark_system(n_integrators=10):
    """Create a chain of integrators to stress the solver."""
    model = MockModel()
    engine = SimulationEngine(model)
    engine.sim_dt = 0.001 # Small step for load
    engine.sim_time = 10.0 # 10 seconds = 10,000 steps
    engine.execution_time = 10.0
    
    blocks = []
    lines = []
    
    # Source: Sine
    sine = DBlock("Sine", 0, QRect(0,0,50,50), "blue")
    sine.params['amplitude'] = 1.0
    sine.params['frequency'] = 1.0
    sine.hierarchy = 0
    sine.block_fn = 'Sine'
    blocks.append(sine)
    
    prev_block = sine
    prev_port = 0
    
    for i in range(n_integrators):
        integ = DBlock("Integrator", i+1, QRect(100+i*60,0,50,50), "green")
        integ.params['init_conds'] = 0.0
        integ.block_fn = 'Integrator' 
        integ.hierarchy = i + 1
        blocks.append(integ)
        
        line = Line(i, prev_block.name, prev_port, integ.name, 0, [])
        lines.append(line)
        
        prev_block = integ
        
    # Scope
    scope = DBlock("Scope", n_integrators+1, QRect(100+n_integrators*60,0,50,50), "black")
    scope.block_fn = 'Scope'
    scope.hierarchy = n_integrators + 1
    blocks.append(scope)
    
    line = Line(n_integrators, prev_block.name, 0, scope.name, 0, [])
    lines.append(line)
    
    model.blocks_list = blocks
    model.line_list = lines
    
    return engine, model, blocks

def run_benchmark():
    engine, model, blocks = create_benchmark_system(n_integrators=10) # 10 serial integrators
    
    dsim = DSim(None, None, None, None, None) # Mock DSim? 
    # DSim constructor is complex (Canvas, etc).
    # Let's just use engine directly or mock execution_batch logic.
    
    logger.info(f"System: 1 Sine -> 10 Integrators -> Scope (dt=0.001, T=10.0s, Steps=10,000)")
    
    # 1. Fast Solver
    start_time = time.time()
    if engine.check_compilability(blocks):
        t_span = (0.0, engine.execution_time)
        engine.run_compiled_simulation(blocks, model.line_list, t_span, engine.sim_dt)
    fast_duration = time.time() - start_time
    logger.info(f"Fast Solver Time: {fast_duration:.4f}s")
    
    # 2. Interpreter
    # Reset engine state
    engine.execution_initialized = False
    
    # We need to manually run the interpreter loop since DSim.execution_loop isn't easily importable without GUI
    # But we can try to use engine.execute_block in a loop.
    # Or simplified:
    start_time = time.time()
    
    # Initialize
    engine.initialize_execution(blocks)
    steps = int(engine.execution_time / engine.sim_dt)
    
    # Simple Interpreter Loop equivalent
    for step in range(steps):
        # Update time
        engine.time_step += engine.sim_dt
        
        # Execute blocks in order
        for block in blocks:
             # Logic from execution_loop reduced
             engine.execute_block(block)
             # Propagate
             out = {0: 0.0} # Simplified prop
             # engine.propagate_outputs(block, out) # Skip overhead of prop for fairness? No, prop is part of cost.
             
    std_duration = time.time() - start_time
    logger.info(f"Interpreter Time (Est): {std_duration:.4f}s")
    
    speedup = std_duration / fast_duration if fast_duration > 0 else 0
    logger.info(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
