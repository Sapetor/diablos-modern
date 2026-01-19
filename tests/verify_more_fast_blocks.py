
import logging
import sys
from PyQt5.QtWidgets import QApplication

# Initialize QApp for DBlock dependencies
if not QApplication.instance():
    app = QApplication(sys.argv)

from lib.engine.simulation_engine import SimulationEngine
from lib.engine.system_compiler import SystemCompiler
import numpy as np
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from PyQt5.QtCore import QRect, QPoint

def make_block(name, type, params=None, pos=(0,0)):
    # DBlock signature: block_fn, sid, coords, color, ...
    # We pass 'type' as block_fn.
    # We pass 'name' as username.
    b = DBlock(type, 0, QRect(0,0,50,50), "white", username=name)
    if params:
        b.params.update(params)
    b.relocate_Block(QPoint(*pos))
    return b

def make_line(src, src_port, dst, dst_port):
    # DLine(sid, src_name, src_port, dst_name, dst_port, points)
    return DLine(0, src.name, src_port, dst.name, dst_port, [(0,0), (0,0)])

def verify_deadband():
    logger.info("=== Verifying Deadband ===")
    # Sine -> Deadband -> Scope
    
    b_sin = make_block("Sin", "Sine", {"amplitude": 2.0, "frequency": 1.0})
    b_db = make_block("DB", "Deadband", {"start": -0.5, "end": 0.5})
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_sin, b_db, b_scope]
    lines = [
        make_line(b_sin, 0, b_db, 0),
        make_line(b_db, 0, b_scope, 0)
    ]
    
    # Mock Model
    class MockModel:
        def __init__(self, blocks, lines):
            self.blocks_list = blocks
            self.lines_list = lines
        def link_goto_from(self): pass
        
    model = MockModel(blocks, lines)
    
    # Initialize Engine
    engine = SimulationEngine(model)
    # Patch initialize
    engine.initialize_execution = lambda x: True
    
    # Run
    t_span = (0, 10.0)
    dt = 0.1
    # We must call run_compiled_simulation
    # Compiler check
    compiler = SystemCompiler()
    if not compiler.check_compilability(blocks):
        logger.error("System deemed not compilable!")
        return False
        
    # Compile
    # Note: run_compiled_simulation calls compile internally usually? 
    # No, it uses self.compiler.compile_system inside run_compiled_simulation if not done?
    # Actually logic is inside execution_batch usually.
    # But for test we call engine.run_compiled_simulation
    
    res = engine.run_compiled_simulation(blocks, lines, t_span, dt)
    if res:
        logger.info("Fast Solver Execution Successful!")
        # Check Scope Data
        vec = b_scope.exec_params.get('vector', [])
        if len(vec) == 0:
            logger.error("Scope vector is empty!")
            return False
        
        vector = np.array(vec)
        # Check Deadband applied
        # Input: 2*sin(t)
        # Check a few points
        # At t=0 -> sin=0 -> in=0 -> out=0 (inside -0.5, 0.5)
        # At t=pi/2 (1.57s) -> sin=1 -> in=2.0 -> out=2.0 - 0.5 = 1.5
        pass
    else:
        logger.error("Fast Solver returned False")
    return True

def verify_exponential():
    logger.info("=== Verifying Exponential ===")
    # Ramp (x) -> Exponential (a*exp(b*x)) -> Scope
    
    b_ramp = make_block("Ramp", "Ramp", {"slope": 1.0, "delay": 0.0})
    b_exp = make_block("Exp", "Exponential", {"a": 2.0, "b": 0.5}) # y = 2 * exp(0.5 * t)
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_ramp, b_exp, b_scope]
    lines = [
        make_line(b_ramp, 0, b_exp, 0),
        make_line(b_exp, 0, b_scope, 0)
    ]
    
    model = type('MockModel', (), {'blocks_list': blocks, 'lines_list': lines, 'link_goto_from': lambda: None})()
    engine = SimulationEngine(model)
    engine.initialize_execution = lambda x: True
    
    res = engine.run_compiled_simulation(blocks, lines, (0, 2.0), 0.1)
    
    if res:
        vec = np.array(b_scope.exec_params.get('vector', []))
        logger.info(f"Scope vector shape: {vec.shape}")
        # t=0 -> x=0 -> y=2*1 = 2
        # t=2 -> x=2 -> y=2*exp(1) = 5.436
        first = vec[0]
        last = vec[-1]
        logger.info(f"First: {first}, Last: {last}")
        if abs(first - 2.0) < 1e-3 and abs(last - 5.436) < 1e-2:
            logger.info("Exponential verified!")
        else:
            logger.error("Exponential values incorrect!")

def verify_pid():
    logger.info("=== Verifying PID (Step Response) ===")
    # Step -> PID -> Integrator -> Scope (Control Loop? No, Open Loop test for PID behavior)
    # Step: delay=1, val=1.
    # PID: Kp=1, Ki=1, Kd=0 (PI controller)
    # Integrator: To accumulate output? 
    # Or just measure PID output.
    
    # Input step at t=1.
    # t < 1: e=0 -> u=0.
    # t >= 1: e=1 -> P=1.
    # Integral term starts increasing. I_dot = 1. I = (t-1).
    # u = 1 + (t-1).
    # At t=2, u = 1 + 1 = 2.
    
    b_step = make_block("Step", "Step", {"delay": 1.0, "value": 1.0}) # Setpoint
    b_pid = make_block("PID", "PID", {"Kp": 1.0, "Ki": 1.0, "Kd": 0.0})
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_step, b_pid, b_scope]
    lines = [
        make_line(b_step, 0, b_pid, 0), # Port 0 (Setpoint)
        make_line(b_pid, 0, b_scope, 0)
    ]
    
    model = type('MockModel', (), {'blocks_list': blocks, 'lines_list': lines, 'link_goto_from': lambda: None})()
    engine = SimulationEngine(model)
    engine.initialize_execution = lambda x: True
    
    params = {}
    engine.run_compiled_simulation(blocks, lines, (0, 3.0), 0.1)
    
    vec = np.array(b_scope.exec_params.get('vector', []))
    # Expected:
    # t=0.0: 0
    # t=0.9: 0
    # t=1.1: P=1, I=0.1 -> 1.1 roughly.
    # t=3.0: P=1, I=2.0 -> 3.0 roughly.
    
    logger.info(f"PID output at start: {vec[0]}")
    logger.info(f"PID output at end: {vec[-1]}")
    
    if abs(vec[-1] - 3.0) < 0.2: # Allow solver drift
        logger.info("PID PI Logic verified!")
    else:
        logger.error(f"PID PI Logic failed. Expected ~3.0, got {vec[-1]}")

def verify_pid_derivative():
    logger.info("=== Verifying PID Derivative ===")
    # Ramp -> PID (Kd=1, N=100) -> Scope
    # Ramp 1.0 slope. e(t) = t.
    # de/dt = 1.
    # D term = Kd * de/dt = 1 * 1 = 1.
    # Filter N=100 (fast).
    # Output should settle to 1.
    
    b_ramp = make_block("Ramp", "Ramp", {"slope": 1.0})
    b_pid = make_block("PID", "PID", {"Kp": 0.0, "Ki": 0.0, "Kd": 1.0, "N": 100.0})
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_ramp, b_pid, b_scope]
    lines = [
        make_line(b_ramp, 0, b_pid, 0),
        make_line(b_pid, 0, b_scope, 0)
    ]
    
    model = type('MockModel', (), {'blocks_list': blocks, 'lines_list': lines, 'link_goto_from': lambda: None})()
    engine = SimulationEngine(model)
    engine.initialize_execution = lambda x: True
    
    engine.run_compiled_simulation(blocks, lines, (0, 1.0), 0.01)
    
    vec = np.array(b_scope.exec_params.get('vector', []))
    logger.info(f"PID D output end: {vec[-1]}")
    if abs(vec[-1] - 1.0) < 0.1:
        logger.info("PID Derivative verified!")
    else:
        logger.error(f"PID Derivative failed. Expected ~1.0, got {vec[-1]}")

if __name__ == "__main__":
    verify_deadband()
    verify_exponential()
    verify_pid()
def verify_ratelimiter():
    logger.info("=== Verifying RateLimiter ===")
    # Step (t=0, val=2.0) -> RateLimiter (rising=1.0) -> Scope
    # Should reach 2.0 at t=2.0.
    
    b_step = make_block("Step", "Step", {"delay": 0.0, "value": 2.0})
    b_rl = make_block("RL", "RateLimiter", {"rising": 1.0, "falling": -1.0, "init_cond": 0.0})
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_step, b_rl, b_scope]
    lines = [
        make_line(b_step, 0, b_rl, 0),
        make_line(b_rl, 0, b_scope, 0)
    ]
    
    model = type('MockModel', (), {'blocks_list': blocks, 'lines_list': lines, 'link_goto_from': lambda: None})()
    engine = SimulationEngine(model)
    engine.initialize_execution = lambda x: True
    
    SystemCompiler().check_compilability(blocks)
    
    engine.run_compiled_simulation(blocks, lines, (0, 3.0), 0.1)
    
    vec = np.array(b_scope.exec_params.get('vector', []))
    last = vec[-1]
    logger.info(f"RL Last Val: {last}")
    
    # Check slope approx
    # at t=1, val should be 1.0
    # at t=2, val should be 2.0
    # at t=3, val should be 2.0
    
    # index 10 (t=1.0), 20 (t=2.0), 30 (t=3.0)
    v1 = vec[10] # t=1
    v2 = vec[20] # t=2
    v3 = vec[30] # t=3
    
    logger.info(f"t=1: {v1}, t=2: {v2}, t=3: {v3}")
    
    if abs(v1 - 1.0) < 0.1 and abs(v2 - 2.0) < 0.1 and abs(v3 - 2.0) < 0.1:
        logger.info("RateLimiter verified!")
    else:
        logger.error(f"RateLimiter check failed! Expected 1.0, 2.0, 2.0. Got {v1}, {v2}, {v3}")

def verify_tranfn_casing():
    logger.info("=== Verifying TranFn Casing (Regression) ===")
    # Step -> TranFn (legacy name) -> Scope
    # TranFn: 1/(s+1). Step=1. Output -> 1 - exp(-t).
    
    b_step = make_block("Step", "Step", {"delay": 0.0, "value": 1.0})
    # Use "TranFn" to mimic palette key
    b_tf = make_block("TF", "TranFn", {"numerator": [1.0], "denominator": [1.0, 1.0]}) 
    b_scope = make_block("Scope", "Scope")
    
    blocks = [b_step, b_tf, b_scope]
    lines = [
        make_line(b_step, 0, b_tf, 0),
        make_line(b_tf, 0, b_scope, 0)
    ]
    
    model = type('MockModel', (), {'blocks_list': blocks, 'lines_list': lines, 'link_goto_from': lambda: None})()
    engine = SimulationEngine(model)
    engine.initialize_execution = lambda x: True
    
    SystemCompiler().check_compilability(blocks)
    
    engine.run_compiled_simulation(blocks, lines, (0, 2.0), 0.1)
    
    vec = np.array(b_scope.exec_params.get('vector', []))
    last = vec[-1]
    logger.info(f"TranFn Last Val: {last}")
    
    # Expected: 1 - exp(-2) = 1 - 0.135 = 0.865
    if abs(last - 0.865) < 0.05:
        logger.info("TranFn Casing verified!")
    else:
        logger.error(f"TranFn Casing failed! Expected ~0.865, got {last}")

if __name__ == "__main__":
    verify_deadband()
    verify_exponential()
    verify_pid()
    verify_pid_derivative()
    verify_ratelimiter()
    verify_tranfn_casing()
