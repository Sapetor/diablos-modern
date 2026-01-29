
import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.engine.system_compiler import SystemCompiler
from lib.engine.simulation_engine import SimulationEngine

class TestFastSolverExpansion(unittest.TestCase):
    def make_line(self, src: str, sport: int, dst: str, dport: int):
        return DLine(0, src, sport, dst, dport, [(0,0), (0,0)])

    def test_ramp_saturation(self):
        print("\nTesting Ramp -> Saturation...")
        # Ramp: slope=2, delay=0.1
        ramp = DBlock(block_type='ramp', name='ramp0')
        ramp.params = {'slope': 2.0, 'delay': 0.1}
        
        # Saturation: min=0, max=1.0
        sat = DBlock(block_type='saturation', name='sat0')
        sat.params = {'min': 0.0, 'max': 1.0}
        
        # Line
        line = self.make_line('ramp0', 0, 'sat0', 0)
        
        blocks = [ramp, sat]
        lines = [line]
        
        # Compile
        compiler = SystemCompiler()
        self.assertTrue(compiler.check_compilability(blocks))
        
        compiled_data = compiler.compile_system(blocks, blocks, lines)
        model_func, y0, state_map, block_matrices = compiled_data
        
        # Better to run SimulationEngine which stores signals
        engine = SimulationEngine()
        engine.load_system(blocks, lines)
        success = engine.run_compiled_simulation(0.0, 1.0, 0.01)
        self.assertTrue(success)

    def test_full_simulation_with_values(self):
        print("\nTesting Simulation Output with Scopes...")
        # 1. Ramp -> Saturation -> Scope
        ramp = DBlock(block_type='ramp', name='ramp0')
        ramp.params = {'slope': 2.0, 'delay': 0.1}
        
        sat = DBlock(block_type='saturation', name='sat0')
        sat.params = {'min': 0.0, 'max': 0.5} # Clip at 0.5
        
        scope = DBlock(block_type='scope', name='scope0')
        
        line1 = self.make_line('ramp0', 0, 'sat0', 0)
        line2 = self.make_line('sat0', 0, 'scope0', 0)
        
        blocks = [ramp, sat, scope]
        lines = [line1, line2]
        
        engine = SimulationEngine()
        engine.load_system(blocks, lines)
        success = engine.run_compiled_simulation(0.0, 1.0, 0.1) # 10 steps
        self.assertTrue(success)
        
        # Scope output should work
        # In Replay Loop, we execute 'Scope' logic which appends to exec_params['vector']
        # But DBlock in this test might not have exec_params initialized unless Simulator initialized it.
        # run_compiled_simulation replay loop checks hasattr.
        # It creates it if missing.
        
        vector = scope.exec_params['vector']
        print(f"Scope Vector: {vector}")
        
        # Expected:
        # t=0.0: Ramp=0, Sat=0
        # t=0.1: Ramp=0, Sat=0
        # t=0.2: Ramp=0.2, Sat=0.2
        # ...
        # t=0.3: Ramp=0.4, Sat=0.4
        # t=0.4: Ramp=0.6, Sat=0.5 (Clipped)
        
        # Verify Clipping
        max_val = np.max(vector)
        self.assertAlmostEqual(max_val, 0.5)
        
    def test_switch(self):
        print("\nTesting Switch...")
        # Ctrl = 2.0 (Threshold 0.0 -> sel=0 -> input 1)
        ctrl = DBlock(block_type='constant', name='ctrl')
        ctrl.params = {'value': 2.0}
        
        in1 = DBlock(block_type='constant', name='in1')
        in1.params = {'value': 100.0}
        
        in2 = DBlock(block_type='constant', name='in2')
        in2.params = {'value': -100.0}
        
        switch = DBlock(block_type='switch', name='switch0')
        switch.params = {'threshold': 0.0, 'n_inputs': 2}
        
        scope = DBlock(block_type='scope', name='scope0')
        
        # switch inputs: 0=ctrl, 1=in1, 2=in2
        l1 = self.make_line('ctrl', 0, 'switch0', 0)
        l2 = self.make_line('in1', 0, 'switch0', 1)
        l3 = self.make_line('in2', 0, 'switch0', 2)
        l4 = self.make_line('switch0', 0, 'scope0', 0)
        
        blocks = [ctrl, in1, in2, switch, scope]
        lines = [l1, l2, l3, l4]
        
        engine = SimulationEngine()
        engine.load_system(blocks, lines)
        engine.run_compiled_simulation(0.0, 0.1, 0.1)
        
        vec = scope.exec_params['vector']
        print(f"Switch Output (Ctrl=2.0 -> In1): {vec[0]}")
        self.assertEqual(vec[0], 100.0)
