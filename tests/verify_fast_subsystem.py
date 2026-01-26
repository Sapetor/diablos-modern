
import sys
import os
import unittest
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)


# Adjust path to find lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.models.simulation_model import SimulationModel
from lib.engine.simulation_engine import SimulationEngine
from blocks.subsystem import Subsystem
from blocks.inport import Inport
from blocks.outport import Outport

class TestFastSubsystem(unittest.TestCase):
    def setUp(self):
        self.model = SimulationModel()
        self.engine = SimulationEngine(self.model)
        
    def test_simple_subsystem_compilability(self):
        """test that a subsystem containing compilable blocks is accepted as compilable."""
        # Create Subsystem
        sub = Subsystem(block_name="Sub1")
        
        # Create Internal Blocks
        inport = Inport(block_name="In1")
        gain = DBlock(block_fn="Gain", name="Gain1")
        gain.params = {'gain': 2.0}
        outport = Outport(block_name="Out1")
        
        # Internal Connections
        # In1 -> Gain1 -> Out1
        line1 = DLine(1, "In1", 0, "Gain1", 0, [])
        line2 = DLine(2, "Gain1", 0, "Out1", 0, [])
        
        sub.sub_blocks = [inport, gain, outport]
        sub.sub_lines = [line1, line2]
        
        # Top Level
        step = DBlock(block_fn="Step", name="Step1")
        step.params = {'value': 1.0, 'delay': 0.0}
        
        scope = DBlock(block_fn="Scope", name="Scope1")
        
        self.model.blocks_list = [step, sub, scope]
        
        # Top Level Connections
        # Step1 -> Sub1 -> Scope1
        # Note: Subsystem needs port definitions?
        # The engine/compiler doesn't check port definition geometry, just connectivity.
        # But Flattener does rely on mapping?
        # Flattener relies on `input_drivers` map built from lines.
        # Check Flattener._resolve_driver logic.
        
        # For Flattener, we need to ensure the line destinations match.
        # Step1 -> Sub1 (Port 0??)
        # Sub1 (Port 0) -> Scope1
        
        top_line1 = DLine(10, "Step1", 0, "Sub1", 0, [])
        top_line2 = DLine(11, "Sub1", 0, "Scope1", 0, [])
        
        self.model.line_list = [top_line1, top_line2]
        
        # Check Compilability
        print("Checking compilability...")
        can_compile = self.engine.check_compilability(self.model.blocks_list)
        self.assertTrue(can_compile, "System with Subsystem should be compilable")
        
    def test_run_fast_subsystem(self):
        """Test execution of the flattened subsystem."""
        # 1. Build Model
        sub = Subsystem(block_name="Sub1")
        inport = Inport(block_name="In1")
        gain = DBlock(block_fn="Gain", name="Gain1")
        gain.params = {'gain': 5.0} # K=5
        outport = Outport(block_name="Out1")
        
        # Need to ensure Inport/Outport "ports_map" or naming expectation works for Flattener.
        # Flattener tries:
        # 1. parent_block.ports_map (if available)
        # 2. Name parsing "In1" -> idx 0.
        # Our Inport name is "In1", so idx 0.
        
        sub.sub_blocks = [inport, gain, outport]
        sub.sub_lines = [
            DLine(1, "In1", 0, "Gain1", 0, []),
            DLine(2, "Gain1", 0, "Out1", 0, [])
        ]
        
        step = DBlock(block_fn="Step", name="Step1")
        step.params = {'value': 2.0, 'delay': 0.0}
        
        scope = DBlock(block_fn="Scope", name="Scope1")
        # Initialize scope vector explicitly just in case
        scope.params['vector'] = []
        
        self.model.blocks_list = [step, sub, scope]
        self.model.line_list = [
            DLine(10, "Step1", 0, "Sub1", 0, []),
            DLine(11, "Sub1", 0, "Scope1", 0, [])
        ]
        
        # 2. Run
        print("Running compiled simulation...")
        t_span = (0.0, 1.0)
        dt = 0.1
        
        # Resolve exec_params (simulating DSim.prepare_execution logic broadly)
        for b in [step, gain, scope]: 
            b.exec_params = b.params.copy()
            
        success = self.engine.run_compiled_simulation(
            self.model.blocks_list,
            self.model.line_list,
            t_span,
            dt
        )
        
        self.assertTrue(success, "Simulation should succeed")
        
        # 3. Verify Results
        # Scope1 should have 2.0 * 5.0 = 10.0
        # Check scope vector
        
        # Find scope in self.model.blocks_list? NO.
        # The Replay Loop updates the block instances in `current_blocks`.
        # Which are flattened copies.
        # Wait, does it propagate back to `self.model.blocks_list`?
        # NO. Flattener makes copies.
        
        # The Plotter reads `engine.active_blocks_list`.
        # So we should check `engine.active_blocks_list`.
        
        found_scope = None
        for b in self.engine.active_blocks_list:
            if "Scope1" in b.name: # Flattened name might be Scope1 (if at top)
                found_scope = b
                break
        
        self.assertIsNotNone(found_scope)
        self.assertTrue(hasattr(found_scope, 'exec_params'))
        self.assertIn('vector', found_scope.exec_params)
        
        vec = found_scope.exec_params['vector']
        print(f"Scope Vector: {vec}")
        
        self.assertTrue(len(vec) > 0)
        self.assertAlmostEqual(vec[-1], 10.0, places=5)

if __name__ == '__main__':
    unittest.main()
