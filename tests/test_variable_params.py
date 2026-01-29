
import unittest
import os
import sys
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.workspace import WorkspaceManager
from blocks.gain import GainBlock
from blocks.gain import GainBlock
from blocks.step import StepBlock
from blocks.base_block import BaseBlock

class TermBlock(BaseBlock):
    def __init__(self):
        super().__init__()
    
    @property
    def block_name(self):
        return "Term"
    
    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]
    
    @property
    def outputs(self):
        return []
    
    @property
    def params(self):
        return {}

    def execute(self, time, inputs, params):
        return {}

from PyQt5.QtWidgets import QApplication

class TestVariableParams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize QApplication for QPixmap
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        # Reset WorkspaceManager
        WorkspaceManager._instance = None
        self.wm = WorkspaceManager()
        self.wm.variables = {'gain_val': 5.0, 'step_amp': 2.0}

        self.sim = DSim()
        self.sim.sim_dt = 0.1
        self.sim.sim_time = 1.0

    def test_variable_gain(self):
        # Create blocks
        # Explicitly set in_ports=0 for Step to avoid issues
        step = DBlock("Step", 1, QRect(0, 0, 100, 100), "blue", block_class=StepBlock, in_ports=0)
        # Populate default params for Step
        step.params = {
            "value": 1.0,
            "delay": 0.0,
            "type": "up",
            "pulse_start_up": True,
            "_init_start_": True,
            "_name_": "step1",
            "_inputs_": 0,
            "_outputs_": 1
        }
        step.params['value'] = 'step_amp'  # Use variable for value
        
        gain = DBlock("Gain", 2, QRect(200, 0, 100, 100), "red", block_class=GainBlock)
        gain.params['gain'] = 'gain_val'  # Use variable

        term = DBlock("Term", 3, QRect(400, 0, 100, 100), "black", in_ports=1, out_ports=0, block_class=TermBlock)

        # Connect blocks
        # Step output 0 -> Gain input 0
        # Gain output 0 -> Term input 0
        self.sim.blocks_list = [step, gain, term]
        # Connect blocks using add_line
        # Step output 0 -> Gain input 0
        from PyQt5.QtCore import QPoint
        # Dummy points for connection
        p1 = QPoint(100, 50)
        p2 = QPoint(200, 50)
        p3 = QPoint(300, 50)
        p4 = QPoint(400, 50)

        self.sim.add_line((step.name, 0, p1), (gain.name, 0, p2))
        self.sim.add_line((gain.name, 0, p3), (term.name, 0, p4))
        
        print(f"Step name: {step.name}, in_ports: {step.in_ports}, out_ports: {step.out_ports}")
        print(f"Gain name: {gain.name}, in_ports: {gain.in_ports}, out_ports: {gain.out_ports}")
        print(f"Term name: {term.name}, in_ports: {term.in_ports}, out_ports: {term.out_ports}")
        # print(f"Signals: {self.sim.signals_list}")
        
        inputs, outputs = self.sim.get_neighbors(step.name)
        print(f"Step neighbors: inputs={len(inputs)}, outputs={len(outputs)}")
        
        # Initialize execution
        # This should resolve parameters
        success = self.sim.execution_init()
        self.assertTrue(success, "Execution initialization failed")

        # Check if parameters were resolved correctly in exec_params
        self.assertEqual(step.exec_params['value'], 2.0)
        self.assertEqual(gain.exec_params['gain'], 5.0)

        # Run simulation step (manually or via run loop)
        # We can just check execution_init for now as that's where resolution happens
        
        # But let's verify that execute uses the resolved value
        # We need to simulate the data flow for one step
        
        # Step execution
        step_out = step.block_instance.execute(time=0.0, inputs={}, params=step.exec_params)
        self.assertEqual(step_out[0], 2.0) # Step starts at 0? Wait, Step default is step_time=1.0. 
        # Let's check Step block logic. Usually step_time=1.0.
        # If time < step_time, output is initial_value (0).
        # Let's set step_time to 0.0 in params.
        step.params['step_time'] = 0.0
        # Re-init to update exec_params
        self.sim.execution_init()
        
        step_out = step.block_instance.execute(time=0.1, inputs={}, params=step.exec_params)
        self.assertEqual(step_out[0], 2.0)

        # Gain execution
        gain_in = {0: step_out[0]}
        gain_out = gain.block_instance.execute(time=0.1, inputs=gain_in, params=gain.exec_params)
        self.assertEqual(gain_out[0], 10.0) # 2.0 * 5.0

if __name__ == '__main__':
    unittest.main()
