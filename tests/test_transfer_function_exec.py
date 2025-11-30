
import unittest
import os
import sys
import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.workspace import WorkspaceManager
from blocks.transfer_function import TransferFunctionBlock

class TestTransferFunctionExec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        WorkspaceManager._instance = None
        self.sim = DSim()
        self.sim.sim_dt = 0.01
        self.sim.sim_time = 1.0

    def test_transfer_function_execution(self):
        # Create Transfer Function block
        tf_block = DBlock("TranFn", 1, QRect(0, 0, 100, 100), "blue", block_class=TransferFunctionBlock)
        tf_block.params = {
            "numerator": [1.0],
            "denominator": [1.0, 1.0],
            "init_conds": [0.0],
            "_name_": "tf1",
            "_inputs_": 1,
            "_outputs_": 1
        }
        
        self.sim.blocks_list = [tf_block]
        self.sim.line_list = [] # No connections needed for this unit test of execution
        
        # Initialize execution - this should inject 'dtime' into exec_params
        success = self.sim.execution_init()
        self.assertTrue(success, "Execution initialization failed")
        
        # Verify dtime is in exec_params
        self.assertIn('dtime', tf_block.exec_params)
        self.assertEqual(tf_block.exec_params['dtime'], 0.01)
        
        # Simulate execution loop call
        # We need to manually trigger what execution_loop does
        # It calls execute() with exec_params
        
        # Mock input
        tf_block.input_queue = {0: 1.0}
        
        # Execute using the block instance directly, but ensuring we use exec_params
        # This mirrors the fix we made in lib.py
        try:
            out = tf_block.block_instance.execute(time=0.0, inputs=tf_block.input_queue, params=tf_block.exec_params)
            print(f"Execution output: {out}")
        except KeyError as e:
            self.fail(f"KeyError during execution: {e}")
        except Exception as e:
            self.fail(f"Exception during execution: {e}")
            
        self.assertFalse(out.get('E', False), f"Block reported error: {out.get('error')}")

if __name__ == '__main__':
    unittest.main()
