import sys
import os
import unittest
from PyQt5.QtCore import QPoint, QRect, QCoreApplication
from PyQt5.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lib.lib import DSim
    from blocks.subsystem import Subsystem
    from blocks.inport import Inport
    from blocks.outport import Outport
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

class TestMIMOSubsystem(unittest.TestCase):
    def setUp(self):
        # Setup headless environment
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        self.dsim = DSim()
        # Ensure we have a clean state
        self.dsim.clear_all()
        # Initialize screen dims to avoid errors in updates if any
        self.dsim.SCREEN_WIDTH = 1000
        self.dsim.SCREEN_HEIGHT = 800

    def test_manual_port_expansion(self):
        """
        Test that adding internal Inport/Outport blocks updates
        the parent Subsystem ports upon exit.
        """
        print("\n--- Test 1: Manual Port Expansion ---")
        
        # 1. Create a Subsystem
        subsys = Subsystem(block_name="MIMO_Sub")
        subsys.sid = 1
        subsys.name = "MIMO_Sub"
        self.dsim.blocks_list.append(subsys)
        
        print(f"Created Subsystem: {subsys.name}")
        print(f"Initial Ports: In={subsys.in_ports}, Out={subsys.out_ports}")
        
        # 2. Enter Subsystem
        self.dsim.enter_subsystem(subsys)
        print("Entered Subsystem")
        
        # 3. Add Multiple Ports
        # 2 Inputs
        in1 = Inport("In1")
        in1.top = 100 # Position determines order
        in2 = Inport("In2")
        in2.top = 200
        
        # 3 Outputs
        out1 = Outport("Out1")
        out1.top = 50
        out2 = Outport("Out2")
        out2.top = 150
        out3 = Outport("Out3")
        out3.top = 250
        
        self.dsim.blocks_list.extend([in1, in2, out1, out2, out3])
        print(f"Added {2} Inports and {3} Outports internally")
        
        # 4. Exit Subsystem (Should trigger sync)
        self.dsim.exit_subsystem()
        print("Exited Subsystem")
        
        # 5. Verify Parent Block Ports
        parent_block = self.dsim.blocks_list[0]
        self.assertEqual(parent_block.name, "MIMO_Sub")
        
        # Check counts
        # Subsystem block updates its in_ports/out_ports based on 'ports' dict in update_Block
        # But DBlock.update_Block only runs if we call it. exit_subsystem calls it.
        
        print(f"Updated Ports: In={parent_block.in_ports}, Out={parent_block.out_ports}")
        
        self.assertEqual(parent_block.in_ports, 2, "Should have 2 input ports")
        self.assertEqual(parent_block.out_ports, 3, "Should have 3 output ports")
        
        # Check coordinates exist
        self.assertEqual(len(parent_block.in_coords), 2)
        self.assertEqual(len(parent_block.out_coords), 3)
        
        print("Port Coordinates verified.")
        
        # Verify ordering (In1 at top 100 vs In2 at top 200)
        # Ports are sorted by top. So name 'In1' (first created) -> index 0?
        # Actually logic sorts by b.top.
        # In1.top=100 -> First. In2.top=200 -> Second.
        
        # Parent ports names are just index 1, 2, ...
        # Check port dictionary structure if needed, but counts suffice for basic MIMO validation.
        ports_in = parent_block.ports['in']
        self.assertEqual(len(ports_in), 2)
        # Verify positions are distributed
        y1 = ports_in[0]['pos'][1]
        y2 = ports_in[1]['pos'][1]
        self.assertTrue(y2 > y1, "Ports should be distributed vertically")
        
        print("Test 1 Passed: MIMO Subsystem Sync Successful")

if __name__ == '__main__':
    unittest.main()
