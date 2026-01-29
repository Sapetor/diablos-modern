import sys
import os
import unittest
from PyQt5.QtCore import QRect

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from lib.lib import DSim
    from blocks.subsystem import Subsystem
    from blocks.inport import Inport
    from blocks.outport import Outport
    from lib.simulation.block import DBlock
except Exception as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

# Mock objects
class MockBlock:
    def __init__(self, name):
        self.name = name
        self.block_fn = "TestBlock"
        self.b_type = 0
        self.params = {}
        self.rect = QRect(0, 0, 50, 50)
        self.in_ports = 1
        self.out_ports = 1
        self.sid = 1

class TestSubsystems(unittest.TestCase):
    def setUp(self):
        # DSim initializes its own model and engine
        self.dsim = DSim()
        self.model = self.dsim.model
        self.engine = self.dsim.engine
        
        # Override screensize to avoid display errors if any
        self.dsim.SCREEN_WIDTH = 800
        self.dsim.SCREEN_HEIGHT = 600
        
    def test_navigation(self):
        """Test enter/exit subsystem navigation."""
        subsys = Subsystem()
        subsys.name = "Subsystem1"
        self.dsim.blocks_list.append(subsys)
        
        # Initial state
        self.assertIsNone(self.dsim.current_subsystem)
        self.assertEqual(len(self.dsim.navigation_stack), 0)
        
        # Enter
        self.dsim.enter_subsystem(subsys)
        self.assertEqual(self.dsim.current_subsystem, "Subsystem1")
        self.assertEqual(len(self.dsim.navigation_stack), 1)
        # Check active list is subsystem list
        self.assertIs(self.dsim.blocks_list, subsys.sub_blocks)
        
        # Exit
        self.dsim.exit_subsystem()
        self.assertIsNone(self.dsim.current_subsystem)
        self.assertNotEqual(self.dsim.blocks_list, subsys.sub_blocks)

    def test_flattening_logic(self):
        """Test flattening of a simple hierarchy."""
        # Root: Const -> Subsystem -> Scope
        # Subsys: Inport -> Gain -> Outport
        
        # 1. Create Subsystem
        subsys = Subsystem()
        subsys.name = "Sub1"
        subsys.sid = 1
        
        # 2. Add Inport/Outport/Gain to Subsystem
        inport = Inport("In1")
        inport.name = "In1" # Enforce name
        inport.sid = 1
        outport = Outport("Out1")
        outport.name = "Out1" # Enforce name
        outport.sid = 2
        
        # Mock Gain (Using DBlock directly for simplicity)
        gain = DBlock("Gain", 3, QRect(0,0,50,50), None, 1, 1, 2, 'both', "gain_fn", {'val': 5}, False)
        gain.name = "Gain1"
        
        subsys.sub_blocks.extend([inport, gain, outport])
        
        # Internal Connections
        # In1 -> Gain1
        # Gain1 -> Out1
        from lib.simulation.connection import DLine
        # Need dummy points for DLine init
        l1 = DLine(1, "In1", 0, "Gain1", 0, [(0,0), (10,10)])
        l2 = DLine(2, "Gain1", 0, "Out1", 0, [(20,20), (30,30)])
        subsys.sub_lines.extend([l1, l2])
        
        # 3. Add to Root
        const = DBlock("Constant", 4, QRect(0,0,50,50), None, 0, 1, 0, 'output', "const_fn", {'val':1}, False)
        const.name = "Const1"
        
        scope = DBlock("Scope", 5, QRect(0,0,50,50), None, 1, 0, 2, 'input', "scope_fn", {}, False)
        scope.name = "Scope1"
        
        self.model.blocks_list.extend([const, subsys, scope])
        
        # Root Connections
        # Const1 -> Sub1 (Port 0 - In1)
        # Sub1 (Port 0 - Out1) -> Scope1
        
        # We need to define ports on Subsystem block so flattener logic? 
        # Flattener assumes logic: "Outport" inside corresponds to Subsystem Output.
        # It references `Out{port+1}` name.
        # My Inport definition above is "In1". 
        
        rl1 = DLine(3, "Const1", 0, "Sub1", 0, [(0,0), (10,10)]) # Const -> Sub1:0 (In1)
        rl2 = DLine(4, "Sub1", 0, "Scope1", 0, [(20,20), (30,30)]) # Sub1:0 (Out1) -> Scope1
        
        self.model.line_list.extend([rl1, rl2])
        
        # 4. Initialize Execution (Triggers Flattening)
        # We need to patch resolve_params because we have incomplete blocks
        # Or test Flattener directly.
        
        from lib.engine.flattener import Flattener
        flattener = Flattener()
        
        flat_blocks, flat_lines = flattener.flatten(self.model.blocks_list, self.model.line_list)
        
        # Verification
        # Should have: Const1, Sub1/Gain1, Scope1
        print("Flat Blocks:", [b.name for b in flat_blocks])
        print("Flat Lines:", [(l.srcblock, l.dstblock) for l in flat_lines])
        
        block_names = [b.name for b in flat_blocks]
        self.assertIn("Const1", block_names)
        self.assertIn("Sub1/Gain1", block_names)
        self.assertIn("Scope1", block_names)
        self.assertNotIn("Sub1", block_names) # Subsystem container gone
        self.assertNotIn("Sub1/In1", block_names)
        
        # Check Lines
        # Should have: Const1 -> Sub1/Gain1
        # and Sub1/Gain1 -> Scope1
        
        connections = [(l.srcblock, l.dstblock) for l in flat_lines]
        self.assertIn(("Const1", "Sub1/Gain1"), connections)
        self.assertIn(("Sub1/Gain1", "Scope1"), connections)

if __name__ == '__main__':
    print("STARTING TEST MAIN", flush=True)
    from PyQt5.QtWidgets import QApplication
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    unittest.main(verbosity=2)
