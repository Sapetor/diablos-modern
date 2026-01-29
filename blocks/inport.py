
from lib.simulation.block import DBlock

class Inport(DBlock):
    """
    Blocks: Inport

    Category: Sources

    Description:
    Represents an input terminal of a Subsystem.
    When placed inside a subsystem, it creates an input port on the parent Subsystem block.
    """
    def __init__(self, block_name="In1", sid=1, coords=(0,0,40,30), color="green"):
        from PyQt5.QtCore import QRect
        if isinstance(coords, tuple):
             rect = QRect(*coords)
        else:
             rect = coords
             
        super().__init__(
            block_fn="Inport",
            sid=sid,
            coords=rect,
            color=color,
            in_ports=0,
            out_ports=1,
            b_type=0, 
            io_edit=False,
            fn_name="inport",
            username=block_name
        )
        self.block_type = "Inport"
        self.width = 40
        self.height = 30
        
        # 0 inputs (comes from outside), 1 output (goes to internal logic)
        self.ports = {
            'out': [{'pos': (self.width, self.height/2), 'type': 'output', 'name': '1'}]
        }
        
    def update(self, t, dt):
        # In a flattened simulation, this block acts as a pass-through 
        # or is optimized away entirely.
        pass
