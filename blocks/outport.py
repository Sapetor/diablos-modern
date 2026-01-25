
from lib.simulation.block import DBlock

class Outport(DBlock):
    """
    Blocks: Outport

    Category: Sinks

    Description:
    Represents an output terminal of a Subsystem.
    When placed inside a subsystem, it creates an output port on the parent Subsystem block.
    """
    def __init__(self, block_name="Out1", sid=1, coords=(0,0,40,30), color="red"):
        from PyQt5.QtCore import QRect
        if isinstance(coords, tuple):
             rect = QRect(*coords)
        else:
             rect = coords
             
        super().__init__(
            block_fn="Outport",
            sid=sid,
            coords=rect,
            color=color,
            in_ports=1,
            out_ports=0,
            b_type=3, 
            io_edit=False,
            fn_name="outport",
            username=block_name
        )
        self.block_type = "Outport"
        self.width = 40
        self.height = 30
        
        # 1 input (comes from internal logic), 0 outputs (goes to outside)
        self.ports = {
            'in': [{'pos': (0, self.height/2), 'type': 'input', 'name': '1'}]
        }
        
    def update(self, t, dt):
        # In a flattened simulation, this block acts as a pass-through 
        # or is optimized away entirely.
        pass
