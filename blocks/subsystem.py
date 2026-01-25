
from lib.simulation.block import DBlock

class Subsystem(DBlock):
    """
    Blocks: Subsystem

    Category: Routing

    Description:
    A container block that can hold other blocks and connections.
    Used to simplify complex diagrams by grouping related functionality.
    Double-click to enter the subsystem and edit its contents.
    """
    def __init__(self, block_name="Subsystem", sid=1, coords=(0,0,100,80), color="lightgray"):
        from PyQt5.QtCore import QRect
        if isinstance(coords, tuple):
             rect = QRect(*coords)
        else:
             rect = coords
             
        super().__init__(
            block_fn="Subsystem",
            sid=sid,
            coords=rect,
            color=color,
            in_ports=0,
            out_ports=0,
            b_type=2,
            io_edit=True,
            fn_name="subsystem",
            username=block_name
        )
        self.block_type = "Subsystem"
        
        # Internal structure
        self.sub_blocks = []
        self.sub_lines = []
        
        # Port Mapping: { primitive_port_idx: (Inport/Outport_name, direction) }
        # This will be used to sync external ports with internal Inport/Outport blocks
        self.ports_map = {}
        
        # Subsystems typically don't have parameters to set directly,
        # but could expose simplified parameters from children in the future.
        self.params = {}

    def update(self, t, dt):
        # Subsystems are containers and generally not executed directly 
        # in the standard loop if flattening is used.
        # If executed hierarchically, this would delegate to sub-blocks.
        pass
