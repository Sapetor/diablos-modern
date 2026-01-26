
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

    def update_Block(self):
        """
        Update subsystem geometry.
        Overrides DBlock.update_Block to assume port positions from self.ports logic
        if available, matching the internal Inport/Outport locations.
        """
        self.in_coords = []
        self.out_coords = []
        
        # Base geometry
        from PyQt5.QtCore import QPoint, QRect
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)
        
        # 1. Input Ports
        if hasattr(self, 'ports') and 'in' in self.ports and self.ports['in']:
            # self.ports['in'] is a list of dicts: {'pos': (x,y), 'type': 'input', 'name': '1'}
            # The 'pos' is relative to the block's top-left (or maybe just Y relative?).
            # In create_subsystem_from_selection, we set pos = (0, relative_y).
            
            for p in self.ports['in']:
                # pos is usually a tuple (x, y) relative to subsystem origin
                # but might need adjustment if logic changed.
                rel_x, rel_y = p['pos']
                
                # Absolute coordinate on canvas
                # Inputs are on the left usually
                abs_x = self.left + rel_x
                abs_y = self.top + rel_y
                
                # If flipped? Subsystems might not flip well yet.
                self.in_coords.append(QPoint(int(abs_x), int(abs_y)))
                
            # Sync count
            self.in_ports = len(self.in_coords)
            
        else:
            # Fallback to default DBlock behavior if no ports map
            super().update_Block()
            
        # 2. Output Ports
        if hasattr(self, 'ports') and 'out' in self.ports and self.ports['out']:
            for p in self.ports['out']:
                rel_x, rel_y = p['pos']
                
                # Absolute coordinate
                abs_x = self.left + rel_x
                abs_y = self.top + rel_y
                
                self.out_coords.append(QPoint(int(abs_x), int(abs_y)))
                
            self.out_ports = len(self.out_coords)
        else:
            # If we didn't have special output ports but used default for input, 
            # we need to be careful not to double-call super().
            # Actually, unconnected outputs should just be 0 if 'out' is empty in ports dict
            if not (hasattr(self, 'ports') and 'in' in self.ports and self.ports['in']):
                 # If we fell back for inputs, we likely fell back for outputs too via super()
                 pass
