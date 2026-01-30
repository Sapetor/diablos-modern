
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
        Overrides DBlock.update_Block to recalculate port positions based on
        current block dimensions, ensuring ports scale properly when resized.
        """
        from PyQt5.QtCore import QPoint, QRect

        self.in_coords = []
        self.out_coords = []

        # Update geometry rectangles
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)
        self.rect = QRect(self.left, self.top, self.width, self.height)

        # Check if we have custom port definitions
        has_input_ports = hasattr(self, 'ports') and 'in' in self.ports and self.ports['in']
        has_output_ports = hasattr(self, 'ports') and 'out' in self.ports and self.ports['out']

        if not has_input_ports and not has_output_ports:
            # No custom ports - use default DBlock behavior
            super().update_Block()
            return

        # Calculate input port positions - evenly distributed along current height
        if has_input_ports:
            num_inputs = len(self.ports['in'])
            self.in_ports = num_inputs

            in_x = self.left if not self.flipped else self.left + self.width

            for i in range(num_inputs):
                # Evenly space ports along height
                port_y = self.top + self.height * (i + 1) / (num_inputs + 1)
                self.in_coords.append(QPoint(int(in_x), int(port_y)))

                # Update stored position for consistency
                self.ports['in'][i]['pos'] = (0 if not self.flipped else self.width,
                                               self.height * (i + 1) / (num_inputs + 1))

        # Calculate output port positions - evenly distributed along current height
        if has_output_ports:
            num_outputs = len(self.ports['out'])
            self.out_ports = num_outputs

            out_x = self.left + self.width if not self.flipped else self.left

            for i in range(num_outputs):
                # Evenly space ports along height
                port_y = self.top + self.height * (i + 1) / (num_outputs + 1)
                self.out_coords.append(QPoint(int(out_x), int(port_y)))

                # Update stored position for consistency
                self.ports['out'][i]['pos'] = (self.width if not self.flipped else 0,
                                                self.height * (i + 1) / (num_outputs + 1))
