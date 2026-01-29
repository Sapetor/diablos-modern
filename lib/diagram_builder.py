"""
DiagramBuilder - Programmatic diagram generation for DiaBloS.

Allows building simulation diagrams from Python code without the GUI.

Example usage:
    from lib.diagram_builder import DiagramBuilder
    
    builder = DiagramBuilder()
    ref = builder.add_block("Step", x=50, y=300, name="ref", params={"h_final": 10.0})
    err = builder.add_block("Sum", x=160, y=100, name="err", params={"sign": "+-"})
    builder.connect(ref, 0, err, 0)
    builder.save("saves/my_diagram.dat")
"""

import json
import os
from typing import Dict, Any, Optional, List

# Block type definitions for defaults
BLOCK_DEFAULTS = {
    "Step": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    "Sine": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    "Ramp": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    "Constant": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    "Noise": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    "PRBS": {"category": "Sources", "color": "#90EE90", "b_type": 0, "in_ports": 0, "out_ports": 1},
    
    "Sum": {"category": "Math", "color": "#87CEEB", "b_type": 2, "in_ports": 2, "out_ports": 1, "io_edit": "both"},
    "Gain": {"category": "Math", "color": "#87CEEB", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "SgProd": {"category": "Math", "color": "#87CEEB", "b_type": 2, "in_ports": 2, "out_ports": 1},
    "Abs": {"category": "Math", "color": "#87CEEB", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "Exp": {"category": "Math", "color": "#87CEEB", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "Mux": {"category": "Routing", "color": "#87CEEB", "b_type": 2, "in_ports": 2, "out_ports": 1, "io_edit": "in"},
    "Demux": {"category": "Routing", "color": "#87CEEB", "b_type": 2, "in_ports": 1, "out_ports": 2, "io_edit": "out"},
    
    "Integrator": {"category": "Control", "color": "#DDA0DD", "b_type": 1, "in_ports": 1, "out_ports": 1},
    "Derivative": {"category": "Control", "color": "#DDA0DD", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "TranFn": {"category": "Control", "color": "#DDA0DD", "b_type": 1, "in_ports": 1, "out_ports": 1},
    "StateSpace": {"category": "Control", "color": "#DDA0DD", "b_type": 1, "in_ports": 1, "out_ports": 1},
    "PID": {"category": "Control", "color": "#DDA0DD", "b_type": 1, "in_ports": 1, "out_ports": 1},
    "Delay": {"category": "Control", "color": "#DDA0DD", "b_type": 1, "in_ports": 1, "out_ports": 1},
    "RateLimiter": {"category": "Control", "color": "#DDA0DD", "b_type": 2, "in_ports": 1, "out_ports": 1},
    
    "Saturation": {"category": "Nonlinear", "color": "#FFD700", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "Deadband": {"category": "Nonlinear", "color": "#FFD700", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "Hysteresis": {"category": "Nonlinear", "color": "#FFD700", "b_type": 2, "in_ports": 1, "out_ports": 1},
    "Switch": {"category": "Nonlinear", "color": "#FFD700", "b_type": 2, "in_ports": 3, "out_ports": 1},
    
    "Scope": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 1, "out_ports": 0, "io_edit": "in"},
    "Display": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 1, "out_ports": 0},
    "Export": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 1, "out_ports": 0},
    "XYGraph": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 2, "out_ports": 0},
    "FFT": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 1, "out_ports": 0},
    "Term": {"category": "Sinks", "color": "#FFB6C1", "b_type": 3, "in_ports": 1, "out_ports": 0},
}


class DiagramBuilder:
    """
    Builds DiaBloS diagrams programmatically.
    
    Usage:
        builder = DiagramBuilder()
        ref = builder.add_block("Step", x=50, y=300, name="ref")
        sum1 = builder.add_block("Sum", x=150, y=300, name="sum1")
        builder.connect(ref, 0, sum1, 0)
        builder.save("diagram.dat")
    """
    
    def __init__(self, sim_time: float = 1.0, sim_dt: float = 0.01):
        """
        Initialize a new diagram builder.
        
        Args:
            sim_time: Total simulation time in seconds
            sim_dt: Simulation time step in seconds
        """
        self.sim_time = sim_time
        self.sim_dt = sim_dt
        self.blocks: List[Dict[str, Any]] = []
        self.lines: List[Dict[str, Any]] = []
        self._block_counter = 0
        self._line_counter = 0
        self._name_to_idx: Dict[str, int] = {}
    
    def add_block(self, block_type: str, x: int, y: int, 
                  name: Optional[str] = None,
                  params: Optional[Dict[str, Any]] = None,
                  width: int = 50, height: int = 40,
                  in_ports: Optional[int] = None,
                  out_ports: Optional[int] = None) -> str:
        """
        Add a block to the diagram.
        
        Args:
            block_type: Type of block (e.g., "Step", "Sum", "Gain", "Scope")
            x: X position in pixels
            y: Y position in pixels  
            name: Optional username for the block (used for connections)
            params: Block-specific parameters (e.g., {"gain": 2.0})
            width: Block width in pixels (default 50)
            height: Block height in pixels (default 40)
            in_ports: Number of input ports (auto-detected from block type if not specified)
            out_ports: Number of output ports (auto-detected if not specified)
            
        Returns:
            Block name/ID for use in connections
        """
        defaults = BLOCK_DEFAULTS.get(block_type, {
            "category": "Other", "color": "#808080", "b_type": 2,
            "in_ports": 1, "out_ports": 1
        })
        
        sid = self._block_counter
        self._block_counter += 1
        
        # Generate name if not provided
        if name is None:
            name = f"{block_type.lower()}{sid}"
        
        # Store mapping
        self._name_to_idx[name] = len(self.blocks)
        
        # Determine port counts
        n_in = in_ports if in_ports is not None else defaults.get("in_ports", 1)
        n_out = out_ports if out_ports is not None else defaults.get("out_ports", 1)
        
        # Get fn_name (lowercase block type)
        fn_name = block_type.lower()
        
        block = {
            "block_fn": block_type,
            "sid": sid,
            "username": name,
            "coords_left": x,
            "coords_top": y,
            "coords_width": width,
            "coords_height": height,
            "coords_height_base": height,
            "in_ports": n_in,
            "out_ports": n_out,
            "dragging": False,
            "selected": False,
            "b_color": defaults.get("color", "#808080"),
            "b_type": defaults.get("b_type", 2),
            "io_edit": defaults.get("io_edit", "none"),
            "fn_name": fn_name,
            "params": params or {},
            "external": False,
            "flipped": False
        }
        
        self.blocks.append(block)
        return name
    
    def connect(self, src_block: str, src_port: int, 
                dst_block: str, dst_port: int) -> str:
        """
        Create a connection between two blocks.
        
        Args:
            src_block: Source block name/ID (returned from add_block)
            src_port: Output port number on source block (0-indexed)
            dst_block: Destination block name/ID
            dst_port: Input port number on destination block (0-indexed)
            
        Returns:
            Line name/ID
        """
        sid = self._line_counter
        self._line_counter += 1
        
        # Get block positions for line endpoints
        src_idx = self._name_to_idx.get(src_block)
        dst_idx = self._name_to_idx.get(dst_block)
        
        # Calculate approximate line points
        if src_idx is not None and dst_idx is not None:
            src_b = self.blocks[src_idx]
            dst_b = self.blocks[dst_idx]
            
            # Source: right edge of block
            src_x = src_b["coords_left"] + src_b["coords_width"]
            src_y = src_b["coords_top"] + src_b["coords_height"] // 2
            
            # Destination: left edge of block, adjust for port
            dst_x = dst_b["coords_left"]
            port_spacing = dst_b["coords_height"] // (dst_b["in_ports"] + 1)
            dst_y = dst_b["coords_top"] + port_spacing * (dst_port + 1)
            
            points = [[src_x, src_y], [dst_x, dst_y]]
        else:
            points = [[0, 0], [100, 100]]
        
        line = {
            "name": f"line{sid}",
            "sid": sid,
            "srcblock": src_block,
            "srcport": src_port,
            "dstblock": dst_block,
            "dstport": dst_port,
            "points": points,
            "cptr": 0,
            "selected": False
        }
        
        self.lines.append(line)
        return line["name"]
    
    def save(self, filepath: str) -> None:
        """
        Save the diagram to a .dat file.
        
        Args:
            filepath: Path to save the file (should end with .dat)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        data = {
            "sim_data": {
                "wind_width": 1280,
                "wind_height": 770,
                "fps": 60,
                "sim_time": self.sim_time,
                "sim_dt": self.sim_dt,
                "sim_trange": 100
            },
            "blocks_data": self.blocks,
            "lines_data": self.lines,
            "version": "2.0"
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Diagram saved to {filepath}")
        print(f"  - {len(self.blocks)} blocks")
        print(f"  - {len(self.lines)} connections")
    
    def get_block(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a block by name."""
        idx = self._name_to_idx.get(name)
        return self.blocks[idx] if idx is not None else None


def create_platoon_diagram(n_vehicles: int = 5, save_path: str = "saves/platoon.dat") -> DiagramBuilder:
    """
    Create a cyclic platoon diagram with N vehicles.
    
    Args:
        n_vehicles: Number of vehicles in the platoon
        save_path: Where to save the diagram
        
    Returns:
        DiagramBuilder instance
    """
    builder = DiagramBuilder(sim_time=5.0, sim_dt=0.01)
    
    # Add reference step input
    ref = builder.add_block("Step", x=50, y=300, name="ref", 
                            params={"start_time": 0.0, "h_start": 0.0, "h_final": 10.0})
    
    # Create vehicle chains
    vehicles = []
    for i in range(n_vehicles):
        y = 100 + i * 80
        
        # For vehicle 1, we need 3 inputs: ref (port 0), pos5 cyclic (port 1), pos1 feedback (port 2)
        # For other vehicles: pos[i-1] (port 0), pos[i] feedback (port 1)
        if i == 0:
            # err1 needs 3 inputs for cyclic: ref + pos5 - pos1
            err = builder.add_block("Sum", x=160, y=y, name=f"err{i+1}", 
                                    params={"sign": "++-"}, in_ports=3)
        else:
            err = builder.add_block("Sum", x=160, y=y, name=f"err{i+1}", 
                                    params={"sign": "+-"}, in_ports=2)
        
        K = builder.add_block("Gain", x=230, y=y, name=f"K{i+1}", 
                              params={"gain": 2.0})
        vel = builder.add_block("Integrator", x=310, y=y, name=f"vel{i+1}",
                                params={"init_conds": 0.0, "method": "FWD_EULER"})
        pos = builder.add_block("Integrator", x=390, y=y, name=f"pos{i+1}",
                                params={"init_conds": 0.0, "method": "FWD_EULER"})
        
        # Internal chain: err -> K -> vel -> pos
        builder.connect(err, 0, K, 0)
        builder.connect(K, 0, vel, 0)
        builder.connect(vel, 0, pos, 0)
        
        # Feedback: pos -> err (last port, negative)
        if i == 0:
            builder.connect(pos, 0, err, 2)  # pos1 to err1 port 2 (negative)
        else:
            builder.connect(pos, 0, err, 1)  # pos[i] to err[i] port 1 (negative)
        
        vehicles.append((err, K, vel, pos))
    
    # Connect reference to first vehicle (port 0)
    builder.connect(ref, 0, "err1", 0)
    
    # Forward coupling: pos[i] -> err[i+1] (port 0)
    for i in range(n_vehicles - 1):
        builder.connect(f"pos{i+1}", 0, f"err{i+2}", 0)
    
    # Cyclic connection: pos[N] -> err1 (port 1, positive for cyclic)
    builder.connect(f"pos{n_vehicles}", 0, "err1", 1)
    
    # Add scope to display all positions
    scope = builder.add_block("Scope", x=520, y=250, name="scope",
                              params={"title": "Vehicle Positions"},
                              in_ports=n_vehicles, height=100)
    
    for i in range(n_vehicles):
        builder.connect(f"pos{i+1}", 0, "scope", i)
    
    builder.save(save_path)
    return builder


# Example usage when run directly
if __name__ == "__main__":
    # Create a simple test diagram
    builder = DiagramBuilder(sim_time=2.0)
    
    step = builder.add_block("Step", x=50, y=200, name="step")
    gain = builder.add_block("Gain", x=150, y=200, name="gain", params={"gain": 2.0})
    scope = builder.add_block("Scope", x=250, y=200, name="scope")
    
    builder.connect(step, 0, gain, 0)
    builder.connect(gain, 0, scope, 0)
    
    builder.save("saves/test_builder.dat")
    
    # Create platoon
    print("\nCreating platoon diagram...")
    create_platoon_diagram(5, "saves/platoon_builder.dat")
