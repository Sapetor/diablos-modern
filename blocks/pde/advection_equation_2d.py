"""
2D Advection-Diffusion Equation Block using Method of Lines (MOL)

Solves: ∂c/∂t = -vx(∂c/∂x) - vy(∂c/∂y) + D∇²c + S(x,y,t)

Where:
- c(x,y,t) is the concentration field
- vx, vy are velocity components
- D is diffusion coefficient (0 for pure advection)
- S(x,y,t) is source term
- ∇²c = ∂²c/∂x² + ∂²c/∂y² (Laplacian)

The domain [0,Lx] × [0,Ly] is discretized into Nx × Ny nodes.
Uses upwind scheme for advection terms (stability) and central differences
for diffusion (accuracy).

State indexing: c[i,j] -> state[k] where k = i + j*Nx (row-major)
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class AdvectionEquation2DBlock(BaseBlock):
    """
    2D Advection-Diffusion Equation solver using Method of Lines.

    Converts the advection-diffusion PDE into a system of ODEs by discretizing space.
    Uses upwind finite differences for advection (stability) and central differences
    for diffusion (accuracy).

    Boundary conditions (for each edge):
    - Dirichlet: c(boundary) = value
    - Neumann: ∂c/∂n(boundary) = value
    - Outflow: ∂c/∂n = 0 (zero gradient)
    """

    @property
    def block_name(self):
        return "AdvectionEquation2D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "cyan"

    @property
    def doc(self):
        return (
            "2D Advection-Diffusion: ∂c/∂t = -v·∇c + D∇²c + S"
            "\n\nSolves the 2D advection-diffusion equation using Method of Lines."
            "\nUses upwind scheme for advection stability."
            "\n\nParameters:"
            "\n- vx, vy: Velocity components [m/s]"
            "\n- D: Diffusion coefficient [m²/s] (0 = pure advection)"
            "\n- Lx, Ly: Domain dimensions [m]"
            "\n- Nx, Ny: Number of nodes in x and y"
            "\n- bc_type_*: Boundary conditions (Dirichlet/Neumann/Outflow)"
            "\n- init_concentration: Initial concentration"
            "\n\nInputs:"
            "\n- source: Source term (scalar or Nx×Ny array)"
            "\n- bc_left, bc_right, bc_bottom, bc_top: BC values"
            "\n\nOutputs:"
            "\n- c_field: Concentration field (Nx×Ny array)"
            "\n- c_avg: Average concentration"
            "\n- c_max: Maximum concentration"
        )

    @property
    def params(self):
        return {
            "vx": {
                "type": "float",
                "default": 1.0,
                "doc": "X-velocity [m/s]"
            },
            "vy": {
                "type": "float",
                "default": 0.0,
                "doc": "Y-velocity [m/s]"
            },
            "D": {
                "type": "float",
                "default": 0.0,
                "doc": "Diffusion coefficient [m²/s]"
            },
            "Lx": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in x [m]"
            },
            "Ly": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in y [m]"
            },
            "Nx": {
                "type": "int",
                "default": 30,
                "doc": "Number of nodes in x direction"
            },
            "Ny": {
                "type": "int",
                "default": 30,
                "doc": "Number of nodes in y direction"
            },
            "bc_type_left": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Left BC: Dirichlet, Neumann, or Outflow"
            },
            "bc_type_right": {
                "type": "string",
                "default": "Outflow",
                "doc": "Right BC: Dirichlet, Neumann, or Outflow"
            },
            "bc_type_bottom": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Bottom BC: Dirichlet, Neumann, or Outflow"
            },
            "bc_type_top": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Top BC: Dirichlet, Neumann, or Outflow"
            },
            "init_concentration": {
                "type": "string",
                "default": "0.0",
                "doc": "Initial concentration: number, 'gaussian', 'step', or 'pulse'"
            },
            "init_amplitude": {
                "type": "float",
                "default": 1.0,
                "doc": "Amplitude for non-uniform initial conditions"
            },
            "_init_start_": {
                "type": "bool",
                "default": True,
                "doc": "Internal: initialization flag"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "source", "type": "array", "doc": "Source term"},
            {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
            {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
            {"name": "bc_bottom", "type": "float", "doc": "Bottom boundary value"},
            {"name": "bc_top", "type": "float", "doc": "Top boundary value"},
        ]

    @property
    def optional_inputs(self):
        """All inputs are optional - default to 0."""
        return [0, 1, 2, 3, 4]

    @property
    def outputs(self):
        return [
            {"name": "c_field", "type": "array", "doc": "Concentration field (Nx×Ny)"},
            {"name": "c_avg", "type": "float", "doc": "Average concentration"},
            {"name": "c_max", "type": "float", "doc": "Maximum concentration"},
        ]

    @property
    def optional_outputs(self):
        """Outputs 1 and 2 (c_avg, c_max) are optional."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw 2D advection icon - arrows showing flow."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()

        # Draw flow arrows
        for row in [0.3, 0.5, 0.7]:
            path.moveTo(0.15, row)
            path.lineTo(0.75, row)
            # Arrow head
            path.moveTo(0.75, row)
            path.lineTo(0.65, row - 0.08)
            path.moveTo(0.75, row)
            path.lineTo(0.65, row + 0.08)

        return path

    def get_initial_state(self, params):
        """Return initial state vector for the 2D concentration field."""
        Nx = int(params.get('Nx', 30))
        Ny = int(params.get('Ny', 30))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        init_conc = params.get('init_concentration', '0.0')
        amplitude = float(params.get('init_amplitude', 1.0))

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

        if isinstance(init_conc, str):
            if init_conc.lower() == 'gaussian':
                # Gaussian pulse at center
                sigma = min(Lx, Ly) / 10
                c0 = amplitude * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / (2*sigma**2))
            elif init_conc.lower() == 'step':
                # Step function: 1 on left quarter, 0 elsewhere
                c0 = amplitude * (X < Lx/4).astype(float)
            elif init_conc.lower() == 'pulse':
                # Localized pulse in corner
                c0 = amplitude * np.exp(-50 * (X**2 + Y**2))
            else:
                # Try to parse as number
                try:
                    c0 = np.full((Ny, Nx), float(init_conc))
                except ValueError:
                    c0 = np.zeros((Ny, Nx))
        else:
            c0 = np.full((Ny, Nx), float(init_conc))

        # State is flattened 2D array in row-major order
        return c0.flatten()

    def get_state_size(self, params):
        """Return the number of state variables."""
        Nx = int(params.get('Nx', 30))
        Ny = int(params.get('Ny', 30))
        return Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute concentration field (for non-compiled execution)."""
        Nx = int(params.get('Nx', 30))
        Ny = int(params.get('Ny', 30))

        # Get current state
        state = kwargs.get('state', None)
        if state is None:
            state = self.get_initial_state(params)

        # Reshape to 2D for output
        c_field = state.reshape((Ny, Nx))
        c_avg = float(np.mean(c_field))
        c_max = float(np.max(c_field))

        return {
            0: c_field,
            1: c_avg,
            2: c_max,
            'E': False
        }

    def compute_derivatives(self, time, state, inputs, params):
        """
        Compute dc/dt for all nodes using upwind scheme for advection
        and central differences for diffusion.
        """
        vx = float(params.get('vx', 1.0))
        vy = float(params.get('vy', 0.0))
        D = float(params.get('D', 0.0))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        Nx = int(params.get('Nx', 30))
        Ny = int(params.get('Ny', 30))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Get boundary conditions
        bc_left = float(inputs.get(1, 0.0)) if inputs.get(1) is not None else 0.0
        bc_right = float(inputs.get(2, 0.0)) if inputs.get(2) is not None else 0.0
        bc_bottom = float(inputs.get(3, 0.0)) if inputs.get(3) is not None else 0.0
        bc_top = float(inputs.get(4, 0.0)) if inputs.get(4) is not None else 0.0

        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Outflow')
        bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
        bc_type_top = params.get('bc_type_top', 'Dirichlet')

        # Get source term
        source = inputs.get(0, 0.0)
        if source is None:
            source = 0.0
        if isinstance(source, np.ndarray):
            if source.size == 1:
                source = float(source.flat[0])
            elif source.shape == (Ny, Nx):
                pass  # Use as-is
            else:
                source = float(source.flat[0])

        # Reshape state to 2D
        c = state.reshape((Ny, Nx))
        dc_dt = np.zeros((Ny, Nx))

        # Interior points: upwind for advection, central for diffusion
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                # Upwind scheme for advection
                if vx >= 0:
                    dc_dx = (c[j, i] - c[j, i-1]) / dx  # backward
                else:
                    dc_dx = (c[j, i+1] - c[j, i]) / dx  # forward

                if vy >= 0:
                    dc_dy = (c[j, i] - c[j-1, i]) / dy  # backward
                else:
                    dc_dy = (c[j+1, i] - c[j, i]) / dy  # forward

                # Central differences for diffusion (Laplacian)
                d2c_dx2 = (c[j, i+1] - 2*c[j, i] + c[j, i-1]) / (dx * dx)
                d2c_dy2 = (c[j+1, i] - 2*c[j, i] + c[j-1, i]) / (dy * dy)

                # Source at this point
                if isinstance(source, np.ndarray):
                    S = source[j, i]
                else:
                    S = source

                # dc/dt = -vx*dc/dx - vy*dc/dy + D*laplacian + source
                dc_dt[j, i] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S

        # Boundary conditions using penalty method for Dirichlet
        penalty = 1000.0

        # Left boundary (i=0)
        if bc_type_left == 'Dirichlet':
            for j in range(Ny):
                dc_dt[j, 0] = penalty * (bc_left - c[j, 0])
        elif bc_type_left == 'Neumann':
            for j in range(1, Ny - 1):
                # Use one-sided difference
                if vx >= 0:
                    dc_dx = bc_left  # prescribed gradient
                else:
                    dc_dx = (c[j, 1] - c[j, 0]) / dx
                if vy >= 0:
                    dc_dy = (c[j, 0] - c[j-1, 0]) / dy
                else:
                    dc_dy = (c[j+1, 0] - c[j, 0]) / dy
                d2c_dx2 = (c[j, 1] - c[j, 0]) / (dx * dx) * 2  # one-sided
                d2c_dy2 = (c[j+1, 0] - 2*c[j, 0] + c[j-1, 0]) / (dy * dy)
                S = source[j, 0] if isinstance(source, np.ndarray) else source
                dc_dt[j, 0] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for j in range(Ny):
                dc_dt[j, 0] = dc_dt[j, 1] if Nx > 1 else 0.0

        # Right boundary (i=Nx-1)
        if bc_type_right == 'Dirichlet':
            for j in range(Ny):
                dc_dt[j, Nx-1] = penalty * (bc_right - c[j, Nx-1])
        elif bc_type_right == 'Neumann':
            for j in range(1, Ny - 1):
                if vx >= 0:
                    dc_dx = (c[j, Nx-1] - c[j, Nx-2]) / dx
                else:
                    dc_dx = bc_right  # prescribed gradient
                if vy >= 0:
                    dc_dy = (c[j, Nx-1] - c[j-1, Nx-1]) / dy
                else:
                    dc_dy = (c[j+1, Nx-1] - c[j, Nx-1]) / dy
                d2c_dx2 = (c[j, Nx-2] - c[j, Nx-1]) / (dx * dx) * 2
                d2c_dy2 = (c[j+1, Nx-1] - 2*c[j, Nx-1] + c[j-1, Nx-1]) / (dy * dy)
                S = source[j, Nx-1] if isinstance(source, np.ndarray) else source
                dc_dt[j, Nx-1] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for j in range(Ny):
                dc_dt[j, Nx-1] = dc_dt[j, Nx-2] if Nx > 1 else 0.0

        # Bottom boundary (j=0)
        if bc_type_bottom == 'Dirichlet':
            for i in range(Nx):
                dc_dt[0, i] = penalty * (bc_bottom - c[0, i])
        elif bc_type_bottom == 'Neumann':
            for i in range(1, Nx - 1):
                if vx >= 0:
                    dc_dx = (c[0, i] - c[0, i-1]) / dx
                else:
                    dc_dx = (c[0, i+1] - c[0, i]) / dx
                if vy >= 0:
                    dc_dy = bc_bottom
                else:
                    dc_dy = (c[1, i] - c[0, i]) / dy
                d2c_dx2 = (c[0, i+1] - 2*c[0, i] + c[0, i-1]) / (dx * dx)
                d2c_dy2 = (c[1, i] - c[0, i]) / (dy * dy) * 2
                S = source[0, i] if isinstance(source, np.ndarray) else source
                dc_dt[0, i] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for i in range(Nx):
                dc_dt[0, i] = dc_dt[1, i] if Ny > 1 else 0.0

        # Top boundary (j=Ny-1)
        if bc_type_top == 'Dirichlet':
            for i in range(Nx):
                dc_dt[Ny-1, i] = penalty * (bc_top - c[Ny-1, i])
        elif bc_type_top == 'Neumann':
            for i in range(1, Nx - 1):
                if vx >= 0:
                    dc_dx = (c[Ny-1, i] - c[Ny-1, i-1]) / dx
                else:
                    dc_dx = (c[Ny-1, i+1] - c[Ny-1, i]) / dx
                if vy >= 0:
                    dc_dy = (c[Ny-1, i] - c[Ny-2, i]) / dy
                else:
                    dc_dy = bc_top
                d2c_dx2 = (c[Ny-1, i+1] - 2*c[Ny-1, i] + c[Ny-1, i-1]) / (dx * dx)
                d2c_dy2 = (c[Ny-2, i] - c[Ny-1, i]) / (dy * dy) * 2
                S = source[Ny-1, i] if isinstance(source, np.ndarray) else source
                dc_dt[Ny-1, i] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for i in range(Nx):
                dc_dt[Ny-1, i] = dc_dt[Ny-2, i] if Ny > 1 else 0.0

        return dc_dt.flatten()
