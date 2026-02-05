"""
2D Heat Equation Block using Method of Lines (MOL)

Solves: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²) + q(x,y,t)

Where:
- T(x,y,t) is the temperature field
- α is thermal diffusivity
- q(x,y,t) is heat source term
- ∇²T = ∂²T/∂x² + ∂²T/∂y² (Laplacian)

The domain [0,Lx] × [0,Ly] is discretized into Nx × Ny nodes.
This converts the PDE into Nx*Ny coupled ODEs.

State indexing: T[i,j] -> state[k] where k = i + j*Nx
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock
from blocks.param_templates import (
    diffusivity_param, domain_params_2d, init_flag_param, pde_2d_init_temp_param
)
from lib.engine.pde_helpers import bc_params_2d

logger = logging.getLogger(__name__)


class HeatEquation2DBlock(BaseBlock):
    """
    2D Heat Equation solver using Method of Lines.

    Converts the 2D heat equation PDE into a system of ODEs by discretizing space.
    Each spatial node (i,j) becomes a state variable.

    Boundary conditions (for each edge):
    - Dirichlet: T(boundary) = value
    - Neumann: ∂T/∂n(boundary) = value (normal derivative)
    """

    @property
    def block_name(self):
        return "HeatEquation2D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "2D Heat Equation: ∂T/∂t = α∇²T + q"
            "\n\nSolves the 2D heat equation using Method of Lines."
            "\nDomain is discretized into Nx × Ny nodes."
            "\n\nParameters:"
            "\n- alpha: Thermal diffusivity [m²/s]"
            "\n- Lx, Ly: Domain dimensions [m]"
            "\n- Nx, Ny: Number of nodes in x and y"
            "\n- bc_type_*: Boundary conditions (Dirichlet/Neumann)"
            "\n- init_temp: Initial temperature"
            "\n\nInputs:"
            "\n- q_src: Heat source (scalar or Nx×Ny array)"
            "\n- bc_left, bc_right, bc_bottom, bc_top: BC values"
            "\n\nOutputs:"
            "\n- T_field: Temperature field (Nx×Ny array)"
            "\n- T_avg: Average temperature"
            "\n- T_max: Maximum temperature"
        )

    @property
    def params(self):
        return {
            **diffusivity_param(default=0.01),
            **domain_params_2d(),
            **bc_params_2d(),
            **pde_2d_init_temp_param(),
            **init_flag_param(),
        }

    @property
    def inputs(self):
        return [
            {"name": "q_src", "type": "array", "doc": "Heat source term"},
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
            {"name": "T_field", "type": "array", "doc": "Temperature field (Nx×Ny)"},
            {"name": "T_avg", "type": "float", "doc": "Average temperature"},
            {"name": "T_max", "type": "float", "doc": "Maximum temperature"},
        ]

    @property
    def optional_outputs(self):
        """Outputs 1 and 2 (T_avg, T_max) are optional."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw 2D heat equation icon - grid with gradient."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()

        # Draw grid pattern
        for i in range(4):
            x = 0.2 + i * 0.2
            path.moveTo(x, 0.2)
            path.lineTo(x, 0.8)
        for j in range(4):
            y = 0.2 + j * 0.2
            path.moveTo(0.2, y)
            path.lineTo(0.8, y)

        # Heat symbol in corner
        path.addEllipse(0.65, 0.15, 0.2, 0.2)

        return path

    def get_initial_state(self, params):
        """Return initial state vector for the 2D field."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        init_temp = params.get('init_temp', '0.0')
        amplitude = float(params.get('init_amplitude', 1.0))

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

        if isinstance(init_temp, str):
            if init_temp.lower() == 'sinusoidal':
                # T = A * sin(πx/Lx) * sin(πy/Ly) - eigenmode of Laplacian
                T0 = amplitude * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
            elif init_temp.lower() == 'gaussian':
                # Gaussian bump at center
                T0 = amplitude * np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
            elif init_temp.lower() == 'hot_spot':
                # Hot spot in corner
                T0 = amplitude * np.exp(-100 * (X**2 + Y**2))
            else:
                # Try to parse as number
                try:
                    T0 = np.full((Ny, Nx), float(init_temp))
                except ValueError:
                    T0 = np.zeros((Ny, Nx))
        else:
            T0 = np.full((Ny, Nx), float(init_temp))

        # State is flattened 2D array in row-major order
        return T0.flatten()

    def get_state_size(self, params):
        """Return the number of state variables."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))
        return Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute temperature field (for non-compiled execution)."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))

        # Get current state
        state = kwargs.get('state', None)
        if state is None:
            # Use get_initial_state to handle string init_temp values
            state = self.get_initial_state(params)

        # Reshape to 2D for output
        T_field = state.reshape((Ny, Nx))
        T_avg = float(np.mean(T_field))
        T_max = float(np.max(T_field))

        return {
            0: T_field,
            1: T_avg,
            2: T_max,
            'E': False
        }

    def compute_derivatives(self, time, state, inputs, params):
        """
        Compute dT/dt for all nodes using 2D finite differences.

        Uses 5-point stencil for Laplacian:
        ∇²T ≈ (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]) / h²
        """
        alpha = float(params.get('alpha', 0.01))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Get boundary conditions
        bc_left = float(inputs.get(1, 0.0)) if inputs.get(1) is not None else 0.0
        bc_right = float(inputs.get(2, 0.0)) if inputs.get(2) is not None else 0.0
        bc_bottom = float(inputs.get(3, 0.0)) if inputs.get(3) is not None else 0.0
        bc_top = float(inputs.get(4, 0.0)) if inputs.get(4) is not None else 0.0

        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')
        bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
        bc_type_top = params.get('bc_type_top', 'Dirichlet')

        # Get heat source
        q_src = inputs.get(0, 0.0)
        if q_src is None:
            q_src = 0.0
        if isinstance(q_src, np.ndarray):
            if q_src.size == 1:
                q_src = float(q_src.flat[0])
            elif q_src.shape == (Ny, Nx):
                pass  # Use as-is
            else:
                q_src = float(q_src.flat[0])

        # Reshape state to 2D
        T = state.reshape((Ny, Nx))
        dT_dt = np.zeros((Ny, Nx))

        # Interior points: 5-point stencil
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                d2Tdx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / (dx * dx)
                d2Tdy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / (dy * dy)

                # Heat source at this point
                if isinstance(q_src, np.ndarray):
                    q = q_src[j, i]
                else:
                    q = q_src

                dT_dt[j, i] = alpha * (d2Tdx2 + d2Tdy2) + q

        # Boundary conditions using penalty method
        penalty = 1000.0

        # Left boundary (i=0)
        if bc_type_left == 'Dirichlet':
            for j in range(Ny):
                dT_dt[j, 0] = penalty * (bc_left - T[j, 0])
        else:  # Neumann
            for j in range(1, Ny - 1):
                # Ghost node approach: T[-1,j] = T[1,j] - 2*dx*flux
                d2Tdx2 = (2*T[j, 1] - 2*T[j, 0] - 2*dx*bc_left) / (dx * dx)
                d2Tdy2 = (T[j+1, 0] - 2*T[j, 0] + T[j-1, 0]) / (dy * dy)
                q = q_src[j, 0] if isinstance(q_src, np.ndarray) else q_src
                dT_dt[j, 0] = alpha * (d2Tdx2 + d2Tdy2) + q

        # Right boundary (i=Nx-1)
        if bc_type_right == 'Dirichlet':
            for j in range(Ny):
                dT_dt[j, Nx-1] = penalty * (bc_right - T[j, Nx-1])
        else:  # Neumann
            for j in range(1, Ny - 1):
                d2Tdx2 = (2*T[j, Nx-2] - 2*T[j, Nx-1] + 2*dx*bc_right) / (dx * dx)
                d2Tdy2 = (T[j+1, Nx-1] - 2*T[j, Nx-1] + T[j-1, Nx-1]) / (dy * dy)
                q = q_src[j, Nx-1] if isinstance(q_src, np.ndarray) else q_src
                dT_dt[j, Nx-1] = alpha * (d2Tdx2 + d2Tdy2) + q

        # Bottom boundary (j=0)
        if bc_type_bottom == 'Dirichlet':
            for i in range(Nx):
                dT_dt[0, i] = penalty * (bc_bottom - T[0, i])
        else:  # Neumann
            for i in range(1, Nx - 1):
                d2Tdx2 = (T[0, i+1] - 2*T[0, i] + T[0, i-1]) / (dx * dx)
                d2Tdy2 = (2*T[1, i] - 2*T[0, i] - 2*dy*bc_bottom) / (dy * dy)
                q = q_src[0, i] if isinstance(q_src, np.ndarray) else q_src
                dT_dt[0, i] = alpha * (d2Tdx2 + d2Tdy2) + q

        # Top boundary (j=Ny-1)
        if bc_type_top == 'Dirichlet':
            for i in range(Nx):
                dT_dt[Ny-1, i] = penalty * (bc_top - T[Ny-1, i])
        else:  # Neumann
            for i in range(1, Nx - 1):
                d2Tdx2 = (T[Ny-1, i+1] - 2*T[Ny-1, i] + T[Ny-1, i-1]) / (dx * dx)
                d2Tdy2 = (2*T[Ny-2, i] - 2*T[Ny-1, i] + 2*dy*bc_top) / (dy * dy)
                q = q_src[Ny-1, i] if isinstance(q_src, np.ndarray) else q_src
                dT_dt[Ny-1, i] = alpha * (d2Tdx2 + d2Tdy2) + q

        return dT_dt.flatten()
