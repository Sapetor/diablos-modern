"""
2D Wave Equation Block using Method of Lines (MOL)

Solves: ∂²u/∂t² = c²∇²u - damping * ∂u/∂t + f(x,y,t)

Where:
- u(x,y,t) is the displacement field
- c is the wave speed
- damping is an optional damping coefficient
- f(x,y,t) is an external forcing term
- ∇²u = ∂²u/∂x² + ∂²u/∂y² (Laplacian)

The second-order PDE is converted to a first-order system:
- ∂u/∂t = v
- ∂v/∂t = c²∇²u - damping*v + f

This results in 2*Nx*Ny state variables (Nx*Ny for u, Nx*Ny for v).

The domain [0,Lx] × [0,Ly] is discretized into Nx × Ny nodes.
State indexing: u[i,j] -> state[k] where k = i + j*Nx
                v[i,j] -> state[Nx*Ny + k]
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class WaveEquation2DBlock(BaseBlock):
    """
    2D Wave Equation solver using Method of Lines.

    Converts the 2D wave equation PDE into a first-order ODE system.
    Uses 2*Nx*Ny states: Nx*Ny for displacement u, Nx*Ny for velocity v = ∂u/∂t.

    Boundary conditions (for each edge):
    - Dirichlet: u(boundary) = value
    - Neumann: ∂u/∂n(boundary) = value (normal derivative)
    """

    @property
    def block_name(self):
        return "WaveEquation2D"

    @property
    def category(self):
        return "PDE Equations"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "2D Wave Equation: ∂²u/∂t² = c²∇²u"
            "\n\nSolves the 2D wave equation using Method of Lines."
            "\nConverted to first-order system with displacement and velocity."
            "\nDomain is discretized into Nx × Ny nodes."
            "\n\nParameters:"
            "\n- c: Wave speed [m/s]"
            "\n- damping: Damping coefficient (0 = undamped)"
            "\n- Lx, Ly: Domain dimensions [m]"
            "\n- Nx, Ny: Number of nodes in x and y"
            "\n- bc_type_*: Boundary conditions (Dirichlet/Neumann)"
            "\n- init_displacement: Initial displacement"
            "\n- init_velocity: Initial velocity"
            "\n\nInputs:"
            "\n- force: External force term (scalar or Nx×Ny array)"
            "\n- bc_left, bc_right, bc_bottom, bc_top: BC values"
            "\n\nOutputs:"
            "\n- u_field: Displacement field (Nx×Ny array)"
            "\n- v_field: Velocity field (Nx×Ny array)"
            "\n- energy: Total wave energy"
        )

    @property
    def params(self):
        return {
            "c": {
                "type": "float",
                "default": 1.0,
                "doc": "Wave speed [m/s]"
            },
            "damping": {
                "type": "float",
                "default": 0.0,
                "doc": "Damping coefficient"
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
                "default": 20,
                "doc": "Number of nodes in x direction"
            },
            "Ny": {
                "type": "int",
                "default": 20,
                "doc": "Number of nodes in y direction"
            },
            "bc_type_left": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Left BC: Dirichlet or Neumann"
            },
            "bc_type_right": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Right BC: Dirichlet or Neumann"
            },
            "bc_type_bottom": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Bottom BC: Dirichlet or Neumann"
            },
            "bc_type_top": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Top BC: Dirichlet or Neumann"
            },
            "init_displacement": {
                "type": "string",
                "default": "0.0",
                "doc": "Initial displacement: number, 'sinusoidal', 'gaussian', or 'radial'"
            },
            "init_velocity": {
                "type": "string",
                "default": "0.0",
                "doc": "Initial velocity: number, 'sinusoidal', 'gaussian', or 'radial'"
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
            {"name": "force", "type": "array", "doc": "External forcing term"},
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
            {"name": "u_field", "type": "array", "doc": "Displacement field (Nx×Ny)"},
            {"name": "v_field", "type": "array", "doc": "Velocity field (Nx×Ny)"},
            {"name": "energy", "type": "float", "doc": "Total wave energy"},
        ]

    @property
    def optional_outputs(self):
        """Outputs 1 and 2 (v_field, energy) are optional."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw 2D wave equation icon - grid with wave pattern."""
        from PyQt5.QtGui import QPainterPath
        import math
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

        # Wave symbol in corner
        path.moveTo(0.65, 0.2)
        for i in range(5):
            x = 0.65 + i * 0.03
            y = 0.2 - 0.05 * math.sin(i * math.pi / 2)
            path.lineTo(x, y)

        return path

    def get_initial_state(self, params):
        """Return initial state vector [u, v] for the 2D field."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        init_disp = params.get('init_displacement', '0.0')
        init_vel = params.get('init_velocity', '0.0')
        amplitude = float(params.get('init_amplitude', 1.0))

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

        # Initialize displacement
        if isinstance(init_disp, str):
            if init_disp.lower() == 'sinusoidal':
                # u = A * sin(πx/Lx) * sin(πy/Ly) - eigenmode
                u0 = amplitude * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
            elif init_disp.lower() == 'gaussian':
                # Gaussian bump at center
                u0 = amplitude * np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
            elif init_disp.lower() == 'radial':
                # Radial wave from corner
                r = np.sqrt(X**2 + Y**2)
                u0 = amplitude * np.exp(-100 * r**2)
            else:
                # Try to parse as number
                try:
                    u0 = np.full((Ny, Nx), float(init_disp))
                except ValueError:
                    u0 = np.zeros((Ny, Nx))
        else:
            u0 = np.full((Ny, Nx), float(init_disp))

        # Initialize velocity
        if isinstance(init_vel, str):
            if init_vel.lower() == 'sinusoidal':
                v0 = amplitude * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
            elif init_vel.lower() == 'gaussian':
                v0 = amplitude * np.exp(-50 * ((X - Lx/2)**2 + (Y - Ly/2)**2))
            elif init_vel.lower() == 'radial':
                r = np.sqrt(X**2 + Y**2)
                v0 = amplitude * np.exp(-100 * r**2)
            else:
                try:
                    v0 = np.full((Ny, Nx), float(init_vel))
                except ValueError:
                    v0 = np.zeros((Ny, Nx))
        else:
            v0 = np.full((Ny, Nx), float(init_vel))

        # State is [u_flat, v_flat] in row-major order
        return np.concatenate([u0.flatten(), v0.flatten()])

    def get_state_size(self, params):
        """Return the number of state variables (2*Nx*Ny)."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))
        return 2 * Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute displacement and velocity fields (for non-compiled execution)."""
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))

        # Get current state
        state = kwargs.get('state', None)
        if state is None:
            state = self.get_initial_state(params)

        # Split state into u and v
        N = Nx * Ny
        u_flat = state[:N]
        v_flat = state[N:]

        # Reshape to 2D for output
        u_field = u_flat.reshape((Ny, Nx))
        v_field = v_flat.reshape((Ny, Nx))

        # Compute energy
        energy = self._compute_energy(u_field, v_field, params)

        return {
            0: u_field,
            1: v_field,
            2: energy,
            'E': False
        }

    def compute_derivatives(self, time, state, inputs, params):
        """
        Compute d[u,v]/dt for all nodes using 2D finite differences.

        Uses 5-point stencil for Laplacian:
        ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h²

        Returns derivatives for first-order system:
        - du/dt = v
        - dv/dt = c²∇²u - damping*v + f
        """
        c = float(params.get('c', 1.0))
        damping = float(params.get('damping', 0.0))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        c_sq = c * c

        # Get boundary conditions
        bc_left = float(inputs.get(1, 0.0)) if inputs.get(1) is not None else 0.0
        bc_right = float(inputs.get(2, 0.0)) if inputs.get(2) is not None else 0.0
        bc_bottom = float(inputs.get(3, 0.0)) if inputs.get(3) is not None else 0.0
        bc_top = float(inputs.get(4, 0.0)) if inputs.get(4) is not None else 0.0

        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')
        bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
        bc_type_top = params.get('bc_type_top', 'Dirichlet')

        # Get force
        force = inputs.get(0, 0.0)
        if force is None:
            force = 0.0
        if isinstance(force, np.ndarray):
            if force.size == 1:
                force = float(force.flat[0])
            elif force.shape == (Ny, Nx):
                pass  # Use as-is
            else:
                force = float(force.flat[0])

        # Split state into u and v
        N = Nx * Ny
        u = state[:N].reshape((Ny, Nx))
        v = state[N:].reshape((Ny, Nx))

        # Derivatives
        du_dt = v.copy()  # du/dt = v
        dv_dt = np.zeros((Ny, Nx))  # dv/dt = c²∇²u - damping*v + f

        # Interior points: 5-point stencil
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                d2udx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx * dx)
                d2udy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy * dy)

                # Force at this point
                if isinstance(force, np.ndarray):
                    f = force[j, i]
                else:
                    f = force

                dv_dt[j, i] = c_sq * (d2udx2 + d2udy2) - damping * v[j, i] + f

        # Boundary conditions using penalty method for Dirichlet
        penalty = 1000.0

        # Left boundary (i=0)
        if bc_type_left == 'Dirichlet':
            for j in range(Ny):
                du_dt[j, 0] = penalty * (bc_left - u[j, 0])
                dv_dt[j, 0] = 0.0
        else:  # Neumann
            for j in range(1, Ny - 1):
                # Ghost node approach for ∂u/∂x = bc_left at i=0
                d2udx2 = (2*u[j, 1] - 2*u[j, 0] - 2*dx*bc_left) / (dx * dx)
                d2udy2 = (u[j+1, 0] - 2*u[j, 0] + u[j-1, 0]) / (dy * dy)
                f = force[j, 0] if isinstance(force, np.ndarray) else force
                dv_dt[j, 0] = c_sq * (d2udx2 + d2udy2) - damping * v[j, 0] + f

        # Right boundary (i=Nx-1)
        if bc_type_right == 'Dirichlet':
            for j in range(Ny):
                du_dt[j, Nx-1] = penalty * (bc_right - u[j, Nx-1])
                dv_dt[j, Nx-1] = 0.0
        else:  # Neumann
            for j in range(1, Ny - 1):
                d2udx2 = (2*u[j, Nx-2] - 2*u[j, Nx-1] + 2*dx*bc_right) / (dx * dx)
                d2udy2 = (u[j+1, Nx-1] - 2*u[j, Nx-1] + u[j-1, Nx-1]) / (dy * dy)
                f = force[j, Nx-1] if isinstance(force, np.ndarray) else force
                dv_dt[j, Nx-1] = c_sq * (d2udx2 + d2udy2) - damping * v[j, Nx-1] + f

        # Bottom boundary (j=0)
        if bc_type_bottom == 'Dirichlet':
            for i in range(Nx):
                du_dt[0, i] = penalty * (bc_bottom - u[0, i])
                dv_dt[0, i] = 0.0
        else:  # Neumann
            for i in range(1, Nx - 1):
                d2udx2 = (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / (dx * dx)
                d2udy2 = (2*u[1, i] - 2*u[0, i] - 2*dy*bc_bottom) / (dy * dy)
                f = force[0, i] if isinstance(force, np.ndarray) else force
                dv_dt[0, i] = c_sq * (d2udx2 + d2udy2) - damping * v[0, i] + f

        # Top boundary (j=Ny-1)
        if bc_type_top == 'Dirichlet':
            for i in range(Nx):
                du_dt[Ny-1, i] = penalty * (bc_top - u[Ny-1, i])
                dv_dt[Ny-1, i] = 0.0
        else:  # Neumann
            for i in range(1, Nx - 1):
                d2udx2 = (u[Ny-1, i+1] - 2*u[Ny-1, i] + u[Ny-1, i-1]) / (dx * dx)
                d2udy2 = (2*u[Ny-2, i] - 2*u[Ny-1, i] + 2*dy*bc_top) / (dy * dy)
                f = force[Ny-1, i] if isinstance(force, np.ndarray) else force
                dv_dt[Ny-1, i] = c_sq * (d2udx2 + d2udy2) - damping * v[Ny-1, i] + f

        # Return flattened derivatives [du_dt, dv_dt]
        return np.concatenate([du_dt.flatten(), dv_dt.flatten()])

    def _compute_energy(self, u_field, v_field, params):
        """
        Compute total wave energy (kinetic + potential).

        Kinetic energy: 0.5 * ∫∫ v² dx dy
        Potential energy: 0.5 * c² * ∫∫ (|∇u|²) dx dy
        """
        Nx = int(params.get('Nx', 20))
        Ny = int(params.get('Ny', 20))
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))
        c = float(params.get('c', 1.0))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)
        dA = dx * dy

        # Kinetic energy: 0.5 * ∫∫ v² dA
        kinetic = 0.5 * np.sum(v_field**2) * dA

        # Potential energy: 0.5 * c² * ∫∫ (∂u/∂x)² + (∂u/∂y)² dA
        du_dx = np.gradient(u_field, dx, axis=1)
        du_dy = np.gradient(u_field, dy, axis=0)
        potential = 0.5 * c**2 * np.sum(du_dx**2 + du_dy**2) * dA

        return float(kinetic + potential)
