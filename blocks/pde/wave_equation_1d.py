"""
1D Wave Equation Block using Method of Lines (MOL)

Solves: ∂²u/∂t² = c²∇²u - damping * ∂u/∂t + f(x,t)

Where:
- u(x,t) is the displacement field
- c is the wave speed
- damping is an optional damping coefficient
- f(x,t) is an external forcing term
- ∇²u = ∂²u/∂x² (second spatial derivative)

The second-order PDE is converted to a first-order system:
- ∂u/∂t = v
- ∂v/∂t = c²∇²u - damping*v + f

This results in 2N state variables for N spatial nodes.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class WaveEquation1DBlock(BaseBlock):
    """
    1D Wave Equation solver using Method of Lines.

    Converts the wave equation PDE into a first-order ODE system.
    Uses 2N states: N for displacement u, N for velocity v = ∂u/∂t.
    """

    @property
    def block_name(self):
        return "WaveEquation1D"

    @property
    def category(self):
        return "PDE Equations"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "1D Wave Equation: ∂²u/∂t² = c²∇²u"
            "\n\nSolves the wave equation using Method of Lines."
            "\nConverted to first-order system with displacement and velocity."
            "\n\nParameters:"
            "\n- c: Wave speed [m/s]"
            "\n- damping: Damping coefficient (0 = undamped)"
            "\n- L: Domain length [m]"
            "\n- N: Number of spatial nodes"
            "\n- bc_type_left/right: 'Dirichlet' or 'Neumann'"
            "\n- init_displacement: Initial displacement"
            "\n- init_velocity: Initial velocity"
            "\n\nInputs:"
            "\n- force: External force term (scalar or array)"
            "\n- bc_left: Left boundary value"
            "\n- bc_right: Right boundary value"
            "\n\nOutputs:"
            "\n- u_field: Displacement field (N values)"
            "\n- v_field: Velocity field (N values)"
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
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
            "N": {
                "type": "int",
                "default": 50,
                "doc": "Number of spatial nodes"
            },
            "bc_type_left": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Left BC type: Dirichlet or Neumann"
            },
            "bc_type_right": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Right BC type: Dirichlet or Neumann"
            },
            "init_displacement": {
                "type": "list",
                "default": [0.0],
                "doc": "Initial displacement (scalar, list, or 'gaussian', 'sine')"
            },
            "init_velocity": {
                "type": "list",
                "default": [0.0],
                "doc": "Initial velocity (scalar or list)"
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
        ]

    @property
    def outputs(self):
        return [
            {"name": "u_field", "type": "array", "doc": "Displacement field"},
            {"name": "v_field", "type": "array", "doc": "Velocity field"},
            {"name": "energy", "type": "float", "doc": "Total wave energy"},
        ]

    @property
    def optional_inputs(self):
        """All inputs are optional: force and BCs default to 0."""
        return [0, 1, 2]

    @property
    def optional_outputs(self):
        """Outputs 1 (v_field) and 2 (energy) are auxiliary."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw wave equation icon - sine wave."""
        from PyQt5.QtGui import QPainterPath
        import math
        path = QPainterPath()
        # Draw a sine wave
        path.moveTo(0.1, 0.5)
        for i in range(20):
            x = 0.1 + i * 0.04
            y = 0.5 - 0.3 * math.sin(i * math.pi / 5)
            path.lineTo(x, y)
        return path

    def get_num_states(self, params):
        """Return number of states (2N for displacement + velocity)."""
        N = int(params.get('N', 50))
        return 2 * N

    def get_initial_conditions(self, params):
        """Return initial condition vector [u, v]."""
        N = int(params.get('N', 50))
        L = float(params.get('L', 1.0))

        # Initial displacement
        init_u = params.get('init_displacement', [0.0])

        if isinstance(init_u, str):
            x = np.linspace(0, L, N)
            if init_u.lower() == 'gaussian':
                # Gaussian pulse centered at L/2
                u0 = np.exp(-100 * (x - L/2)**2)
            elif init_u.lower() == 'sine':
                # Single sine mode
                u0 = np.sin(np.pi * x / L)
            else:
                u0 = np.zeros(N)
        elif isinstance(init_u, (int, float)):
            u0 = np.full(N, float(init_u))
        else:
            u0 = np.array(init_u, dtype=float).flatten()
            if len(u0) == 1:
                u0 = np.full(N, u0[0])
            elif len(u0) != N:
                # Interpolate
                x_old = np.linspace(0, 1, len(u0))
                x_new = np.linspace(0, 1, N)
                u0 = np.interp(x_new, x_old, u0)

        # Initial velocity
        init_v = params.get('init_velocity', [0.0])

        if isinstance(init_v, (int, float)):
            v0 = np.full(N, float(init_v))
        else:
            v0 = np.array(init_v, dtype=float).flatten()
            if len(v0) == 1:
                v0 = np.full(N, v0[0])
            elif len(v0) != N:
                x_old = np.linspace(0, 1, len(v0))
                x_new = np.linspace(0, 1, N)
                v0 = np.interp(x_new, x_old, v0)

        # Combine into state vector [u, v]
        return np.concatenate([u0, v0])

    def execute(self, time, inputs, params, **kwargs):
        """Execute the wave equation block."""
        output_only = kwargs.get('output_only', False)

        # Initialization
        if params.get('_init_start_', True):
            N = int(params.get('N', 50))
            L = float(params.get('L', 1.0))
            y0 = self.get_initial_conditions(params)
            params['u'] = y0[:N]
            params['v'] = y0[N:]
            params['_init_start_'] = False
            params['dx'] = L / (N - 1)

        N = int(params.get('N', 50))

        if output_only:
            u = params.get('u', np.zeros(N))
            v = params.get('v', np.zeros(N))
            energy = self._compute_energy(u, v, params)
            return {0: u, 1: v, 2: energy, 'E': False}

        # Get parameters
        c = float(params.get('c', 1.0))
        damping = float(params.get('damping', 0.0))
        L = float(params.get('L', 1.0))
        dx = params.get('dx', L / (N - 1))
        dtime = float(params.get('dtime', 0.01))

        # Get current state
        u = params.get('u', np.zeros(N))
        v = params.get('v', np.zeros(N))

        # Get inputs
        force = inputs.get(0, 0.0)
        bc_left = inputs.get(1, 0.0)
        bc_right = inputs.get(2, 0.0)

        # Ensure force is array
        if isinstance(force, (int, float)):
            force = np.full(N, float(force))
        else:
            force = np.atleast_1d(force).flatten()
            if len(force) != N:
                force = np.full(N, force[0] if len(force) > 0 else 0.0)

        # Compute derivatives
        du_dt = v.copy()
        dv_dt = np.zeros(N)

        c_sq = c * c

        # Interior nodes
        for i in range(1, N-1):
            d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx * dx)
            dv_dt[i] = c_sq * d2u_dx2 - damping * v[i] + force[i]

        # Boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')

        if bc_type_left == 'Dirichlet':
            u[0] = bc_left
            v[0] = 0.0
            du_dt[0] = 0.0
            dv_dt[0] = 0.0
        elif bc_type_left == 'Neumann':
            d2u_dx2 = (2*u[1] - 2*u[0] - 2*dx*bc_left) / (dx * dx)
            dv_dt[0] = c_sq * d2u_dx2 - damping * v[0] + force[0]

        if bc_type_right == 'Dirichlet':
            u[N-1] = bc_right
            v[N-1] = 0.0
            du_dt[N-1] = 0.0
            dv_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            d2u_dx2 = (2*u[N-2] - 2*u[N-1] + 2*dx*bc_right) / (dx * dx)
            dv_dt[N-1] = c_sq * d2u_dx2 - damping * v[N-1] + force[N-1]

        # Forward Euler update
        u_new = u + du_dt * dtime
        v_new = v + dv_dt * dtime

        params['u'] = u_new
        params['v'] = v_new

        energy = self._compute_energy(u_new, v_new, params)

        return {0: u_new, 1: v_new, 2: energy, 'E': False}

    def _compute_energy(self, u, v, params):
        """Compute total wave energy (kinetic + potential)."""
        N = len(u)
        L = float(params.get('L', 1.0))
        c = float(params.get('c', 1.0))
        dx = L / (N - 1)

        # Kinetic energy: 0.5 * ∫ v² dx
        kinetic = 0.5 * np.sum(v**2) * dx

        # Potential energy: 0.5 * c² * ∫ (∂u/∂x)² dx
        du_dx = np.gradient(u, dx)
        potential = 0.5 * c**2 * np.sum(du_dx**2) * dx

        return kinetic + potential

    def compute_derivatives(self, state, params, inputs):
        """
        Compute d[u,v]/dt for the ODE solver.

        Args:
            state: Current state vector [u (N values), v (N values)]
            params: Block parameters
            inputs: Dict with force, bc_left, bc_right

        Returns:
            derivatives: [du_dt, dv_dt] flattened
        """
        N = int(params.get('N', 50))
        c = float(params.get('c', 1.0))
        damping = float(params.get('damping', 0.0))
        L = float(params.get('L', 1.0))
        dx = L / (N - 1)

        u = state[:N]
        v = state[N:]

        # Get inputs
        force = inputs.get('force', 0.0)
        bc_left = inputs.get('bc_left', 0.0)
        bc_right = inputs.get('bc_right', 0.0)

        if isinstance(force, (int, float)):
            force = np.full(N, float(force))
        else:
            force = np.atleast_1d(force).flatten()
            if len(force) != N:
                force = np.full(N, force[0] if len(force) > 0 else 0.0)

        c_sq = c * c

        # du/dt = v
        du_dt = v.copy()

        # dv/dt = c²∇²u - damping*v + force
        dv_dt = np.zeros(N)

        for i in range(1, N-1):
            d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / (dx * dx)
            dv_dt[i] = c_sq * d2u_dx2 - damping * v[i] + force[i]

        # Boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')

        if bc_type_left == 'Dirichlet':
            du_dt[0] = 0.0
            dv_dt[0] = 0.0
        elif bc_type_left == 'Neumann':
            d2u_dx2 = (2*u[1] - 2*u[0] - 2*dx*bc_left) / (dx * dx)
            dv_dt[0] = c_sq * d2u_dx2 - damping * v[0] + force[0]

        if bc_type_right == 'Dirichlet':
            du_dt[N-1] = 0.0
            dv_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            d2u_dx2 = (2*u[N-2] - 2*u[N-1] + 2*dx*bc_right) / (dx * dx)
            dv_dt[N-1] = c_sq * d2u_dx2 - damping * v[N-1] + force[N-1]

        return np.concatenate([du_dt, dv_dt])
