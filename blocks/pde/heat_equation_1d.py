"""
1D Heat Equation Block using Method of Lines (MOL)

Solves: ∂T/∂t = α∇²T + q(x,t)

Where:
- T(x,t) is the temperature field
- α is thermal diffusivity
- q(x,t) is heat source term
- ∇²T = ∂²T/∂x² (second spatial derivative)

The domain [0, L] is discretized into N nodes using finite differences.
This converts the PDE into N coupled ODEs that the solver handles.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class HeatEquation1DBlock(BaseBlock):
    """
    1D Heat Equation solver using Method of Lines.

    Converts the heat equation PDE into a system of ODEs by discretizing space.
    Each spatial node becomes a state variable in the ODE system.

    Boundary conditions:
    - Dirichlet: T(boundary) = value
    - Neumann: ∂T/∂x(boundary) = value
    - Robin: -k∂T/∂x = h(T - T_inf)
    """

    @property
    def block_name(self):
        return "HeatEquation1D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "orange"

    @property
    def doc(self):
        return (
            "1D Heat Equation: ∂T/∂t = α∇²T + q"
            "\n\nSolves the heat/diffusion equation using Method of Lines."
            "\nSpace is discretized; time is handled by the ODE solver."
            "\n\nParameters:"
            "\n- alpha: Thermal diffusivity [m²/s]"
            "\n- L: Domain length [m]"
            "\n- N: Number of spatial nodes"
            "\n- bc_type_left/right: 'Dirichlet', 'Neumann', or 'Robin'"
            "\n- init_conds: Initial temperature distribution"
            "\n\nInputs:"
            "\n- q_src: Heat source term (scalar or array)"
            "\n- bc_left: Left boundary value"
            "\n- bc_right: Right boundary value"
            "\n\nOutputs:"
            "\n- T_field: Full temperature field (N values)"
            "\n- T_avg: Average temperature (scalar)"
        )

    @property
    def params(self):
        return {
            "alpha": {
                "type": "float",
                "default": 1.0,
                "doc": "Thermal diffusivity [m²/s]"
            },
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
            "N": {
                "type": "int",
                "default": 20,
                "doc": "Number of spatial nodes"
            },
            "bc_type_left": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Left BC type: Dirichlet, Neumann, or Robin"
            },
            "bc_type_right": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Right BC type: Dirichlet, Neumann, or Robin"
            },
            "h_left": {
                "type": "float",
                "default": 10.0,
                "doc": "Left Robin coefficient (heat transfer coeff)"
            },
            "h_right": {
                "type": "float",
                "default": 10.0,
                "doc": "Right Robin coefficient (heat transfer coeff)"
            },
            "k_thermal": {
                "type": "float",
                "default": 1.0,
                "doc": "Thermal conductivity for Robin BC [W/(m·K)]"
            },
            "init_conds": {
                "type": "list",
                "default": [0.0],
                "doc": "Initial conditions (scalar or list of N values)"
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
            {"name": "q_src", "type": "array", "doc": "Heat source term"},
            {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
            {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "T_field", "type": "array", "doc": "Temperature field (N nodes)"},
            {"name": "T_avg", "type": "float", "doc": "Average temperature"},
        ]

    @property
    def optional_outputs(self):
        """Output 1 (T_avg) is optional - doesn't need to be connected."""
        return [1]

    def draw_icon(self, block_rect):
        """Draw heat equation icon - temperature profile curve."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw a decaying temperature profile curve
        path.moveTo(0.1, 0.8)
        path.cubicTo(0.3, 0.3, 0.5, 0.4, 0.7, 0.5)
        path.lineTo(0.9, 0.5)
        # Add heat waves below
        path.moveTo(0.2, 0.85)
        path.cubicTo(0.3, 0.9, 0.4, 0.85, 0.5, 0.9)
        path.moveTo(0.5, 0.9)
        path.cubicTo(0.6, 0.85, 0.7, 0.9, 0.8, 0.85)
        return path

    def get_num_states(self, params):
        """Return number of states (= number of spatial nodes)."""
        return int(params.get('N', 20))

    def get_initial_conditions(self, params):
        """Return initial condition vector for the temperature field."""
        N = int(params.get('N', 20))
        ic = params.get('init_conds', [0.0])

        if isinstance(ic, (int, float)):
            return np.full(N, float(ic))

        ic_arr = np.array(ic, dtype=float).flatten()

        if len(ic_arr) == 1:
            return np.full(N, ic_arr[0])
        elif len(ic_arr) == N:
            return ic_arr
        elif len(ic_arr) < N:
            # Interpolate to N points
            x_old = np.linspace(0, 1, len(ic_arr))
            x_new = np.linspace(0, 1, N)
            return np.interp(x_new, x_old, ic_arr)
        else:
            # Downsample
            indices = np.linspace(0, len(ic_arr)-1, N, dtype=int)
            return ic_arr[indices]

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute the heat equation block.

        For the fast solver, this is only used during replay.
        The actual ODE integration is done by the SystemCompiler.
        """
        output_only = kwargs.get('output_only', False)

        # Initialization
        if params.get('_init_start_', True):
            N = int(params.get('N', 20))
            params['T'] = self.get_initial_conditions(params)
            params['_init_start_'] = False
            params['dx'] = float(params.get('L', 1.0)) / (N - 1)

        if output_only:
            T = params.get('T', np.zeros(int(params.get('N', 20))))
            return {0: T, 1: np.mean(T), 'E': False}

        # Get parameters
        alpha = float(params.get('alpha', 1.0))
        L = float(params.get('L', 1.0))
        N = int(params.get('N', 20))
        dx = params.get('dx', L / (N - 1))
        dtime = float(params.get('dtime', 0.01))

        # Get current state
        T = params.get('T', np.zeros(N))

        # Get inputs
        q_src = inputs.get(0, 0.0)
        bc_left_val = inputs.get(1, 0.0)
        bc_right_val = inputs.get(2, 0.0)

        # Ensure q_src is array of correct size
        if isinstance(q_src, (int, float)):
            q_src = np.full(N, float(q_src))
        else:
            q_src = np.atleast_1d(q_src).flatten()
            if len(q_src) != N:
                if len(q_src) == 1:
                    q_src = np.full(N, q_src[0])
                else:
                    q_src = np.interp(np.linspace(0, 1, N),
                                      np.linspace(0, 1, len(q_src)), q_src)

        # Compute spatial derivative (∇²T) using central differences
        dT_dt = np.zeros(N)

        # Interior nodes: central difference
        for i in range(1, N-1):
            d2T_dx2 = (T[i+1] - 2*T[i] + T[i-1]) / (dx * dx)
            dT_dt[i] = alpha * d2T_dx2 + q_src[i]

        # Apply boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')

        # Left boundary (i=0)
        if bc_type_left == 'Dirichlet':
            T[0] = bc_left_val
            dT_dt[0] = 0.0
        elif bc_type_left == 'Neumann':
            # Ghost node: T[-1] = T[1] - 2*dx*bc_left_val
            # ∂²T/∂x² at i=0: (T[1] - 2*T[0] + T[-1]) / dx²
            #                = (2*T[1] - 2*T[0] - 2*dx*bc_left_val) / dx²
            d2T_dx2 = (2*T[1] - 2*T[0] - 2*dx*bc_left_val) / (dx * dx)
            dT_dt[0] = alpha * d2T_dx2 + q_src[0]
        elif bc_type_left == 'Robin':
            # -k * ∂T/∂x = h * (T - T_inf) at x=0
            h = params.get('h_left', 10.0)
            k = params.get('k_thermal', 1.0)
            # Forward difference: ∂T/∂x ≈ (T[1] - T[0]) / dx
            # Robin: T[0] = (k*T[1]/dx + h*bc_left_val) / (k/dx + h)
            T[0] = (k * T[1] / dx + h * bc_left_val) / (k / dx + h)
            dT_dt[0] = 0.0

        # Right boundary (i=N-1)
        if bc_type_right == 'Dirichlet':
            T[N-1] = bc_right_val
            dT_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            # Ghost node approach
            d2T_dx2 = (2*T[N-2] - 2*T[N-1] + 2*dx*bc_right_val) / (dx * dx)
            dT_dt[N-1] = alpha * d2T_dx2 + q_src[N-1]
        elif bc_type_right == 'Robin':
            h = params.get('h_right', 10.0)
            k = params.get('k_thermal', 1.0)
            T[N-1] = (k * T[N-2] / dx + h * bc_right_val) / (k / dx + h)
            dT_dt[N-1] = 0.0

        # Forward Euler time step (simple, for interpreter mode)
        T_new = T + dT_dt * dtime
        params['T'] = T_new

        # Compute outputs
        T_avg = np.mean(T_new)

        return {0: T_new, 1: T_avg, 'E': False}

    def compute_derivatives(self, T, params, inputs):
        """
        Compute dT/dt for the ODE solver.

        This method is called by the SystemCompiler to get the time derivatives
        of all state variables for the current spatial discretization.

        Args:
            T: Current temperature field (N values)
            params: Block parameters
            inputs: Dict with q_src, bc_left, bc_right

        Returns:
            dT_dt: Time derivatives (N values)
        """
        alpha = float(params.get('alpha', 1.0))
        L = float(params.get('L', 1.0))
        N = int(params.get('N', 20))
        dx = L / (N - 1)

        # Get inputs
        q_src = inputs.get('q_src', 0.0)
        bc_left_val = inputs.get('bc_left', 0.0)
        bc_right_val = inputs.get('bc_right', 0.0)

        # Ensure q_src is array
        if isinstance(q_src, (int, float)):
            q_src = np.full(N, float(q_src))
        else:
            q_src = np.atleast_1d(q_src).flatten()
            if len(q_src) != N:
                q_src = np.full(N, q_src[0] if len(q_src) > 0 else 0.0)

        # Initialize derivatives
        dT_dt = np.zeros(N)

        # Interior nodes
        for i in range(1, N-1):
            d2T_dx2 = (T[i+1] - 2*T[i] + T[i-1]) / (dx * dx)
            dT_dt[i] = alpha * d2T_dx2 + q_src[i]

        # Boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')

        # Left boundary
        if bc_type_left == 'Dirichlet':
            dT_dt[0] = 0.0  # Fixed value, no change
        elif bc_type_left == 'Neumann':
            d2T_dx2 = (2*T[1] - 2*T[0] - 2*dx*bc_left_val) / (dx * dx)
            dT_dt[0] = alpha * d2T_dx2 + q_src[0]
        elif bc_type_left == 'Robin':
            dT_dt[0] = 0.0  # Algebraically determined

        # Right boundary
        if bc_type_right == 'Dirichlet':
            dT_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            d2T_dx2 = (2*T[N-2] - 2*T[N-1] + 2*dx*bc_right_val) / (dx * dx)
            dT_dt[N-1] = alpha * d2T_dx2 + q_src[N-1]
        elif bc_type_right == 'Robin':
            dT_dt[N-1] = 0.0

        return dT_dt

    def apply_boundary_conditions(self, T, params, inputs):
        """
        Apply boundary conditions to the temperature field.
        Called during ODE solution to enforce Dirichlet/Robin BCs.

        Returns:
            Modified T array with BCs applied
        """
        N = len(T)
        L = float(params.get('L', 1.0))
        dx = L / (N - 1)

        bc_left_val = inputs.get('bc_left', 0.0)
        bc_right_val = inputs.get('bc_right', 0.0)
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Dirichlet')

        T_mod = T.copy()

        if bc_type_left == 'Dirichlet':
            T_mod[0] = bc_left_val
        elif bc_type_left == 'Robin':
            h = params.get('h_left', 10.0)
            k = params.get('k_thermal', 1.0)
            T_mod[0] = (k * T[1] / dx + h * bc_left_val) / (k / dx + h)

        if bc_type_right == 'Dirichlet':
            T_mod[N-1] = bc_right_val
        elif bc_type_right == 'Robin':
            h = params.get('h_right', 10.0)
            k = params.get('k_thermal', 1.0)
            T_mod[N-1] = (k * T[N-2] / dx + h * bc_right_val) / (k / dx + h)

        return T_mod
