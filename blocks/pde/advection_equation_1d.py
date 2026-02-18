"""
1D Advection Equation Block using Method of Lines (MOL)

Solves: ∂c/∂t + v * ∂c/∂x = 0

Where:
- c(x,t) is the concentration/quantity being advected
- v is the advection velocity (can be positive or negative)

The advection equation describes the transport of a quantity by a flow
without diffusion (pure transport/convection).

Uses upwind scheme for stability.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock
from blocks.param_templates import (
    advection_velocity_param, domain_params_1d, init_flag_param
)

logger = logging.getLogger(__name__)


class AdvectionEquation1DBlock(BaseBlock):
    """
    1D Advection Equation solver using Method of Lines.

    Uses upwind differencing for numerical stability:
    - Positive velocity: backward difference
    - Negative velocity: forward difference
    """

    @property
    def block_name(self):
        return "AdvectionEquation1D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "cyan"

    @property
    def doc(self):
        return (
            "1D Advection Equation: ∂c/∂t + v * ∂c/∂x = 0"
            "\n\nSolves the advection (transport) equation using upwind scheme."
            "\nDescribes pure transport without diffusion."
            "\n\nParameters:"
            "\n- velocity: Advection velocity [m/s] (positive = rightward)"
            "\n- L: Domain length [m]"
            "\n- N: Number of spatial nodes"
            "\n- bc_type: 'Dirichlet' (fixed inlet) or 'Periodic'"
            "\n- init_conds: Initial concentration distribution"
            "\n\nInputs:"
            "\n- c_inlet: Inlet concentration (for Dirichlet BC)"
            "\n\nOutputs:"
            "\n- c_field: Concentration field (N values)"
            "\n- c_total: Total mass (integral of c)"
        )

    @property
    def params(self):
        return {
            **advection_velocity_param(default=1.0, param_name="velocity",
                                       doc="Advection velocity [m/s] (positive = rightward)"),
            **domain_params_1d(default_length=1.0, default_nodes=50),
            "bc_type": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "BC type: Dirichlet or Periodic"
            },
            "init_conds": {
                "type": "list",
                "default": [0.0],
                "doc": "Initial concentration (scalar, list, or 'gaussian', 'step')"
            },
            **init_flag_param(),
        }

    @property
    def inputs(self):
        return [
            {"name": "c_inlet", "type": "float", "doc": "Inlet concentration value"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "c_field", "type": "array", "doc": "Concentration field"},
            {"name": "c_total", "type": "float", "doc": "Total mass (integral)"},
        ]

    @property
    def optional_inputs(self):
        """Input 0 (c_inlet) is optional when using periodic BC."""
        return [0]

    @property
    def optional_outputs(self):
        """Output 1 (c_total) is auxiliary/diagnostic."""
        return [1]

    def draw_icon(self, block_rect):
        """Draw advection icon - arrow with flowing profile."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw a bell curve being transported
        path.moveTo(0.15, 0.7)
        path.cubicTo(0.25, 0.7, 0.3, 0.3, 0.4, 0.3)
        path.cubicTo(0.5, 0.3, 0.55, 0.7, 0.65, 0.7)
        # Arrow showing flow direction
        path.moveTo(0.7, 0.5)
        path.lineTo(0.9, 0.5)
        path.moveTo(0.85, 0.4)
        path.lineTo(0.9, 0.5)
        path.lineTo(0.85, 0.6)
        return path

    def get_num_states(self, params):
        """Return number of states (= number of spatial nodes)."""
        return int(params.get('N', 50))

    def get_initial_conditions(self, params):
        """Return initial condition vector for the concentration field."""
        N = int(params.get('N', 50))
        L = float(params.get('L', 1.0))
        ic = params.get('init_conds', [0.0])

        x = np.linspace(0, L, N)

        if isinstance(ic, str):
            if ic.lower() == 'gaussian':
                # Gaussian pulse centered at L/4
                # Width sigma = 0.14m (~7 grid points at N=101, L=2)
                c0 = np.exp(-25 * (x - L/4)**2)
            elif ic.lower() == 'step':
                # Step function at L/4
                c0 = np.where(x < L/4, 1.0, 0.0)
            elif ic.lower() in ('sin', 'sine'):
                c0 = 0.5 * (1 + np.sin(2 * np.pi * x / L))
            else:
                c0 = np.zeros(N)
        elif isinstance(ic, (int, float)):
            c0 = np.full(N, float(ic))
        else:
            c0 = np.array(ic, dtype=float).flatten()
            if len(c0) == 1:
                c0 = np.full(N, c0[0])
            elif len(c0) != N:
                x_old = np.linspace(0, 1, len(c0))
                x_new = np.linspace(0, 1, N)
                c0 = np.interp(x_new, x_old, c0)

        return c0

    def execute(self, time, inputs, params, **kwargs):
        """Execute the advection equation block."""
        output_only = kwargs.get('output_only', False)

        # Initialization
        if params.get('_init_start_', True):
            N = int(params.get('N', 50))
            L = float(params.get('L', 1.0))
            params['c'] = self.get_initial_conditions(params)
            params['_init_start_'] = False
            params['dx'] = L / (N - 1)

        N = int(params.get('N', 50))

        if output_only:
            c = params.get('c', np.zeros(N))
            dx = params.get('dx', 1.0 / (N - 1))
            c_total = np.sum(c) * dx
            return {0: c, 1: c_total, 'E': False}

        # Get parameters
        v = float(params.get('velocity', 1.0))
        L = float(params.get('L', 1.0))
        dx = params.get('dx', L / (N - 1))
        dtime = float(params.get('dtime', 0.01))
        bc_type = params.get('bc_type', 'Dirichlet')

        # Get current state
        c = params.get('c', np.zeros(N))

        # Get inputs
        c_inlet = inputs.get(0, 0.0)

        # Compute spatial derivative using second-order upwind scheme
        # This reduces numerical diffusion compared to first-order upwind
        dc_dt = np.zeros(N)

        if v >= 0:
            # Positive velocity: second-order backward difference (upwind from left)
            # Interior points: (3*c[i] - 4*c[i-1] + c[i-2]) / (2*dx)
            for i in range(2, N):
                dc_dx = (3*c[i] - 4*c[i-1] + c[i-2]) / (2*dx)
                dc_dt[i] = -v * dc_dx
            # First interior point: first-order fallback
            if N > 1:
                dc_dx = (c[1] - c[0]) / dx
                dc_dt[1] = -v * dc_dx

            # Left boundary (inlet)
            if bc_type == 'Dirichlet':
                dc_dt[0] = 0.0  # Fixed by inlet
            elif bc_type == 'Periodic':
                dc_dx = (3*c[0] - 4*c[N-1] + c[N-2]) / (2*dx)
                dc_dt[0] = -v * dc_dx
        else:
            # Negative velocity: second-order forward difference (upwind from right)
            # Interior points: (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*dx)
            for i in range(N-2):
                dc_dx = (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*dx)
                dc_dt[i] = -v * dc_dx
            # Last interior point: first-order fallback
            if N > 1:
                dc_dx = (c[N-1] - c[N-2]) / dx
                dc_dt[N-2] = -v * dc_dx

            # Right boundary (inlet for negative velocity)
            if bc_type == 'Dirichlet':
                dc_dt[N-1] = 0.0
            elif bc_type == 'Periodic':
                dc_dx = (-3*c[N-1] + 4*c[0] - c[1]) / (2*dx)
                dc_dt[N-1] = -v * dc_dx

        # Apply inlet BC
        if bc_type == 'Dirichlet':
            if v >= 0:
                c[0] = c_inlet
            else:
                c[N-1] = c_inlet

        # RK4 time integration for better accuracy
        def compute_rhs(c_state):
            """Compute dc/dt for given state."""
            rhs = np.zeros(N)
            if v >= 0:
                for i in range(2, N):
                    dc_dx = (3*c_state[i] - 4*c_state[i-1] + c_state[i-2]) / (2*dx)
                    rhs[i] = -v * dc_dx
                if N > 1:
                    dc_dx = (c_state[1] - c_state[0]) / dx
                    rhs[1] = -v * dc_dx
                if bc_type == 'Dirichlet':
                    rhs[0] = 0.0
                elif bc_type == 'Periodic':
                    dc_dx = (3*c_state[0] - 4*c_state[N-1] + c_state[N-2]) / (2*dx)
                    rhs[0] = -v * dc_dx
            else:
                for i in range(N-2):
                    dc_dx = (-3*c_state[i] + 4*c_state[i+1] - c_state[i+2]) / (2*dx)
                    rhs[i] = -v * dc_dx
                if N > 1:
                    dc_dx = (c_state[N-1] - c_state[N-2]) / dx
                    rhs[N-2] = -v * dc_dx
                if bc_type == 'Dirichlet':
                    rhs[N-1] = 0.0
                elif bc_type == 'Periodic':
                    dc_dx = (-3*c_state[N-1] + 4*c_state[0] - c_state[1]) / (2*dx)
                    rhs[N-1] = -v * dc_dx
            return rhs

        # RK4 stages
        k1 = compute_rhs(c)
        k2 = compute_rhs(c + 0.5 * dtime * k1)
        k3 = compute_rhs(c + 0.5 * dtime * k2)
        k4 = compute_rhs(c + dtime * k3)

        c_new = c + (dtime / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        params['c'] = c_new

        c_total = np.sum(c_new) * dx

        return {0: c_new, 1: c_total, 'E': False}

    def compute_derivatives(self, c, params, inputs):
        """
        Compute dc/dt for the ODE solver.

        Uses second-order upwind differencing for reduced numerical diffusion.
        """
        N = int(params.get('N', 50))
        v = float(params.get('velocity', 1.0))
        L = float(params.get('L', 1.0))
        dx = L / (N - 1)
        bc_type = params.get('bc_type', 'Dirichlet')

        c_inlet = inputs.get('c_inlet', 0.0)

        dc_dt = np.zeros(N)

        if v >= 0:
            # Second-order backward difference
            for i in range(2, N):
                dc_dx = (3*c[i] - 4*c[i-1] + c[i-2]) / (2*dx)
                dc_dt[i] = -v * dc_dx
            if N > 1:
                dc_dx = (c[1] - c[0]) / dx
                dc_dt[1] = -v * dc_dx

            if bc_type == 'Dirichlet':
                dc_dt[0] = 0.0
            elif bc_type == 'Periodic':
                dc_dx = (3*c[0] - 4*c[N-1] + c[N-2]) / (2*dx)
                dc_dt[0] = -v * dc_dx
        else:
            # Second-order forward difference
            for i in range(N-2):
                dc_dx = (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*dx)
                dc_dt[i] = -v * dc_dx
            if N > 1:
                dc_dx = (c[N-1] - c[N-2]) / dx
                dc_dt[N-2] = -v * dc_dx

            if bc_type == 'Dirichlet':
                dc_dt[N-1] = 0.0
            elif bc_type == 'Periodic':
                dc_dx = (-3*c[N-1] + 4*c[0] - c[1]) / (2*dx)
                dc_dt[N-1] = -v * dc_dx

        return dc_dt

    def apply_boundary_conditions(self, c, params, inputs):
        """Apply inlet boundary condition."""
        N = len(c)
        v = float(params.get('velocity', 1.0))
        bc_type = params.get('bc_type', 'Dirichlet')
        c_inlet = inputs.get('c_inlet', 0.0)

        c_mod = c.copy()

        if bc_type == 'Dirichlet':
            if v >= 0:
                c_mod[0] = c_inlet
            else:
                c_mod[N-1] = c_inlet

        return c_mod
