"""
1D Diffusion-Reaction Equation Block using Method of Lines (MOL)

Solves: ∂c/∂t = D * ∇²c - k * c^n + S(x,t)

Where:
- c(x,t) is the concentration
- D is the diffusion coefficient
- k is the reaction rate constant
- n is the reaction order (1 = first order, 2 = second order)
- S(x,t) is a source term

Common applications:
- Chemical reactions with diffusion
- Population dynamics with spatial spread
- Heat conduction with heat sinks/sources
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock
from blocks.param_templates import (
    diffusivity_param, domain_params_1d, init_flag_param
)
from lib.engine.pde_helpers import bc_params_1d

logger = logging.getLogger(__name__)


class DiffusionReaction1DBlock(BaseBlock):
    """
    1D Diffusion-Reaction Equation solver using Method of Lines.

    Combines diffusion (spreading) with reaction (consumption/production).
    Supports first and second-order reactions.
    """

    @property
    def block_name(self):
        return "DiffusionReaction1D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "green"

    @property
    def doc(self):
        return (
            "1D Diffusion-Reaction: ∂c/∂t = D∇²c - kc^n + S"
            "\n\nCombines diffusion with chemical/physical reactions."
            "\n\nParameters:"
            "\n- D: Diffusion coefficient [m²/s]"
            "\n- k: Reaction rate constant [1/s for n=1]"
            "\n- n: Reaction order (1 or 2)"
            "\n- L: Domain length [m]"
            "\n- N: Number of spatial nodes"
            "\n- bc_type_left/right: 'Dirichlet', 'Neumann', or 'Robin'"
            "\n- init_conds: Initial concentration"
            "\n\nInputs:"
            "\n- source: Source term S(x,t)"
            "\n- bc_left: Left boundary value"
            "\n- bc_right: Right boundary value"
            "\n\nOutputs:"
            "\n- c_field: Concentration field"
            "\n- c_total: Total mass"
            "\n- reaction_rate: Current reaction rate"
        )

    @property
    def params(self):
        return {
            **diffusivity_param(default=0.01, param_name="D", doc="Diffusion coefficient [m²/s]"),
            "k": {
                "type": "float",
                "default": 0.1,
                "doc": "Reaction rate constant"
            },
            "n": {
                "type": "int",
                "default": 1,
                "doc": "Reaction order (1 or 2)"
            },
            **domain_params_1d(default_length=1.0, default_nodes=30),
            **bc_params_1d(left_default="Dirichlet", right_default="Neumann", include_robin=False),
            "h_mass_transfer": {
                "type": "float",
                "default": 1.0,
                "doc": "Mass transfer coefficient for Robin BC"
            },
            "init_conds": {
                "type": "list",
                "default": [1.0],
                "doc": "Initial concentration"
            },
            **init_flag_param(),
        }

    @property
    def inputs(self):
        return [
            {"name": "source", "type": "array", "doc": "Source term S(x,t)"},
            {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
            {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "c_field", "type": "array", "doc": "Concentration field"},
            {"name": "c_total", "type": "float", "doc": "Total mass"},
            {"name": "reaction_rate", "type": "float", "doc": "Total reaction rate"},
        ]

    @property
    def optional_inputs(self):
        """Inputs 0-2 are optional: source term and BCs depend on use case."""
        return [0, 1, 2]

    @property
    def optional_outputs(self):
        """Outputs 1 (c_total) and 2 (reaction_rate) are auxiliary."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw diffusion-reaction icon - spreading profile with reaction."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw spreading Gaussian profile
        path.moveTo(0.1, 0.75)
        path.cubicTo(0.2, 0.75, 0.35, 0.25, 0.5, 0.25)
        path.cubicTo(0.65, 0.25, 0.8, 0.75, 0.9, 0.75)
        # Small circles representing reaction/particles
        path.addEllipse(0.3, 0.55, 0.08, 0.08)
        path.addEllipse(0.5, 0.45, 0.08, 0.08)
        path.addEllipse(0.65, 0.55, 0.08, 0.08)
        return path

    def get_num_states(self, params):
        """Return number of states (= number of spatial nodes)."""
        return int(params.get('N', 30))

    def get_initial_conditions(self, params):
        """Return initial condition vector."""
        N = int(params.get('N', 30))
        L = float(params.get('L', 1.0))
        ic = params.get('init_conds', [1.0])

        x = np.linspace(0, L, N)

        if isinstance(ic, str):
            if ic.lower() == 'uniform':
                c0 = np.ones(N)
            elif ic.lower() == 'gaussian':
                c0 = np.exp(-50 * (x - L/2)**2)
            elif ic.lower() == 'linear':
                c0 = 1 - x/L
            else:
                c0 = np.ones(N)
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
        """Execute the diffusion-reaction equation block."""
        output_only = kwargs.get('output_only', False)

        # Initialization
        if params.get('_init_start_', True):
            N = int(params.get('N', 30))
            L = float(params.get('L', 1.0))
            params['c'] = self.get_initial_conditions(params)
            params['_init_start_'] = False
            params['dx'] = L / (N - 1)

        N = int(params.get('N', 30))

        if output_only:
            c = params.get('c', np.zeros(N))
            dx = params.get('dx', 1.0 / (N - 1))
            k = float(params.get('k', 0.1))
            n = int(params.get('n', 1))
            c_total = np.sum(c) * dx
            reaction_rate = np.sum(k * np.power(np.maximum(c, 0), n)) * dx
            return {0: c, 1: c_total, 2: reaction_rate, 'E': False}

        # Get parameters
        D = float(params.get('D', 0.01))
        k = float(params.get('k', 0.1))
        n = int(params.get('n', 1))
        L = float(params.get('L', 1.0))
        dx = params.get('dx', L / (N - 1))
        dtime = float(params.get('dtime', 0.01))

        # Get current state
        c = params.get('c', np.zeros(N))

        # Get inputs
        source = inputs.get(0, 0.0)
        bc_left = inputs.get(1, 0.0)
        bc_right = inputs.get(2, 0.0)

        # Ensure source is array
        if isinstance(source, (int, float)):
            source = np.full(N, float(source))
        else:
            source = np.atleast_1d(source).flatten()
            if len(source) != N:
                source = np.full(N, source[0] if len(source) > 0 else 0.0)

        # Compute derivative
        dc_dt = np.zeros(N)

        # Interior nodes
        for i in range(1, N-1):
            d2c_dx2 = (c[i+1] - 2*c[i] + c[i-1]) / (dx * dx)
            reaction = k * np.power(max(c[i], 0), n)
            dc_dt[i] = D * d2c_dx2 - reaction + source[i]

        # Boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Neumann')

        if bc_type_left == 'Dirichlet':
            c[0] = bc_left
            dc_dt[0] = 0.0
        elif bc_type_left == 'Neumann':
            d2c_dx2 = (2*c[1] - 2*c[0] - 2*dx*bc_left) / (dx * dx)
            reaction = k * np.power(max(c[0], 0), n)
            dc_dt[0] = D * d2c_dx2 - reaction + source[0]
        elif bc_type_left == 'Robin':
            h = params.get('h_mass_transfer', 1.0)
            c[0] = (D * c[1] / dx + h * bc_left) / (D / dx + h)
            dc_dt[0] = 0.0

        if bc_type_right == 'Dirichlet':
            c[N-1] = bc_right
            dc_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            d2c_dx2 = (2*c[N-2] - 2*c[N-1] + 2*dx*bc_right) / (dx * dx)
            reaction = k * np.power(max(c[N-1], 0), n)
            dc_dt[N-1] = D * d2c_dx2 - reaction + source[N-1]
        elif bc_type_right == 'Robin':
            h = params.get('h_mass_transfer', 1.0)
            c[N-1] = (D * c[N-2] / dx + h * bc_right) / (D / dx + h)
            dc_dt[N-1] = 0.0

        # Forward Euler update
        c_new = c + dc_dt * dtime

        # Ensure non-negativity
        c_new = np.maximum(c_new, 0.0)

        params['c'] = c_new

        c_total = np.sum(c_new) * dx
        reaction_rate = np.sum(k * np.power(np.maximum(c_new, 0), n)) * dx

        return {0: c_new, 1: c_total, 2: reaction_rate, 'E': False}

    def compute_derivatives(self, c, params, inputs):
        """
        Compute dc/dt for the ODE solver.
        """
        N = int(params.get('N', 30))
        D = float(params.get('D', 0.01))
        k = float(params.get('k', 0.1))
        n = int(params.get('n', 1))
        L = float(params.get('L', 1.0))
        dx = L / (N - 1)

        source = inputs.get('source', 0.0)
        bc_left = inputs.get('bc_left', 0.0)
        bc_right = inputs.get('bc_right', 0.0)

        if isinstance(source, (int, float)):
            source = np.full(N, float(source))
        else:
            source = np.atleast_1d(source).flatten()
            if len(source) != N:
                source = np.full(N, source[0] if len(source) > 0 else 0.0)

        dc_dt = np.zeros(N)

        # Interior nodes
        for i in range(1, N-1):
            d2c_dx2 = (c[i+1] - 2*c[i] + c[i-1]) / (dx * dx)
            reaction = k * np.power(max(c[i], 0), n)
            dc_dt[i] = D * d2c_dx2 - reaction + source[i]

        # Boundary conditions
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Neumann')

        if bc_type_left == 'Dirichlet':
            dc_dt[0] = 0.0
        elif bc_type_left == 'Neumann':
            d2c_dx2 = (2*c[1] - 2*c[0] - 2*dx*bc_left) / (dx * dx)
            reaction = k * np.power(max(c[0], 0), n)
            dc_dt[0] = D * d2c_dx2 - reaction + source[0]
        elif bc_type_left == 'Robin':
            dc_dt[0] = 0.0

        if bc_type_right == 'Dirichlet':
            dc_dt[N-1] = 0.0
        elif bc_type_right == 'Neumann':
            d2c_dx2 = (2*c[N-2] - 2*c[N-1] + 2*dx*bc_right) / (dx * dx)
            reaction = k * np.power(max(c[N-1], 0), n)
            dc_dt[N-1] = D * d2c_dx2 - reaction + source[N-1]
        elif bc_type_right == 'Robin':
            dc_dt[N-1] = 0.0

        return dc_dt

    def apply_boundary_conditions(self, c, params, inputs):
        """Apply boundary conditions."""
        N = len(c)
        L = float(params.get('L', 1.0))
        D = float(params.get('D', 0.01))
        dx = L / (N - 1)

        bc_left = inputs.get('bc_left', 0.0)
        bc_right = inputs.get('bc_right', 0.0)
        bc_type_left = params.get('bc_type_left', 'Dirichlet')
        bc_type_right = params.get('bc_type_right', 'Neumann')

        c_mod = c.copy()

        if bc_type_left == 'Dirichlet':
            c_mod[0] = bc_left
        elif bc_type_left == 'Robin':
            h = params.get('h_mass_transfer', 1.0)
            c_mod[0] = (D * c[1] / dx + h * bc_left) / (D / dx + h)

        if bc_type_right == 'Dirichlet':
            c_mod[N-1] = bc_right
        elif bc_type_right == 'Robin':
            h = params.get('h_mass_transfer', 1.0)
            c_mod[N-1] = (D * c[N-2] / dx + h * bc_right) / (D / dx + h)

        return c_mod
