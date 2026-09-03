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
from blocks.pde._compat import as_scalar
from blocks.param_templates import (
    diffusivity_param,
    domain_params_1d,
    init_flag_param,
    robin_bc_params,
)
from lib.engine.pde_helpers import bc_params_1d, parse_pde_initial_condition
from lib.engine.pde_ops import heat_rhs_1d, is_periodic, robin_boundary_value

logger = logging.getLogger(__name__)


class HeatEquation1DBlock(BaseBlock):
    """
    1D Heat Equation solver using Method of Lines.

    Converts the heat equation PDE into a system of ODEs by discretizing space.
    Each spatial node becomes a state variable in the ODE system.

    Boundary conditions:
    - Dirichlet: T(boundary) = value
    - Neumann: ∂T/∂x(boundary) = value
    - Robin: -k∂T/∂n = h(T - T_inf), n the outward normal
    - Periodic: the N nodes wrap as a ring (set on either end)
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
            "\n- bc_type_left/right: 'Dirichlet', 'Neumann', 'Robin', or 'Periodic'"
            "\n  ('Periodic' on either end wraps the whole rod into a ring;"
            "\n  the opposite end's BC type and value are then ignored)"
            "\n- h_left/h_right, k_thermal: Robin coefficients (static defaults)"
            "\n- init_conds: Initial temperature -- a number, a list, or one of"
            "\n  'sine', 'gaussian', 'uniform', 'step', 'linear', 'random'"
            "\n- seed: Seed for the 'random' IC (0 = not reproducible)"
            "\n\nInputs:"
            "\n- q_src: Heat source term (scalar or array)"
            "\n- bc_left: Left boundary value (Dirichlet value, Neumann flux,"
            "\n  or Robin ambient temperature T_inf -- time-varying)"
            "\n- bc_right: Right boundary value (same meaning)"
            "\n- h_left, h_right: OPTIONAL time-varying Robin coefficients."
            "\n  Leave them unconnected to use the h_left / h_right params."
            "\n  Both execution paths read these per time step, so a Robin"
            "\n  boundary can model a fan switching on mid-run."
            "\n\nOutputs:"
            "\n- T_field: Full temperature field (N values)"
            "\n- T_avg: Average temperature (scalar)"
        )

    @property
    def params(self):
        return {
            **diffusivity_param(default=1.0),
            **domain_params_1d(default_length=1.0, default_nodes=20),
            **bc_params_1d(include_robin=True),
            **robin_bc_params(),
            "init_conds": {
                "type": "list",
                "default": [0.0],
                "doc": (
                    "Initial conditions: scalar, list of N values, or one of "
                    "'sine', 'gaussian', 'uniform', 'step', 'linear', 'random'"
                ),
            },
            "seed": {
                "type": "int",
                "default": 0,
                "doc": "Random seed for the 'random' initial condition (0 = random).",
            },
            **init_flag_param(),
        }

    @property
    def inputs(self):
        return [
            {"name": "q_src", "type": "array", "doc": "Heat source term"},
            {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
            {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
            {"name": "h_left", "type": "float", "doc": "Left Robin coefficient (optional)"},
            {"name": "h_right", "type": "float", "doc": "Right Robin coefficient (optional)"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "T_field", "type": "array", "doc": "Temperature field (N nodes)"},
            {"name": "T_avg", "type": "float", "doc": "Average temperature"},
        ]

    @property
    def optional_inputs(self):
        """q_src (0) and the Robin coefficient ports (3, 4) are optional.

        The h ports default to the ``h_left`` / ``h_right`` params when left
        unconnected, so diagrams saved before those ports existed (in_ports=3)
        keep working unchanged.
        """
        return [0, 3, 4]

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
        return int(params.get("N", 20))

    def get_initial_conditions(self, params):
        """Return initial condition vector for the temperature field.

        Delegates to the shared ``parse_pde_initial_condition`` so the
        interpreter and the compiled path (which calls the same helper from
        ``SystemCompiler.compile_system``) build the SAME field -- including a
        seeded 'random' one. This block used to carry its own copy of the
        parsing, which recognised fewer patterns and gave 'gaussian' a narrower
        width (exp(-100 r^2)) than the compiled path's exp(-50 r^2).
        """
        return parse_pde_initial_condition(
            params.get("init_conds", [0.0]),
            int(params.get("N", 20)),
            float(params.get("L", 1.0)),
            pde_type="heat",
            seed=params.get("seed", 0),
        )

    @staticmethod
    def _robin_coeffs(params, h_left_in, h_right_in):
        """Resolve the Robin h coefficients, input port overriding param.

        ``None`` means "port not connected" -- the static param then applies.
        Called on every time step / RHS evaluation, never cached, so a connected
        port makes the coefficient genuinely time-varying.
        """
        h_left = float(params.get("h_left", 10.0)) if h_left_in is None else as_scalar(h_left_in)
        h_right = (
            float(params.get("h_right", 10.0)) if h_right_in is None else as_scalar(h_right_in)
        )
        return h_left, h_right

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute the heat equation block.

        For the fast solver, this is only used during replay.
        The actual ODE integration is done by the SystemCompiler.
        """
        output_only = kwargs.get("output_only", False)

        # Initialization
        if params.get("_init_start_", True):
            N = max(2, int(params.get("N", 20)))
            params["N"] = N
            params["T"] = self.get_initial_conditions(params)
            params["_init_start_"] = False
            params["dx"] = float(params.get("L", 1.0)) / (N - 1)
            # Re-arm the CFL warning for this run.  reset_memblocks() only
            # re-sets _init_start_ and drops _prev/mem/output, and an unchanged
            # re-run is served from the cached exec_params, so a flag left set
            # here would silence the warning for every later run in the process
            # -- the second run of a diverging diagram would look clean.
            params["_cfl_warned_"] = False

        if output_only:
            T = params.get("T", np.zeros(int(params.get("N", 20))))
            return {0: T, 1: np.mean(T), "E": False}

        # Get parameters
        alpha = float(params.get("alpha", 1.0))
        L = float(params.get("L", 1.0))
        N = int(params.get("N", 20))
        dx = params.get("dx", L / (N - 1))
        dtime = float(params.get("dtime", 0.01))

        # Get current state. Copy so boundary-condition algebra below does not
        # mutate the stored previous-state array in place (the BC updates assign
        # to T[0]/T[N-1] and would otherwise corrupt params['T'] mid-step).
        T = np.array(params.get("T", np.zeros(N)), dtype=float)

        # Get inputs
        q_src = inputs.get(0, 0.0)
        bc_left_val = as_scalar(inputs.get(1, 0.0))
        bc_right_val = as_scalar(inputs.get(2, 0.0))

        # Ensure q_src is array of correct size
        if isinstance(q_src, (int, float)):
            q_src = np.full(N, float(q_src))
        else:
            q_src = np.atleast_1d(q_src).flatten()
            if len(q_src) != N:
                if len(q_src) == 1:
                    q_src = np.full(N, q_src[0])
                else:
                    q_src = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, len(q_src)), q_src)

        # Boundary conditions
        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Dirichlet")
        h_left, h_right = self._robin_coeffs(params, inputs.get(3), inputs.get(4))
        k = float(params.get("k_thermal", 1.0))

        # Spatial discretisation + boundary derivatives are single-sourced in
        # lib.engine.pde_ops. The interpreter integrates Dirichlet/Robin nodes
        # algebraically (it overwrites the field value below), so it uses the
        # 'hold' boundary mode: dT/dt = 0 at those nodes, ghost-node flux at
        # Neumann nodes. dT_dt is computed from the pre-update field T.
        dT_dt = heat_rhs_1d(
            T,
            alpha,
            dx,
            q_src,
            bc_type_left,
            bc_left_val,
            bc_type_right,
            bc_right_val,
            h_left,
            h_right,
            k,
            boundary_mode="hold",
        )

        # Set Dirichlet/Robin boundary values directly on the field (dT_dt is 0
        # there, so the Forward Euler step below leaves them at these values).
        # A periodic rod has no boundary node to pin -- heat_rhs_1d gives both
        # end nodes a real wrapped stencil, so leave the field alone.
        if bc_type_left == "Dirichlet":
            T[0] = bc_left_val
        elif bc_type_left == "Robin":
            T[0] = robin_boundary_value(T[1], bc_left_val, h_left, k, dx)

        if bc_type_right == "Dirichlet":
            T[N - 1] = bc_right_val
        elif bc_type_right == "Robin":
            T[N - 1] = robin_boundary_value(T[N - 2], bc_right_val, h_right, k, dx)

        # Forward Euler time step (simple, for interpreter mode).
        # FTCS is only stable for dtime <= dx^2 / (2*alpha); beyond that the
        # explicit update diverges silently. Warn once if the step is too large
        # so the user can reduce dtime (or use the compiled solver).
        if alpha > 0:
            cfl_limit = dx * dx / (2.0 * alpha)
            if dtime > cfl_limit and not params.get("_cfl_warned_", False):
                logger.warning(
                    "HeatEquation1D: interpreter dtime=%g exceeds FTCS "
                    "stability limit dx^2/(2*alpha)=%g (dx=%g, alpha=%g); "
                    "explicit Forward Euler may diverge. Reduce dtime or use "
                    "the compiled solver.",
                    dtime,
                    cfl_limit,
                    dx,
                    alpha,
                )
                params["_cfl_warned_"] = True
        T_new = T + dT_dt * dtime
        params["T"] = T_new

        # Compute outputs
        T_avg = np.mean(T_new)

        return {0: T_new, 1: T_avg, "E": False}

    def compute_derivatives(self, time, state, inputs, params):
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
        T = state  # signature unified with the 2D PDE blocks
        alpha = float(params.get("alpha", 1.0))
        L = float(params.get("L", 1.0))
        N = int(params.get("N", 20))
        dx = L / (N - 1)

        # Get inputs
        q_src = inputs.get("q_src", 0.0)
        bc_left_val = as_scalar(inputs.get("bc_left", 0.0))
        bc_right_val = as_scalar(inputs.get("bc_right", 0.0))

        # Ensure q_src is array
        if isinstance(q_src, (int, float)):
            q_src = np.full(N, float(q_src))
        else:
            q_src = np.atleast_1d(q_src).flatten()
            if len(q_src) != N:
                q_src = np.full(N, q_src[0] if len(q_src) > 0 else 0.0)

        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Dirichlet")
        h_left, h_right = self._robin_coeffs(params, inputs.get("h_left"), inputs.get("h_right"))
        k = float(params.get("k_thermal", 1.0))

        # 'hold' boundary mode: Dirichlet/Robin nodes are algebraically
        # determined (dT/dt = 0), Neumann nodes use the ghost-node flux.
        return heat_rhs_1d(
            T,
            alpha,
            dx,
            q_src,
            bc_type_left,
            bc_left_val,
            bc_type_right,
            bc_right_val,
            h_left,
            h_right,
            k,
            boundary_mode="hold",
        )

    def apply_boundary_conditions(self, T, params, inputs):
        """
        Apply boundary conditions to the temperature field.
        Called during ODE solution to enforce Dirichlet/Robin BCs.

        A periodic rod is returned untouched: both end nodes are genuine degrees
        of freedom there, not slaved boundary values.

        Returns:
            Modified T array with BCs applied
        """
        N = len(T)
        L = float(params.get("L", 1.0))
        dx = L / (N - 1)

        bc_left_val = inputs.get("bc_left", 0.0)
        bc_right_val = inputs.get("bc_right", 0.0)
        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Dirichlet")

        T_mod = T.copy()
        if is_periodic(bc_type_left, bc_type_right):
            return T_mod

        k = float(params.get("k_thermal", 1.0))
        h_left, h_right = self._robin_coeffs(params, inputs.get("h_left"), inputs.get("h_right"))

        if bc_type_left == "Dirichlet":
            T_mod[0] = bc_left_val
        elif bc_type_left == "Robin":
            T_mod[0] = robin_boundary_value(T[1], bc_left_val, h_left, k, dx)

        if bc_type_right == "Dirichlet":
            T_mod[N - 1] = bc_right_val
        elif bc_type_right == "Robin":
            T_mod[N - 1] = robin_boundary_value(T[N - 2], bc_right_val, h_right, k, dx)

        return T_mod
