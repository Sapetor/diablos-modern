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
from blocks.pde._compat import as_scalar, as_scalar_opt
from blocks.param_templates import (
    diffusivity_param,
    domain_params_2d,
    init_flag_param,
    pde_2d_init_temp_param,
)
from lib.engine.pde_helpers import bc_params_2d, parse_pde_2d_initial_condition
from lib.engine.pde_ops import heat_rhs_2d

logger = logging.getLogger(__name__)


class HeatEquation2DBlock(BaseBlock):
    """
    2D Heat Equation solver using Method of Lines.

    Converts the 2D heat equation PDE into a system of ODEs by discretizing space.
    Each spatial node (i,j) becomes a state variable.

    Boundary conditions (for each edge):
    - Dirichlet: T(boundary) = value
    - Neumann: ∂T/∂n(boundary) = value (normal derivative)
    - Robin: -k∂T/∂n = h_edge(T - T_inf), per-edge h, T_inf from the bc_* port
    - Periodic: wraps a whole axis (left/right -> x, bottom/top -> y)
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
            "\n- bc_type_*: 'Dirichlet', 'Neumann', 'Robin', or 'Periodic' per edge"
            "\n  ('Periodic' on the left OR right wraps x; on the bottom OR top"
            "\n  wraps y; the opposite edge's setting is then ignored)"
            "\n- h_left/h_right/h_bottom/h_top, k_thermal: per-edge Robin coeffs"
            "\n- init_temp: Initial temperature -- a number or one of 'sinusoidal',"
            "\n  'gaussian', 'hot_spot', 'radial', 'linear', 'step', 'random',"
            "\n  'checkerboard'"
            "\n- seed: Seed for the 'random' IC (0 = not reproducible)"
            "\n\nInputs:"
            "\n- q_src: Heat source (scalar or Nx×Ny array)"
            "\n- bc_left, bc_right, bc_bottom, bc_top: BC values (for a Robin"
            "\n  edge this is the ambient temperature T_inf, so it is already"
            "\n  time-varying)"
            "\n- h_left, h_right, h_bottom, h_top: OPTIONAL time-varying Robin"
            "\n  coefficients; unconnected ports fall back to the h_* params."
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
            **bc_params_2d(include_robin=True),
            **pde_2d_init_temp_param(),
            "seed": {
                "type": "int",
                "default": 0,
                "doc": "Random seed for the 'random' initial condition (0 = random).",
            },
            **init_flag_param(),
        }

    # Input-port index of each edge's optional Robin-coefficient port, in the
    # order the edges appear in the BC ports above.
    _H_PORTS = {"h_left": 5, "h_right": 6, "h_bottom": 7, "h_top": 8}

    @property
    def inputs(self):
        return [
            {"name": "q_src", "type": "array", "doc": "Heat source term"},
            {"name": "bc_left", "type": "float", "doc": "Left boundary value"},
            {"name": "bc_right", "type": "float", "doc": "Right boundary value"},
            {"name": "bc_bottom", "type": "float", "doc": "Bottom boundary value"},
            {"name": "bc_top", "type": "float", "doc": "Top boundary value"},
            {"name": "h_left", "type": "float", "doc": "Left Robin coefficient (optional)"},
            {"name": "h_right", "type": "float", "doc": "Right Robin coefficient (optional)"},
            {"name": "h_bottom", "type": "float", "doc": "Bottom Robin coefficient (optional)"},
            {"name": "h_top", "type": "float", "doc": "Top Robin coefficient (optional)"},
        ]

    @property
    def optional_inputs(self):
        """All inputs are optional - default to 0 (or, for the h ports, to the
        matching h_* param), so diagrams saved with the original five ports load
        and run unchanged."""
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
        """Return initial state vector for the 2D field.

        Delegates to the shared ``parse_pde_2d_initial_condition`` -- the same
        helper ``SystemCompiler.compile_system`` uses to seed the compiled
        state -- so both paths start from an identical field, including a seeded
        'random' one.
        """
        T0 = parse_pde_2d_initial_condition(
            params.get("init_temp", "0.0"),
            int(params.get("Nx", 20)),
            int(params.get("Ny", 20)),
            float(params.get("Lx", 1.0)),
            float(params.get("Ly", 1.0)),
            float(params.get("init_amplitude", 1.0)),
            seed=params.get("seed", 0),
        )
        # State is flattened 2D array in row-major order
        return T0.flatten()

    def _robin_coeffs(self, params, inputs):
        """Resolve the four per-edge Robin h coefficients for this step.

        A connected ``h_*`` input port (indices in ``_H_PORTS``) overrides the
        matching static param; an unconnected one reads ``None`` and falls back.
        Re-resolved on every call, so a connected port is genuinely
        time-varying.
        """
        resolved = []
        for name in ("h_left", "h_right", "h_bottom", "h_top"):
            port_val = inputs.get(self._H_PORTS[name])
            if port_val is None:
                resolved.append(float(params.get(name, 10.0)))
            else:
                resolved.append(as_scalar(port_val))
        return resolved

    def get_state_size(self, params):
        """Return the number of state variables."""
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))
        return Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute temperature field (for non-compiled execution).

        Two callers: the compiled replay supplies the already-integrated field
        via the ``state`` kwarg (just reshape it); pure interpreter mode gets no
        ``state`` and must advance the field itself. The 1D PDE blocks
        self-integrate with Forward Euler and persist their field in ``params``;
        the 2D blocks now do the same, so the interpreter no longer leaves the
        field frozen at its initial condition.
        """
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))

        state = kwargs.get("state", None)
        if state is None:
            state = self._interp_step(time, inputs, params)

        T_field = np.asarray(state, dtype=float).reshape((Ny, Nx))
        return {
            0: T_field,
            1: float(np.mean(T_field)),
            2: float(np.max(T_field)),
            "E": False,
        }

    def _interp_step(self, time, inputs, params):
        """Return the current interpreter-mode field, then advance and persist it
        by one Forward-Euler step for the next call. The first call returns the
        initial condition unstepped so the field is sample-aligned with the
        compiled path (which records the IC at t=0). FTCS is only stable for
        dtime <= min(dx,dy)^2 / (4*alpha); beyond that the explicit update
        diverges -- use the compiled solver for stiff / fine grids."""
        if params.get("_init_start_", True):
            params["_interp_state_"] = self.get_initial_state(params)
            params["_init_start_"] = False
            return params["_interp_state_"]

        state = np.asarray(params["_interp_state_"], dtype=float)
        dtime = float(params.get("dtime", 0.01))
        dstate = self.compute_derivatives(time, state, inputs, params)
        state = state + np.asarray(dstate, dtype=float) * dtime
        params["_interp_state_"] = state
        return state

    def compute_derivatives(self, time, state, inputs, params):
        """
        Compute dT/dt for all nodes using 2D finite differences.

        Uses 5-point stencil for Laplacian:
        ∇²T ≈ (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4*T[i,j]) / h²
        """
        alpha = float(params.get("alpha", 0.01))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Get boundary conditions. Signals arrive as 1-element arrays from a
        # connected source, so coerce via as_scalar -- a bare float() raises
        # "only 0-dimensional arrays can be converted to Python scalars" under
        # NumPy 2.x and silently took element 0 under 1.x.
        bc_left = as_scalar_opt(inputs.get(1))
        bc_right = as_scalar_opt(inputs.get(2))
        bc_bottom = as_scalar_opt(inputs.get(3))
        bc_top = as_scalar_opt(inputs.get(4))

        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Dirichlet")
        bc_type_bottom = params.get("bc_type_bottom", "Dirichlet")
        bc_type_top = params.get("bc_type_top", "Dirichlet")

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

        # Reshape state to 2D. Spatial discretisation + boundary conditions are
        # single-sourced in lib.engine.pde_ops (shared with the compiled kernel).
        h_left, h_right, h_bottom, h_top = self._robin_coeffs(params, inputs)
        T = state.reshape((Ny, Nx))
        dT_dt = heat_rhs_2d(
            T,
            alpha,
            dx,
            dy,
            q_src,
            bc_type_left,
            bc_type_right,
            bc_type_bottom,
            bc_type_top,
            bc_left,
            bc_right,
            bc_bottom,
            bc_top,
            h_left,
            h_right,
            h_bottom,
            h_top,
            float(params.get("k_thermal", 1.0)),
        )

        return dT_dt.flatten()
