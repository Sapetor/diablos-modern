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
from lib.engine.pde_ops import advection_rhs_2d

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
            "vx": {"type": "float", "default": 1.0, "doc": "X-velocity [m/s]"},
            "vy": {"type": "float", "default": 0.0, "doc": "Y-velocity [m/s]"},
            "D": {"type": "float", "default": 0.0, "doc": "Diffusion coefficient [m²/s]"},
            "Lx": {"type": "float", "default": 1.0, "doc": "Domain length in x [m]"},
            "Ly": {"type": "float", "default": 1.0, "doc": "Domain length in y [m]"},
            "Nx": {"type": "int", "default": 30, "doc": "Number of nodes in x direction"},
            "Ny": {"type": "int", "default": 30, "doc": "Number of nodes in y direction"},
            "bc_type_left": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Left BC: Dirichlet, Neumann, or Outflow",
            },
            "bc_type_right": {
                "type": "string",
                "default": "Outflow",
                "doc": "Right BC: Dirichlet, Neumann, or Outflow",
            },
            "bc_type_bottom": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Bottom BC: Dirichlet, Neumann, or Outflow",
            },
            "bc_type_top": {
                "type": "string",
                "default": "Dirichlet",
                "doc": "Top BC: Dirichlet, Neumann, or Outflow",
            },
            "init_concentration": {
                "type": "string",
                "default": "0.0",
                "doc": "Initial concentration: number, 'gaussian', 'step', or 'pulse'",
            },
            "init_amplitude": {
                "type": "float",
                "default": 1.0,
                "doc": "Amplitude for non-uniform initial conditions",
            },
            "_init_start_": {
                "type": "bool",
                "default": True,
                "doc": "Internal: initialization flag",
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
        Nx = int(params.get("Nx", 30))
        Ny = int(params.get("Ny", 30))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        init_conc = params.get("init_concentration", "0.0")
        amplitude = float(params.get("init_amplitude", 1.0))

        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)

        if isinstance(init_conc, str):
            if init_conc.lower() == "gaussian":
                # Gaussian pulse at center
                sigma = min(Lx, Ly) / 10
                c0 = amplitude * np.exp(-((X - Lx / 2) ** 2 + (Y - Ly / 2) ** 2) / (2 * sigma**2))
            elif init_conc.lower() == "step":
                # Step function: 1 on left quarter, 0 elsewhere
                c0 = amplitude * (X < Lx / 4).astype(float)
            elif init_conc.lower() == "pulse":
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
        Nx = int(params.get("Nx", 30))
        Ny = int(params.get("Ny", 30))
        return Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute concentration field (for non-compiled execution).

        The compiled replay supplies the integrated field via the ``state``
        kwarg; pure interpreter mode gets none and advances the field itself
        with Forward Euler, persisting it in ``params`` (the 1D PDE blocks do
        the same). Without this the interpreter left the field frozen at its
        initial condition.
        """
        Nx = int(params.get("Nx", 30))
        Ny = int(params.get("Ny", 30))

        state = kwargs.get("state", None)
        if state is None:
            state = self._interp_step(time, inputs, params)

        c_field = np.asarray(state, dtype=float).reshape((Ny, Nx))
        return {0: c_field, 1: float(np.mean(c_field)), 2: float(np.max(c_field)), "E": False}

    def _interp_step(self, time, inputs, params):
        """Return the current interpreter-mode concentration field, then advance
        and persist it by one Forward-Euler step. The first call returns the
        initial condition unstepped so samples align with the compiled path."""
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
        Compute dc/dt for all nodes using upwind scheme for advection
        and central differences for diffusion.
        """
        vx = float(params.get("vx", 1.0))
        vy = float(params.get("vy", 0.0))
        D = float(params.get("D", 0.0))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        Nx = int(params.get("Nx", 30))
        Ny = int(params.get("Ny", 30))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Get boundary conditions
        bc_left = float(inputs.get(1, 0.0)) if inputs.get(1) is not None else 0.0
        bc_right = float(inputs.get(2, 0.0)) if inputs.get(2) is not None else 0.0
        bc_bottom = float(inputs.get(3, 0.0)) if inputs.get(3) is not None else 0.0
        bc_top = float(inputs.get(4, 0.0)) if inputs.get(4) is not None else 0.0

        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Outflow")
        bc_type_bottom = params.get("bc_type_bottom", "Dirichlet")
        bc_type_top = params.get("bc_type_top", "Dirichlet")

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

        # Reshape state to 2D and delegate the spatial discretisation + BCs to
        # the shared operator (single source of truth with the compiled path).
        c = state.reshape((Ny, Nx))
        dc_dt = advection_rhs_2d(
            c,
            vx,
            vy,
            D,
            dx,
            dy,
            source,
            bc_type_left,
            bc_type_right,
            bc_type_bottom,
            bc_type_top,
            bc_left,
            bc_right,
            bc_bottom,
            bc_top,
        )

        return dc_dt.flatten()
