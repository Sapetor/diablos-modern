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
from blocks.pde._compat import as_scalar_opt
from blocks.param_templates import wave_speed_param, domain_params_2d, init_flag_param
from lib.engine.pde_helpers import bc_params_2d, companion_seed, parse_pde_2d_initial_condition
from lib.engine.pde_ops import wave_rhs_2d

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
        return "PDE"

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
            "\n- bc_type_*: 'Dirichlet', 'Neumann', or 'Periodic' per edge"
            "\n  ('Periodic' on the left OR right wraps x; on the bottom OR top"
            "\n  wraps y; the opposite edge's setting is then ignored)"
            "\n- init_displacement: Initial displacement (number, 'sinusoidal',"
            "\n  'gaussian', 'radial', 'linear', 'step', 'random', 'checkerboard')"
            "\n- init_velocity: Initial velocity (same named patterns)"
            "\n- seed: Seed for the 'random' IC (0 = not reproducible)"
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
            **wave_speed_param(default=1.0),
            "damping": {"type": "float", "default": 0.0, "doc": "Damping coefficient"},
            **domain_params_2d(),
            **bc_params_2d(),
            "init_displacement": {
                "type": "string",
                "default": "0.0",
                "doc": (
                    "Initial displacement: number, 'sinusoidal', 'gaussian', "
                    "'radial', 'linear', 'step', 'random', or 'checkerboard'"
                ),
            },
            "init_velocity": {
                "type": "string",
                "default": "0.0",
                "doc": "Initial velocity: number or the same named patterns",
            },
            "init_amplitude": {
                "type": "float",
                "default": 1.0,
                "doc": "Amplitude for non-uniform initial conditions",
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
        """Return initial state vector [u, v] for the 2D field.

        Delegates to the shared ``parse_pde_2d_initial_condition``. The compiled
        path seeds its state by calling THIS method, so there is one
        implementation for both paths; the velocity field takes a
        ``companion_seed`` so a 'random' displacement and a 'random' velocity
        are independent rather than the same array.
        """
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        amplitude = float(params.get("init_amplitude", 1.0))
        seed = params.get("seed", 0)

        u0 = parse_pde_2d_initial_condition(
            params.get("init_displacement", "0.0"), Nx, Ny, Lx, Ly, amplitude, seed=seed
        )
        v0 = parse_pde_2d_initial_condition(
            params.get("init_velocity", "0.0"),
            Nx,
            Ny,
            Lx,
            Ly,
            amplitude,
            seed=companion_seed(seed),
        )

        # State is [u_flat, v_flat] in row-major order
        return np.concatenate([u0.flatten(), v0.flatten()])

    def get_state_size(self, params):
        """Return the number of state variables (2*Nx*Ny)."""
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))
        return 2 * Nx * Ny

    def execute(self, time, inputs, params, **kwargs):
        """Compute displacement and velocity fields (for non-compiled execution).

        The compiled replay supplies the integrated [u, v] state via the
        ``state`` kwarg; pure interpreter mode gets none and advances the state
        itself with Forward Euler, persisting it in ``params`` (the 1D PDE
        blocks do the same). Without this the interpreter left the field frozen
        at its initial condition.
        """
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))

        state = kwargs.get("state", None)
        if state is None:
            state = self._interp_step(time, inputs, params)
        state = np.asarray(state, dtype=float)

        # Split state into u and v
        N = Nx * Ny
        u_field = state[:N].reshape((Ny, Nx))
        v_field = state[N:].reshape((Ny, Nx))

        energy = self._compute_energy(u_field, v_field, params)

        return {0: u_field, 1: v_field, 2: energy, "E": False}

    def _interp_step(self, time, inputs, params):
        """Return the current interpreter-mode [u, v] state, then advance and
        persist it by one Forward-Euler step. The first call returns the initial
        condition unstepped so samples align with the compiled path."""
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
        Compute d[u,v]/dt for all nodes using 2D finite differences.

        Uses 5-point stencil for Laplacian:
        ∇²u ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h²

        Returns derivatives for first-order system:
        - du/dt = v
        - dv/dt = c²∇²u - damping*v + f
        """
        c = float(params.get("c", 1.0))
        damping = float(params.get("damping", 0.0))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))

        dx = Lx / (Nx - 1)
        dy = Ly / (Ny - 1)

        # Get boundary conditions. A connected source emits a 1-element array,
        # which a bare float() cannot convert under NumPy 2.x; as_scalar_opt
        # also maps an unconnected (None) port to 0.0 rather than NaN.
        bc_left = as_scalar_opt(inputs.get(1))
        bc_right = as_scalar_opt(inputs.get(2))
        bc_bottom = as_scalar_opt(inputs.get(3))
        bc_top = as_scalar_opt(inputs.get(4))

        bc_type_left = params.get("bc_type_left", "Dirichlet")
        bc_type_right = params.get("bc_type_right", "Dirichlet")
        bc_type_bottom = params.get("bc_type_bottom", "Dirichlet")
        bc_type_top = params.get("bc_type_top", "Dirichlet")

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

        # Spatial discretisation + BC math is single-sourced in lib.engine.pde_ops.
        du_dt, dv_dt = wave_rhs_2d(
            u,
            v,
            c,
            damping,
            dx,
            dy,
            force,
            bc_type_left,
            bc_type_right,
            bc_type_bottom,
            bc_type_top,
            bc_left,
            bc_right,
            bc_bottom,
            bc_top,
        )

        # Return flattened derivatives [du_dt, dv_dt]
        return np.concatenate([du_dt.flatten(), dv_dt.flatten()])

    def _compute_energy(self, u_field, v_field, params):
        """
        Compute total wave energy (kinetic + potential).

        Kinetic energy: 0.5 * ∫∫ v² dx dy
        Potential energy: 0.5 * c² * ∫∫ (|∇u|²) dx dy
        """
        Nx = int(params.get("Nx", 20))
        Ny = int(params.get("Ny", 20))
        Lx = float(params.get("Lx", 1.0))
        Ly = float(params.get("Ly", 1.0))
        c = float(params.get("c", 1.0))

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
