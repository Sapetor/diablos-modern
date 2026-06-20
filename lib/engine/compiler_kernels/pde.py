"""Partial-differential-equation block kernels for the compiled path.

Covers the 1D families (Heat, Wave, Advection, DiffusionReaction) and the 2D
families (Heat, Wave, Advection). Each builds the *output executor* that writes
the field/state into the signal dict and the spatial-discretisation derivative
into dy_vec; state allocation and initial conditions stay in
``SystemCompiler.compile_system``. Bodies are verbatim extractions of the
corresponding branches from ``_create_block_executor`` (dedented; shared locals
unpacked from the BuildContext at the top).
"""
import numpy as np

from lib.engine.compiler_kernels import kernel


def _fill_neumann_corners(arr, ny, nx, left_open, right_open, bottom_open, top_open):
    """Assign 2D-PDE corner derivatives for the all-Neumann case.

    The Neumann edge loops use range(1, N-1) and skip the endpoints, so a corner
    is left unset only when BOTH its adjacent edges are "open" (Neumann) -- a
    Dirichlet/Outflow edge already covers corners via its full-range loop. For
    such corners, approximate the derivative as the mean of the two edge
    neighbors so they evolve instead of staying frozen at the initial value.
    The ``*_open`` flags are computed by the caller per its own BC convention
    (heat/wave: ``!= 'Dirichlet'``; advection: ``== 'Neumann'``).
    """
    if ny < 3 or nx < 3:
        return
    if left_open and bottom_open:
        arr[0, 0] = 0.5 * (arr[0, 1] + arr[1, 0])
    if right_open and bottom_open:
        arr[0, nx - 1] = 0.5 * (arr[0, nx - 2] + arr[1, nx - 1])
    if left_open and top_open:
        arr[ny - 1, 0] = 0.5 * (arr[ny - 1, 1] + arr[ny - 2, 0])
    if right_open and top_open:
        arr[ny - 1, nx - 1] = 0.5 * (arr[ny - 1, nx - 2] + arr[ny - 2, nx - 1])


@kernel("Heatequation1D")
def build_heatequation1d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    alpha = float(params.get('alpha', 1.0))
    L = float(params.get('L', 1.0))
    N = int(params.get('N', 20))
    dx = L / (N - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Dirichlet')
    h_left = float(params.get('h_left', 10.0))
    h_right = float(params.get('h_right', 10.0))
    k_thermal = float(params.get('k_thermal', 1.0))

    q_src_key = input_sources[0] if len(input_sources) > 0 else None
    bc_left_key = input_sources[1] if len(input_sources) > 1 else None
    bc_right_key = input_sources[2] if len(input_sources) > 2 else None

    def exec_heat1d(t, y, dy_vec, signals,
                   _start=start, _N=N, _alpha=alpha, _dx=dx,
                   _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                   _h_left=h_left, _h_right=h_right, _k=k_thermal,
                   _q_key=q_src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
        T = y[_start:_start + _N]

        # Get inputs
        q_src = signals.get(_q_key, 0.0) if _q_key else 0.0
        bc_left_val = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right_val = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

        # Ensure q_src is array
        if isinstance(q_src, (int, float)):
            q_src = np.full(_N, float(q_src))
        else:
            q_src = np.atleast_1d(q_src).flatten()
            if len(q_src) != _N:
                q_src = np.full(_N, q_src[0] if len(q_src) > 0 else 0.0)

        dT_dt = np.zeros(_N)
        dx_sq = _dx * _dx

        # Interior nodes: central difference (vectorized; identical to the
        # per-node stencil but avoids the Python loop in the ODE RHS).
        dT_dt[1:-1] = _alpha * (T[2:] - 2 * T[1:-1] + T[:-2]) / dx_sq + q_src[1:-1]

        # Left boundary
        if _bc_type_left == 'Dirichlet':
            # Force boundary to match input value using penalty method
            dT_dt[0] = 1000.0 * (bc_left_val - T[0])
        elif _bc_type_left == 'Neumann':
            d2T_dx2 = (2*T[1] - 2*T[0] - 2*_dx*bc_left_val) / dx_sq
            dT_dt[0] = _alpha * d2T_dx2 + q_src[0]
        elif _bc_type_left == 'Robin':
            # Convective Robin BC (-k dT/dx = h (T - T_inf)). Drive the
            # boundary node toward the Robin-consistent value -- the same
            # relation blocks/pde/heat_equation_1d.py solves algebraically
            # -- via the penalty method, instead of forcing T_inf as a
            # Dirichlet value (which was the wrong physics and diverged
            # from the interpreted path).
            t0_robin = (_k * T[1] / _dx + _h_left * bc_left_val) / (_k / _dx + _h_left)
            dT_dt[0] = 1000.0 * (t0_robin - T[0])

        # Right boundary
        if _bc_type_right == 'Dirichlet':
            # Force boundary to match input value using penalty method
            dT_dt[_N-1] = 1000.0 * (bc_right_val - T[_N-1])
        elif _bc_type_right == 'Neumann':
            d2T_dx2 = (2*T[_N-2] - 2*T[_N-1] + 2*_dx*bc_right_val) / dx_sq
            dT_dt[_N-1] = _alpha * d2T_dx2 + q_src[_N-1]
        elif _bc_type_right == 'Robin':
            # Convective Robin BC, mirror of the left boundary: drive the
            # node toward the Robin-consistent value rather than forcing
            # the ambient temperature as a Dirichlet value.
            tN_robin = (_k * T[_N-2] / _dx + _h_right * bc_right_val) / (_k / _dx + _h_right)
            dT_dt[_N-1] = 1000.0 * (tN_robin - T[_N-1])

        # Output: temperature field and average
        signals[b_name] = T
        signals[b_name + '_avg'] = np.mean(T)

        dy_vec[_start:_start + _N] = dT_dt
    return exec_heat1d


@kernel("Waveequation1D")
def build_waveequation1d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    c = float(params.get('c', 1.0))
    damping = float(params.get('damping', 0.0))
    L = float(params.get('L', 1.0))
    N = int(params.get('N', 50))
    dx = L / (N - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Dirichlet')

    force_key = input_sources[0] if len(input_sources) > 0 else None
    bc_left_key = input_sources[1] if len(input_sources) > 1 else None
    bc_right_key = input_sources[2] if len(input_sources) > 2 else None

    def exec_wave1d(t, y, dy_vec, signals,
                   _start=start, _N=N, _c=c, _damping=damping, _dx=dx,
                   _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                   _f_key=force_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
        u = y[_start:_start + _N]
        v = y[_start + _N:_start + 2*_N]

        # Get inputs
        force = signals.get(_f_key, 0.0) if _f_key else 0.0
        bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

        if isinstance(force, (int, float)):
            force = np.full(_N, float(force))
        else:
            force = np.atleast_1d(force).flatten()
            if len(force) != _N:
                force = np.full(_N, force[0] if len(force) > 0 else 0.0)

        c_sq = _c * _c
        dx_sq = _dx * _dx

        du_dt = v.copy()
        dv_dt = np.zeros(_N)

        # Interior (vectorized; identical to the per-node stencil)
        dv_dt[1:-1] = (c_sq * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx_sq
                       - _damping * v[1:-1] + force[1:-1])

        # Boundaries
        if _bc_type_left == 'Dirichlet':
            du_dt[0] = 0.0
            dv_dt[0] = 0.0
        elif _bc_type_left == 'Neumann':
            d2u_dx2 = (2*u[1] - 2*u[0] - 2*_dx*bc_left) / dx_sq
            dv_dt[0] = c_sq * d2u_dx2 - _damping * v[0] + force[0]

        if _bc_type_right == 'Dirichlet':
            du_dt[_N-1] = 0.0
            dv_dt[_N-1] = 0.0
        elif _bc_type_right == 'Neumann':
            d2u_dx2 = (2*u[_N-2] - 2*u[_N-1] + 2*_dx*bc_right) / dx_sq
            dv_dt[_N-1] = c_sq * d2u_dx2 - _damping * v[_N-1] + force[_N-1]

        signals[b_name] = u
        signals[b_name + '_v'] = v

        dy_vec[_start:_start + _N] = du_dt
        dy_vec[_start + _N:_start + 2*_N] = dv_dt
    return exec_wave1d


@kernel("Advectionequation1D")
def build_advectionequation1d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    velocity = float(params.get('velocity', 1.0))
    L = float(params.get('L', 1.0))
    N = int(params.get('N', 50))
    dx = L / (N - 1)
    bc_type = params.get('bc_type', 'Dirichlet')

    inlet_key = input_sources[0] if len(input_sources) > 0 else None

    def exec_advection1d(t, y, dy_vec, signals,
                        _start=start, _N=N, _v=velocity, _dx=dx,
                        _bc_type=bc_type, _inlet_key=inlet_key):
        c = y[_start:_start + _N]

        c_inlet = signals.get(_inlet_key, 0.0) if _inlet_key else 0.0

        dc_dt = np.zeros(_N)

        if _v >= 0:
            # Second-order backward difference (upwind) - reduces numerical diffusion
            # Interior: (3*c[i] - 4*c[i-1] + c[i-2]) / (2*dx)  (vectorized)
            dc_dt[2:] = -_v * (3 * c[2:] - 4 * c[1:-1] + c[:-2]) / (2 * _dx)
            # First interior point: first-order fallback
            if _N > 1:
                dc_dx = (c[1] - c[0]) / _dx
                dc_dt[1] = -_v * dc_dx
            if _bc_type == 'Dirichlet':
                dc_dt[0] = 1000.0 * (c_inlet - c[0])  # Penalty method for inlet BC
            elif _bc_type == 'Periodic':
                dc_dx = (3*c[0] - 4*c[_N-1] + c[_N-2]) / (2*_dx)
                dc_dt[0] = -_v * dc_dx
        else:
            # Second-order forward difference (upwind)
            # Interior: (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*dx)  (vectorized)
            dc_dt[:-2] = -_v * (-3 * c[:-2] + 4 * c[1:-1] - c[2:]) / (2 * _dx)
            # Last interior point: first-order fallback
            if _N > 1:
                dc_dx = (c[_N-1] - c[_N-2]) / _dx
                dc_dt[_N-2] = -_v * dc_dx
            if _bc_type == 'Dirichlet':
                dc_dt[_N-1] = 1000.0 * (c_inlet - c[_N-1])  # Penalty method for outlet BC
            elif _bc_type == 'Periodic':
                dc_dx = (-3*c[_N-1] + 4*c[0] - c[1]) / (2*_dx)
                dc_dt[_N-1] = -_v * dc_dx

        signals[b_name] = c
        signals[b_name + '_total'] = np.sum(c) * _dx

        dy_vec[_start:_start + _N] = dc_dt
    return exec_advection1d


@kernel("Diffusionreaction1D")
def build_diffusionreaction1d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    D = float(params.get('D', 0.01))
    k = float(params.get('k', 0.1))
    n = int(params.get('n', 1))
    L = float(params.get('L', 1.0))
    N = int(params.get('N', 30))
    dx = L / (N - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Neumann')

    src_key = input_sources[0] if len(input_sources) > 0 else None
    bc_left_key = input_sources[1] if len(input_sources) > 1 else None
    bc_right_key = input_sources[2] if len(input_sources) > 2 else None

    def exec_diffreact1d(t, y, dy_vec, signals,
                        _start=start, _N=N, _D=D, _k=k, _n=n, _dx=dx,
                        _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                        _s_key=src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
        c = y[_start:_start + _N]

        source = signals.get(_s_key, 0.0) if _s_key else 0.0
        bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

        if isinstance(source, (int, float)):
            source = np.full(_N, float(source))
        else:
            source = np.atleast_1d(source).flatten()
            if len(source) != _N:
                source = np.full(_N, source[0] if len(source) > 0 else 0.0)

        dc_dt = np.zeros(_N)
        dx_sq = _dx * _dx

        # Interior (vectorized; identical to the per-node stencil).
        # np.maximum matches the per-element max(c[i], 0) reaction clamp.
        d2c_dx2 = (c[2:] - 2 * c[1:-1] + c[:-2]) / dx_sq
        reaction = _k * np.power(np.maximum(c[1:-1], 0.0), _n)
        dc_dt[1:-1] = _D * d2c_dx2 - reaction + source[1:-1]

        # Boundaries - use penalty method for Dirichlet to force value
        if _bc_type_left == 'Dirichlet':
            dc_dt[0] = 1000.0 * (bc_left - c[0])  # Force c[0] → bc_left
        elif _bc_type_left == 'Neumann':
            d2c_dx2 = (2*c[1] - 2*c[0] - 2*_dx*bc_left) / dx_sq
            reaction = _k * np.power(max(c[0], 0), _n)
            dc_dt[0] = _D * d2c_dx2 - reaction + source[0]

        if _bc_type_right == 'Dirichlet':
            dc_dt[_N-1] = 1000.0 * (bc_right - c[_N-1])  # Force c[N-1] → bc_right
        elif _bc_type_right == 'Neumann':
            d2c_dx2 = (2*c[_N-2] - 2*c[_N-1] + 2*_dx*bc_right) / dx_sq
            reaction = _k * np.power(max(c[_N-1], 0), _n)
            dc_dt[_N-1] = _D * d2c_dx2 - reaction + source[_N-1]

        signals[b_name] = c
        signals[b_name + '_total'] = np.sum(c) * _dx

        dy_vec[_start:_start + _N] = dc_dt
    return exec_diffreact1d


@kernel("Heatequation2D")
def build_heatequation2d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    alpha = float(params.get('alpha', 0.01))
    Lx = float(params.get('Lx', 1.0))
    Ly = float(params.get('Ly', 1.0))
    Nx = int(params.get('Nx', 20))
    Ny = int(params.get('Ny', 20))
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Dirichlet')
    bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
    bc_type_top = params.get('bc_type_top', 'Dirichlet')

    q_src_key = input_sources[0] if len(input_sources) > 0 else None
    bc_left_key = input_sources[1] if len(input_sources) > 1 else None
    bc_right_key = input_sources[2] if len(input_sources) > 2 else None
    bc_bottom_key = input_sources[3] if len(input_sources) > 3 else None
    bc_top_key = input_sources[4] if len(input_sources) > 4 else None

    def exec_heat2d(t, y, dy_vec, signals,
                   _start=start, _Nx=Nx, _Ny=Ny, _alpha=alpha, _dx=dx, _dy=dy,
                   _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                   _bc_type_bottom=bc_type_bottom, _bc_type_top=bc_type_top,
                   _q_key=q_src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key,
                   _bc_b_key=bc_bottom_key, _bc_t_key=bc_top_key):
        n_states = _Nx * _Ny
        T_flat = y[_start:_start + n_states]
        T = T_flat.reshape((_Ny, _Nx))

        # Get inputs
        q_src = signals.get(_q_key, 0.0) if _q_key else 0.0
        bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0
        bc_bottom = signals.get(_bc_b_key, 0.0) if _bc_b_key else 0.0
        bc_top = signals.get(_bc_t_key, 0.0) if _bc_t_key else 0.0

        # Ensure q_src is scalar (simplified)
        if isinstance(q_src, np.ndarray):
            q_src = float(q_src.flat[0]) if q_src.size > 0 else 0.0

        dT_dt = np.zeros((_Ny, _Nx))
        dx_sq = _dx * _dx
        dy_sq = _dy * _dy
        penalty = 1000.0

        # Interior nodes: 5-point stencil (vectorized; identical per-node math)
        dT_dt[1:-1, 1:-1] = _alpha * (
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx_sq
            + (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy_sq
        ) + q_src

        # Left boundary (i=0)
        if _bc_type_left == 'Dirichlet':
            for j in range(_Ny):
                dT_dt[j, 0] = penalty * (bc_left - T[j, 0])
        else:  # Neumann
            for j in range(1, _Ny - 1):
                d2Tdx2 = (2*T[j, 1] - 2*T[j, 0] - 2*_dx*bc_left) / dx_sq
                d2Tdy2 = (T[j+1, 0] - 2*T[j, 0] + T[j-1, 0]) / dy_sq
                dT_dt[j, 0] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

        # Right boundary (i=Nx-1)
        if _bc_type_right == 'Dirichlet':
            for j in range(_Ny):
                dT_dt[j, _Nx-1] = penalty * (bc_right - T[j, _Nx-1])
        else:  # Neumann
            for j in range(1, _Ny - 1):
                d2Tdx2 = (2*T[j, _Nx-2] - 2*T[j, _Nx-1] + 2*_dx*bc_right) / dx_sq
                d2Tdy2 = (T[j+1, _Nx-1] - 2*T[j, _Nx-1] + T[j-1, _Nx-1]) / dy_sq
                dT_dt[j, _Nx-1] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

        # Bottom boundary (j=0)
        if _bc_type_bottom == 'Dirichlet':
            for i in range(_Nx):
                dT_dt[0, i] = penalty * (bc_bottom - T[0, i])
        else:  # Neumann
            for i in range(1, _Nx - 1):
                d2Tdx2 = (T[0, i+1] - 2*T[0, i] + T[0, i-1]) / dx_sq
                d2Tdy2 = (2*T[1, i] - 2*T[0, i] - 2*_dy*bc_bottom) / dy_sq
                dT_dt[0, i] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

        # Top boundary (j=Ny-1)
        if _bc_type_top == 'Dirichlet':
            for i in range(_Nx):
                dT_dt[_Ny-1, i] = penalty * (bc_top - T[_Ny-1, i])
        else:  # Neumann
            for i in range(1, _Nx - 1):
                d2Tdx2 = (T[_Ny-1, i+1] - 2*T[_Ny-1, i] + T[_Ny-1, i-1]) / dx_sq
                d2Tdy2 = (2*T[_Ny-2, i] - 2*T[_Ny-1, i] + 2*_dy*bc_top) / dy_sq
                dT_dt[_Ny-1, i] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

        # Fill the all-Neumann corners (skipped by the Neumann edge loops)
        # from their edge neighbors. Dirichlet edges already cover corners.
        _fill_neumann_corners(
            dT_dt, _Ny, _Nx,
            _bc_type_left != 'Dirichlet', _bc_type_right != 'Dirichlet',
            _bc_type_bottom != 'Dirichlet', _bc_type_top != 'Dirichlet')

        # Output: temperature field (2D), average, max
        signals[b_name] = T
        signals[b_name + '_avg'] = np.mean(T)
        signals[b_name + '_max'] = np.max(T)

        dy_vec[_start:_start + n_states] = dT_dt.flatten()
    return exec_heat2d


@kernel("Waveequation2D")
def build_waveequation2d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    c_wave = float(params.get('c', 1.0))
    damping = float(params.get('damping', 0.0))
    Lx = float(params.get('Lx', 1.0))
    Ly = float(params.get('Ly', 1.0))
    Nx = int(params.get('Nx', 20))
    Ny = int(params.get('Ny', 20))
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Dirichlet')
    bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
    bc_type_top = params.get('bc_type_top', 'Dirichlet')

    f_key = input_sources[0] if len(input_sources) > 0 else None
    bc_l_key = input_sources[1] if len(input_sources) > 1 else None
    bc_r_key = input_sources[2] if len(input_sources) > 2 else None
    bc_b_key = input_sources[3] if len(input_sources) > 3 else None
    bc_t_key = input_sources[4] if len(input_sources) > 4 else None

    def exec_wave2d(t, y, dy_vec, signals,
                    _start=start, _Nx=Nx, _Ny=Ny, _c_sq=c_wave*c_wave,
                    _damping=damping, _dx=dx, _dy=dy,
                    _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                    _bc_type_bottom=bc_type_bottom, _bc_type_top=bc_type_top,
                    _f_key=f_key, _bc_l_key=bc_l_key, _bc_r_key=bc_r_key,
                    _bc_b_key=bc_b_key, _bc_t_key=bc_t_key):
        N = _Nx * _Ny
        u_flat = y[_start:_start + N]
        v_flat = y[_start + N:_start + 2*N]
        u = u_flat.reshape((_Ny, _Nx))
        v = v_flat.reshape((_Ny, _Nx))

        force = signals.get(_f_key, 0.0) if _f_key else 0.0
        bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0
        bc_bottom = signals.get(_bc_b_key, 0.0) if _bc_b_key else 0.0
        bc_top = signals.get(_bc_t_key, 0.0) if _bc_t_key else 0.0

        if isinstance(force, np.ndarray):
            if force.size == 1:
                force = float(force.flat[0])
            elif force.shape != (_Ny, _Nx):
                # Downstream indexes force[j, i] as a (Ny, Nx) grid. A
                # connected source of any other shape would mis-index or
                # raise inside the RHS, so broadcast it to the grid when
                # possible, else fall back to a scalar (its first value).
                try:
                    force = np.broadcast_to(force, (_Ny, _Nx))
                except ValueError:
                    force = float(np.atleast_1d(force).flat[0])

        du_dt = v.copy()
        dv_dt = np.zeros((_Ny, _Nx))
        dx_sq = _dx * _dx
        dy_sq = _dy * _dy
        penalty = 1000.0

        # Interior: 5-point stencil (vectorized; identical per-node math)
        _f_int = force[1:-1, 1:-1] if isinstance(force, np.ndarray) else force
        dv_dt[1:-1, 1:-1] = (_c_sq * (
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx_sq
            + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy_sq)
            - _damping * v[1:-1, 1:-1] + _f_int)

        # Left boundary (i=0)
        if _bc_type_left == 'Dirichlet':
            for j in range(_Ny):
                du_dt[j, 0] = penalty * (bc_left - u[j, 0])
                dv_dt[j, 0] = 0.0
        else:
            for j in range(1, _Ny - 1):
                d2udx2 = (2*u[j, 1] - 2*u[j, 0] - 2*_dx*bc_left) / dx_sq
                d2udy2 = (u[j+1, 0] - 2*u[j, 0] + u[j-1, 0]) / dy_sq
                f = force[j, 0] if isinstance(force, np.ndarray) else force
                dv_dt[j, 0] = _c_sq * (d2udx2 + d2udy2) - _damping * v[j, 0] + f

        # Right boundary (i=Nx-1)
        if _bc_type_right == 'Dirichlet':
            for j in range(_Ny):
                du_dt[j, _Nx-1] = penalty * (bc_right - u[j, _Nx-1])
                dv_dt[j, _Nx-1] = 0.0
        else:
            for j in range(1, _Ny - 1):
                d2udx2 = (2*u[j, _Nx-2] - 2*u[j, _Nx-1] + 2*_dx*bc_right) / dx_sq
                d2udy2 = (u[j+1, _Nx-1] - 2*u[j, _Nx-1] + u[j-1, _Nx-1]) / dy_sq
                f = force[j, _Nx-1] if isinstance(force, np.ndarray) else force
                dv_dt[j, _Nx-1] = _c_sq * (d2udx2 + d2udy2) - _damping * v[j, _Nx-1] + f

        # Bottom boundary (j=0)
        if _bc_type_bottom == 'Dirichlet':
            for i in range(_Nx):
                du_dt[0, i] = penalty * (bc_bottom - u[0, i])
                dv_dt[0, i] = 0.0
        else:
            for i in range(1, _Nx - 1):
                d2udx2 = (u[0, i+1] - 2*u[0, i] + u[0, i-1]) / dx_sq
                d2udy2 = (2*u[1, i] - 2*u[0, i] - 2*_dy*bc_bottom) / dy_sq
                f = force[0, i] if isinstance(force, np.ndarray) else force
                dv_dt[0, i] = _c_sq * (d2udx2 + d2udy2) - _damping * v[0, i] + f

        # Top boundary (j=Ny-1)
        if _bc_type_top == 'Dirichlet':
            for i in range(_Nx):
                du_dt[_Ny-1, i] = penalty * (bc_top - u[_Ny-1, i])
                dv_dt[_Ny-1, i] = 0.0
        else:
            for i in range(1, _Nx - 1):
                d2udx2 = (u[_Ny-1, i+1] - 2*u[_Ny-1, i] + u[_Ny-1, i-1]) / dx_sq
                d2udy2 = (2*u[_Ny-2, i] - 2*u[_Ny-1, i] + 2*_dy*bc_top) / dy_sq
                f = force[_Ny-1, i] if isinstance(force, np.ndarray) else force
                dv_dt[_Ny-1, i] = _c_sq * (d2udx2 + d2udy2) - _damping * v[_Ny-1, i] + f

        # Fill the all-Neumann corners of dv_dt from their edge neighbors
        # (du_dt corners are already correct via du_dt = v.copy()).
        _fill_neumann_corners(
            dv_dt, _Ny, _Nx,
            _bc_type_left != 'Dirichlet', _bc_type_right != 'Dirichlet',
            _bc_type_bottom != 'Dirichlet', _bc_type_top != 'Dirichlet')

        signals[b_name] = u
        signals[b_name + '_v'] = v
        # Energy: 0.5 * sum(v^2) * dA + 0.5 * c^2 * sum(|grad u|^2) * dA
        dA = _dx * _dy
        du_dx_arr = np.gradient(u, _dx, axis=1)
        du_dy_arr = np.gradient(u, _dy, axis=0)
        energy = 0.5 * np.sum(v**2) * dA + 0.5 * _c_sq * np.sum(du_dx_arr**2 + du_dy_arr**2) * dA
        signals[b_name + '_energy'] = float(energy)

        dy_vec[_start:_start + N] = du_dt.flatten()
        dy_vec[_start + N:_start + 2*N] = dv_dt.flatten()
    return exec_wave2d


@kernel("Advectionequation2D")
def build_advectionequation2d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    vx = float(params.get('vx', 1.0))
    vy = float(params.get('vy', 0.0))
    D_coeff = float(params.get('D', 0.0))
    Lx = float(params.get('Lx', 1.0))
    Ly = float(params.get('Ly', 1.0))
    Nx = int(params.get('Nx', 30))
    Ny = int(params.get('Ny', 30))
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    bc_type_left = params.get('bc_type_left', 'Dirichlet')
    bc_type_right = params.get('bc_type_right', 'Outflow')
    bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
    bc_type_top = params.get('bc_type_top', 'Dirichlet')

    s_key = input_sources[0] if len(input_sources) > 0 else None
    bc_l_key = input_sources[1] if len(input_sources) > 1 else None
    bc_r_key = input_sources[2] if len(input_sources) > 2 else None
    bc_b_key = input_sources[3] if len(input_sources) > 3 else None
    bc_t_key = input_sources[4] if len(input_sources) > 4 else None

    def exec_advection2d(t, y, dy_vec, signals,
                         _start=start, _Nx=Nx, _Ny=Ny, _vx=vx, _vy=vy,
                         _D=D_coeff, _dx=dx, _dy=dy,
                         _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                         _bc_type_bottom=bc_type_bottom, _bc_type_top=bc_type_top,
                         _s_key=s_key, _bc_l_key=bc_l_key, _bc_r_key=bc_r_key,
                         _bc_b_key=bc_b_key, _bc_t_key=bc_t_key):
        n_states = _Nx * _Ny
        c_flat = y[_start:_start + n_states]
        c = c_flat.reshape((_Ny, _Nx))

        source = signals.get(_s_key, 0.0) if _s_key else 0.0
        bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
        bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0
        bc_bottom = signals.get(_bc_b_key, 0.0) if _bc_b_key else 0.0
        bc_top = signals.get(_bc_t_key, 0.0) if _bc_t_key else 0.0

        if isinstance(source, np.ndarray):
            if source.size == 1:
                source = float(source.flat[0])
            elif source.shape != (_Ny, _Nx):
                # Downstream indexes source[j, i] as a (Ny, Nx) grid.
                # Broadcast any other shape to the grid when possible,
                # else fall back to a scalar (its first value), so the
                # RHS never mis-indexes or raises.
                try:
                    source = np.broadcast_to(source, (_Ny, _Nx))
                except ValueError:
                    source = float(np.atleast_1d(source).flat[0])

        dc_dt = np.zeros((_Ny, _Nx))
        dx_sq = _dx * _dx
        dy_sq = _dy * _dy
        penalty = 1000.0

        # Interior: upwind advection + central diffusion (vectorized;
        # _vx/_vy are constants so the upwind branch is hoisted out).
        _ci = c[1:-1, 1:-1]
        if _vx >= 0:
            _dc_dx = (_ci - c[1:-1, :-2]) / _dx
        else:
            _dc_dx = (c[1:-1, 2:] - _ci) / _dx
        if _vy >= 0:
            _dc_dy = (_ci - c[:-2, 1:-1]) / _dy
        else:
            _dc_dy = (c[2:, 1:-1] - _ci) / _dy
        _d2c_dx2 = (c[1:-1, 2:] - 2 * _ci + c[1:-1, :-2]) / dx_sq
        _d2c_dy2 = (c[2:, 1:-1] - 2 * _ci + c[:-2, 1:-1]) / dy_sq
        _S_int = source[1:-1, 1:-1] if isinstance(source, np.ndarray) else source
        dc_dt[1:-1, 1:-1] = (-_vx * _dc_dx - _vy * _dc_dy
                             + _D * (_d2c_dx2 + _d2c_dy2) + _S_int)

        # Left boundary (i=0)
        if _bc_type_left == 'Dirichlet':
            for j in range(_Ny):
                dc_dt[j, 0] = penalty * (bc_left - c[j, 0])
        elif _bc_type_left == 'Neumann':
            # NOTE: the prescribed-gradient (bc_*) is applied to the
            # advective term (dc_dx) only. The diffusive Laplacian here
            # uses a zero-gradient ghost-node form and intentionally
            # ignores bc_* — i.e. advection Neumann diffusion is treated
            # as zero-gradient, unlike Heat2D's flux-carrying ghost node.
            for j in range(1, _Ny - 1):
                dc_dx = bc_left if _vx >= 0 else (c[j, 1] - c[j, 0]) / _dx
                dc_dy = (c[j, 0] - c[j-1, 0]) / _dy if _vy >= 0 else (c[j+1, 0] - c[j, 0]) / _dy
                d2c_dx2 = (c[j, 1] - c[j, 0]) / dx_sq * 2
                d2c_dy2 = (c[j+1, 0] - 2*c[j, 0] + c[j-1, 0]) / dy_sq
                S = source[j, 0] if isinstance(source, np.ndarray) else source
                dc_dt[j, 0] = -_vx * dc_dx - _vy * dc_dy + _D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for j in range(_Ny):
                dc_dt[j, 0] = dc_dt[j, 1] if _Nx > 1 else 0.0

        # Right boundary (i=Nx-1)
        if _bc_type_right == 'Dirichlet':
            for j in range(_Ny):
                dc_dt[j, _Nx-1] = penalty * (bc_right - c[j, _Nx-1])
        elif _bc_type_right == 'Neumann':
            for j in range(1, _Ny - 1):
                dc_dx = (c[j, _Nx-1] - c[j, _Nx-2]) / _dx if _vx >= 0 else bc_right
                dc_dy = (c[j, _Nx-1] - c[j-1, _Nx-1]) / _dy if _vy >= 0 else (c[j+1, _Nx-1] - c[j, _Nx-1]) / _dy
                d2c_dx2 = (c[j, _Nx-2] - c[j, _Nx-1]) / dx_sq * 2
                d2c_dy2 = (c[j+1, _Nx-1] - 2*c[j, _Nx-1] + c[j-1, _Nx-1]) / dy_sq
                S = source[j, _Nx-1] if isinstance(source, np.ndarray) else source
                dc_dt[j, _Nx-1] = -_vx * dc_dx - _vy * dc_dy + _D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for j in range(_Ny):
                dc_dt[j, _Nx-1] = dc_dt[j, _Nx-2] if _Nx > 1 else 0.0

        # Bottom boundary (j=0)
        if _bc_type_bottom == 'Dirichlet':
            for i in range(_Nx):
                dc_dt[0, i] = penalty * (bc_bottom - c[0, i])
        elif _bc_type_bottom == 'Neumann':
            for i in range(1, _Nx - 1):
                dc_dx = (c[0, i] - c[0, i-1]) / _dx if _vx >= 0 else (c[0, i+1] - c[0, i]) / _dx
                dc_dy = bc_bottom if _vy >= 0 else (c[1, i] - c[0, i]) / _dy
                d2c_dx2 = (c[0, i+1] - 2*c[0, i] + c[0, i-1]) / dx_sq
                d2c_dy2 = (c[1, i] - c[0, i]) / dy_sq * 2
                S = source[0, i] if isinstance(source, np.ndarray) else source
                dc_dt[0, i] = -_vx * dc_dx - _vy * dc_dy + _D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for i in range(_Nx):
                dc_dt[0, i] = dc_dt[1, i] if _Ny > 1 else 0.0

        # Top boundary (j=Ny-1)
        if _bc_type_top == 'Dirichlet':
            for i in range(_Nx):
                dc_dt[_Ny-1, i] = penalty * (bc_top - c[_Ny-1, i])
        elif _bc_type_top == 'Neumann':
            for i in range(1, _Nx - 1):
                dc_dx = (c[_Ny-1, i] - c[_Ny-1, i-1]) / _dx if _vx >= 0 else (c[_Ny-1, i+1] - c[_Ny-1, i]) / _dx
                dc_dy = (c[_Ny-1, i] - c[_Ny-2, i]) / _dy if _vy >= 0 else bc_top
                d2c_dx2 = (c[_Ny-1, i+1] - 2*c[_Ny-1, i] + c[_Ny-1, i-1]) / dx_sq
                d2c_dy2 = (c[_Ny-2, i] - c[_Ny-1, i]) / dy_sq * 2
                S = source[_Ny-1, i] if isinstance(source, np.ndarray) else source
                dc_dt[_Ny-1, i] = -_vx * dc_dx - _vy * dc_dy + _D * (d2c_dx2 + d2c_dy2) + S
        else:  # Outflow
            for i in range(_Nx):
                dc_dt[_Ny-1, i] = dc_dt[_Ny-2, i] if _Ny > 1 else 0.0

        # Fill the all-Neumann corners from their edge neighbors. Dirichlet
        # and Outflow edges already cover corners via their full-range loops.
        _fill_neumann_corners(
            dc_dt, _Ny, _Nx,
            _bc_type_left == 'Neumann', _bc_type_right == 'Neumann',
            _bc_type_bottom == 'Neumann', _bc_type_top == 'Neumann')

        signals[b_name] = c
        signals[b_name + '_avg'] = np.mean(c)
        signals[b_name + '_max'] = np.max(c)

        dy_vec[_start:_start + n_states] = dc_dt.flatten()
    return exec_advection2d
