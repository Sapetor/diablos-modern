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
from lib.engine.pde_ops import (
    advection_rhs_1d, advection_rhs_2d,
    diffusion_reaction_rhs_1d,
    heat_rhs_1d, heat_rhs_2d, wave_rhs_1d, wave_rhs_2d,
)


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

        # Spatial discretisation + boundary conditions are single-sourced in
        # lib.engine.pde_ops. The compiled path integrates the boundary nodes as
        # stiff ODEs, so it uses the 'penalty' boundary mode (Robin/Dirichlet
        # nodes are driven toward their prescribed / Robin-consistent value).
        dT_dt = heat_rhs_1d(
            T, _alpha, _dx, q_src,
            _bc_type_left, bc_left_val, _bc_type_right, bc_right_val,
            _h_left, _h_right, _k, boundary_mode='penalty')

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

        # Spatial discretisation + boundary conditions single-sourced in
        # lib.engine.pde_ops (shared with the interpreter block).
        du_dt, dv_dt = wave_rhs_1d(
            u, v, _c, _damping, _dx, force,
            _bc_type_left, bc_left, _bc_type_right, bc_right)

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

        dc_dt = advection_rhs_1d(c, _v, _dx, c_inlet, _bc_type,
                                 boundary_mode='penalty')

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

        # Spatial discretisation + boundary conditions single-sourced in
        # lib.engine.pde_ops. The compiled path integrates the boundary nodes as
        # stiff ODEs, so Dirichlet uses the 'penalty' boundary mode.
        dc_dt = diffusion_reaction_rhs_1d(
            c, _D, _k, _n, _dx, source,
            _bc_type_left, bc_left, _bc_type_right, bc_right,
            boundary_mode='penalty')

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

        # Spatial discretisation + boundary conditions single-sourced in
        # lib.engine.pde_ops (shared with the interpreter block).
        dT_dt = heat_rhs_2d(
            T, _alpha, _dx, _dy, q_src,
            _bc_type_left, _bc_type_right, _bc_type_bottom, _bc_type_top,
            bc_left, bc_right, bc_bottom, bc_top)

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
                    _start=start, _Nx=Nx, _Ny=Ny, _c=c_wave, _c_sq=c_wave*c_wave,
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

        # Spatial discretisation + boundary conditions single-sourced in
        # lib.engine.pde_ops (shared with the interpreter block).
        du_dt, dv_dt = wave_rhs_2d(
            u, v, _c, _damping, _dx, _dy, force,
            _bc_type_left, _bc_type_right, _bc_type_bottom, _bc_type_top,
            bc_left, bc_right, bc_bottom, bc_top)

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

        dc_dt = advection_rhs_2d(
            c, _vx, _vy, _D, _dx, _dy, source,
            _bc_type_left, _bc_type_right, _bc_type_bottom, _bc_type_top,
            bc_left, bc_right, bc_bottom, bc_top)

        signals[b_name] = c
        signals[b_name + '_avg'] = np.mean(c)
        signals[b_name + '_max'] = np.max(c)

        dy_vec[_start:_start + n_states] = dc_dt.flatten()
    return exec_advection2d
