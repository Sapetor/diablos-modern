"""Shared finite-difference / boundary-condition operators for the PDE blocks.

Single source of truth for the spatial discretisation and boundary-condition
math used by BOTH execution paths of the PDE block families:

* the interpreter blocks in ``blocks/pde/`` call these per time step
  (``execute`` / ``compute_derivatives``), and
* the compiled kernel builders in ``lib/engine/compiler_kernels/pde.py`` close
  over them to build the ODE right-hand side.

Keeping one implementation prevents the two paths from drifting apart -- the
Robin-BC correctness divergence recorded in ``tasks/todo.md`` was caused by two
independent copies of this math. Every function here is pure: it takes plain
arrays / scalars and returns a freshly allocated derivative array. Nothing here
touches compiler internals, block instance state, or signal dicts.
"""

import numpy as np

# Dirichlet/Robin penalty stiffness used by the compiled path to drive a
# boundary node toward its prescribed (or Robin-consistent) value.
PENALTY = 1000.0


def robin_boundary_value(T_neighbor, bc_val, h, k, dx):
    """Robin-consistent boundary node value for ``-k dT/dx = h (T - T_inf)``.

    Discretising the gradient with a one-sided difference toward the first
    interior node gives ``T_b = (k T_neighbor / dx + h T_inf) / (k / dx + h)``.
    ``bc_val`` is the ambient value ``T_inf``.
    """
    return (k * T_neighbor / dx + h * bc_val) / (k / dx + h)


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


def heat_rhs_1d(
    T,
    alpha,
    dx,
    q_src,
    bc_type_left,
    bc_val_left,
    bc_type_right,
    bc_val_right,
    h_left,
    h_right,
    k_thermal,
    boundary_mode,
):
    """dT/dt for the 1D heat equation ``T_t = alpha T_xx + q``.

    Args:
        T: length-N temperature field (read only, never mutated).
        alpha: thermal diffusivity.
        dx: grid spacing.
        q_src: length-N heat-source array.
        bc_type_left/right: 'Dirichlet', 'Neumann', or 'Robin'.
        bc_val_left/right: prescribed boundary value (Dirichlet value, Neumann
            flux, or Robin ambient temperature).
        h_left/h_right, k_thermal: Robin convection / conduction coefficients.
        boundary_mode: how Dirichlet & Robin boundaries drive the RHS:
            * 'penalty' -- ``dT/dt = PENALTY (target - T_b)`` (compiled path,
              which integrates the boundary node as a stiff ODE), or
            * 'hold' -- ``dT/dt = 0`` at Dirichlet/Robin nodes (interpreter,
              which instead sets the boundary value on the field directly).
            Neumann boundaries are identical in both modes (ghost-node flux).

    Returns:
        A new length-N ``dT_dt`` array.
    """
    N = len(T)
    dT_dt = np.zeros(N)
    dx_sq = dx * dx

    # Interior nodes: central difference (vectorized; identical to the per-node
    # stencil but avoids the Python loop in the ODE RHS).
    dT_dt[1:-1] = alpha * (T[2:] - 2 * T[1:-1] + T[:-2]) / dx_sq + q_src[1:-1]

    # Left boundary
    if bc_type_left == "Dirichlet":
        dT_dt[0] = PENALTY * (bc_val_left - T[0]) if boundary_mode == "penalty" else 0.0
    elif bc_type_left == "Neumann":
        d2T_dx2 = (2 * T[1] - 2 * T[0] - 2 * dx * bc_val_left) / dx_sq
        dT_dt[0] = alpha * d2T_dx2 + q_src[0]
    elif bc_type_left == "Robin":
        if boundary_mode == "penalty":
            target = robin_boundary_value(T[1], bc_val_left, h_left, k_thermal, dx)
            dT_dt[0] = PENALTY * (target - T[0])
        else:
            dT_dt[0] = 0.0

    # Right boundary
    if bc_type_right == "Dirichlet":
        dT_dt[N - 1] = PENALTY * (bc_val_right - T[N - 1]) if boundary_mode == "penalty" else 0.0
    elif bc_type_right == "Neumann":
        d2T_dx2 = (2 * T[N - 2] - 2 * T[N - 1] + 2 * dx * bc_val_right) / dx_sq
        dT_dt[N - 1] = alpha * d2T_dx2 + q_src[N - 1]
    elif bc_type_right == "Robin":
        if boundary_mode == "penalty":
            target = robin_boundary_value(T[N - 2], bc_val_right, h_right, k_thermal, dx)
            dT_dt[N - 1] = PENALTY * (target - T[N - 1])
        else:
            dT_dt[N - 1] = 0.0

    return dT_dt


def heat_rhs_2d(
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
):
    """dT/dt for the 2D heat equation ``T_t = alpha (T_xx + T_yy) + q``.

    Args:
        T: (Ny, Nx) temperature field (read only, never mutated).
        alpha: thermal diffusivity.
        dx, dy: grid spacing in x and y.
        q_src: heat source -- a scalar or an (Ny, Nx) array.
        bc_type_*: 'Dirichlet' or 'Neumann' per edge.
        bc_left/right/bottom/top: prescribed edge value (Dirichlet value or
            Neumann normal flux).

    Dirichlet edges use the penalty method (both execution paths agree in 2D);
    Neumann edges use the ghost-node flux stencil. All-Neumann corners left
    unset by the edge loops are filled from their edge neighbors.

    Returns:
        A new (Ny, Nx) ``dT_dt`` array.
    """
    Ny, Nx = T.shape
    dT_dt = np.zeros((Ny, Nx))
    dx_sq = dx * dx
    dy_sq = dy * dy
    q_is_arr = isinstance(q_src, np.ndarray)
    q_int = q_src[1:-1, 1:-1] if q_is_arr else q_src

    # Interior nodes: 5-point stencil (vectorized; identical per-node math)
    dT_dt[1:-1, 1:-1] = (
        alpha
        * (
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx_sq
            + (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy_sq
        )
        + q_int
    )

    # Left boundary (i=0)
    if bc_type_left == "Dirichlet":
        for j in range(Ny):
            dT_dt[j, 0] = PENALTY * (bc_left - T[j, 0])
    else:  # Neumann
        for j in range(1, Ny - 1):
            d2Tdx2 = (2 * T[j, 1] - 2 * T[j, 0] - 2 * dx * bc_left) / dx_sq
            d2Tdy2 = (T[j + 1, 0] - 2 * T[j, 0] + T[j - 1, 0]) / dy_sq
            q = q_src[j, 0] if q_is_arr else q_src
            dT_dt[j, 0] = alpha * (d2Tdx2 + d2Tdy2) + q

    # Right boundary (i=Nx-1)
    if bc_type_right == "Dirichlet":
        for j in range(Ny):
            dT_dt[j, Nx - 1] = PENALTY * (bc_right - T[j, Nx - 1])
    else:  # Neumann
        for j in range(1, Ny - 1):
            d2Tdx2 = (2 * T[j, Nx - 2] - 2 * T[j, Nx - 1] + 2 * dx * bc_right) / dx_sq
            d2Tdy2 = (T[j + 1, Nx - 1] - 2 * T[j, Nx - 1] + T[j - 1, Nx - 1]) / dy_sq
            q = q_src[j, Nx - 1] if q_is_arr else q_src
            dT_dt[j, Nx - 1] = alpha * (d2Tdx2 + d2Tdy2) + q

    # Bottom boundary (j=0)
    if bc_type_bottom == "Dirichlet":
        for i in range(Nx):
            dT_dt[0, i] = PENALTY * (bc_bottom - T[0, i])
    else:  # Neumann
        for i in range(1, Nx - 1):
            d2Tdx2 = (T[0, i + 1] - 2 * T[0, i] + T[0, i - 1]) / dx_sq
            d2Tdy2 = (2 * T[1, i] - 2 * T[0, i] - 2 * dy * bc_bottom) / dy_sq
            q = q_src[0, i] if q_is_arr else q_src
            dT_dt[0, i] = alpha * (d2Tdx2 + d2Tdy2) + q

    # Top boundary (j=Ny-1)
    if bc_type_top == "Dirichlet":
        for i in range(Nx):
            dT_dt[Ny - 1, i] = PENALTY * (bc_top - T[Ny - 1, i])
    else:  # Neumann
        for i in range(1, Nx - 1):
            d2Tdx2 = (T[Ny - 1, i + 1] - 2 * T[Ny - 1, i] + T[Ny - 1, i - 1]) / dx_sq
            d2Tdy2 = (2 * T[Ny - 2, i] - 2 * T[Ny - 1, i] + 2 * dy * bc_top) / dy_sq
            q = q_src[Ny - 1, i] if q_is_arr else q_src
            dT_dt[Ny - 1, i] = alpha * (d2Tdx2 + d2Tdy2) + q

    # Fill the all-Neumann corners (skipped by the Neumann edge loops) from their
    # edge neighbors. Dirichlet edges already cover corners via full-range loops.
    _fill_neumann_corners(
        dT_dt,
        Ny,
        Nx,
        bc_type_left != "Dirichlet",
        bc_type_right != "Dirichlet",
        bc_type_bottom != "Dirichlet",
        bc_type_top != "Dirichlet",
    )

    return dT_dt


def wave_rhs_1d(
    u, v, c, damping, dx, force, bc_type_left, bc_val_left, bc_type_right, bc_val_right
):
    """d[u,v]/dt for the 1D wave equation ``u_tt = c^2 u_xx - damping u_t + f``.

    The second-order PDE is a first-order system ``u_t = v``,
    ``v_t = c^2 u_xx - damping v + f``.

    Args:
        u, v: length-N displacement and velocity fields (read only).
        c: wave speed.
        damping: velocity damping coefficient.
        dx: grid spacing.
        force: length-N forcing array.
        bc_type_left/right: 'Dirichlet' or 'Neumann'.
        bc_val_left/right: prescribed boundary value (Dirichlet value or Neumann
            flux).

    Unlike the heat family, both execution paths hold Dirichlet nodes the same
    way here (``du/dt = dv/dt = 0`` at the node; the interpreter's ``execute``
    additionally sets the field value directly), so no boundary-mode kwarg is
    needed. Neumann boundaries use the ghost-node flux stencil.

    Returns:
        A ``(du_dt, dv_dt)`` tuple of freshly allocated length-N arrays.
    """
    N = len(u)
    c_sq = c * c
    dx_sq = dx * dx

    du_dt = v.copy()
    dv_dt = np.zeros(N)

    # Interior (vectorized; matches the compiled kernel's associativity)
    dv_dt[1:-1] = c_sq * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx_sq - damping * v[1:-1] + force[1:-1]

    # Left boundary
    if bc_type_left == "Dirichlet":
        du_dt[0] = 0.0
        dv_dt[0] = 0.0
    elif bc_type_left == "Neumann":
        d2u_dx2 = (2 * u[1] - 2 * u[0] - 2 * dx * bc_val_left) / dx_sq
        dv_dt[0] = c_sq * d2u_dx2 - damping * v[0] + force[0]

    # Right boundary
    if bc_type_right == "Dirichlet":
        du_dt[N - 1] = 0.0
        dv_dt[N - 1] = 0.0
    elif bc_type_right == "Neumann":
        d2u_dx2 = (2 * u[N - 2] - 2 * u[N - 1] + 2 * dx * bc_val_right) / dx_sq
        dv_dt[N - 1] = c_sq * d2u_dx2 - damping * v[N - 1] + force[N - 1]

    return du_dt, dv_dt


def wave_rhs_2d(
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
):
    """d[u,v]/dt for the 2D wave equation ``u_tt = c^2 (u_xx + u_yy) - damping u_t + f``.

    First-order system ``u_t = v``, ``v_t = c^2 (u_xx + u_yy) - damping v + f``.

    Args:
        u, v: (Ny, Nx) displacement and velocity fields (read only).
        c: wave speed.
        damping: velocity damping coefficient.
        dx, dy: grid spacing in x and y.
        force: forcing -- a scalar or an (Ny, Nx) array.
        bc_type_*: 'Dirichlet' or 'Neumann' per edge.
        bc_left/right/bottom/top: prescribed edge value (Dirichlet value or
            Neumann normal flux).

    Both execution paths agree in 2D: Dirichlet edges use the penalty method
    (``du/dt = PENALTY (target - u)``, ``dv/dt = 0``) and Neumann edges use the
    ghost-node flux stencil, so no boundary-mode kwarg is needed. All-Neumann
    corners left unset by the edge loops are filled from their edge neighbors.

    Returns:
        A ``(du_dt, dv_dt)`` tuple of freshly allocated (Ny, Nx) arrays.
    """
    Ny, Nx = u.shape
    c_sq = c * c
    dx_sq = dx * dx
    dy_sq = dy * dy
    f_is_arr = isinstance(force, np.ndarray)

    du_dt = v.copy()
    dv_dt = np.zeros((Ny, Nx))

    # Interior: 5-point stencil (vectorized; identical per-node math)
    f_int = force[1:-1, 1:-1] if f_is_arr else force
    dv_dt[1:-1, 1:-1] = (
        c_sq
        * (
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx_sq
            + (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy_sq
        )
        - damping * v[1:-1, 1:-1]
        + f_int
    )

    # Left boundary (i=0)
    if bc_type_left == "Dirichlet":
        for j in range(Ny):
            du_dt[j, 0] = PENALTY * (bc_left - u[j, 0])
            dv_dt[j, 0] = 0.0
    else:  # Neumann
        for j in range(1, Ny - 1):
            d2udx2 = (2 * u[j, 1] - 2 * u[j, 0] - 2 * dx * bc_left) / dx_sq
            d2udy2 = (u[j + 1, 0] - 2 * u[j, 0] + u[j - 1, 0]) / dy_sq
            f = force[j, 0] if f_is_arr else force
            dv_dt[j, 0] = c_sq * (d2udx2 + d2udy2) - damping * v[j, 0] + f

    # Right boundary (i=Nx-1)
    if bc_type_right == "Dirichlet":
        for j in range(Ny):
            du_dt[j, Nx - 1] = PENALTY * (bc_right - u[j, Nx - 1])
            dv_dt[j, Nx - 1] = 0.0
    else:  # Neumann
        for j in range(1, Ny - 1):
            d2udx2 = (2 * u[j, Nx - 2] - 2 * u[j, Nx - 1] + 2 * dx * bc_right) / dx_sq
            d2udy2 = (u[j + 1, Nx - 1] - 2 * u[j, Nx - 1] + u[j - 1, Nx - 1]) / dy_sq
            f = force[j, Nx - 1] if f_is_arr else force
            dv_dt[j, Nx - 1] = c_sq * (d2udx2 + d2udy2) - damping * v[j, Nx - 1] + f

    # Bottom boundary (j=0)
    if bc_type_bottom == "Dirichlet":
        for i in range(Nx):
            du_dt[0, i] = PENALTY * (bc_bottom - u[0, i])
            dv_dt[0, i] = 0.0
    else:  # Neumann
        for i in range(1, Nx - 1):
            d2udx2 = (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1]) / dx_sq
            d2udy2 = (2 * u[1, i] - 2 * u[0, i] - 2 * dy * bc_bottom) / dy_sq
            f = force[0, i] if f_is_arr else force
            dv_dt[0, i] = c_sq * (d2udx2 + d2udy2) - damping * v[0, i] + f

    # Top boundary (j=Ny-1)
    if bc_type_top == "Dirichlet":
        for i in range(Nx):
            du_dt[Ny - 1, i] = PENALTY * (bc_top - u[Ny - 1, i])
            dv_dt[Ny - 1, i] = 0.0
    else:  # Neumann
        for i in range(1, Nx - 1):
            d2udx2 = (u[Ny - 1, i + 1] - 2 * u[Ny - 1, i] + u[Ny - 1, i - 1]) / dx_sq
            d2udy2 = (2 * u[Ny - 2, i] - 2 * u[Ny - 1, i] + 2 * dy * bc_top) / dy_sq
            f = force[Ny - 1, i] if f_is_arr else force
            dv_dt[Ny - 1, i] = c_sq * (d2udx2 + d2udy2) - damping * v[Ny - 1, i] + f

    # Fill the all-Neumann corners of dv_dt from their edge neighbors (du_dt
    # corners are already correct via du_dt = v.copy()).
    _fill_neumann_corners(
        dv_dt,
        Ny,
        Nx,
        bc_type_left != "Dirichlet",
        bc_type_right != "Dirichlet",
        bc_type_bottom != "Dirichlet",
        bc_type_top != "Dirichlet",
    )

    return du_dt, dv_dt


def advection_rhs_1d(c, v, dx, c_inlet, bc_type, boundary_mode):
    """dc/dt for the 1D advection equation ``c_t + v c_x = 0``.

    Second-order upwind differencing (backward for ``v >= 0``, forward for
    ``v < 0``) with a first-order fallback at the single node next to the inlet.

    Args:
        c: length-N concentration field (read only, never mutated).
        v: advection velocity (sign selects the upwind direction).
        dx: grid spacing.
        c_inlet: prescribed inlet concentration (used only by the Dirichlet
            'penalty' branch).
        bc_type: 'Dirichlet' (fixed inlet) or 'Periodic'.
        boundary_mode: how the Dirichlet inlet drives the RHS:
            * 'penalty' -- ``dc/dt = PENALTY (c_inlet - c_b)`` (compiled path,
              which integrates the inlet node as a stiff ODE), or
            * 'hold' -- ``dc/dt = 0`` at the inlet (interpreter, which instead
              sets the inlet value on the field directly).
            'Periodic' boundaries are identical in both modes.

    Returns:
        A new length-N ``dc_dt`` array.
    """
    N = len(c)
    dc_dt = np.zeros(N)

    if v >= 0:
        # Second-order backward difference (upwind); vectorized interior.
        dc_dt[2:] = -v * (3 * c[2:] - 4 * c[1:-1] + c[:-2]) / (2 * dx)
        # First interior point: first-order fallback.
        if N > 1:
            dc_dt[1] = -v * ((c[1] - c[0]) / dx)
        # Left boundary (inlet)
        if bc_type == "Dirichlet":
            dc_dt[0] = PENALTY * (c_inlet - c[0]) if boundary_mode == "penalty" else 0.0
        elif bc_type == "Periodic":
            dc_dt[0] = -v * ((3 * c[0] - 4 * c[N - 1] + c[N - 2]) / (2 * dx))
    else:
        # Second-order forward difference (upwind); vectorized interior.
        dc_dt[:-2] = -v * (-3 * c[:-2] + 4 * c[1:-1] - c[2:]) / (2 * dx)
        # Last interior point: first-order fallback.
        if N > 1:
            dc_dt[N - 2] = -v * ((c[N - 1] - c[N - 2]) / dx)
        # Right boundary (inlet for negative velocity)
        if bc_type == "Dirichlet":
            dc_dt[N - 1] = PENALTY * (c_inlet - c[N - 1]) if boundary_mode == "penalty" else 0.0
        elif bc_type == "Periodic":
            dc_dt[N - 1] = -v * ((-3 * c[N - 1] + 4 * c[0] - c[1]) / (2 * dx))

    return dc_dt


def advection_rhs_2d(
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
):
    """dc/dt for the 2D advection-diffusion equation.

    ``c_t = -vx c_x - vy c_y + D (c_xx + c_yy) + S`` with upwind differencing
    for the advective terms (stability) and central differences for diffusion.

    Args:
        c: (Ny, Nx) concentration field (read only, never mutated).
        vx, vy: velocity components (sign selects the upwind direction).
        D: diffusion coefficient (0 = pure advection).
        dx, dy: grid spacing in x and y.
        source: source term -- a scalar or an (Ny, Nx) array.
        bc_type_*: 'Dirichlet', 'Neumann', or 'Outflow' per edge.
        bc_left/right/bottom/top: prescribed edge value (Dirichlet value or
            Neumann normal gradient).

    Both execution paths agree in 2D: Dirichlet edges use the penalty method,
    Neumann edges use a one-sided stencil (the prescribed gradient enters the
    advective term; the diffusive Laplacian uses a zero-gradient ghost node),
    and Outflow edges copy the edge-normal interior neighbor. So no
    boundary-mode kwarg is needed. The Dirichlet and Outflow edge loops run the
    full index range, so corners take the value of whichever edge is processed
    last (order: left, right, bottom, top); all-Neumann corners left unset are
    filled from their edge neighbors.

    Returns:
        A new (Ny, Nx) ``dc_dt`` array.
    """
    Ny, Nx = c.shape
    dc_dt = np.zeros((Ny, Nx))
    dx_sq = dx * dx
    dy_sq = dy * dy
    src_is_arr = isinstance(source, np.ndarray)

    # Interior: upwind advection + central diffusion (vectorized; vx/vy are
    # constants so the upwind branch is hoisted out).
    ci = c[1:-1, 1:-1]
    if vx >= 0:
        dc_dx = (ci - c[1:-1, :-2]) / dx
    else:
        dc_dx = (c[1:-1, 2:] - ci) / dx
    if vy >= 0:
        dc_dy = (ci - c[:-2, 1:-1]) / dy
    else:
        dc_dy = (c[2:, 1:-1] - ci) / dy
    d2c_dx2 = (c[1:-1, 2:] - 2 * ci + c[1:-1, :-2]) / dx_sq
    d2c_dy2 = (c[2:, 1:-1] - 2 * ci + c[:-2, 1:-1]) / dy_sq
    S_int = source[1:-1, 1:-1] if src_is_arr else source
    dc_dt[1:-1, 1:-1] = -vx * dc_dx - vy * dc_dy + D * (d2c_dx2 + d2c_dy2) + S_int

    # Left boundary (i=0)
    if bc_type_left == "Dirichlet":
        for j in range(Ny):
            dc_dt[j, 0] = PENALTY * (bc_left - c[j, 0])
    elif bc_type_left == "Neumann":
        # NOTE: the prescribed gradient (bc_*) is applied to the advective term
        # (dc_dx) only. The diffusive Laplacian uses a zero-gradient ghost-node
        # form and intentionally ignores bc_* -- i.e. advection Neumann
        # diffusion is treated as zero-gradient, unlike Heat2D's flux-carrying
        # ghost node.
        for j in range(1, Ny - 1):
            dcx = bc_left if vx >= 0 else (c[j, 1] - c[j, 0]) / dx
            dcy = (c[j, 0] - c[j - 1, 0]) / dy if vy >= 0 else (c[j + 1, 0] - c[j, 0]) / dy
            d2x = (c[j, 1] - c[j, 0]) / dx_sq * 2
            d2y = (c[j + 1, 0] - 2 * c[j, 0] + c[j - 1, 0]) / dy_sq
            S = source[j, 0] if src_is_arr else source
            dc_dt[j, 0] = -vx * dcx - vy * dcy + D * (d2x + d2y) + S
    else:  # Outflow
        for j in range(Ny):
            dc_dt[j, 0] = dc_dt[j, 1] if Nx > 1 else 0.0

    # Right boundary (i=Nx-1)
    if bc_type_right == "Dirichlet":
        for j in range(Ny):
            dc_dt[j, Nx - 1] = PENALTY * (bc_right - c[j, Nx - 1])
    elif bc_type_right == "Neumann":
        for j in range(1, Ny - 1):
            dcx = (c[j, Nx - 1] - c[j, Nx - 2]) / dx if vx >= 0 else bc_right
            dcy = (
                (c[j, Nx - 1] - c[j - 1, Nx - 1]) / dy
                if vy >= 0
                else (c[j + 1, Nx - 1] - c[j, Nx - 1]) / dy
            )
            d2x = (c[j, Nx - 2] - c[j, Nx - 1]) / dx_sq * 2
            d2y = (c[j + 1, Nx - 1] - 2 * c[j, Nx - 1] + c[j - 1, Nx - 1]) / dy_sq
            S = source[j, Nx - 1] if src_is_arr else source
            dc_dt[j, Nx - 1] = -vx * dcx - vy * dcy + D * (d2x + d2y) + S
    else:  # Outflow
        for j in range(Ny):
            dc_dt[j, Nx - 1] = dc_dt[j, Nx - 2] if Nx > 1 else 0.0

    # Bottom boundary (j=0)
    if bc_type_bottom == "Dirichlet":
        for i in range(Nx):
            dc_dt[0, i] = PENALTY * (bc_bottom - c[0, i])
    elif bc_type_bottom == "Neumann":
        for i in range(1, Nx - 1):
            dcx = (c[0, i] - c[0, i - 1]) / dx if vx >= 0 else (c[0, i + 1] - c[0, i]) / dx
            dcy = bc_bottom if vy >= 0 else (c[1, i] - c[0, i]) / dy
            d2x = (c[0, i + 1] - 2 * c[0, i] + c[0, i - 1]) / dx_sq
            d2y = (c[1, i] - c[0, i]) / dy_sq * 2
            S = source[0, i] if src_is_arr else source
            dc_dt[0, i] = -vx * dcx - vy * dcy + D * (d2x + d2y) + S
    else:  # Outflow
        for i in range(Nx):
            dc_dt[0, i] = dc_dt[1, i] if Ny > 1 else 0.0

    # Top boundary (j=Ny-1)
    if bc_type_top == "Dirichlet":
        for i in range(Nx):
            dc_dt[Ny - 1, i] = PENALTY * (bc_top - c[Ny - 1, i])
    elif bc_type_top == "Neumann":
        for i in range(1, Nx - 1):
            dcx = (
                (c[Ny - 1, i] - c[Ny - 1, i - 1]) / dx
                if vx >= 0
                else (c[Ny - 1, i + 1] - c[Ny - 1, i]) / dx
            )
            dcy = (c[Ny - 1, i] - c[Ny - 2, i]) / dy if vy >= 0 else bc_top
            d2x = (c[Ny - 1, i + 1] - 2 * c[Ny - 1, i] + c[Ny - 1, i - 1]) / dx_sq
            d2y = (c[Ny - 2, i] - c[Ny - 1, i]) / dy_sq * 2
            S = source[Ny - 1, i] if src_is_arr else source
            dc_dt[Ny - 1, i] = -vx * dcx - vy * dcy + D * (d2x + d2y) + S
    else:  # Outflow
        for i in range(Nx):
            dc_dt[Ny - 1, i] = dc_dt[Ny - 2, i] if Ny > 1 else 0.0

    # Fill the all-Neumann corners from their edge neighbors. Dirichlet and
    # Outflow edges already cover corners via their full-range loops.
    _fill_neumann_corners(
        dc_dt,
        Ny,
        Nx,
        bc_type_left == "Neumann",
        bc_type_right == "Neumann",
        bc_type_bottom == "Neumann",
        bc_type_top == "Neumann",
    )

    return dc_dt


def diffusion_reaction_rhs_1d(
    c, D, k, n, dx, source, bc_type_left, bc_val_left, bc_type_right, bc_val_right, boundary_mode
):
    """dc/dt for the 1D diffusion-reaction equation ``c_t = D c_xx - k c^n + S``.

    Diffusion is a central-difference Laplacian; the reaction sink ``k c^n`` uses
    a non-negativity clamp (``max(c, 0)``) so a slightly negative concentration
    cannot feed back a spurious source.

    Args:
        c: length-N concentration field (read only, never mutated).
        D: diffusion coefficient.
        k: reaction rate constant.
        n: reaction order.
        dx: grid spacing.
        source: length-N source array.
        bc_type_left/right: 'Dirichlet', 'Neumann', or 'Robin'.
        bc_val_left/right: prescribed boundary value (Dirichlet value, Neumann
            flux, or Robin ambient value).
        boundary_mode: how the Dirichlet boundary drives the RHS:
            * 'penalty' -- ``dc/dt = PENALTY (bc - c_b)`` (compiled path, which
              integrates the boundary node as a stiff ODE), or
            * 'hold' -- ``dc/dt = 0`` at the Dirichlet node (interpreter, which
              instead sets the boundary value on the field directly).
            Neumann boundaries are identical in both modes (ghost-node flux).
            Robin holds ``dc/dt = 0`` in both modes: the compiled path never
            discretised a Robin flux here (Robin is not selectable in the UI),
            and the interpreter drives the Robin node purely by setting its
            field value (via ``robin_boundary_value``) rather than the RHS.

    Returns:
        A new length-N ``dc_dt`` array.
    """
    N = len(c)
    dc_dt = np.zeros(N)
    dx_sq = dx * dx

    # Interior nodes: diffusion (central difference) - reaction sink + source.
    d2c_dx2 = (c[2:] - 2 * c[1:-1] + c[:-2]) / dx_sq
    reaction = k * np.power(np.maximum(c[1:-1], 0.0), n)
    dc_dt[1:-1] = D * d2c_dx2 - reaction + source[1:-1]

    # Left boundary
    if bc_type_left == "Dirichlet":
        dc_dt[0] = PENALTY * (bc_val_left - c[0]) if boundary_mode == "penalty" else 0.0
    elif bc_type_left == "Neumann":
        d2c_dx2_l = (2 * c[1] - 2 * c[0] - 2 * dx * bc_val_left) / dx_sq
        reaction_l = k * np.power(max(c[0], 0), n)
        dc_dt[0] = D * d2c_dx2_l - reaction_l + source[0]
    elif bc_type_left == "Robin":
        dc_dt[0] = 0.0

    # Right boundary
    if bc_type_right == "Dirichlet":
        dc_dt[N - 1] = PENALTY * (bc_val_right - c[N - 1]) if boundary_mode == "penalty" else 0.0
    elif bc_type_right == "Neumann":
        d2c_dx2_r = (2 * c[N - 2] - 2 * c[N - 1] + 2 * dx * bc_val_right) / dx_sq
        reaction_r = k * np.power(max(c[N - 1], 0), n)
        dc_dt[N - 1] = D * d2c_dx2_r - reaction_r + source[N - 1]
    elif bc_type_right == "Robin":
        dc_dt[N - 1] = 0.0

    return dc_dt
