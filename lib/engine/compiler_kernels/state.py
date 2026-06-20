"""State-block output kernels for the compiled path.

Covers the *output executor* of the ODE-state blocks: Integrator, StateSpace,
TransferFcn, PID and RateLimiter. State allocation (y0 / state_map /
block_matrices) and the derivative function still live in
``SystemCompiler.compile_system``; only the per-block output closure built by
``_create_block_executor`` is migrated here. Bodies are verbatim extractions;
shared locals (including ``state_map`` and ``block_matrices``) are unpacked from
the BuildContext at the top.
"""
import numpy as np

from lib.engine.compiler_kernels import kernel


@kernel("Integrator")
def build_integrator(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    src = input_sources[0] if input_sources else None

    def exec_integrator(t, y, dy_vec, signals):
        # Output y = state
        if size == 1:
            signals[b_name] = y[start]
            # Derivative dx/dt = input. Upstream signals may be 1-element
            # arrays (e.g. a LogicalOperator/Demux output, or a length-1
            # Constant); in NumPy 2.0 assigning an array into the scalar
            # slot dy_vec[start] raises, so reduce to a scalar here (same
            # guard the scalar StateSpace branch uses).
            val = signals.get(src, 0.0) if src else 0.0
            dy_vec[start] = np.ravel(val)[0] if np.ndim(val) else val
        else:
            signals[b_name] = y[start : start + size]
            # Vector input?
            # Assume input is scalar broadcast or vector
            # For now scalar broadcast if missing logic
            val = signals.get(src, 0.0) if src else 0.0
            dy_vec[start : start + size] = np.atleast_1d(val).flatten()
    return exec_integrator


@kernel("StateSpace", "TransferFcn")
def build_statespace(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    block_matrices = ctx.block_matrices
    if b_name in block_matrices:
        A, B, C, D = block_matrices[b_name]
        start, size = state_map[b_name]
        n_inputs = B.shape[1]
        # Capture all input sources for multi-input blocks
        all_srcs = list(input_sources) if input_sources else []
        src = all_srcs[0] if all_srcs else None

        if size == 1 and n_inputs == 1:
            # Fast scalar path (SISO, 1st-order)
            def exec_ss(t, y, dy_vec, signals):
                u_val = signals.get(src, 0.0) if src else 0.0
                # Upstream signals may be 1-element arrays; in NumPy 2.0
                # assigning an array into dy_vec[start] (a scalar slot)
                # is an error, so reduce to a scalar here.
                u = np.ravel(u_val)[0] if np.ndim(u_val) else u_val
                x_s = y[start]
                dx = A[0,0]*x_s + B[0,0]*u
                y_out = C[0,0]*x_s + D[0,0]*u
                signals[b_name] = y_out
                dy_vec[start] = dx
            return exec_ss
        elif n_inputs == 1:
            # Multi-state, single input
            def exec_ss(t, y, dy_vec, signals):
                x = y[start : start + size].reshape(-1, 1)
                u_val = signals.get(src, 0.0) if src else 0.0
                u = np.atleast_1d(u_val).reshape(-1, 1)
                dx = A @ x + B @ u
                y_out = C @ x + D @ u
                signals[b_name] = y_out.item() if y_out.size == 1 else y_out.flatten()
                dy_vec[start : start + size] = dx.flatten()
            return exec_ss
        else:
            # Multi-input: assemble u vector from all input ports
            # Sources may provide vectors (e.g. Mux output), so unpack
            # them into consecutive u slots.
            def exec_ss(t, y, dy_vec, signals):
                x = y[start : start + size].reshape(-1, 1)
                u = np.zeros((n_inputs, 1))
                idx = 0
                for s in all_srcs:
                    if s:
                        val = signals.get(s, 0.0)
                        v = np.atleast_1d(val).flatten()
                        for j in range(len(v)):
                            if idx < n_inputs:
                                u[idx, 0] = v[j]
                                idx += 1
                    else:
                        idx += 1
                dx = A @ x + B @ u
                y_out = C @ x + D @ u
                signals[b_name] = y_out.item() if y_out.size == 1 else y_out.flatten()
                dy_vec[start : start + size] = dx.flatten()
            return exec_ss

    # No matrices were built for this block (e.g. compilation produced none):
    # mirror the legacy fall-through to the generic no-op executor.
    def exec_noop(t, y, dy_vec, signals):
        pass
    return exec_noop


@kernel("PID")
def build_pid(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    sp_src = input_sources[0] if len(input_sources)>0 else None
    meas_src = input_sources[1] if len(input_sources)>1 else None

    Kp = float(params.get('Kp', 1.0))
    Ki = float(params.get('Ki', 0.0))
    Kd = float(params.get('Kd', 0.0))
    N = float(params.get('N', 20.0))
    u_min = float(params.get('u_min', -np.inf))
    u_max = float(params.get('u_max', np.inf))

    def exec_pid(t, y, dy_vec, signals):
        # PID state is scalar (two slots, dy_vec[start]/[start+1]), so
        # reduce vector inputs to their first element; this also keeps
        # the anti-windup comparison below scalar (a vector `e` would
        # make `if e > 0` ambiguous).
        sp_raw = signals.get(sp_src, 0.0) if sp_src else 0.0
        meas_raw = signals.get(meas_src, 0.0) if meas_src else 0.0
        sp = float(np.ravel(sp_raw)[0])
        meas = float(np.ravel(meas_raw)[0])
        e = sp - meas

        # x_i, x_d
        x_i = y[start]
        x_d = y[start + 1]

        dx_i = e
        dx_d = N * (e - x_d)
        d_term = Kd * dx_d
        i_term = Ki * x_i
        p_term = Kp * e

        u_unsat = p_term + i_term + d_term
        u_out = np.clip(u_unsat, u_min, u_max)
        signals[b_name] = u_out

        # Anti-windup
        if (u_unsat > u_max and e > 0) or (u_unsat < u_min and e < 0):
            dx_i = 0.0

        dy_vec[start] = dx_i
        dy_vec[start+1] = dx_d
    return exec_pid


@kernel("RateLimiter")
def build_ratelimiter(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    state_map = ctx.state_map
    start, size = state_map[b_name]
    src = input_sources[0] if input_sources else None
    # Match the interpreted RateLimiterBlock param keys
    # (rising_slew/falling_slew, default infinite, magnitude-only).
    rising = abs(float(params.get('rising_slew', np.inf)))
    falling = -abs(float(params.get('falling_slew', np.inf)))
    # Compiled approximation: the interpreted block applies exact
    # per-step slew clamping, but inside a continuous ODE RHS we model
    # the limiter as a stiff first-order chase dy = clip((u-y)*K, ...).
    # K is a large stiffness gain so the output tracks u closely while
    # the clip enforces the slew bounds. This diverges slightly from the
    # interpreted path; rerun without the fast solver for exact parity.
    K = 1000.0

    def exec_ratelimiter(t, y, dy_vec, signals):
        u_raw = signals.get(src, 0.0) if src else 0.0
        # RateLimiter has one scalar state slot; reduce a vector input to
        # its first element (mirrors exec_integrator / scalar exec_ss) so
        # dy_vec[start] = dy never receives a sequence.
        u = np.ravel(u_raw)[0] if np.ndim(u_raw) else u_raw
        y_val = y[start]
        signals[b_name] = y_val

        # Rate calculation
        err = u - y_val
        rate = err * K

        # Clamp rate
        dy = np.clip(rate, falling, rising)
        dy_vec[start] = dy
    return exec_ratelimiter
