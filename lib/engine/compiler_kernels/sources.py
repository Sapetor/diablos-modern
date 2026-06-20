"""Source-block kernels (Constant, Step, Ramp, Sine) for the compiled path.

Bodies are verbatim extractions of the corresponding branches that used to live
in ``SystemCompiler._create_block_executor``; only the shared locals (``b_name``,
``params``) are unpacked from the BuildContext at the top.
"""
import numpy as np

from lib.engine.compiler_kernels import kernel


@kernel("Constant")
def build_constant(ctx):
    b_name = ctx.b_name
    params = ctx.params
    raw_val = params.get('value', 0.0)
    # Handle both scalar and array values
    if isinstance(raw_val, (list, tuple)):
        val = np.atleast_1d(raw_val)
    elif hasattr(raw_val, '__iter__') and not isinstance(raw_val, str):
        val = np.atleast_1d(raw_val)
    else:
        val = float(raw_val)

    def exec_constant(t, y, dy_vec, signals):
        signals[b_name] = val
    return exec_constant


@kernel("Sine")
def build_sine(ctx):
    b_name = ctx.b_name
    params = ctx.params
    amp = float(params.get('amplitude', 1.0))
    freq = float(params.get('frequency', params.get('omega', 1.0)))
    phase = float(params.get('phase', params.get('init_angle', 0.0)))
    bias = float(params.get('bias', 0.0))

    def exec_sine(t, y, dy_vec, signals):
        signals[b_name] = amp * np.sin(freq * t + phase) + bias
    return exec_sine


@kernel("Step")
def build_step(ctx):
    b_name = ctx.b_name
    params = ctx.params
    step_t = float(params.get('delay', 0.0))
    val = float(params.get('value', 1.0))
    step_type = params.get('type', 'up')

    if step_type == 'impulse':
        dt = float(params.get('dtime', 0.01))
        eps = dt * 1e-3
        impulse_end = step_t + eps
        impulse_height = val / eps

        def exec_step(t, y, dy_vec, signals):
            signals[b_name] = impulse_height if step_t <= t < impulse_end else 0.0
        return exec_step
    else:
        def exec_step(t, y, dy_vec, signals):
            signals[b_name] = val if t >= step_t else 0.0
        return exec_step


@kernel("Ramp")
def build_ramp(ctx):
    b_name = ctx.b_name
    params = ctx.params
    slope = float(params.get('slope', 1.0))
    delay = float(params.get('delay', 0.0))

    def exec_ramp(t, y, dy_vec, signals):
        if slope > 0:
            val = max(0.0, slope * (t - delay))
        elif slope < 0:
            val = min(0.0, slope * (t - delay))
        else:
            val = 0.0
        signals[b_name] = val
    return exec_ramp
