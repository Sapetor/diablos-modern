"""Source-block kernels for the compiled path.

Covers Constant, Step, Ramp, Sine, Wavegenerator, Impulse, Prbs and Noise.
Bodies are verbatim extractions of the corresponding branches that used to live
in ``SystemCompiler._create_block_executor``; only the shared locals (``b_name``,
``params``) are unpacked from the BuildContext at the top.
"""
import numpy as np
from scipy import signal

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


@kernel("Wavegenerator")
def build_wavegenerator(ctx):
    b_name = ctx.b_name
    params = ctx.params
    waveform = params.get('waveform', 'Sine')
    amp = float(params.get('amplitude', 1.0))
    freq = float(params.get('frequency', 1.0))
    phase = float(params.get('phase', 0.0))
    bias = float(params.get('bias', 0.0))

    if waveform == 'Sine':
        def exec_wavegen_sine(t, y, dy_vec, signals):
            arg = 2 * np.pi * freq * t + phase
            signals[b_name] = bias + amp * np.sin(arg)
        return exec_wavegen_sine
    elif waveform == 'Square':
        def exec_wavegen_square(t, y, dy_vec, signals):
            arg = 2 * np.pi * freq * t + phase
            signals[b_name] = bias + amp * signal.square(arg)
        return exec_wavegen_square
    elif waveform == 'Triangle':
        def exec_wavegen_triangle(t, y, dy_vec, signals):
            arg = 2 * np.pi * freq * t + phase
            signals[b_name] = bias + amp * signal.sawtooth(arg, width=0.5)
        return exec_wavegen_triangle
    elif waveform == 'Sawtooth':
        def exec_wavegen_sawtooth(t, y, dy_vec, signals):
            arg = 2 * np.pi * freq * t + phase
            signals[b_name] = bias + amp * signal.sawtooth(arg, width=1.0)
        return exec_wavegen_sawtooth
    else:
        # Default to sine
        def exec_wavegen_default(t, y, dy_vec, signals):
            arg = 2 * np.pi * freq * t + phase
            signals[b_name] = bias + amp * np.sin(arg)
        return exec_wavegen_default


@kernel("Impulse")
def build_impulse(ctx):
    b_name = ctx.b_name
    params = ctx.params
    imp_t = float(params.get('delay', 0.0))
    imp_val = float(params.get('value', 1.0))
    imp_dt = float(params.get('dtime', 0.01))
    # Use a very narrow pulse so the response is not visibly delayed.
    # Width = dt/1000 keeps the shift negligible while RK45 handles it.
    eps = imp_dt * 1e-3
    imp_end = imp_t + eps
    imp_height = imp_val / eps

    def exec_impulse(t, y, dy_vec, signals):
        signals[b_name] = imp_height if imp_t <= t < imp_end else 0.0
    return exec_impulse


@kernel("Prbs")
def build_prbs(ctx):
    b_name = ctx.b_name
    params = ctx.params
    high = float(params.get('high', 1.0))
    low = float(params.get('low', 0.0))
    bit_time = float(params.get('bit_time', 0.1))
    order = int(params.get('order', 7))
    seed = int(params.get('seed', 1)) & ((1 << order) - 1)
    if seed == 0:
        seed = 1

    # Primitive polynomial taps for left-shift Fibonacci LFSR
    # (verified with Mathematica)
    _primitive_taps = {
        2: [1, 0], 3: [2, 0], 4: [3, 0], 5: [4, 2],
        6: [5, 0], 7: [6, 5], 8: [7, 5, 4, 3], 9: [8, 4],
        10: [9, 6], 11: [10, 8], 12: [11, 10, 9, 3], 13: [12, 11, 8, 6],
        14: [13, 12, 10, 8], 15: [14, 13], 16: [15, 13, 12, 10], 17: [16, 13],
        18: [17, 10], 19: [18, 17, 16, 13], 20: [19, 16], 21: [20, 18],
        22: [21, 20], 23: [22, 17], 24: [23, 22, 21, 16],
    }
    taps = _primitive_taps.get(order, [1, 0])
    mask = (1 << order) - 1
    period = (1 << order) - 1

    # Precompute full LFSR sequence (period = 2^order - 1)
    lfsr = seed
    sequence = np.empty(period, dtype=np.float64)
    for i in range(period):
        sequence[i] = high if (lfsr & 1) else low
        feedback = 0
        for p in taps:
            feedback ^= (lfsr >> p) & 1
        lfsr = ((lfsr << 1) & mask) | feedback

    def exec_prbs(t, y, dy_vec, signals):
        bit_index = int(t / bit_time) % period
        signals[b_name] = sequence[bit_index]
    return exec_prbs


@kernel("Noise")
def build_noise(ctx):
    b_name = ctx.b_name
    params = ctx.params
    mu = float(params.get('mu', 0.0))
    sigma = float(params.get('sigma', 1.0))

    def exec_noise(t, y, dy_vec, signals):
        signals[b_name] = mu + sigma * np.random.randn()
    return exec_noise
