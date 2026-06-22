"""Field-probe / field-reduction block kernels for the compiled path.

Covers FieldProbe, FieldIntegral, FieldMax, FieldGradient, FieldLaplacian,
FieldScope, FieldProbe2D, FieldScope2D and FieldSlice -- blocks that read a PDE
field signal and emit a scalar/derived signal. Bodies are verbatim extractions
of the corresponding branches from ``SystemCompiler._create_block_executor``
(dedented; shared locals unpacked from the BuildContext at the top).
"""
import numpy as np

from lib.engine.compiler_kernels import kernel

# np.trapz was renamed to np.trapezoid in NumPy 2.0 (old name removed).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@kernel("Fieldprobe")
def build_fieldprobe(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    pos_src = input_sources[1] if len(input_sources) > 1 else None
    position = float(params.get('position', 0.5))
    mode = params.get('position_mode', 'normalized')
    L = float(params.get('L', 1.0))

    def exec_fieldprobe(t, y, dy_vec, signals, _src=src, _pos_src=pos_src,
                       _position=position, _mode=mode, _L=L):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            signals[b_name] = 0.0
            return

        pos = signals.get(_pos_src, _position) if _pos_src else _position
        # Probe position is scalar; reduce a mis-wired vector to its first
        # element so int(np.floor(idx_float)) below stays safe.
        pos = float(np.ravel(pos)[0]) if np.ndim(pos) else pos

        if _mode == 'absolute':
            pos_norm = pos / _L
        else:
            pos_norm = pos

        pos_norm = np.clip(pos_norm, 0.0, 1.0)
        N = len(field)
        idx_float = pos_norm * (N - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, N - 1)
        frac = idx_float - idx_low

        value = field[idx_low] * (1 - frac) + field[idx_high] * frac
        signals[b_name] = float(value)
    return exec_fieldprobe


@kernel("Fieldintegral")
def build_fieldintegral(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    L = float(params.get('L', 1.0))
    normalize = params.get('normalize', False)

    def exec_fieldintegral(t, y, dy_vec, signals, _src=src, _L=L, _norm=normalize):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            signals[b_name] = 0.0
            return

        N = len(field)
        dx = _L / (N - 1) if N > 1 else _L
        integral = _trapezoid(field, dx=dx)

        if _norm:
            integral = integral / _L

        signals[b_name] = float(integral)
    return exec_fieldintegral


@kernel("Fieldmax")
def build_fieldmax(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    mode = params.get('mode', 'max')
    L = float(params.get('L', 1.0))

    def exec_fieldmax(t, y, dy_vec, signals, _src=src, _mode=mode, _L=L):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            signals[b_name] = 0.0
            return

        if _mode == 'min':
            idx = int(np.argmin(field))
        else:
            idx = int(np.argmax(field))

        value = field[idx]
        N = len(field)
        location = (idx / (N - 1)) * _L if N > 1 else 0.0

        signals[b_name] = float(value)
        signals[b_name + '_loc'] = float(location)
        signals[b_name + '_idx'] = idx
    return exec_fieldmax


@kernel("Fieldgradient")
def build_fieldgradient(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    L = float(params.get('L', 1.0))

    def exec_fieldgradient(t, y, dy_vec, signals, _src=src, _L=L):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        field = np.atleast_1d(field).flatten()

        if len(field) < 2:
            signals[b_name] = np.array([0.0])
            return

        N = len(field)
        dx = _L / (N - 1)
        gradient = np.gradient(field, dx)
        signals[b_name] = gradient
    return exec_fieldgradient


@kernel("Fieldlaplacian")
def build_fieldlaplacian(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    L = float(params.get('L', 1.0))

    def exec_fieldlaplacian(t, y, dy_vec, signals, _src=src, _L=L):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        field = np.atleast_1d(field).flatten()

        if len(field) < 3:
            signals[b_name] = np.zeros(len(field))
            return

        N = len(field)
        dx = _L / (N - 1)
        dx_sq = dx * dx

        laplacian = np.zeros(N)
        for i in range(1, N-1):
            laplacian[i] = (field[i+1] - 2*field[i] + field[i-1]) / dx_sq

        laplacian[0] = (field[2] - 2*field[1] + field[0]) / dx_sq
        laplacian[N-1] = (field[N-1] - 2*field[N-2] + field[N-3]) / dx_sq

        signals[b_name] = laplacian
    return exec_fieldlaplacian


@kernel("Fieldscope")
def build_fieldscope(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    # FieldScope is a sink - just pass through
    src = input_sources[0] if input_sources else None

    def exec_fieldscope(t, y, dy_vec, signals, _src=src):
        field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
        signals[b_name] = np.atleast_1d(field).flatten()
    return exec_fieldscope


@kernel("Fieldprobe2D")
def build_fieldprobe2d(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    x_pos_src = input_sources[1] if len(input_sources) > 1 else None
    y_pos_src = input_sources[2] if len(input_sources) > 2 else None
    x_position = float(params.get('x_position', 0.5))
    y_position = float(params.get('y_position', 0.5))
    position_mode = params.get('position_mode', 'normalized')
    Lx = float(params.get('Lx', 1.0))
    Ly = float(params.get('Ly', 1.0))

    def exec_fieldprobe2d(t, y, dy_vec, signals, _src=src,
                          _x_pos_src=x_pos_src, _y_pos_src=y_pos_src,
                          _x_pos=x_position, _y_pos=y_position,
                          _mode=position_mode, _Lx=Lx, _Ly=Ly):
        field = signals.get(_src, None) if _src else None
        if field is None:
            signals[b_name] = 0.0
            return

        field = np.atleast_2d(field)
        Ny, Nx = field.shape

        # Get positions
        x_pos = signals.get(_x_pos_src, _x_pos) if _x_pos_src else _x_pos
        y_pos = signals.get(_y_pos_src, _y_pos) if _y_pos_src else _y_pos
        # Probe positions are scalar; reduce mis-wired vectors to their
        # first element so min/max and int(np.floor(...)) below stay safe.
        x_pos = float(np.ravel(x_pos)[0]) if np.ndim(x_pos) else x_pos
        y_pos = float(np.ravel(y_pos)[0]) if np.ndim(y_pos) else y_pos

        # Convert to normalized
        if _mode == 'absolute':
            x_norm = x_pos / _Lx
            y_norm = y_pos / _Ly
        else:
            x_norm = x_pos
            y_norm = y_pos

        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        # Bilinear interpolation
        i_float = x_norm * (Nx - 1)
        j_float = y_norm * (Ny - 1)
        i0 = int(np.floor(i_float))
        i1 = min(i0 + 1, Nx - 1)
        j0 = int(np.floor(j_float))
        j1 = min(j0 + 1, Ny - 1)
        di = i_float - i0
        dj = j_float - j0

        val = (field[j0, i0] * (1 - di) * (1 - dj) +
               field[j0, i1] * di * (1 - dj) +
               field[j1, i0] * (1 - di) * dj +
               field[j1, i1] * di * dj)

        signals[b_name] = float(val)
    return exec_fieldprobe2d


@kernel("Fieldscope2D")
def build_fieldscope2d(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    # FieldScope2D is a sink - pass through 2D field
    src = input_sources[0] if input_sources else None

    def exec_fieldscope2d(t, y, dy_vec, signals, _src=src):
        field = signals.get(_src, np.zeros((1, 1))) if _src else np.zeros((1, 1))
        signals[b_name] = np.atleast_2d(field)
    return exec_fieldscope2d


@kernel("Fieldslice")
def build_fieldslice(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    pos_src = input_sources[1] if len(input_sources) > 1 else None
    slice_direction = params.get('slice_direction', 'x')
    slice_position = float(params.get('slice_position', 0.5))

    def exec_fieldslice(t, y, dy_vec, signals, _src=src, _pos_src=pos_src,
                       _direction=slice_direction, _pos=slice_position):
        field = signals.get(_src, None) if _src else None
        if field is None:
            signals[b_name] = np.array([0.0])
            return

        field = np.atleast_2d(field)
        Ny, Nx = field.shape

        position = signals.get(_pos_src, _pos) if _pos_src else _pos
        # Slice position is scalar; reduce a mis-wired vector to its first
        # element so int(position * ...) below stays safe.
        position = float(np.ravel(position)[0]) if np.ndim(position) else position

        if _direction.lower() == 'x':
            j = int(position * (Ny - 1))
            j = max(0, min(Ny - 1, j))
            slice_arr = field[j, :]
        else:
            i = int(position * (Nx - 1))
            i = max(0, min(Nx - 1, i))
            slice_arr = field[:, i]

        signals[b_name] = slice_arr
    return exec_fieldslice
