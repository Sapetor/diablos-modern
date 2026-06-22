"""Algebraic math-block kernels for the compiled path.

Covers Gain, Sum, Exponential, Abs, SgProd, Product, MatrixGain and
MathFunction. Bodies are verbatim extractions of the corresponding branches
from ``SystemCompiler._create_block_executor``; only the shared locals are
unpacked from the BuildContext at the top.
"""
import logging

import numpy as np

from lib.engine.compiler_kernels import kernel
from lib.safe_eval import compile_expr

logger = logging.getLogger(__name__)


@kernel("Gain")
def build_gain(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    gain = float(params.get('gain', 1.0))
    # Optimization: If only 1 input
    src = input_sources[0] if input_sources else None

    def exec_gain(t, y, dy_vec, signals):
        val = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = val * gain
    return exec_gain


@kernel("Sum")
def build_sum(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    signs = params.get('sign', params.get('inputs', '++'))
    # Bake signs and sources. Iterate over the connected input ports
    # (not just the sign string) so extra wired inputs are not silently
    # dropped; missing sign characters default to '+'.
    n_terms = max(len(signs), len(input_sources))
    ops = []
    for i in range(n_terms):
        char = signs[i] if i < len(signs) else '+'
        src = input_sources[i] if i < len(input_sources) else None
        ops.append((src, 1.0 if char == '+' else -1.0))

    def exec_sum(t, y, dy_vec, signals):
        res = 0.0
        for src, mul in ops:
            val = signals.get(src, 0.0) if src else 0.0
            res += val * mul
        signals[b_name] = res
    return exec_sum


@kernel("Exponential")
def build_exponential(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    a = float(params.get('a', 1.0))
    b_coef = float(params.get('b', 1.0))  # avoid local var 'b'

    def exec_exp(t, y, dy_vec, signals):
        x_in = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = a * np.exp(b_coef * x_in)
    return exec_exp


@kernel("Abs", "Absblock")
def build_abs(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None

    def exec_abs(t, y, dy_vec, signals):
        val = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = abs(val)
    return exec_abs


@kernel("SgProd", "Sgprod", "SigProduct")
def build_sgprod(ctx):
    # 'Sgprod' covers the block_fn.title() form (block_fn 'SgProd' ->
    # 'SgProd'.title() == 'Sgprod'); without it SgProd silently compiled
    # to a no-op, zeroing its output in the fast solver (same .title()
    # normalization already handled for MatrixGain/StateSpace above).
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    baked_srcs = [s for s in input_sources]

    def exec_sgprod(t, y, dy_vec, signals):
        res = 1.0
        if baked_srcs:
            for src in baked_srcs:
                if src:
                    res *= signals.get(src, 0.0)
                else:
                    res *= 0.0  # Connected to 0
        else:
            res = 1.0  # Or 1.0 for identity
        signals[b_name] = res
    return exec_sgprod


@kernel("Product", "product")
def build_product(ctx):
    # Product block with configurable * and / operations
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    baked_srcs = [s for s in input_sources]
    ops = params.get('ops', '**')

    def exec_product(t, y, dy_vec, signals, _ops=ops, _srcs=baked_srcs):
        res = 1.0
        for i, src in enumerate(_srcs):
            op = _ops[i] if i < len(_ops) else '*'
            val = signals.get(src, 0.0) if src else 0.0
            if op == '*':
                res *= val
            elif op == '/':
                # Element-wise so vector divisors work (`if val != 0` is
                # ambiguous on an array). Mirror Product.execute() / the
                # compiled-replay path: divide-by-zero keeps the numerator
                # sign as a large finite value (not inf, which would break
                # the ODE RHS) and 0/0 -> 0.
                res = np.asarray(res, dtype=float)
                v = np.asarray(val, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    res = res / v
                    res = np.where(np.isinf(res), np.sign(res) * 1e308, res)
                    res = np.where(np.isnan(res), 0.0, res)
        signals[b_name] = res
    return exec_product


@kernel("Matrixgain", "MatrixGain")
def build_matrixgain(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    import ast as _ast
    K_raw = params.get('gain', '1.0')
    if isinstance(K_raw, str):
        try:
            K = np.array(_ast.literal_eval(K_raw), dtype=float)
        except (ValueError, SyntaxError):
            K = np.array([float(K_raw)], dtype=float)
    else:
        K = np.atleast_1d(np.array(K_raw, dtype=float))
    src = input_sources[0] if input_sources else None

    if K.ndim == 2:
        def exec_mgain(t, y, dy_vec, signals):
            u = np.atleast_1d(signals.get(src, 0.0) if src else 0.0).astype(float).flatten()
            if len(u) < K.shape[1]:
                u = np.pad(u, (0, K.shape[1] - len(u)))
            elif len(u) > K.shape[1]:
                u = u[:K.shape[1]]
            signals[b_name] = K @ u
    elif K.ndim == 1 and K.size > 1:
        _warned = [False]

        def exec_mgain(t, y, dy_vec, signals, _warned=_warned):
            u = np.atleast_1d(signals.get(src, 0.0) if src else 0.0).astype(float)
            if len(K) == len(u):
                signals[b_name] = K * u
            else:
                # Dimension mismatch is almost always a wiring/config
                # error; warn once instead of silently truncating.
                if not _warned[0]:
                    logger.warning(
                        "MatrixGain %s: vector gain length %d does not match "
                        "input length %d; truncating to the overlapping prefix.",
                        b_name, len(K), len(u))
                    _warned[0] = True
                m = min(len(K), len(u))
                signals[b_name] = K[:m] * u[:m]
    else:
        k_scalar = float(K.flatten()[0])

        def exec_mgain(t, y, dy_vec, signals):
            val = signals.get(src, 0.0) if src else 0.0
            signals[b_name] = np.atleast_1d(val).astype(float) * k_scalar
    return exec_mgain


@kernel("Mathfunction")
def build_mathfunction(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    # Check both 'function' and 'expression' keys for backward compatibility
    func_raw = params.get('function', params.get('expression', 'sin'))
    func = str(func_raw).lower()

    # Pre-select the function to avoid string comparison at runtime
    np_func = None
    use_expr = False
    if func == 'sin':
        np_func = np.sin
    elif func == 'cos':
        np_func = np.cos
    elif func == 'tan':
        np_func = np.tan
    elif func == 'asin':
        np_func = np.arcsin
    elif func == 'acos':
        np_func = np.arccos
    elif func == 'atan':
        np_func = np.arctan
    elif func == 'exp':
        np_func = np.exp
    elif func == 'log':
        np_func = np.log
    elif func == 'log10':
        np_func = np.log10
    elif func == 'sqrt':
        np_func = np.sqrt
    elif func == 'square':
        np_func = lambda x: x * x
    elif func == 'sign':
        np_func = np.sign
    elif func == 'abs':
        np_func = np.abs
    elif func == 'ceil':
        np_func = np.ceil
    elif func == 'floor':
        np_func = np.floor
    elif func == 'reciprocal':
        def _reciprocal(x):
            if isinstance(x, np.ndarray):
                m = x != 0
                # Build the non-zero mask once and reuse it.
                return np.where(m, 1.0 / np.where(m, x, 1.0), 0.0)
            return 1.0 / x if x != 0 else 0.0
        np_func = _reciprocal
    elif func == 'cube':
        np_func = lambda x: x * x * x
    else:
        # Python expression fallback
        use_expr = True
        expr_str = str(func_raw)  # Use raw string for eval

    if use_expr:
        # Compile expression once at compile time for hot-loop performance
        _compiled_expr = compile_expr(expr_str)

        # Dry-run once on a representative input so structural failures
        # (undefined names, bad subscripts, type errors) surface at
        # compile time as a WARNING instead of being silently swallowed
        # to 0.0 on every step for the whole run.
        try:
            _compiled_expr({"u": 1.0, "t": 0.0})
        except Exception as _e:  # noqa: BLE001 - report any setup failure once
            logger.warning(
                "MathFunction %s: expression %r failed to evaluate on a "
                "sample input (%s); it will fall back to 0.0 each step.",
                b_name, expr_str, _e)

        def exec_mathfunc_expr(t, y, dy_vec, signals, _src=src, _compiled=_compiled_expr):
            val = signals.get(_src, 0.0) if _src else 0.0
            try:
                signals[b_name] = float(_compiled({"u": val, "t": t}))
            except Exception as _e:
                # User expression failed this step (e.g. domain/type error);
                # fall back to 0.0. Per-timestep hot loop -> debug level to
                # avoid flooding while still being diagnosable. Persistent
                # structural failures are already surfaced once at compile
                # time via the dry-run WARNING above.
                logger.debug("MathFunction expr eval failed for %s: %s", b_name, _e)
                signals[b_name] = 0.0
        return exec_mathfunc_expr
    else:
        def exec_mathfunc(t, y, dy_vec, signals, _func=np_func, _src=src):
            val = signals.get(_src, 0.0) if _src else 0.0
            try:
                signals[b_name] = _func(val)
            except (ValueError, ZeroDivisionError):
                signals[b_name] = 0.0
        return exec_mathfunc
