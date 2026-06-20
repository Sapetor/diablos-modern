"""Algebraic math-block kernels (Gain, Sum) for the compiled path.

Bodies are verbatim extractions of the corresponding branches from
``SystemCompiler._create_block_executor``; only the shared locals are unpacked
from the BuildContext at the top.
"""
from lib.engine.compiler_kernels import kernel


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
