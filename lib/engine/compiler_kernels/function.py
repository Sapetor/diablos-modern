"""Compiled kernel for the Function (user expression) block.

Same evaluator and restrictions as ``blocks/function.py``: the hardened
``lib.safe_eval`` walker with numpy math, parsed and validated once here by
``compile_expr`` instead of on every RHS call. The variables mirror
``FunctionBlock.execute``: ``u`` is the list of input ports (unconnected ports
read 0), ``u1``, ``u2``, ... are 1-indexed aliases and ``t`` is the time.
Inputs arrive as 1-D arrays, the shape most blocks output in the interpreter
(a few, like Sine, output 0-d arrays there; the compiled signals do not keep
that difference). An expression that only works on one of the two shapes
therefore either gives the same value on both paths or fails here.

A failing expression raises instead of outputting 0: the compiled run then
fails and the run falls back to the interpreter, which evaluates it with its
own input shapes and reports any error exactly as without the fast solver.
"""

import numpy as np

from lib.engine.compiler_kernels import kernel
from lib.safe_eval import SafeEvalError, compile_expr


@kernel("Function")
def build_function(ctx):
    b_name = ctx.b_name
    expr = str(ctx.params.get("expression", "u[0]"))
    compiled = compile_expr(expr)  # raises SafeEvalError on a forbidden construct
    # The interpreter sizes ``u`` from the engine-managed port count.
    n = max(int(ctx.params.get("_inputs_", 0) or 0), len(ctx.input_sources), 1)
    sources = list(ctx.input_sources) + [None] * (n - len(ctx.input_sources))
    aliases = ["u{}".format(i + 1) for i in range(n)]

    def exec_function(t, y, dy_vec, signals):
        u = [
            np.atleast_1d(np.asarray(signals.get(src, 0.0) if src else 0.0, dtype=float))
            for src in sources
        ]
        variables = dict(zip(aliases, u))
        variables["u"] = u
        variables["t"] = t
        try:
            out = np.atleast_1d(np.asarray(compiled(variables), dtype=float))
        except (SafeEvalError, ValueError, TypeError, IndexError) as exc:
            raise SafeEvalError(
                "Function block {!r}: expression {!r} failed: {}".format(b_name, expr, exc)
            ) from exc
        # A one-element result is a scalar signal on the compiled path (the
        # interpreter's atleast_1d makes the two indistinguishable).
        signals[b_name] = float(out[0]) if out.size == 1 else out

    return exec_function
