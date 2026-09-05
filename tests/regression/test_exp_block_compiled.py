"""Regression: the ``Exp`` block must be accepted by the fast solver.

The palette stores ``block_fn = "Exp"`` (the block's ``block_name``), but the
compiler's allowlist, the kernel registry and the replay set only knew the
spelling ``"Exponential"``. A diagram with a real Exp block therefore silently
fell back to the interpreter, and the kernel was only reachable from tests
that constructed the context by hand.

The block is also algebraic (``y = a * exp(b * x)`` of its *input*), so it must
run with the middle group, not with the sources.
"""

import numpy as np
import pytest


def _block(block_fn, name, in_ports, out_ports, params, b_type):
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor
    from lib.simulation.block import DBlock

    blk = DBlock(
        block_fn=block_fn,
        sid=0,
        coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150),
        in_ports=in_ports,
        out_ports=out_ports,
        params=dict(params),
        username="",
        b_type=b_type,
    )
    blk.name = name
    return blk


def _line(src, dst):
    from PyQt5.QtCore import QPoint
    from lib.simulation.connection import DLine

    return DLine(
        sid=0,
        srcblock=src,
        srcport=0,
        dstblock=dst,
        dstport=0,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


@pytest.mark.regression
class TestExpBlockCompiled:
    def test_exp_block_fn_is_compilable(self, qapp):
        from lib.engine.block_names import canonical_fn
        from lib.engine.compiler_kernels import get_kernel_builder
        from lib.engine.compiled_runner import _KERNEL_REPLAY_FNS
        from lib.engine.system_compiler import SystemCompiler

        blocks = [
            _block("Constant", "constant0", 0, 1, {"value": 2.0}, 0),
            _block("Exp", "exp0", 1, 1, {"a": 1.0, "b": 0.5}, 2),
            _block("Scope", "scope0", 1, 0, {}, 3),
        ]
        assert SystemCompiler().check_compilability(blocks)
        assert get_kernel_builder(canonical_fn("Exp")) is not None
        assert canonical_fn("Exp") in _KERNEL_REPLAY_FNS

    def test_exp_runs_after_its_input_in_compiled_path(self, qapp):
        """Constant(2) -> Exp(a=1, b=0.5) -> 1/(s+1): steady state is e^1."""
        from scipy.integrate import solve_ivp
        from lib.engine.system_compiler import SystemCompiler

        blocks = [
            _block("Constant", "constant0", 0, 1, {"value": 2.0}, 0),
            _block("Exp", "exp0", 1, 1, {"a": 1.0, "b": 0.5}, 2),
            _block("TranFn", "tranfn0", 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]}, 1),
        ]
        lines = [_line("constant0", "exp0"), _line("exp0", "tranfn0")]

        model_func, y0, state_map, _ = SystemCompiler().compile_system(blocks, blocks, lines)
        sol = solve_ivp(model_func, (0, 10), y0, method="RK45", rtol=1e-8, atol=1e-10)
        assert sol.success

        start, _ = state_map["tranfn0"]
        # If Exp ran in the source group it would read a stale/zero input and
        # drive the filter toward exp(0) = 1 instead of exp(1).
        assert sol.y[start, -1] == pytest.approx(np.e, rel=1e-3)
