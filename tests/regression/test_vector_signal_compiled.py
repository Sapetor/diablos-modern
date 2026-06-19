"""Regression tests for vector-signal safety in the compiled fast solver.

These guard the sibling bugs found after the tester-reported vector-signal
crashes (commits b4a88a5 "vector-safe Product/MathFunction/Exponential in
compiled replay" and 3d0cbb5 "harden Deadband/Switch/PID/Hysteresis"). The
earlier fixes touched the compiled *replay* path; an audit found the same class
of bug still live in the compiled *ODE* closures:

  * ``exec_product`` ('/' branch) raised "truth value of an array is ambiguous"
    on a vector divisor (``system_compiler.py``).
  * ``exec_ratelimiter`` raised "setting an array element with a sequence" when a
    vector reached its single scalar state slot (``system_compiler.py``).

Both run inside the compiled ODE right-hand side, so the tests below drive a real
vector signal through ``SystemCompiler.compile_system`` + the model RHS and would
crash on the pre-fix code.

A third fix in the same audit hardened the compiled-*replay* StateSpace branch
(``simulation_engine.py``: ``float(<vector>)`` -> ``float(np.ravel(u_val)[0])``).
That path is only reached for a length-mismatched signal, which the interpreted
initialization pass already rejects with a clean "dimension mismatch" error, so
it cannot be provoked from a minimal compilable diagram without contrived setup;
it is covered by the audit's direct repro instead. The change is a no-op for the
scalar inputs every existing test/example uses (``np.ravel(s)[0] == s``).
"""

import numpy as np
import pytest
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


def _mk(fn, sid, inp, outp, params, b_type=2):
    from lib.simulation.block import DBlock
    return DBlock(
        block_fn=fn, sid=sid, coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150), in_ports=inp, out_ports=outp,
        params=params, username="", b_type=b_type,
    )


def _ln(sid, src, sp, dst, dp):
    from lib.simulation.connection import DLine
    return DLine(sid=sid, srcblock=src, srcport=sp, dstblock=dst, dstport=dp,
                 points=[QPoint(0, 0), QPoint(1, 0)])


def _resolve(blocks):
    """Populate exec_params the way DSim/initialize_execution would."""
    from lib.workspace import WorkspaceManager
    WorkspaceManager._instance = None
    wm = WorkspaceManager()
    for b in blocks:
        b.exec_params = wm.resolve_params(b.params)
        b.exec_params.update({k: v for k, v in b.params.items() if k.startswith("_")})
        b.exec_params["dtime"] = 0.01
    return blocks


@pytest.mark.regression
class TestVectorSignalCompiled:
    def test_product_division_vector_ode_rhs(self, qapp):
        """Constant[2,4,6] (*) / Constant[1,2,3] (/) -> Integrator.

        The Product '/' branch runs inside the compiled ODE right-hand side; a
        vector divisor used to raise 'truth value of an array is ambiguous'. The
        element-wise quotient must be [2, 2, 2]."""
        from lib.engine.system_compiler import SystemCompiler

        blocks = _resolve([
            _mk("Constant", 0, 0, 1, {"value": [2.0, 4.0, 6.0]}, b_type=0),
            _mk("Constant", 1, 0, 1, {"value": [1.0, 2.0, 3.0]}, b_type=0),
            _mk("Product", 0, 2, 1, {"ops": "*/"}),
            _mk("Integrator", 0, 1, 1, {"init_conds": [0.0, 0.0, 0.0]}, b_type=1),
        ])
        lines = [
            _ln(0, "constant0", 0, "product0", 0),
            _ln(1, "constant1", 0, "product0", 1),
            _ln(2, "product0", 0, "integrator0", 0),
        ]
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)
        model_func, y0, state_map, _ = compiler.compile_system(blocks, blocks, lines)

        dy = np.asarray(model_func(0.0, y0))  # pre-fix: ValueError (ambiguous truth)
        start, size = state_map["integrator0"]
        assert size == 3, f"expected 3 integrator slots, got {size}"
        np.testing.assert_allclose(dy[start:start + size], [2.0, 2.0, 2.0], atol=1e-9)

    def test_ratelimiter_vector_input_ode_rhs(self, qapp):
        """Constant[5,3,1] -> RateLimiter.

        RateLimiter has a single scalar state slot; a vector input used to raise
        'setting an array element with a sequence' at dy_vec[start] = dy. The fix
        reduces the input to its first element (5.0); the derivative must be a
        finite scalar."""
        from lib.engine.system_compiler import SystemCompiler

        blocks = _resolve([
            _mk("Constant", 0, 0, 1, {"value": [5.0, 3.0, 1.0]}, b_type=0),
            _mk("RateLimiter", 0, 1, 1,
                {"rising_slew": 10.0, "falling_slew": 10.0}, b_type=1),
        ])
        lines = [_ln(0, "constant0", 0, "ratelimiter0", 0)]
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)
        model_func, y0, state_map, _ = compiler.compile_system(blocks, blocks, lines)

        dy = np.asarray(model_func(0.0, y0))  # pre-fix: ValueError (sequence)
        start, size = state_map["ratelimiter0"]
        assert size == 1
        assert np.ndim(dy[start]) == 0 and np.isfinite(dy[start])
