"""
Regression test for feedback loop with Sum block sign parameter.

Bug: SystemCompiler read block.params.get('inputs', '++') for Sum blocks,
but Sum blocks only define 'sign' as their param. Palette-created blocks
would default to '++' (all positive), breaking feedback subtraction.

Fix: Changed to block.params.get('sign', block.params.get('inputs', '++'))
"""

import pytest
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


def _make_block(block_fn, sid, username, in_ports, out_ports, params, b_type=2):
    """Helper to create a DBlock with minimal boilerplate."""
    from lib.simulation.block import DBlock
    return DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150),
        in_ports=in_ports,
        out_ports=out_ports,
        params=params,
        username=username,
        b_type=b_type,
    )


def _make_line(sid, src, srcport, dst, dstport):
    """Helper to create a DLine."""
    from lib.simulation.connection import DLine
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


@pytest.mark.regression
class TestFeedbackLoopSumSign:
    """Test that feedback loops work correctly with Sum block 'sign' parameter."""

    def test_closed_loop_steady_state(self, qapp):
        """
        Closed-loop: Step(8) -> Sum('+-') -> Gain(1) -> TranFn(1/(s+0.2)) -> feedback

        CL transfer function: 1/(s+1.2), steady-state = 8/1.2 ~ 6.667
        If feedback is broken (signs='++'), output would diverge or be wrong.
        """
        from lib.engine.system_compiler import SystemCompiler
        from scipy.integrate import solve_ivp

        # Create blocks — Sum has ONLY 'sign', not 'inputs'
        # block.name = block_fn.lower() + str(sid), so names are: step0, sum0, gain0, tranfn0
        blocks = [
            _make_block('Step', 0, '', 0, 1, {'value': 8.0, 'delay': 0.0, 'type': 'up'}, b_type=0),
            _make_block('Sum', 0, '', 2, 1, {'sign': '+-'}),
            _make_block('Gain', 0, '', 1, 1, {'gain': 1.0}),
            _make_block('TranFn', 0, '', 1, 1, {'numerator': [1.0], 'denominator': [1.0, 0.2]}),
        ]

        # Connections use block.name: step0, sum0, gain0, tranfn0
        lines = [
            _make_line(0, 'step0', 0, 'sum0', 0),
            _make_line(1, 'sum0', 0, 'gain0', 0),
            _make_line(2, 'gain0', 0, 'tranfn0', 0),
            _make_line(3, 'tranfn0', 0, 'sum0', 1),  # feedback
        ]

        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)

        # compile_system(blocks, sorted_order, lines) -> (model_func, y0, state_map)
        model_func, y0, state_map, block_matrices = compiler.compile_system(blocks, blocks, lines)

        assert 'tranfn0' in state_map, "TranFn should have state"
        start, size = state_map['tranfn0']

        sol = solve_ivp(model_func, (0, 50), y0, method='RK45',
                        t_eval=np.linspace(0, 50, 1000), rtol=1e-8, atol=1e-10)
        assert sol.success

        # For 1/(s+0.2) in state-space: A=-0.2, B=1, C=1, D=0
        # Output y = C*x = x (the state IS the output)
        final_value = sol.y[start, -1]

        expected_ss = 8.0 / 1.2  # ≈ 6.667
        assert abs(final_value - expected_ss) < 0.05, \
            f"Closed-loop SS should be ~{expected_ss:.3f}, got {final_value:.3f}"

        # Must NOT be the open-loop value (would mean feedback is broken)
        # Open-loop SS of 1/(s+0.2) to step of 8 = 8/0.2 = 40
        assert final_value < 10.0, \
            f"Output {final_value:.1f} too high — feedback may be broken (open-loop SS=40)"

    def test_sum_sign_param_only(self, qapp):
        """Verify palette-created Sum block only has 'sign', not 'inputs'."""
        block = _make_block('Sum', 0, 'sum0', 2, 1, {'sign': '+-'})

        assert 'sign' in block.params
        assert 'inputs' not in block.params

        # Compiler lookup must find 'sign'
        signs = block.params.get('sign', block.params.get('inputs', '++'))
        assert signs == '+-'

    def test_sum_with_legacy_inputs_key(self, qapp):
        """Sum block from old file format with 'inputs' key should still work."""
        block = _make_block('Sum', 0, 'sum0', 2, 1, {'sign': '+-', 'inputs': '+-'})

        signs = block.params.get('sign', block.params.get('inputs', '++'))
        assert signs == '+-'

    def test_sum_with_only_inputs_key(self, qapp):
        """Sum block with only legacy 'inputs' key (backward compat)."""
        block = _make_block('Sum', 0, 'sum0', 2, 1, {'inputs': '+-'})

        signs = block.params.get('sign', block.params.get('inputs', '++'))
        assert signs == '+-'
