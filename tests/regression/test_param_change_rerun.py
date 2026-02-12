"""
Regression tests for parameter change and re-simulation.

These tests verify that changing block parameters and recompiling/re-simulating
produces different (correct) results. This is critical for ensuring that the
SystemCompiler correctly picks up parameter changes on subsequent runs.

Run with: pytest tests/regression/test_param_change_rerun.py -v
"""

import pytest
import numpy as np
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.engine.system_compiler import SystemCompiler
from scipy.integrate import solve_ivp


def _make_block(block_fn, sid, username, in_ports, out_ports, params, b_type=2):
    """Helper to create a DBlock for testing."""
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
    """Helper to create a DLine connection."""
    return DLine(
        sid=sid,
        srcblock=src,
        srcport=srcport,
        dstblock=dst,
        dstport=dstport,
        points=[QPoint(0, 0), QPoint(100, 0)],
    )


@pytest.mark.regression
class TestParameterChangeRerun:
    """Test that parameter changes produce different simulation results."""

    def test_transfer_function_denominator_change(self, qapp):
        """
        Test that changing TranFn denominator produces different step response.

        Setup: Step → TranFn → Scope
        - First run: denominator = [1.0, 1.0] (time constant τ=1)
        - Second run: denominator = [1.0, 0.5] (time constant τ=2)

        These should produce significantly different step responses.
        """
        # Create blocks
        step_block = _make_block(
            block_fn='Step',
            sid=0,
            username='Step',
            in_ports=0,
            out_ports=1,
            params={'value': 1.0, 'delay': 0.0, 'type': 'up'},
        )

        tranfn_block = _make_block(
            block_fn='TranFn',
            sid=1,
            username='TranFn',
            in_ports=1,
            out_ports=1,
            params={
                'numerator': [1.0],
                'denominator': [1.0, 1.0],  # 1/(s+1), τ=1
            },
        )

        scope_block = _make_block(
            block_fn='Scope',
            sid=2,
            username='Scope',
            in_ports=1,
            out_ports=0,
            params={'labels': 'output'},
        )

        # Create connections
        line1 = _make_line(sid=0, src='step0', srcport=0, dst='tranfn1', dstport=0)
        line2 = _make_line(sid=1, src='tranfn1', srcport=0, dst='scope2', dstport=0)

        blocks = [step_block, tranfn_block, scope_block]
        lines = [line1, line2]

        # First simulation with denominator = [1.0, 1.0]
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks), "Blocks should be compilable"

        model_func, y0, state_map, block_matrices = compiler.compile_system(
            blocks, blocks, lines
        )

        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 500)
        sol1 = solve_ivp(model_func, t_span, y0, method='RK45', t_eval=t_eval)

        # Extract output (TranFn state)
        output1 = sol1.y[0, :]  # First state variable is TranFn output

        # Change denominator to [1.0, 0.5] (τ=2, slower response)
        tranfn_block.params['denominator'] = [1.0, 0.5]

        # Re-compile and simulate with SAME block objects
        model_func2, y0_2, state_map2, block_matrices2 = compiler.compile_system(
            blocks, blocks, lines
        )

        sol2 = solve_ivp(model_func2, t_span, y0_2, method='RK45', t_eval=t_eval)
        output2 = sol2.y[0, :]

        # Verify outputs are different
        diff = np.abs(output1 - output2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        assert max_diff > 0.1, (
            f"Outputs should be significantly different after parameter change. "
            f"Max diff: {max_diff:.6f}"
        )
        assert mean_diff > 0.05, (
            f"Mean difference should be substantial. Mean diff: {mean_diff:.6f}"
        )

        # Verify the responses match expected behavior
        # For 1/(s+1) with unit step: DC gain = 1/1 = 1, τ=1, so 63% of 1 at t=1
        # For 1/(s+0.5) with unit step: DC gain = 1/0.5 = 2, τ=2, so 63% of 2 at t=2
        idx_t1 = np.argmin(np.abs(t_eval - 1.0))
        idx_t2 = np.argmin(np.abs(t_eval - 2.0))

        # First system (τ=1, DC gain=1) should be ~63% of 1.0 at t=1
        assert 0.55 < output1[idx_t1] < 0.70, (
            f"τ=1 system should be ~63% of 1.0 at t=1, got {output1[idx_t1]:.3f}"
        )

        # Second system (τ=2, DC gain=2) should be ~63% of 2.0 at t=2
        assert 1.1 < output2[idx_t2] < 1.4, (
            f"τ=2 system should be ~63% of 2.0 (≈1.26) at t=2, got {output2[idx_t2]:.3f}"
        )

    def test_gain_parameter_change(self, qapp):
        """
        Test that changing Gain parameter produces different output.

        Setup: Step → Gain → Integrator → Scope
        Use Integrator to capture gain effect in ODE state.
        - First run: gain = 2.0 → integrator ramps at slope 2
        - Second run: gain = 5.0 → integrator ramps at slope 5
        """
        step_block = _make_block('Step', 0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'})
        gain_block = _make_block('Gain', 1, '', 1, 1, {'gain': 2.0})
        integ_block = _make_block('Integrator', 2, '', 1, 1, {'init_conds': 0.0, 'method': 'RK45'})
        scope_block = _make_block('Scope', 3, '', 1, 0, {'labels': 'output'})

        lines = [
            _make_line(0, 'step0', 0, 'gain1', 0),
            _make_line(1, 'gain1', 0, 'integrator2', 0),
            _make_line(2, 'integrator2', 0, 'scope3', 0),
        ]
        blocks = [step_block, gain_block, integ_block, scope_block]

        # First simulation with gain = 2.0
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)
        model_func, y0, state_map, _ = compiler.compile_system(blocks, blocks, lines)

        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        sol1 = solve_ivp(model_func, t_span, y0, method='RK45', t_eval=t_eval)
        final1 = sol1.y[0, -1]  # Integrator of 2.0 over 5s = 10.0

        # Change gain to 5.0
        gain_block.params['gain'] = 5.0

        # Re-compile and simulate
        model_func2, y0_2, _, _ = compiler.compile_system(blocks, blocks, lines)
        sol2 = solve_ivp(model_func2, t_span, y0_2, method='RK45', t_eval=t_eval)
        final2 = sol2.y[0, -1]  # Integrator of 5.0 over 5s = 25.0

        # Verify outputs are different and in correct ratio
        assert abs(final1 - 10.0) < 0.1, f"Gain=2 should give ~10.0, got {final1:.3f}"
        assert abs(final2 - 25.0) < 0.1, f"Gain=5 should give ~25.0, got {final2:.3f}"
        assert abs(final2 / final1 - 2.5) < 0.1, f"Ratio should be ~2.5, got {final2/final1:.3f}"

    def test_step_value_change(self, qapp):
        """
        Test that changing Step value produces different output.

        Setup: Step → Integrator → Scope
        - First run: step value = 1.0
        - Second run: step value = 3.0

        The integrator output should scale proportionally.
        """
        # Create blocks
        step_block = _make_block(
            block_fn='Step',
            sid=0,
            username='Step',
            in_ports=0,
            out_ports=1,
            params={'value': 1.0, 'delay': 0.0, 'type': 'up'},
        )

        integrator_block = _make_block(
            block_fn='Integrator',
            sid=1,
            username='Integrator',
            in_ports=1,
            out_ports=1,
            params={'init_conds': 0.0, 'method': 'RK45'},
        )

        scope_block = _make_block(
            block_fn='Scope',
            sid=2,
            username='Scope',
            in_ports=1,
            out_ports=0,
            params={'labels': 'output'},
        )

        # Create connections
        line1 = _make_line(sid=0, src='step0', srcport=0, dst='integrator1', dstport=0)
        line2 = _make_line(sid=1, src='integrator1', srcport=0, dst='scope2', dstport=0)

        blocks = [step_block, integrator_block, scope_block]
        lines = [line1, line2]

        # First simulation with step value = 1.0
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks), "Blocks should be compilable"

        model_func, y0, state_map, block_matrices = compiler.compile_system(
            blocks, blocks, lines
        )

        t_span = (0, 5)
        t_eval = np.linspace(0, 5, 100)
        sol1 = solve_ivp(model_func, t_span, y0, method='RK45', t_eval=t_eval)

        # Extract integrator output (first state)
        output1 = sol1.y[0, :]
        final_value1 = output1[-1]

        # Change step value to 3.0
        step_block.params['value'] = 3.0

        # Re-compile and simulate
        model_func2, y0_2, state_map2, block_matrices2 = compiler.compile_system(
            blocks, blocks, lines
        )

        sol2 = solve_ivp(model_func2, t_span, y0_2, method='RK45', t_eval=t_eval)
        output2 = sol2.y[0, :]
        final_value2 = output2[-1]

        # Verify outputs are different
        diff = np.abs(output1 - output2)
        max_diff = np.max(diff)

        assert max_diff > 0.5, (
            f"Outputs should be significantly different. Max diff: {max_diff:.6f}"
        )

        # Verify the ratio is approximately 3.0 (integral of 3.0 vs 1.0)
        ratio = final_value2 / final_value1
        assert 2.5 < ratio < 3.5, (
            f"Final value ratio should be ~3.0, got {ratio:.3f} "
            f"(final1={final_value1:.3f}, final2={final_value2:.3f})"
        )

    def test_transfer_function_numerator_change(self, qapp):
        """
        Test that changing TranFn numerator produces different output.

        For 1/(s+1): state x_ss=1, output y=Cx=1*1=1
        For 2/(s+1): state x_ss=1, output y=Cx=2*1=2

        Since sol.y gives the raw state, we compute y=Cx+Du manually
        using the block_matrices returned by the compiler.
        """
        step_block = _make_block('Step', 0, '', 0, 1, {'value': 1.0, 'delay': 0.0, 'type': 'up'})
        tranfn_block = _make_block('TranFn', 1, '', 1, 1, {
            'numerator': [1.0], 'denominator': [1.0, 1.0],
        })
        scope_block = _make_block('Scope', 2, '', 1, 0, {'labels': 'output'})

        lines = [
            _make_line(0, 'step0', 0, 'tranfn1', 0),
            _make_line(1, 'tranfn1', 0, 'scope2', 0),
        ]
        blocks = [step_block, tranfn_block, scope_block]

        # First simulation: num=[1.0]
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks)
        model_func, y0, state_map, block_matrices = compiler.compile_system(blocks, blocks, lines)

        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 500)
        sol1 = solve_ivp(model_func, t_span, y0, method='RK45', t_eval=t_eval)

        # Compute output y = C*x + D*u for TranFn
        A1, B1, C1, D1 = block_matrices['tranfn1']
        x1_final = sol1.y[0, -1]
        y1_final = float(C1[0, 0] * x1_final + D1[0, 0] * 1.0)

        # Change numerator to [2.0]
        tranfn_block.params['numerator'] = [2.0]

        model_func2, y0_2, state_map2, block_matrices2 = compiler.compile_system(blocks, blocks, lines)
        sol2 = solve_ivp(model_func2, t_span, y0_2, method='RK45', t_eval=t_eval)

        A2, B2, C2, D2 = block_matrices2['tranfn1']
        x2_final = sol2.y[0, -1]
        y2_final = float(C2[0, 0] * x2_final + D2[0, 0] * 1.0)

        # Verify C matrix changed
        assert C1[0, 0] != C2[0, 0], f"C matrix should change: C1={C1[0,0]}, C2={C2[0,0]}"

        # Verify output steady-state differs by factor of 2
        assert np.isclose(y1_final, 1.0, atol=0.1), f"num=[1] output should be ~1.0, got {y1_final:.3f}"
        assert np.isclose(y2_final, 2.0, atol=0.1), f"num=[2] output should be ~2.0, got {y2_final:.3f}"

    def test_multiple_parameter_changes(self, qapp):
        """
        Test changing multiple parameters simultaneously.

        Setup: Step → Gain → TranFn → Scope
        Change both gain and transfer function denominator.
        """
        # Create blocks
        step_block = _make_block(
            block_fn='Step',
            sid=0,
            username='Step',
            in_ports=0,
            out_ports=1,
            params={'value': 1.0, 'delay': 0.0, 'type': 'up'},
        )

        gain_block = _make_block(
            block_fn='Gain',
            sid=1,
            username='Gain',
            in_ports=1,
            out_ports=1,
            params={'gain': 1.0},
        )

        tranfn_block = _make_block(
            block_fn='TranFn',
            sid=2,
            username='TranFn',
            in_ports=1,
            out_ports=1,
            params={
                'numerator': [1.0],
                'denominator': [1.0, 1.0],
            },
        )

        scope_block = _make_block(
            block_fn='Scope',
            sid=3,
            username='Scope',
            in_ports=1,
            out_ports=0,
            params={'labels': 'output'},
        )

        # Create connections
        line1 = _make_line(sid=0, src='step0', srcport=0, dst='gain1', dstport=0)
        line2 = _make_line(sid=1, src='gain1', srcport=0, dst='tranfn2', dstport=0)
        line3 = _make_line(sid=2, src='tranfn2', srcport=0, dst='scope3', dstport=0)

        blocks = [step_block, gain_block, tranfn_block, scope_block]
        lines = [line1, line2, line3]

        # First simulation
        compiler = SystemCompiler()
        assert compiler.check_compilability(blocks), "Blocks should be compilable"

        model_func, y0, state_map, block_matrices = compiler.compile_system(
            blocks, blocks, lines
        )

        t_span = (0, 10)
        t_eval = np.linspace(0, 10, 500)
        sol1 = solve_ivp(model_func, t_span, y0, method='RK45', t_eval=t_eval)

        output1 = sol1.y[0, :]

        # Change both parameters
        gain_block.params['gain'] = 3.0
        tranfn_block.params['denominator'] = [1.0, 0.5]

        # Re-compile and simulate
        model_func2, y0_2, state_map2, block_matrices2 = compiler.compile_system(
            blocks, blocks, lines
        )

        sol2 = solve_ivp(model_func2, t_span, y0_2, method='RK45', t_eval=t_eval)
        output2 = sol2.y[0, :]

        # Verify outputs are significantly different
        diff = np.abs(output1 - output2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        assert max_diff > 0.5, (
            f"Outputs should be very different after multiple parameter changes. "
            f"Max diff: {max_diff:.6f}"
        )
        assert mean_diff > 0.2, (
            f"Mean difference should be substantial. Mean diff: {mean_diff:.6f}"
        )

        # Verify steady-state reflects parameter changes
        # First: gain=1, den=[1,1] → DC = 1*1/1 = 1.0
        # Second: gain=3, den=[1,0.5] → DC = 3*1/0.5 = 6.0
        # Note: sol.y[0] is the TranFn state. Output y = C*x + D*u.
        # For 1/(s+a): A=-a, C=1, so output ≈ state at steady state.
        # Compute actual output using block_matrices
        A1, B1, C1, D1 = block_matrices['tranfn2']
        y1_ss = float(C1[0, 0] * output1[-1])

        A2, B2, C2, D2 = block_matrices2['tranfn2']
        y2_ss = float(C2[0, 0] * output2[-1])

        assert np.isclose(y1_ss, 1.0, atol=0.1), (
            f"First system output should settle to ~1.0, got {y1_ss:.3f}"
        )
        assert np.isclose(y2_ss, 6.0, atol=0.3), (
            f"Second system output should settle to ~6.0 (gain=3 * DC=2), got {y2_ss:.3f}"
        )
