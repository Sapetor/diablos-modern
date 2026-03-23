"""Unit tests for MatrixGain block."""

import pytest
import numpy as np


@pytest.mark.unit
class TestMatrixGainBlock:
    """Tests for standalone MatrixGain block."""

    def test_block_properties(self):
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        assert block.block_name == "MatrixGain"
        assert block.category == "Math"
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1

    def test_scalar_gain_string(self):
        """String '2.5' parsed as scalar gain."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': '2.5'}
        result = block.execute(0, {0: np.array([3.0])}, params)
        assert result[0] == pytest.approx([7.5])

    def test_vector_gain_string(self):
        """String '[2, 3]' parsed as element-wise vector gain."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': '[2, 3]'}
        result = block.execute(0, {0: np.array([1.0, 2.0])}, params)
        np.testing.assert_array_almost_equal(result[0], [2.0, 6.0])

    def test_matrix_gain_string(self):
        """String '[[1, 2], [3, 4]]' parsed as matrix gain (K @ u)."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': '[[1, 2], [3, 4]]'}
        result = block.execute(0, {0: np.array([1.0, 0.0])}, params)
        np.testing.assert_array_almost_equal(result[0], [1.0, 3.0])

    def test_numeric_gain(self):
        """Direct numeric value (already resolved by workspace)."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': 3.0}
        result = block.execute(0, {0: np.array([2.0])}, params)
        assert result[0] == pytest.approx([6.0])

    def test_numeric_list_gain(self):
        """Already-resolved list (from workspace)."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': [[1, 0], [0, 2]]}
        result = block.execute(0, {0: np.array([3.0, 4.0])}, params)
        np.testing.assert_array_almost_equal(result[0], [3.0, 8.0])

    def test_workspace_variable_resolution(self):
        """Workspace variable name resolves to matrix at sim time."""
        from blocks.matrix_gain import MatrixGainBlock
        from lib.workspace import WorkspaceManager

        # Set up workspace
        wm = WorkspaceManager()
        old_vars = wm.variables.copy()
        wm.variables['K'] = [[2, 0], [0, 3]]

        try:
            block = MatrixGainBlock()
            # Simulate what the engine does: resolve_params before execute
            params = {'gain': 'K'}
            resolved = wm.resolve_params(params)
            result = block.execute(0, {0: np.array([1.0, 1.0])}, resolved)
            np.testing.assert_array_almost_equal(result[0], [2.0, 3.0])
        finally:
            wm.variables = old_vars

    def test_default_is_string(self):
        """Default gain is a string so property editor uses QLineEdit."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        meta = block.params['gain']
        assert isinstance(meta['default'], str)

    def test_dimension_mismatch_pads(self):
        """Matrix with more columns than input length pads with zeros."""
        from blocks.matrix_gain import MatrixGainBlock
        block = MatrixGainBlock()
        params = {'gain': '[[1, 2, 3]]'}
        result = block.execute(0, {0: np.array([1.0])}, params)
        # u padded to [1, 0, 0], K@u = [1*1 + 2*0 + 3*0] = [1]
        np.testing.assert_array_almost_equal(result[0], [1.0])

    def test_compiled_solver(self, qapp):
        """MatrixGain works in compiled solver with scalar string gain."""
        from PyQt5.QtCore import QRect, QPoint
        from PyQt5.QtGui import QColor
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine
        from lib.engine.system_compiler import SystemCompiler
        from scipy.integrate import solve_ivp

        blocks = [
            DBlock(block_fn='Step', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=0, out_ports=1,
                   params={'value': 1.0, 'delay': 0.0, 'type': 'up'},
                   username='', b_type=0),
            DBlock(block_fn='MatrixGain', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=1, out_ports=1,
                   params={'gain': '3.0'},
                   username='', b_type=2),
            DBlock(block_fn='TranFn', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=1, out_ports=1,
                   params={'numerator': [1.0], 'denominator': [1.0, 1.0]},
                   username='', b_type=1),
        ]
        lines = [
            DLine(sid=0, srcblock='step0', srcport=0,
                  dstblock='matrixgain0', dstport=0,
                  points=[QPoint(0, 0), QPoint(100, 0)]),
            DLine(sid=1, srcblock='matrixgain0', srcport=0,
                  dstblock='tranfn0', dstport=0,
                  points=[QPoint(0, 0), QPoint(100, 0)]),
        ]

        compiler = SystemCompiler()
        model_func, y0, state_map, _ = compiler.compile_system(
            blocks, blocks, lines
        )
        sol = solve_ivp(model_func, (0, 10), y0, method='RK45',
                        t_eval=np.linspace(0, 10, 1000), rtol=1e-8, atol=1e-10)
        assert sol.success

        # Step(1) -> Gain(3) -> 1/(s+1): SS = 3.0
        start, _ = state_map['tranfn0']
        assert sol.y[start, -1] == pytest.approx(3.0, abs=0.05)
