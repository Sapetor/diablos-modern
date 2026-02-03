"""
Unit tests for Math block implementations.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestGainBlock:
    """Tests for Gain block."""

    def test_scalar_gain(self):
        """Test scalar gain multiplication."""
        from blocks.gain import GainBlock
        block = GainBlock()
        params = {'gain': 2.0}
        inputs = {0: np.array([3.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 6.0, "Scalar gain should multiply input"

    def test_negative_gain(self):
        """Test negative gain (inversion)."""
        from blocks.gain import GainBlock
        block = GainBlock()
        params = {'gain': -1.0}
        inputs = {0: np.array([5.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == -5.0, "Negative gain should invert signal"

    def test_vector_input_scalar_gain(self):
        """Test scalar gain applied to vector input."""
        from blocks.gain import GainBlock
        block = GainBlock()
        params = {'gain': 3.0}
        inputs = {0: np.array([1.0, 2.0, 3.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        expected = np.array([3.0, 6.0, 9.0])
        assert np.allclose(result[0], expected), "Scalar gain on vector"

    def test_matrix_gain(self):
        """Test matrix gain for MIMO systems."""
        from blocks.gain import GainBlock
        block = GainBlock()
        # 2x2 matrix gain
        params = {'gain': [[1, 2], [3, 4]]}
        inputs = {0: np.array([1.0, 1.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        # y = [[1,2],[3,4]] @ [1,1]^T = [3, 7]
        expected = np.array([3.0, 7.0])
        assert np.allclose(result[0], expected), "Matrix gain multiplication"

    def test_zero_gain(self):
        """Test zero gain produces zero output."""
        from blocks.gain import GainBlock
        block = GainBlock()
        params = {'gain': 0.0}
        inputs = {0: np.array([100.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 0.0, "Zero gain should zero output"


@pytest.mark.unit
class TestSumBlock:
    """Tests for Sum block."""

    def test_sum_two_inputs(self):
        """Test summing two inputs."""
        from blocks.sum import SumBlock
        block = SumBlock()
        params = {'sign': '++'}
        inputs = {0: np.array([3.0]), 1: np.array([5.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 8.0, "Sum of two positive inputs"

    def test_sum_subtraction(self):
        """Test subtraction (+-) operation."""
        from blocks.sum import SumBlock
        block = SumBlock()
        params = {'sign': '+-'}
        inputs = {0: np.array([10.0]), 1: np.array([3.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 7.0, "Subtraction: 10 - 3 = 7"

    def test_sum_three_inputs(self):
        """Test summing three inputs."""
        from blocks.sum import SumBlock
        block = SumBlock()
        params = {'sign': '++-'}
        inputs = {0: np.array([1.0]), 1: np.array([2.0]), 2: np.array([0.5])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 2.5, "Sum: 1 + 2 - 0.5 = 2.5"

    def test_sum_vectors(self):
        """Test summing vector inputs."""
        from blocks.sum import SumBlock
        block = SumBlock()
        params = {'sign': '++'}
        inputs = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        expected = np.array([4.0, 6.0])
        assert np.allclose(result[0], expected), "Vector sum"


@pytest.mark.unit
class TestMathFunctionBlock:
    """Tests for MathFunction block."""

    def test_sin_function(self):
        """Test sine function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'sin'}
        inputs = {0: np.array([np.pi/2])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - 1.0) < 1e-10, "sin(pi/2) = 1"

    def test_cos_function(self):
        """Test cosine function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'cos'}
        inputs = {0: np.array([0.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - 1.0) < 1e-10, "cos(0) = 1"

    def test_exp_function(self):
        """Test exponential function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'exp'}
        inputs = {0: np.array([1.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - np.e) < 1e-10, "exp(1) = e"

    def test_log_function(self):
        """Test natural log function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'log'}
        inputs = {0: np.array([np.e])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - 1.0) < 1e-10, "log(e) = 1"

    def test_sqrt_function(self):
        """Test square root function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'sqrt'}
        inputs = {0: np.array([16.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - 4.0) < 1e-10, "sqrt(16) = 4"

    def test_square_function(self):
        """Test square function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'square'}
        inputs = {0: np.array([5.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert abs(result[0][0] - 25.0) < 1e-10, "square(5) = 25"

    def test_abs_function(self):
        """Test absolute value function."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'abs'}
        inputs = {0: np.array([-7.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert result[0][0] == 7.0, "abs(-7) = 7"

    def test_function_on_vector(self):
        """Test math function applied element-wise to vector."""
        from blocks.math_function import MathFunctionBlock
        block = MathFunctionBlock()
        params = {'function': 'square'}
        inputs = {0: np.array([1.0, 2.0, 3.0])}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        expected = np.array([1.0, 4.0, 9.0])
        assert np.allclose(result[0], expected), "square([1,2,3]) = [1,4,9]"


@pytest.mark.unit
class TestDerivativeBlock:
    """Tests for Derivative block."""

    def test_derivative_constant_is_zero(self):
        """Test derivative of constant is zero."""
        from blocks.derivative import DerivativeBlock
        block = DerivativeBlock()
        params = {'dt': 0.01, '_last_value_': None, '_last_time_': None}

        # Feed constant value
        block.execute(time=0.0, inputs={0: np.array([5.0])}, params=params)
        result = block.execute(time=0.01, inputs={0: np.array([5.0])}, params=params)

        assert abs(result[0][0]) < 1e-10, "Derivative of constant should be 0"

    def test_derivative_linear_ramp(self):
        """Test derivative of linear ramp is constant."""
        from blocks.derivative import DerivativeBlock
        block = DerivativeBlock()
        params = {'dt': 0.01, '_last_value_': None, '_last_time_': None}

        # Linear ramp with slope 2
        block.execute(time=0.0, inputs={0: np.array([0.0])}, params=params)
        result = block.execute(time=0.01, inputs={0: np.array([0.02])}, params=params)

        # Derivative should be slope = 2
        assert abs(result[0][0] - 2.0) < 0.1, "Derivative of ramp should equal slope"


@pytest.mark.unit
class TestProductBlock:
    """Tests for Signal Product block."""

    def test_product_two_scalars(self):
        """Test product of two scalar inputs."""
        from blocks.sigproduct import SigProductBlock
        block = SigProductBlock()
        params = {}
        inputs = {0: 3.0, 1: 4.0}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert np.isclose(result[0], 12.0), "3 * 4 = 12"

    def test_product_with_zero(self):
        """Test product with zero gives zero."""
        from blocks.sigproduct import SigProductBlock
        block = SigProductBlock()
        params = {}
        inputs = {0: 5.0, 1: 0.0}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert np.isclose(result[0], 0.0), "5 * 0 = 0"

    def test_product_negative_numbers(self):
        """Test product with negative numbers."""
        from blocks.sigproduct import SigProductBlock
        block = SigProductBlock()
        params = {}
        inputs = {0: -3.0, 1: 4.0}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert np.isclose(result[0], -12.0), "-3 * 4 = -12"

    def test_product_identity(self):
        """Test product with 1 is identity."""
        from blocks.sigproduct import SigProductBlock
        block = SigProductBlock()
        params = {}
        inputs = {0: 7.5, 1: 1.0}

        result = block.execute(time=0.0, inputs=inputs, params=params)
        assert np.isclose(result[0], 7.5), "7.5 * 1 = 7.5"
