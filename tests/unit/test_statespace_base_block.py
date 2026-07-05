"""Unit tests for StateSpaceBaseBlock helper methods.

StateSpaceBaseBlock is abstract, so it is exercised here through a minimal
concrete subclass that supplies the required abstract members. The tests target
the pure computational helpers shared by TranFn / StateSpace subclasses: matrix
dimension validation/reshaping, state-vector initialization (pad/truncate),
input processing, the y = Cx + Du output, the x[k+1] = Ax + Bu update, and
output formatting.
"""

import numpy as np
import pytest

from blocks.statespace_base import StateSpaceBaseBlock


class _ConcreteStateSpace(StateSpaceBaseBlock):
    """Minimal instantiable subclass for exercising the base helpers."""

    @property
    def block_name(self):
        return "ConcreteStateSpace"

    @property
    def fn_name(self):
        return "ConcreteStateSpace"

    @property
    def params(self):
        return {}

    def execute(self, time, inputs, params, **kwargs):
        return {}


@pytest.fixture
def block():
    return _ConcreteStateSpace()


@pytest.mark.unit
class TestValidateMatrices:
    def test_valid_siso_reshapes_1d_matrices(self, block):
        # SISO first-order: A=[-1], B=[1], C=[1], D=[0] given as scalars/1D.
        result = block._validate_state_space_matrices([[-1.0]], [1.0], [1.0], [0.0])
        assert not isinstance(result, dict), "valid matrices must not error"
        A, B, C, D, n, m, p = result
        assert (n, m, p) == (1, 1, 1)
        assert B.shape == (1, 1)
        assert C.shape == (1, 1)
        assert D.shape == (1, 1)

    def test_valid_mimo_dimensions(self, block):
        A = [[0.0, 1.0], [-2.0, -3.0]]
        B = [[0.0], [1.0]]
        C = [[1.0, 0.0]]
        D = [[0.0]]
        A, B, C, D, n, m, p = block._validate_state_space_matrices(A, B, C, D)
        assert (n, m, p) == (2, 1, 1)

    def test_non_square_a_errors(self, block):
        result = block._validate_state_space_matrices(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [1.0], [1.0], [0.0]
        )
        assert isinstance(result, dict) and result['E'] is True
        assert 'square' in result['error']

    def test_b_row_mismatch_errors(self, block):
        # A is 2x2 but B has only 1 row.
        result = block._validate_state_space_matrices(
            [[0.0, 1.0], [-1.0, -1.0]], [[1.0]], [[1.0, 0.0]], [[0.0]]
        )
        assert isinstance(result, dict) and result['E'] is True
        assert 'B matrix' in result['error']

    def test_c_column_mismatch_errors(self, block):
        # A is 2x2 but C has only 1 column.
        result = block._validate_state_space_matrices(
            [[0.0, 1.0], [-1.0, -1.0]], [[0.0], [1.0]], [[1.0]], [[0.0]]
        )
        assert isinstance(result, dict) and result['E'] is True
        assert 'C matrix' in result['error']

    def test_d_shape_mismatch_errors(self, block):
        # p=1, m=1 expected, but D is 2x1.
        result = block._validate_state_space_matrices(
            [[-1.0]], [[1.0]], [[1.0]], [[0.0], [0.0]]
        )
        assert isinstance(result, dict) and result['E'] is True
        assert 'D matrix' in result['error']


@pytest.mark.unit
class TestInitializeStateVector:
    def test_exact_length(self, block):
        x = block._initialize_state_vector(2, [1.0, 2.0])
        assert x.shape == (2, 1)
        assert np.allclose(x.flatten(), [1.0, 2.0])

    def test_pads_with_zeros(self, block):
        x = block._initialize_state_vector(3, [5.0])
        assert x.shape == (3, 1)
        assert np.allclose(x.flatten(), [5.0, 0.0, 0.0])

    def test_truncates_extra(self, block):
        x = block._initialize_state_vector(2, [1.0, 2.0, 3.0, 4.0])
        assert x.shape == (2, 1)
        assert np.allclose(x.flatten(), [1.0, 2.0])

    def test_scalar_init(self, block):
        x = block._initialize_state_vector(1, 7.0)
        assert x.shape == (1, 1)
        assert x.item() == 7.0


@pytest.mark.unit
class TestProcessInput:
    def test_scalar_input(self, block):
        u, err = block._process_input({0: 3.0}, 1)
        assert err is None
        assert u.shape == (1, 1)
        assert u.item() == 3.0

    def test_array_input(self, block):
        u, err = block._process_input({0: np.array([1.0, 2.0])}, 2)
        assert err is None
        assert u.shape == (2, 1)
        assert np.allclose(u.flatten(), [1.0, 2.0])

    def test_missing_input_defaults_zero(self, block):
        u, err = block._process_input({}, 1)
        assert err is None
        assert u.item() == 0.0

    def test_output_only_returns_zeros(self, block):
        u, err = block._process_input({0: 5.0}, 3, output_only=True)
        assert err is None
        assert u.shape == (3, 1)
        assert np.allclose(u, 0.0)

    def test_dimension_mismatch_errors(self, block):
        u, err = block._process_input({0: np.array([1.0, 2.0])}, 3)
        assert u is None
        assert err['E'] is True
        assert 'dimension mismatch' in err['error']


@pytest.mark.unit
class TestComputeOutput:
    def test_output_equation(self, block):
        C = np.array([[2.0, 0.0]])
        D = np.array([[1.0]])
        x = np.array([[3.0], [4.0]])
        u = np.array([[5.0]])
        y, err = block._compute_output(C, D, x, u)
        assert err is None
        # y = 2*3 + 0*4 + 1*5 = 11
        assert np.isclose(y.item(), 11.0)

    def test_incompatible_shapes_errors(self, block):
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])
        x = np.array([[1.0]])  # wrong length for C
        u = np.array([[0.0]])
        y, err = block._compute_output(C, D, x, u)
        assert y is None
        assert err['E'] is True


@pytest.mark.unit
class TestUpdateState:
    def test_state_update_equation(self, block):
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        x = np.array([[1.0], [2.0]])
        u = np.array([[3.0]])
        params = {}
        err = block._update_state(A, B, x, u, params)
        assert err is None
        # x_next = A x + B u = [2, 0] + [0, 3] = [2, 3]
        assert np.allclose(params['_x_'].flatten(), [2.0, 3.0])

    def test_incompatible_shapes_errors(self, block):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        B = np.array([[1.0]])
        x = np.array([[1.0]])  # wrong length for A
        u = np.array([[1.0]])
        params = {}
        err = block._update_state(A, B, x, u, params)
        assert err['E'] is True
        assert '_x_' not in params


@pytest.mark.unit
class TestFormatOutput:
    def test_scalar_output(self, block):
        y = np.array([[42.0]])
        out = block._format_output(y)
        assert np.isscalar(out) or isinstance(out, float)
        assert out == 42.0

    def test_vector_output_flattened(self, block):
        y = np.array([[1.0], [2.0], [3.0]])
        out = block._format_output(y)
        assert out.shape == (3,)
        assert np.allclose(out, [1.0, 2.0, 3.0])
