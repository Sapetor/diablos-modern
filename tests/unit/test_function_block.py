"""
Unit tests for the Function block (blocks/function.py).

The Function block evaluates a user-supplied Python expression against its
input ports. Inputs are exposed both as the 0-indexed list ``u`` (``u[0]``,
``u[1]``, ...) and as the 1-indexed aliases ``u1``, ``u2``, ...; the current
simulation time is bound to ``t``. Evaluation goes through the hardened
``safe_expr`` AST walker, so arbitrary Python (imports, attribute access,
comprehensions) is rejected with an error dict rather than executed.
"""

import numpy as np
import pytest

from blocks.function import FunctionBlock


@pytest.mark.unit
class TestFunctionBlockMetadata:
    def test_block_name(self):
        assert FunctionBlock().block_name == "Function"

    def test_category(self):
        assert FunctionBlock().category == "Math"

    def test_io_editable_is_input(self):
        # Variable number of input ports, edited via the property-editor spinner.
        assert FunctionBlock().io_editable == "input"

    def test_single_default_input_port(self):
        assert len(FunctionBlock().inputs) == 1

    def test_single_output_port(self):
        assert len(FunctionBlock().outputs) == 1


@pytest.mark.unit
class TestFunctionBlockSingleInput:
    def test_identity(self):
        block = FunctionBlock()
        params = {"expression": "u[0]", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([3.0])}, params=params)
        assert np.isclose(result[0][0], 3.0)

    def test_square(self):
        block = FunctionBlock()
        params = {"expression": "u[0]**2", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([4.0])}, params=params)
        assert np.isclose(result[0][0], 16.0)

    def test_time_dependence(self):
        block = FunctionBlock()
        params = {"expression": "u[0] + t", "_inputs_": 1}
        result = block.execute(time=2.0, inputs={0: np.array([1.0])}, params=params)
        assert np.isclose(result[0][0], 3.0)

    def test_bare_numpy_function(self):
        block = FunctionBlock()
        params = {"expression": "sin(u[0])", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([np.pi / 2])}, params=params)
        assert np.isclose(result[0][0], 1.0)

    def test_np_prefixed_function(self):
        block = FunctionBlock()
        params = {"expression": "np.cos(u[0])", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([0.0])}, params=params)
        assert np.isclose(result[0][0], 1.0)

    def test_input_count_inferred_when_no_inputs_param(self):
        # Direct unit calls may omit the engine-managed '_inputs_' key.
        block = FunctionBlock()
        result = block.execute(time=0.0, inputs={0: np.array([5.0])}, params={"expression": "u[0] * 2"})
        assert np.isclose(result[0][0], 10.0)


@pytest.mark.unit
class TestFunctionBlockMultiInput:
    def test_sum_two_inputs_zero_indexed(self):
        block = FunctionBlock()
        params = {"expression": "u[0] + u[1]", "_inputs_": 2}
        result = block.execute(time=0.0, inputs={0: np.array([1.0]), 1: np.array([2.0])}, params=params)
        assert np.isclose(result[0][0], 3.0)

    def test_product_one_indexed_aliases(self):
        block = FunctionBlock()
        params = {"expression": "u1 * u2", "_inputs_": 2}
        result = block.execute(time=0.0, inputs={0: np.array([2.0]), 1: np.array([3.0])}, params=params)
        assert np.isclose(result[0][0], 6.0)

    def test_nonlinear_two_input(self):
        block = FunctionBlock()
        params = {"expression": "sin(u[0]) + u[1]**2", "_inputs_": 2}
        result = block.execute(time=0.0, inputs={0: np.array([0.0]), 1: np.array([3.0])}, params=params)
        assert np.isclose(result[0][0], 9.0)

    def test_missing_input_defaults_to_zero(self):
        block = FunctionBlock()
        params = {"expression": "u[0] + u[1]", "_inputs_": 2}
        result = block.execute(time=0.0, inputs={0: np.array([5.0])}, params=params)
        assert np.isclose(result[0][0], 5.0)


@pytest.mark.unit
class TestFunctionBlockVectorOutput:
    def test_list_expression_yields_vector(self):
        block = FunctionBlock()
        params = {"expression": "[u[0], u[0] * 2]", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([5.0])}, params=params)
        assert np.allclose(np.asarray(result[0]).flatten(), [5.0, 10.0])


@pytest.mark.unit
class TestFunctionBlockErrors:
    def test_syntax_error_returns_error_dict(self):
        block = FunctionBlock()
        params = {"expression": "u[0] +", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params)
        assert result.get("E") is True
        assert "error" in result

    def test_import_is_rejected(self):
        block = FunctionBlock()
        params = {"expression": "__import__('os')", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params)
        assert result.get("E") is True

    def test_attribute_escape_is_rejected(self):
        # Attribute access is only allowed on np/math, not arbitrary objects.
        block = FunctionBlock()
        params = {"expression": "t.__class__", "_inputs_": 1}
        result = block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params)
        assert result.get("E") is True
