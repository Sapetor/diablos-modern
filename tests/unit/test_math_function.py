"""Error-reporting tests for the MathFunction block.

Regression focus: any exception inside ``execute`` used to be swallowed and
replaced with ``np.zeros_like(u)``. An all-zero signal is indistinguishable from
a legitimate result, so a typo in an expression or a non-numeric input silently
falsified the whole run instead of stopping it.
"""

import numpy as np
import pytest

from blocks.math_function import MathFunctionBlock


@pytest.mark.unit
class TestMathFunctionErrorReporting:
    def test_non_numeric_input_returns_an_error_dict(self):
        block = MathFunctionBlock()
        result = block.execute(time=0.0, inputs={0: "not-a-number"}, params={"function": "sin"})

        assert result.get("E") is True
        assert "error" in result
        # And emphatically not a plausible-looking zero signal.
        assert 0 not in result

    def test_unparsable_expression_returns_an_error_dict(self):
        block = MathFunctionBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0])},
            params={"function": "u ** ("},
        )

        assert result.get("E") is True
        assert "error" in result

    def test_expression_with_an_unknown_name_returns_an_error_dict(self):
        block = MathFunctionBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0])},
            params={"function": "no_such_function(u)"},
        )

        assert result.get("E") is True

    def test_error_message_names_the_offending_function(self):
        block = MathFunctionBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0])},
            params={"function": "u ** (", "_name_": "MF1"},
        )

        assert "u ** (" in result["error"]

    def test_valid_expression_still_evaluates(self):
        """The error path must not have swallowed the Python-expression fallback."""
        block = MathFunctionBlock()
        result = block.execute(
            time=2.0,
            inputs={0: np.array([3.0])},
            params={"function": "u**2 + t"},
        )

        assert result.get("E") is not True
        assert np.isclose(np.atleast_1d(result[0])[0], 11.0)

    def test_named_function_on_a_vector_is_unaffected(self):
        block = MathFunctionBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0, 4.0, 9.0])},
            params={"function": "sqrt"},
        )

        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])
