"""Unit tests for the continuous TransferFunction block.

Regression focus: the block is SISO but did not check its input width. A vector
input reached ``y.item()`` and raised ``ValueError: can only convert an array of
size 1 to a Python scalar`` out of ``execute()``, crashing the run instead of
returning the ``{'E': True, ...}`` dict the engine knows how to report. It now
uses the same ``_process_input`` check as StateSpace.
"""

import numpy as np
import pytest

from blocks.transfer_function import TransferFunctionBlock


def _params(**ov):
    p = {
        "numerator": [1.0],
        "denominator": [1.0, 1.0],
        "_init_start_": True,
        "init_conds": [0.0],
    }
    p.update(ov)
    return p


@pytest.mark.unit
class TestTransferFunctionScalarPath:
    def test_first_order_lag_step_response_starts_at_zero(self):
        block = TransferFunctionBlock()
        params = _params()

        first = block.execute(0.0, {0: 1.0}, params, dtime=0.01)
        assert first["E"] is False
        assert np.isclose(first[0], 0.0)

    def test_first_order_lag_approaches_unity(self):
        block = TransferFunctionBlock()
        params = _params()

        dt = 0.01
        y = 0.0
        for k in range(500):
            y = block.execute(k * dt, {0: 1.0}, params, dtime=dt)[0]
        # 1 - e^{-5} = 0.9933
        assert np.isclose(y, 1.0 - np.exp(-5.0), atol=2e-2)

    def test_size_one_array_input_is_accepted(self):
        block = TransferFunctionBlock()
        params = _params()

        result = block.execute(0.0, {0: np.array([1.0])}, params, dtime=0.01)
        assert result["E"] is False
        assert np.isscalar(result[0]) or np.ndim(result[0]) == 0

    def test_output_only_does_not_advance_the_state(self):
        block = TransferFunctionBlock()
        params = _params()

        block.execute(0.0, {0: 1.0}, params, dtime=0.01)
        state_before = np.array(params["_x_"], copy=True)
        block.execute(0.01, {0: 1.0}, params, dtime=0.01, output_only=True)
        np.testing.assert_allclose(params["_x_"], state_before)


@pytest.mark.unit
class TestTransferFunctionVectorInput:
    """A vector into a SISO transfer function must be a clear error, not a crash."""

    @pytest.mark.parametrize("u", [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])
    def test_vector_input_returns_an_error_dict(self, u):
        block = TransferFunctionBlock()
        params = _params()

        result = block.execute(0.0, {0: u}, params, dtime=0.01)

        assert result.get("E") is True
        assert "dimension" in result["error"].lower()
        assert 0 not in result

    def test_error_matches_the_statespace_wording(self):
        """Same check, same message as blocks/statespace_base.py::_process_input."""
        block = TransferFunctionBlock()
        params = _params()

        result = block.execute(0.0, {0: np.array([1.0, 2.0])}, params, dtime=0.01)

        assert result["error"] == "Input dimension mismatch: expected 1, got 2"

    def test_vector_input_does_not_corrupt_the_state(self):
        block = TransferFunctionBlock()
        params = _params()

        block.execute(0.0, {0: 1.0}, params, dtime=0.01)
        state_before = np.array(params["_x_"], copy=True)

        block.execute(0.01, {0: np.array([1.0, 2.0])}, params, dtime=0.01)
        np.testing.assert_allclose(params["_x_"], state_before)
