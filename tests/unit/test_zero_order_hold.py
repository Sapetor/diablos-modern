"""Unit tests for the ZeroOrderHold block.

Regression focus: a ``sampling_time <= 0`` used to hang the whole application.
``while next_sample <= time + eps: next_sample += sampling_time`` never
terminates when ``sampling_time`` is 0 or negative, and the GUI calls
``execute()`` from the repaint timer, so the window froze with no way out.
"""

import numpy as np
import pytest

from blocks.zero_order_hold import ZeroOrderHoldBlock


@pytest.mark.unit
class TestZeroOrderHoldSampling:
    def test_holds_between_samples(self):
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": 0.1, "_init_start_": True}

        first = block.execute(0.0, {0: np.array([1.0])}, params)
        assert np.isclose(first[0][0], 1.0)

        # Before the next sample instant the old value is held.
        held = block.execute(0.05, {0: np.array([9.0])}, params)
        assert np.isclose(held[0][0], 1.0)

        # At the next sample instant the new value is taken.
        sampled = block.execute(0.1, {0: np.array([9.0])}, params)
        assert np.isclose(sampled[0][0], 9.0)

    def test_schedule_advances_past_a_large_time_jump(self):
        """A single call far past the schedule must catch the schedule up."""
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": 0.1, "_init_start_": True}

        block.execute(0.0, {0: np.array([1.0])}, params)
        block.execute(5.0, {0: np.array([2.0])}, params)
        assert params["_next_sample_time_"] > 5.0


@pytest.mark.unit
@pytest.mark.timeout(10)
class TestZeroOrderHoldDegenerateSampleTime:
    """``sampling_time <= 0`` must return promptly, not spin forever."""

    @pytest.mark.parametrize("sampling_time", [0.0, -0.1, -1.0])
    def test_returns_promptly_and_passes_input_through(self, sampling_time):
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": sampling_time, "_init_start_": True}

        result = block.execute(0.0, {0: np.array([3.0])}, params)
        assert np.isclose(result[0][0], 3.0)

        # ... and keeps following the input on later calls (pass-through).
        result = block.execute(0.01, {0: np.array([4.0])}, params)
        assert np.isclose(result[0][0], 4.0)

    @pytest.mark.parametrize("sampling_time", [0.0, -0.5])
    def test_output_only_does_not_consume_the_input(self, sampling_time):
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": sampling_time, "_init_start_": True}

        block.execute(0.0, {0: np.array([3.0])}, params)
        held = block.execute(0.01, {0: np.array([7.0])}, params, output_only=True)
        assert np.isclose(held[0][0], 3.0)

    def test_nan_sample_time_returns_promptly(self):
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": float("nan"), "_init_start_": True}

        result = block.execute(0.0, {0: np.array([2.5])}, params)
        assert np.isclose(result[0][0], 2.5)

    def test_unparsable_sample_time_falls_back_to_the_default_rate(self):
        block = ZeroOrderHoldBlock()
        params = {"sampling_time": "not-a-number", "_init_start_": True}

        result = block.execute(0.0, {0: np.array([2.5])}, params)
        assert np.isclose(result[0][0], 2.5)
        # safe_float default (0.1) is used, so a real schedule is set up.
        assert params["_next_sample_time_"] > 0.0
