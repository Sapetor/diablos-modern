"""Unit tests for the FirstOrderHold block.

Regression focus: ``input_sample_time = 0`` used to hang the application. The
block guarded only ``sampling_time < 0`` (remapping the continuous marker to
0.1), so exactly 0 fell through to
``while next_sample <= time + eps: next_sample += 0`` and spun forever.
"""

import numpy as np
import pytest

from blocks.first_order_hold import FirstOrderHoldBlock


@pytest.mark.unit
class TestFirstOrderHoldSampling:
    def test_continuous_marker_still_maps_to_the_default_rate(self):
        """A negative period is the "continuous" marker, not a degenerate rate."""
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": -1.0, "_init_start_": True}

        block.execute(0.0, {0: 1.0}, params)
        # The -1 marker is remapped to 0.1, so a real schedule is set up.
        assert np.isclose(params["_next_sample_time_"], 0.1)

    def test_extrapolates_between_samples(self):
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": 0.1, "_init_start_": True}

        block.execute(0.0, {0: 0.0}, params)
        block.execute(0.1, {0: 1.0}, params)
        # Slope is 1.0/0.1 = 10 per second; half a period later -> 1.0 + 0.5.
        mid = block.execute(0.15, {0: 1.0}, params)
        assert np.isclose(float(np.atleast_1d(mid[0])[0]), 1.5)


@pytest.mark.unit
@pytest.mark.timeout(10)
class TestFirstOrderHoldDegenerateSampleTime:
    """``input_sample_time == 0`` must return promptly, not spin forever."""

    def test_zero_sample_time_returns_promptly_and_passes_input_through(self):
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": 0.0, "_init_start_": True}

        result = block.execute(0.0, {0: 2.0}, params)
        assert np.isclose(float(np.atleast_1d(result[0])[0]), 2.0)

        result = block.execute(0.01, {0: 5.0}, params)
        assert np.isclose(float(np.atleast_1d(result[0])[0]), 5.0)

    def test_zero_sample_time_never_leaves_a_ramp_armed(self):
        """Pass-through must not extrapolate off a stale sample pair."""
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": 0.0, "_init_start_": True}

        block.execute(0.0, {0: 0.0}, params)
        block.execute(1.0, {0: 1.0}, params)
        far = block.execute(100.0, {0: 1.0}, params)
        assert np.isclose(float(np.atleast_1d(far[0])[0]), 1.0)

    def test_nan_sample_time_returns_promptly(self):
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": float("nan"), "_init_start_": True}

        result = block.execute(0.0, {0: 2.0}, params)
        assert np.isclose(float(np.atleast_1d(result[0])[0]), 2.0)

    def test_zero_sample_time_with_output_only_does_not_consume_input(self):
        block = FirstOrderHoldBlock()
        params = {"input_sample_time": 0.0, "_init_start_": True}

        block.execute(0.0, {0: 2.0}, params)
        held = block.execute(0.01, {0: 9.0}, params, output_only=True)
        assert np.isclose(float(np.atleast_1d(held[0])[0]), 2.0)
