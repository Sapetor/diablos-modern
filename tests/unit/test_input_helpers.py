"""Unit tests for the shared sample-time gating helpers in ``blocks.input_helpers``.

Sample-instant gating used to be copy-pasted into every discrete-rate block with
two different tolerances (1e-9 in ZeroOrderHold/FirstOrderHold/RateTransition,
1e-12 in PacketLoss/NetworkChannel/RandomSource) and, in two of them, with no
guard against a non-positive period. Both now live here.
"""

import pytest

from blocks.input_helpers import SAMPLE_TIME_EPS, advance_sample_time, sample_due


@pytest.mark.unit
class TestSampleDue:
    def test_due_at_and_after_the_scheduled_instant(self):
        assert sample_due(0.1, 0.1)
        assert sample_due(0.2, 0.1)

    def test_not_due_before_the_scheduled_instant(self):
        assert not sample_due(0.09, 0.1)

    def test_tolerates_accumulated_floating_point_drift(self):
        """0.1 summed ten times lands just below 1.0; the sample must still fire."""
        scheduled = 0.0
        for _ in range(10):
            scheduled += 0.1
        assert scheduled != 1.0  # the drift this tolerance exists for
        assert sample_due(1.0, scheduled)

    def test_one_epsilon_for_every_block(self):
        assert SAMPLE_TIME_EPS == 1e-9


@pytest.mark.unit
@pytest.mark.timeout(10)
class TestAdvanceSampleTime:
    def test_advances_by_whole_periods_past_the_current_time(self):
        params = {"_next_": 0.0}
        advance_sample_time(params, "_next_", 0.0, 0.1)
        assert params["_next_"] == pytest.approx(0.1)

    def test_catches_up_after_a_large_time_jump(self):
        params = {"_next_": 0.0}
        advance_sample_time(params, "_next_", 1.0, 0.1)
        assert params["_next_"] > 1.0
        assert params["_next_"] == pytest.approx(1.1, abs=1e-9)

    @pytest.mark.parametrize("step", [0.0, -0.1, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_period_returns_promptly(self, step):
        """This is the ZOH/FOH hang: the naive loop never terminates here."""
        params = {"_next_": 0.0}
        advance_sample_time(params, "_next_", 0.25, step)
        assert params["_next_"] == pytest.approx(0.25)

    def test_missing_key_is_seeded_from_the_current_time(self):
        params = {}
        advance_sample_time(params, "_next_", 0.5, 0.1)
        assert params["_next_"] == pytest.approx(0.6)
