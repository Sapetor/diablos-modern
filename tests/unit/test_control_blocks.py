import numpy as np
from blocks.saturation import SaturationBlock
from blocks.rate_limiter import RateLimiterBlock
from blocks.pid import PIDBlock


def test_saturation_clips_values():
    block = SaturationBlock()
    params = {"min": -1.0, "max": 1.0}
    out = block.execute(0.0, {0: np.array([-2.0, 0.5, 2.0])}, params)[0]
    assert np.allclose(out, [-1.0, 0.5, 1.0])


def test_rate_limiter_limits_slew():
    block = RateLimiterBlock()
    params = {"rising_slew": 1.0, "falling_slew": 1.0, "_init_start_": True, "dtime": 0.1}
    # First call initializes state
    y0 = block.execute(0.0, {0: np.array([0.0])}, params)[0]
    assert y0 == 0.0
    # Second call attempts a step to 10, but should limit to 0.1
    y1 = block.execute(0.1, {0: np.array([10.0])}, params)[0]
    assert np.isclose(y1, 0.1)


def test_pid_proportional_only():
    block = PIDBlock()
    params = {
        "Kp": 2.0,
        "Ki": 0.0,
        "Kd": 0.0,
        "_init_start_": True,
        "dtime": 0.1,
    }
    out = block.execute(0.0, {0: np.array([1.0]), 1: np.array([0.25])}, params)[0]
    # error = 0.75; u = 2 * 0.75 = 1.5
    assert np.isclose(out, 1.5)
