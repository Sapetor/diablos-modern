import numpy as np
from blocks.saturation import SaturationBlock
from blocks.rate_limiter import RateLimiterBlock
from blocks.pid import PIDBlock
from blocks.hysteresis import HysteresisBlock
from blocks.deadband import DeadbandBlock
from blocks.switch import SwitchBlock


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


def test_hysteresis_switches_and_holds():
    block = HysteresisBlock()
    p = {"upper": 1.0, "lower": -1.0, "high": 2.0, "low": -2.0, "_init_start_": True}
    out1 = block.execute(0.0, {0: np.array([0.0])}, p)[0][0]
    assert out1 == -2.0
    out2 = block.execute(0.1, {0: np.array([1.2])}, p)[0][0]
    assert out2 == 2.0
    out3 = block.execute(0.2, {0: np.array([0.5])}, p)[0][0]
    # Should hold high state inside band
    assert out3 == 2.0


def test_deadband_zeroes_small_signal():
    block = DeadbandBlock()
    p = {"deadband": 0.5, "center": 0.0}
    out = block.execute(0.0, {0: np.array([-0.4, 0.0, 0.6])}, p)[0]
    assert np.allclose(out, [0.0, 0.0, 0.6])


def test_switch_selects_true_branch():
    block = SwitchBlock()
    p = {"threshold": 0.0}
    out = block.execute(0.0, {0: np.array([0.5]), 1: np.array([10.0]), 2: np.array([-10.0])}, p)[0][0]
    assert out == 10.0  # ctrl>=thr -> first data input
    out2 = block.execute(0.0, {0: np.array([-0.1]), 1: np.array([10.0]), 2: np.array([-10.0])}, p)[0][0]
    assert out2 == -10.0


def test_switch_index_mode_multiway():
    block = SwitchBlock()
    p = {"mode": "index", "n_inputs": 3}
    out0 = block.execute(0.0, {0: np.array([0.1]), 1: np.array([1.0]), 2: np.array([2.0]), 3: np.array([3.0])}, p)[0][0]
    assert out0 == 1.0  # round(0.1)=0 -> in0
    out2 = block.execute(0.0, {0: np.array([2.2]), 1: np.array([1.0]), 2: np.array([2.0]), 3: np.array([3.0])}, p)[0][0]
    assert out2 == 3.0  # clamp to last
