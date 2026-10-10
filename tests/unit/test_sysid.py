"""Transfer-function identification (lib/analysis/sysid.py)."""

import numpy as np
import pytest
from scipy import signal

from lib.analysis.sysid import SysIdError, fit_percent, format_tf, identify_tf


def _data(num, den, t, u, noise=0.0, seed=0):
    _, y, _ = signal.lsim(signal.lti(num, den), u, t)
    rng = np.random.default_rng(seed)
    return y + noise * rng.standard_normal(y.size)


def _prbs(t, period=0.5, seed=1):
    rng = np.random.default_rng(seed)
    levels = rng.choice([-1.0, 1.0], size=int(t[-1] / period) + 2)
    return levels[(t / period).astype(int)]


@pytest.mark.unit
class TestIdentifyTf:
    def test_second_order_from_prbs(self):
        t = np.linspace(0, 20, 2001)
        u = _prbs(t)
        y = _data([2.0], [1.0, 3.0, 2.0], t, u)
        r = identify_tf(t, u, y, num_order=0, den_order=2)
        np.testing.assert_allclose(r["den"], [1.0, 3.0, 2.0], rtol=2e-2)
        np.testing.assert_allclose(r["num"], [2.0], rtol=2e-2)
        assert r["fit_percent"] > 99

    def test_first_order_step_with_offset_and_noise(self):
        t = np.linspace(0, 5, 501)
        u = np.where(t >= 0.5, 1.0, 0.0)
        y = 3.0 + _data([4.0], [0.5, 1.0], t, u, noise=0.02)  # 8/(s+2)
        r = identify_tf(t, u, y, num_order=0, den_order=1)
        np.testing.assert_allclose(r["den"], [1.0, 2.0], rtol=5e-2)
        np.testing.assert_allclose(r["num"], [8.0], rtol=5e-2)
        assert r["fit_percent"] > 95
        assert r["y_model"][0] == pytest.approx(3.0, abs=0.05)

    def test_zero_and_irregular_sampling(self):
        fine = np.linspace(0, 15, 15001)
        u_fine = _prbs(fine, period=0.7, seed=4)
        y_fine = _data([1.0, 2.0], [1.0, 2.0, 5.0], fine, u_fine)  # (s+2)/(s^2+2s+5)
        rng = np.random.default_rng(3)
        idx = np.unique(np.concatenate([[0], rng.choice(fine.size, 1500, replace=False)]))
        t, u, y = fine[idx], u_fine[idx], y_fine[idx]
        r = identify_tf(t, u, y, num_order=1, den_order=2)
        assert r["fit_percent"] > 97
        np.testing.assert_allclose(r["den"], [1.0, 2.0, 5.0], rtol=0.1)

    def test_bad_inputs(self):
        t = np.linspace(0, 1, 100)
        with pytest.raises(SysIdError):
            identify_tf(t, t, t, num_order=3, den_order=2)
        with pytest.raises(SysIdError):
            identify_tf(t[:5], t[:5], t[:5], num_order=0, den_order=1)
        with pytest.raises(SysIdError):
            identify_tf(t, t, t[:50], num_order=0, den_order=1)


@pytest.mark.unit
def test_helpers():
    assert fit_percent([0, 1, 2], [0, 1, 2]) == 100.0
    assert format_tf([2.0, 1.0], [1.0, 3.0, 2.0]) == "(2 s + 1) / (s^2 + 3 s + 2)"
    assert format_tf([-0.5], [1.0, -1.0]) == "(-0.5) / (s - 1)"


@pytest.mark.unit
def test_long_record_is_resampled_and_fast():
    import time

    t = np.linspace(0, 50, 100001)
    u = _prbs(t, period=1.0)
    y = _data([1.0], [1.0, 1.0], t, u)
    start = time.perf_counter()
    r = identify_tf(t, u, y, num_order=0, den_order=1)
    assert time.perf_counter() - start < 10
    assert r["grid"].size <= 5000
    np.testing.assert_allclose(r["den"], [1.0, 1.0], rtol=3e-2)
