"""Unit tests for the RandomSource block."""

import numpy as np
import pytest

from blocks.random_source import RandomSourceBlock


def _fresh_params(**overrides):
    """Build a default param dict from the block's declared defaults."""
    block = RandomSourceBlock()
    params = {k: v["default"] for k, v in block.params.items()}
    params.update(overrides)
    return params


def _run(params, n, dtime=0.1):
    """Drive a fresh block n steps; return a flat ndarray of scalar samples."""
    block = RandomSourceBlock()
    out = []
    for i in range(n):
        res = block.execute(time=i * dtime, inputs={}, params=params, dtime=dtime)
        out.append(float(np.atleast_1d(res[0])[0]))
    return np.asarray(out)


@pytest.mark.unit
class TestRandomSource:
    def test_block_contract(self):
        block = RandomSourceBlock()
        assert block.block_name == "RandomSource"
        assert block.category == "Sources"
        assert block.b_type == 0
        assert block.requires_inputs is False
        assert len(block.inputs) == 0
        assert len(block.outputs) == 1

    def test_output_is_atleast_1d(self):
        params = _fresh_params(seed=42)
        block = RandomSourceBlock()
        res = block.execute(time=0.0, inputs={}, params=params, dtime=0.1)
        val = res[0]
        assert isinstance(val, np.ndarray)
        assert val.ndim >= 1

    def test_uniform_in_range_and_mean(self):
        """Uniform draws fall in [low, high]; mean ~ (low+high)/2."""
        low, high = 2.0, 8.0
        params = _fresh_params(distribution="uniform", low=low, high=high, seed=42)
        samples = _run(params, 5000)
        assert np.all(samples >= low)
        assert np.all(samples <= high)
        assert abs(samples.mean() - (low + high) / 2.0) < 0.15

    def test_bernoulli_values_and_mean(self):
        """Bernoulli yields only {0,1}; mean ~ p."""
        p = 0.3
        params = _fresh_params(distribution="bernoulli", p=p, seed=42)
        samples = _run(params, 5000)
        assert set(np.unique(samples)).issubset({0.0, 1.0})
        assert abs(samples.mean() - p) < 0.03

    def test_normal_mean(self):
        """Normal draws have mean ~ mu within tolerance."""
        mu, sigma = 5.0, 2.0
        params = _fresh_params(distribution="normal", mu=mu, sigma=sigma, seed=42)
        samples = _run(params, 8000)
        assert abs(samples.mean() - mu) < 0.1
        assert abs(samples.std() - sigma) < 0.1

    def test_randint_integer_in_range(self):
        """randint yields integers in [low, high] inclusive."""
        low, high = 1, 6  # like a die
        params = _fresh_params(distribution="randint", low=low, high=high, seed=42)
        samples = _run(params, 3000)
        assert np.all(samples >= low)
        assert np.all(samples <= high)
        # all values are integer-valued
        assert np.allclose(samples, np.round(samples))
        # both endpoints are reachable
        assert samples.min() == low
        assert samples.max() == high

    def test_same_seed_reproducible(self):
        """Same nonzero seed => identical sample stream."""
        s1 = _run(_fresh_params(distribution="uniform", seed=42), 500)
        s2 = _run(_fresh_params(distribution="uniform", seed=42), 500)
        assert np.array_equal(s1, s2)

    def test_different_seed_differs(self):
        """Different seeds => different sample stream."""
        s1 = _run(_fresh_params(distribution="uniform", seed=42), 500)
        s2 = _run(_fresh_params(distribution="uniform", seed=123), 500)
        assert not np.array_equal(s1, s2)

    def test_sample_time_holds_between_samples(self):
        """With sample_time>dtime the value is held between sample instants."""
        params = _fresh_params(distribution="uniform", low=0.0, high=1.0,
                               seed=42, sample_time=0.5)
        block = RandomSourceBlock()
        dtime = 0.1
        v0 = float(np.atleast_1d(
            block.execute(time=0.0, inputs={}, params=params, dtime=dtime)[0])[0])
        # t = 0.1 .. 0.4 should all repeat the t=0 sample.
        for k in range(1, 5):
            vk = float(np.atleast_1d(
                block.execute(time=k * dtime, inputs={}, params=params,
                              dtime=dtime)[0])[0])
            assert vk == v0
        # t = 0.5 is a new sample instant -> a fresh draw is allowed (may differ).
        v5 = float(np.atleast_1d(
            block.execute(time=0.5, inputs={}, params=params, dtime=dtime)[0])[0])
        # still within range regardless
        assert 0.0 <= v5 <= 1.0

    def test_zero_seed_runs_without_error(self):
        """seed=0 => entropy-seeded; should still produce in-range output."""
        params = _fresh_params(distribution="uniform", low=-1.0, high=1.0, seed=0)
        samples = _run(params, 100)
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)
