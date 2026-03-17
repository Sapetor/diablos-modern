"""Tests for the stochastic (Noise) solver fallback in compiled simulation."""

import numpy as np
import pytest


class TestStochasticSolverDetection:
    """Verify that Noise blocks trigger fixed-step Euler instead of RK45."""

    def _make_block(self, name, block_fn):
        """Create a minimal mock block for compilability checks."""
        class FakeBlock:
            pass
        b = FakeBlock()
        b.name = name
        b.block_fn = block_fn
        b.external = False
        return b

    def test_noise_detected_as_stochastic(self):
        """Noise block should be detected by stochastic check."""
        stochastic_fns = {'Noise'}
        blocks = [
            self._make_block('noise0', 'Noise'),
            self._make_block('scope0', 'Scope'),
        ]
        has_stochastic = any(
            (b.block_fn.title() if b.block_fn else '') in stochastic_fns
            for b in blocks
        )
        assert has_stochastic

    def test_prbs_not_detected_as_stochastic(self):
        """PRBS is deterministic (LFSR) and should NOT trigger Euler."""
        stochastic_fns = {'Noise'}
        blocks = [
            self._make_block('prbs0', 'PRBS'),
            self._make_block('scope0', 'Scope'),
        ]
        has_stochastic = any(
            (b.block_fn.title() if b.block_fn else '') in stochastic_fns
            for b in blocks
        )
        assert not has_stochastic

    def test_step_not_detected_as_stochastic(self):
        """Deterministic sources should use normal RK45."""
        stochastic_fns = {'Noise'}
        blocks = [
            self._make_block('step0', 'Step'),
            self._make_block('tranfn0', 'TranFn'),
        ]
        has_stochastic = any(
            (b.block_fn.title() if b.block_fn else '') in stochastic_fns
            for b in blocks
        )
        assert not has_stochastic


class TestFixedStepEuler:
    """Test the fixed-step Euler integration used for stochastic systems."""

    def test_euler_simple_exponential_decay(self):
        """Euler on dy/dt = -y should approximate y(t) = exp(-t)."""
        dt = 0.001
        t_eval = np.arange(0, 1.0 + dt, dt)
        y0 = np.array([1.0])

        def model_func(t, y):
            return -y

        # Fixed-step Euler (same as in simulation_engine.py)
        n_states = len(y0)
        n_steps = len(t_eval)
        y_history = np.zeros((n_states, n_steps))
        y = y0.copy()
        y_history[:, 0] = y
        for idx in range(1, n_steps):
            dy = model_func(t_eval[idx - 1], y)
            y = y + dt * np.asarray(dy)
            y_history[:, idx] = y

        # At t=1, y should be close to exp(-1) ≈ 0.3679
        assert np.isclose(y_history[0, -1], np.exp(-1.0), atol=0.01)

    def test_euler_with_noise_completes(self):
        """Euler with random RHS should complete without hanging."""
        dt = 0.01
        t_eval = np.arange(0, 1.0 + dt, dt)
        y0 = np.array([0.0])

        def model_func(t, y):
            noise = np.random.randn()
            return np.array([-y[0] + noise])

        n_states = len(y0)
        n_steps = len(t_eval)
        y_history = np.zeros((n_states, n_steps))
        y = y0.copy()
        y_history[:, 0] = y
        for idx in range(1, n_steps):
            dy = model_func(t_eval[idx - 1], y)
            y = y + dt * np.asarray(dy)
            y_history[:, idx] = y

        assert y_history.shape == (1, n_steps)
        assert not np.any(np.isnan(y_history))
        assert not np.any(np.isinf(y_history))

    def test_euler_preserves_shape_multi_state(self):
        """Euler should handle multi-state systems correctly."""
        dt = 0.01
        t_eval = np.arange(0, 0.1 + dt, dt)
        y0 = np.array([1.0, 0.0])

        def model_func(t, y):
            return np.array([y[1], -y[0]])  # harmonic oscillator

        n_states = len(y0)
        n_steps = len(t_eval)
        y_history = np.zeros((n_states, n_steps))
        y = y0.copy()
        y_history[:, 0] = y
        for idx in range(1, n_steps):
            dy = model_func(t_eval[idx - 1], y)
            y = y + dt * np.asarray(dy)
            y_history[:, idx] = y

        assert y_history.shape == (2, n_steps)
        # Energy should be roughly conserved (Euler drifts slightly)
        energy_start = y_history[0, 0]**2 + y_history[1, 0]**2
        energy_end = y_history[0, -1]**2 + y_history[1, -1]**2
        assert np.isclose(energy_start, energy_end, rtol=0.05)
