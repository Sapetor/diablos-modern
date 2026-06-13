"""Unit tests for the Bode/Nyquist stability-margin and auto-range helpers.

These exercise the pure computational helpers shared by the Bode and Nyquist
analyzers (``lib/analysis/analyzers/base_analyzer.py``): the frequency
auto-range derived from the system poles/zeros and the gain/phase margin
computation from the magnitude/phase arrays. They avoid creating Qt windows so
they are fast and headless-safe.
"""

import numpy as np
import pytest
import scipy.signal as signal

from lib.analysis.analyzers.base_analyzer import BaseAnalyzer


@pytest.mark.unit
class TestBodeMargins:
    def test_second_order_auto_range_and_margins(self):
        """Known 2nd-order system 1/(s^2 + s + 1): wn = 1 rad/s, zeta = 0.5."""
        num = [1.0]
        den = [1.0, 1.0, 1.0]

        # --- Auto-range must straddle the natural frequency (1 rad/s) ---
        w = BaseAnalyzer._auto_frequency_range(num, den, dt=0.0)
        assert w[0] > 0
        assert w[0] < 1.0 < w[-1], "auto-range must cover the natural frequency"
        # Roughly two decades below/above the pole magnitude (|pole| ~ 1).
        assert w[0] == pytest.approx(1e-2, rel=1e-6)
        assert w[-1] == pytest.approx(1e2, rel=1e-6)

        # --- Margins from the Bode arrays ---
        sys = signal.TransferFunction(num, den)
        w, mag, phase = sys.bode(w=w)
        m = BaseAnalyzer._compute_stability_margins(w, mag, phase)

        # Phase margin is finite and positive (system is stable, ~90 deg here).
        assert np.isfinite(m['phase_margin_deg'])
        assert m['phase_margin_deg'] == pytest.approx(90.0, abs=2.0)
        # The 0 dB gain crossover sits at the natural frequency (~1 rad/s).
        assert m['phase_crossover_w'] is not None
        assert m['phase_crossover_w'] == pytest.approx(1.0, abs=0.05)

        # All margin keys are present.
        for key in ('gain_margin_db', 'gain_crossover_w',
                    'phase_margin_deg', 'phase_crossover_w'):
            assert key in m

    def test_finite_gain_and_phase_margin(self):
        """1/(s(s+1)(s+2)) has well-known finite GM (~15.6 dB) and PM (~53 deg)."""
        num = [1.0]
        den = np.polymul([1.0, 0.0], np.polymul([1.0, 1.0], [1.0, 2.0]))

        w = BaseAnalyzer._auto_frequency_range(num, den, dt=0.0)
        sys = signal.TransferFunction(num, den)
        w, mag, phase = sys.bode(w=w)
        m = BaseAnalyzer._compute_stability_margins(w, mag, phase)

        assert np.isfinite(m['gain_margin_db'])
        assert m['gain_margin_db'] == pytest.approx(15.56, abs=0.5)
        # -180 deg crossing occurs at w = sqrt(2) rad/s for this system.
        assert m['gain_crossover_w'] == pytest.approx(np.sqrt(2.0), abs=0.05)

        assert np.isfinite(m['phase_margin_deg'])
        assert m['phase_margin_deg'] == pytest.approx(53.4, abs=1.0)
        assert m['phase_crossover_w'] is not None

    def test_auto_range_uses_pole_zero_magnitudes(self):
        """Fast and slow dynamics: 1/((s+0.1)(s+1000))."""
        den = np.polymul([1.0, 0.1], [1.0, 1000.0])
        w = BaseAnalyzer._auto_frequency_range([1.0], den, dt=0.0)
        # Two decades below slowest pole (0.1 -> 1e-3) and above fastest (1000 -> 1e5).
        assert w[0] == pytest.approx(1e-3, rel=1e-6)
        assert w[-1] == pytest.approx(1e5, rel=1e-6)

    def test_auto_range_fallback_for_pure_gain(self):
        """No finite, non-origin dynamics -> historical default band."""
        w = BaseAnalyzer._auto_frequency_range([2.0], [1.0], dt=0.0)
        assert w[0] == pytest.approx(1e-2, rel=1e-6)
        assert w[-1] == pytest.approx(1e2, rel=1e-6)

    def test_auto_range_ignores_origin_poles(self):
        """A lone integrator (pole at origin) falls back rather than collapsing."""
        w = BaseAnalyzer._auto_frequency_range([1.0], [1.0, 0.0], dt=0.0)
        assert w[0] == pytest.approx(1e-2, rel=1e-6)
        assert w[-1] == pytest.approx(1e2, rel=1e-6)

    def test_auto_range_discrete_clamped_to_nyquist(self):
        """Discrete sweeps never exceed the Nyquist frequency pi/dt."""
        dt = 0.1
        w = BaseAnalyzer._auto_frequency_range([1.0], [1.0, -0.5], dt=dt)
        assert w[-1] <= np.pi / dt + 1e-9
        assert w[0] > 0

    def test_margins_no_crossing_returns_inf(self):
        """A flat, sub-0 dB, low-phase response has no crossings -> inf margins."""
        w = np.logspace(-2, 2, 100)
        mag = np.full_like(w, -10.0)   # always below 0 dB
        phase = np.full_like(w, -30.0)  # never reaches -180
        m = BaseAnalyzer._compute_stability_margins(w, mag, phase)
        assert m['gain_margin_db'] == float('inf')
        assert m['gain_crossover_w'] is None
        assert m['phase_margin_deg'] == float('inf')
        assert m['phase_crossover_w'] is None
