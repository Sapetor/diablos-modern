"""Guard the WaveEquation1D energy single-sourcing.

The compiled solver's post-solve replay once computed a kinetic-only
"Simplified" energy that diverged from the block's execute() (full
kinetic+potential). Both paths now call ``pde_ops.wave_energy_1d``; these tests
lock that in so the divergence cannot silently return.
"""

import numpy as np
import pytest

from lib.engine.pde_ops import wave_energy_1d


@pytest.mark.unit
class TestWaveEnergy1D:
    def test_potential_term_present_when_at_rest(self):
        # v = 0 everywhere (string at rest) but curved displacement: kinetic
        # energy is 0, so a kinetic-only formula would return ~0. The real
        # total energy must be strictly positive from the potential term.
        N = 51
        x = np.linspace(0, 1, N)
        u = np.sin(np.pi * x)  # curved -> nonzero du/dx
        v = np.zeros(N)
        params = {"L": 1.0, "c": 1.0}
        energy = wave_energy_1d(u, v, params)
        assert energy > 0.1, "energy must include the potential term, not kinetic only"

    def test_pure_kinetic_when_flat(self):
        # Flat displacement (du/dx = 0) -> potential 0, energy == kinetic.
        N = 41
        u = np.full(N, 2.0)  # constant -> zero gradient
        v = np.full(N, 3.0)
        params = {"L": 1.0, "c": 1.0}
        dx = 1.0 / (N - 1)
        expected_kinetic = 0.5 * np.sum(v**2) * dx
        assert np.isclose(wave_energy_1d(u, v, params), expected_kinetic)

    def test_matches_block_compute_energy(self):
        # The block's _compute_energy must delegate to the shared helper.
        from blocks.pde.wave_equation_1d import WaveEquation1DBlock

        block = WaveEquation1DBlock()
        N = 61
        x = np.linspace(0, 1, N)
        u = np.sin(2 * np.pi * x)
        v = np.cos(np.pi * x)
        params = {"L": 1.0, "c": 2.0}
        assert np.isclose(block._compute_energy(u, v, params), wave_energy_1d(u, v, params))

    def test_scales_with_wave_speed(self):
        # Potential energy carries a c^2 factor; higher c -> more energy for the
        # same curved displacement at rest.
        N = 51
        x = np.linspace(0, 1, N)
        u = np.sin(np.pi * x)
        v = np.zeros(N)
        e1 = wave_energy_1d(u, v, {"L": 1.0, "c": 1.0})
        e2 = wave_energy_1d(u, v, {"L": 1.0, "c": 2.0})
        assert e2 > e1
