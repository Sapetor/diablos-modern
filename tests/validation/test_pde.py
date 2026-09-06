"""PDE blocks against exact solutions of the equations they discretize.

Two 1-D problems with a closed form. The heat block starts from an
eigenfunction of the Laplacian under homogeneous Dirichlet ends, which decays in
place at a rate written down exactly. The advection block carries a Gaussian
pulse, which pure transport translates rigidly.

Both are checked twice: once against the exact solution on a fine grid, and once
across a sequence of grids, because the *order* is the sharper claim. A single
tolerance on one grid can be met by a scheme that is accidentally close; only a
convergence rate of two says the spatial discretization is the second-order
stencil it is meant to be.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_heat_eigenmode_decays_at_the_analytic_rate():
    rows = cases.case_heat_eigenmode()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_heat_field_keeps_its_mode_shape():
    """The eigenmode decays without changing shape.

    ``sin(pi x/L)`` is an eigenfunction, so at every instant the normalized
    field must still be that half-sine; a scheme that leaked into other modes
    would still pass a decay-rate check on the spatial mean.
    """
    cfg = cases.HEAT
    N = 81
    builder = cases.build_heat_1d(cfg["alpha"], cfg["L"], N, cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True)
    _t, field = result.field("field")
    x = np.linspace(0.0, cfg["L"], N)
    shape = np.sin(np.pi * x / cfg["L"])
    final = field[-1]
    assert np.max(np.abs(final)) > 1e-6, "the mode decayed to nothing; nothing to compare"
    normalized = final / np.max(np.abs(final))
    assert np.max(np.abs(normalized - shape)) < 2e-3
    H.release(result)


def test_advection_transports_a_gaussian_without_changing_it():
    rows = cases.case_advection_pulse()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_advected_pulse_moves_at_the_prescribed_velocity():
    """The peak travels ``v t``: transport, not just a plausible smear.

    Measured as the centre of mass of the field, which is robust to the grid
    and to the small amount of numerical dispersion a second-order upwind
    stencil leaves behind.
    """
    cfg = cases.ADVECTION
    N = 401
    builder = cases.build_advection_1d(cfg["velocity"], cfg["L"], N, cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True)
    t, field = result.field("field")
    x = np.linspace(0.0, cfg["L"], N)
    start = float(np.sum(x * field[0]) / np.sum(field[0]))
    end = float(np.sum(x * field[-1]) / np.sum(field[-1]))
    assert end - start == pytest.approx(cfg["velocity"] * t[-1], abs=2e-3)
    H.release(result)
