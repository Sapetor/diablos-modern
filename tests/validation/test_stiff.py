"""A stiff system through the compiled solver's stiff methods.

Van der Pol at ``mu = 1000`` is the standard stiff benchmark: the relaxation
oscillator spends most of its period on a slow manifold and crosses between
branches in a layer roughly ``1/mu`` wide. It is built here from Integrators and
math blocks rather than from a single block, so the compiled system really is
assembled from the diagram, and compared against ``solve_ivp`` driven directly
at ``rtol=1e-10`` -- two orders tighter than the run under test.

The interpreter is deliberately absent: it is fixed-step, and an explicit
fixed-step method needs ``dt`` of order ``1e-6`` here to stay stable, which is
not a comparison, it is a different experiment.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_van_der_pol_matches_a_tight_radau_reference():
    rows = cases.case_van_der_pol_stiff()
    failures = cases.format_failures(rows)
    assert not failures, failures
    assert {row.method for row in rows} == {"Radau", "LSODA"}


def test_the_stiff_run_is_actually_stiff():
    """Guard the premise: this system really does punish an explicit method.

    If it were not stiff, agreement with Radau would say nothing about stiff
    solving. The Jacobian at the initial state has an eigenvalue near
    ``mu(1 - x1^2) = -3000``, three orders faster than the 1 s horizon, so an
    explicit method is step-limited by stability rather than by accuracy.
    """
    cfg = cases.VDP
    mu = cfg["mu"]
    x1, x2 = cfg["x0"]
    jacobian = np.array([[0.0, 1.0], [-2.0 * mu * x1 * x2 - 1.0, mu * (1.0 - x1**2)]])
    fastest = np.min(np.real(np.linalg.eigvals(jacobian)))
    assert fastest * cfg["sim_time"] < -1000.0, (
        "Jacobian eigenvalue {0:.1f} is too gentle to be a stiffness test".format(fastest)
    )


def test_solver_method_reaches_the_compiled_run():
    """``solver_method`` is honoured, not silently replaced by the default."""
    cfg = cases.VDP
    builder = cases.build_van_der_pol(cfg["mu"], cfg["x0"], cfg["sim_time"], cfg["sim_dt"])
    result = H.run(builder, compiled=True, solver_method="Radau", rtol=1e-9, atol=1e-12)
    assert result.diagnostics.get("method_used") == "Radau", result.diagnostics
    assert result.diagnostics.get("method_requested") == "Radau", result.diagnostics
    H.release(result)
