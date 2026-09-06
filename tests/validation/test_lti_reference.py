"""LTI blocks against independent scipy implementations.

Where a closed form is inconvenient, the reference is a different program
solving the same problem: ``scipy.linalg.expm`` for the exact zero-order-hold
discretization of a state-space model under a step, and ``scipy.signal.lsim``
for a transfer function driven by a sine.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases

pytestmark = pytest.mark.validation


def test_state_space_matches_exact_zoh_discretization():
    """StateSpace vs ``expm``, which is exact for a piecewise-constant input."""
    rows = cases.case_state_space_vs_expm()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_zoh_discretization_helper_is_the_matrix_exponential():
    """``Ad = e^{A dt}``, and ``Bd`` integrates ``B`` over one step.

    The reference itself is checked: for an invertible ``A`` the augmented
    exponential must agree with the textbook ``A^{-1}(Ad - I)B``.
    """
    A, B = cases.SS_A, cases.SS_B
    dt = cases.SS["sim_dt"]
    Ad, Bd = cases.zoh_discretization(A, B, dt)
    from scipy.linalg import expm

    assert np.allclose(Ad, expm(A * dt), atol=1e-14)
    assert np.allclose(Bd, np.linalg.solve(A, (Ad - np.eye(A.shape[0])).dot(B)), atol=1e-12)


def test_transfer_function_matches_lsim():
    """TranFn under a sine vs ``scipy.signal.lsim`` on the same grid."""
    rows = cases.case_transfer_function_vs_lsim()
    failures = cases.format_failures(rows)
    assert not failures, failures
