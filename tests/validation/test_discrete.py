"""Discrete-time blocks: the recursion, and the rate it runs at.

A discrete block has two things to get right, and they fail independently. The
recursion itself is checked against ``scipy.signal.dlsim`` at the sample
instants. The *rate* is checked between them: a block gated to ``Ts`` must hold
its output over the whole inter-sample interval, and must move at the sample
instants and nowhere else. A block that quietly advances once per solver step
still matches dlsim on a grid where ``dt == Ts``, and only the hold check
catches it.
"""

import numpy as np
import pytest

from tests.validation import _cases as cases
from tests.validation import _harness as H

pytestmark = pytest.mark.validation


def test_discrete_transfer_function_matches_dlsim():
    rows = cases.case_discrete_transfer_function()
    failures = cases.format_failures(rows)
    assert not failures, failures


def test_discrete_block_moves_only_at_sample_instants():
    """The output's jump times are exactly the multiples of ``Ts``.

    Complements the "constant between samples" tolerance above: that one would
    also pass for a block that never updated at all.
    """
    cfg = cases.DISCRETE
    num, den = cases.discrete_plant()
    builder = H.build(cfg["sim_time"], cfg["sim_dt"])
    H.add(builder, "Step", "u", {"value": 1.0, "delay": 0.0, "type": "up"})
    H.add(
        builder,
        "DiscreteTranFn",
        "G",
        {"numerator": num, "denominator": den, "sampling_time": cfg["Ts"]},
    )
    H.add(builder, "Scope", "sc", {"labels": "y"})
    builder.connect("u", 0, "G", 0)
    builder.connect("G", 0, "sc", 0)

    result = H.run(builder, compiled=False)
    t, y = result.signal("y")
    jumps = t[1:][np.abs(np.diff(y)) > 1e-12]
    steps_per_sample = int(round(cfg["Ts"] / cfg["sim_dt"]))
    expected = t[steps_per_sample::steps_per_sample]
    # The recursion reaches steady state, after which consecutive samples stop
    # differing measurably; every jump that does happen must be on the rate.
    assert len(jumps) > 5
    assert np.all(np.isin(np.round(jumps, 9), np.round(expected, 9)))
    H.release(result)


def test_zero_order_hold_matches_the_analytic_staircase():
    rows = cases.case_zero_order_hold()
    failures = cases.format_failures(rows)
    assert not failures, failures
