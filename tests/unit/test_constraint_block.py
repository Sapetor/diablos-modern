"""Unit tests for the optimization Constraint block (blocks/optimization/constraint.py).

The block plays two roles and both were untested:

1. during simulation, ``execute()`` tracks max/min/final/integral of the signal
   and emits the *instantaneous* violation on port 0;
2. after the run, ``get_constraint_value()`` converts the tracked statistic into
   the ``('ineq'|'eq', value)`` form scipy expects (``value >= 0`` == satisfied
   for an inequality), and ``get_penalty()`` into a quadratic penalty.

The sign conventions in (2) are the easy thing to get backwards -- a flipped
sign silently inverts a constraint instead of failing -- so they are pinned here
for all three constraint types and all four evaluation modes.
"""

import numpy as np
import pytest

from blocks.optimization.constraint import ConstraintBlock


def _defaults(**overrides):
    params = {name: spec["default"] for name, spec in ConstraintBlock().params.items()}
    params.update(overrides)
    return params


def _drive(params, signal_seq, dtime=0.1):
    """Run the block over a signal sequence; return the per-step violations."""
    block = ConstraintBlock()
    params.setdefault("dtime", dtime)
    violations = []
    for step, value in enumerate(signal_seq):
        result = block.execute(
            time=step * dtime,
            inputs={0: np.array([float(value)])},
            params=params,
            dtime=dtime,
        )
        assert result["E"] is False
        violations.append(result[0])
    return violations


@pytest.mark.unit
class TestConstraintContract:
    def test_identity(self):
        block = ConstraintBlock()
        assert block.block_name == "Constraint"
        assert block.category == "Optimization"

    def test_ports(self):
        block = ConstraintBlock()
        assert [p["name"] for p in block.inputs] == ["signal"]
        assert [p["name"] for p in block.outputs] == ["violation"]
        # Typically a terminal block: nothing consumes the violation signal.
        assert block.requires_outputs is False

    def test_params(self):
        params = ConstraintBlock().params
        assert set(params) == {
            "type",
            "bound",
            "mode",
            "tolerance",
            "penalty_weight",
            "_init_start_",
        }
        assert params["type"]["default"] == "<="
        assert params["mode"]["default"] == "max"
        assert params["_init_start_"]["default"] is True


@pytest.mark.unit
class TestConstraintExecution:
    def test_init_flag_seeds_the_trackers_and_clears_itself(self):
        params = _defaults()
        _drive(params, [0.5])

        assert params["_init_start_"] is False
        assert params["_max_value_"] == pytest.approx(0.5)
        assert params["_min_value_"] == pytest.approx(0.5)
        assert params["_final_value_"] == pytest.approx(0.5)

    def test_tracks_max_min_final_and_integral(self):
        params = _defaults()
        _drive(params, [0.0, 2.0, -1.0, 0.5], dtime=0.1)

        assert params["_max_value_"] == pytest.approx(2.0)
        assert params["_min_value_"] == pytest.approx(-1.0)
        assert params["_final_value_"] == pytest.approx(0.5)
        # Rectangle rule over the four samples.
        assert params["_integral_"] == pytest.approx((0.0 + 2.0 - 1.0 + 0.5) * 0.1)

    def test_le_violation_is_the_overshoot_and_zero_when_satisfied(self):
        params = _defaults(type="<=", bound=1.0)
        violations = _drive(params, [0.0, 1.0, 1.5, 0.25])

        assert violations == pytest.approx([0.0, 0.0, 0.5, 0.0])

    def test_ge_violation_is_the_shortfall(self):
        params = _defaults(type=">=", bound=1.0)
        violations = _drive(params, [0.0, 1.0, 1.5])

        assert violations == pytest.approx([1.0, 0.0, 0.0])

    def test_eq_violation_respects_the_tolerance_band(self):
        params = _defaults(type="==", bound=1.0, tolerance=0.1)
        violations = _drive(params, [1.0, 1.05, 1.5])

        assert violations[0] == pytest.approx(0.0)
        assert violations[1] == pytest.approx(0.0), "inside the tolerance band"
        assert violations[2] == pytest.approx(0.4), "|1.5 - 1.0| - 0.1"

    def test_unknown_constraint_type_reports_no_violation(self):
        params = _defaults(type="~=", bound=1.0)
        assert _drive(params, [99.0]) == pytest.approx([0.0])

    def test_accepts_a_scalar_or_an_array_input(self):
        block = ConstraintBlock()
        params = _defaults(type="<=", bound=1.0)
        scalar = block.execute(time=0.0, inputs={0: 3.0}, params=params, dtime=0.1)

        params = _defaults(type="<=", bound=1.0)
        vector = block.execute(time=0.0, inputs={0: np.array([3.0])}, params=params, dtime=0.1)

        assert scalar[0] == pytest.approx(vector[0]) == pytest.approx(2.0)


@pytest.mark.unit
class TestConstraintValueForTheOptimizer:
    """``get_constraint_value`` -> scipy: ``('ineq', v)`` is satisfied when v >= 0."""

    @pytest.mark.parametrize(
        "mode,expected_signal",
        [("max", 2.0), ("min", -1.0), ("final", 0.5), ("integral", 0.15)],
    )
    def test_mode_selects_which_tracked_statistic_is_constrained(self, mode, expected_signal):
        params = _defaults(type="<=", bound=10.0, mode=mode)
        _drive(params, [0.0, 2.0, -1.0, 0.5], dtime=0.1)

        kind, value = ConstraintBlock().get_constraint_value(params)

        assert kind == "ineq"
        # value = bound - signal for a '<=' constraint
        assert value == pytest.approx(10.0 - expected_signal)

    def test_le_is_satisfied_when_the_tracked_max_is_under_the_bound(self):
        params = _defaults(type="<=", bound=3.0, mode="max")
        _drive(params, [0.0, 2.0, 1.0])

        kind, value = ConstraintBlock().get_constraint_value(params)
        assert kind == "ineq"
        assert value > 0

    def test_le_is_violated_when_the_tracked_max_exceeds_the_bound(self):
        params = _defaults(type="<=", bound=1.0, mode="max")
        _drive(params, [0.0, 2.0, 1.0])

        _, value = ConstraintBlock().get_constraint_value(params)
        assert value < 0

    def test_ge_sign_convention_is_the_mirror_image(self):
        params = _defaults(type=">=", bound=1.0, mode="min")
        _drive(params, [2.0, 3.0])

        kind, value = ConstraintBlock().get_constraint_value(params)
        assert kind == "ineq"
        assert value == pytest.approx(1.0)  # min 2.0 - bound 1.0

    def test_eq_returns_an_eq_constraint_zeroed_inside_the_tolerance(self):
        params = _defaults(type="==", bound=1.0, mode="final", tolerance=0.1)
        _drive(params, [1.05])

        kind, value = ConstraintBlock().get_constraint_value(params)
        assert kind == "eq"
        assert value == pytest.approx(0.0)

    def test_eq_outside_the_tolerance_keeps_the_deviation_sign(self):
        block = ConstraintBlock()

        high = _defaults(type="==", bound=1.0, mode="final", tolerance=0.1)
        _drive(high, [1.5])
        low = _defaults(type="==", bound=1.0, mode="final", tolerance=0.1)
        _drive(low, [0.5])

        assert block.get_constraint_value(high)[1] == pytest.approx(0.4)
        assert block.get_constraint_value(low)[1] == pytest.approx(-0.4)

    def test_unknown_type_degrades_to_a_satisfied_inequality(self):
        params = _defaults(type="~=", mode="final")
        _drive(params, [42.0])

        assert ConstraintBlock().get_constraint_value(params) == ("ineq", 0.0)


@pytest.mark.unit
class TestConstraintPenaltyAndReset:
    def test_penalty_is_zero_while_satisfied(self):
        params = _defaults(type="<=", bound=10.0, mode="max")
        _drive(params, [1.0, 2.0])

        assert ConstraintBlock().get_penalty(params) == pytest.approx(0.0)

    def test_penalty_is_weight_times_violation_squared(self):
        params = _defaults(type="<=", bound=1.0, mode="max", penalty_weight=100.0)
        _drive(params, [1.0, 3.0])

        # violation = 3.0 - 1.0 = 2.0  ->  100 * 4
        assert ConstraintBlock().get_penalty(params) == pytest.approx(400.0)

    def test_reset_rearms_the_block_for_the_next_optimizer_iteration(self):
        """Without this, run N+1 of an optimization inherits run N's extrema."""
        block = ConstraintBlock()
        params = _defaults(type="<=", bound=1.0, mode="max")
        _drive(params, [5.0])
        assert params["_max_value_"] == pytest.approx(5.0)

        block.reset(params)

        assert params["_init_start_"] is True
        assert params["_max_value_"] == -np.inf
        assert params["_min_value_"] == np.inf
        assert params["_final_value_"] == 0.0
        assert params["_integral_"] == 0.0

        _drive(params, [0.5])
        assert params["_max_value_"] == pytest.approx(0.5), "stale extremum leaked"

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = ConstraintBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
