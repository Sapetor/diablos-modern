"""Unit tests for lib/engine/verification_report.py.

The report used to be assembled inline in ``SimulationController`` (GUI only);
these tests pin the collection, the pass/fail judgement per verify_mode and the
assembled text, using plain stub blocks.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.engine import verification_report as vr


def _display(username, value, label=""):
    return SimpleNamespace(
        block_fn="Display",
        username=username,
        name=username,
        params={"_display_value_": value, "label": label},
    )


def _state_variable(name, final, initial=None):
    ep = {"_state_": final}
    if initial is not None:
        ep["initial_value"] = initial
    return SimpleNamespace(block_fn="StateVariable", username=name, name=name, exec_params=ep)


def _scope(name, vector, verify_mode="auto", vec_dim=1, labels=None):
    ep = {"vector": vector, "verify_mode": verify_mode, "vec_dim": vec_dim}
    if labels is not None:
        ep["vec_labels"] = labels
    return SimpleNamespace(block_fn="Scope", username=name, name=name, exec_params=ep)


def _other():
    return SimpleNamespace(block_fn="Gain", username="g", name="g", params={}, exec_params={})


@pytest.mark.unit
class TestCollection:
    def test_display_values_prefer_label(self):
        blocks = [_display("d0", "42", label="answer"), _display("d1", 3.5), _other()]
        assert vr.collect_display_values(blocks) == {"answer": "42", "d1": 3.5}

    def test_display_without_value_shows_placeholder(self):
        b = SimpleNamespace(block_fn="Display", username="d", name="d", params=None)
        assert vr.collect_display_values([b]) == {"d": "---"}

    def test_display_value_is_read_from_exec_params_after_a_run(self):
        b = SimpleNamespace(
            block_fn="Display",
            username="d",
            name="d",
            params={"label": ""},
            exec_params={"label": "err", "_display_value_": "err: 0.0007"},
        )
        assert vr.collect_display_values([b]) == {"err": "err: 0.0007"}

    def test_state_variables_need_a_state(self):
        blocks = [
            _state_variable("x", [0.0, 0.0], [1.0, 1.0]),
            SimpleNamespace(block_fn="StateVariable", username="empty", name="e", exec_params={}),
        ]
        states = vr.collect_state_variables(blocks)
        assert list(states) == ["x"]
        assert np.array_equal(states["x"]["final"], [0.0, 0.0])
        assert np.array_equal(states["x"]["initial"], [1.0, 1.0])

    def test_scalar_scope_first_last_and_count(self):
        info = vr.collect_scope_convergence([_scope("s", [1.0, 0.5, 0.25])])["s"]
        assert info["first"] == 1.0 and info["last"] == 0.25 and info["samples"] == 3
        assert info["verify_mode"] == "auto"

    def test_interleaved_vector_scope_is_reshaped(self):
        info = vr.collect_scope_convergence([_scope("s", [1, 2, 3, 4, 5, 6, 7], vec_dim=2)])["s"]
        assert info["samples"] == 3  # 7 values -> 3 complete (x, y) samples
        assert np.array_equal(info["first"], [1, 2])
        assert np.array_equal(info["last"], [5, 6])

    def test_empty_scope_is_skipped(self):
        assert vr.collect_scope_convergence([_scope("s", []), _scope("n", None)]) == {}

    def test_username_falls_back_to_name(self):
        b = SimpleNamespace(
            block_fn="Scope", username="", name="scope7", exec_params={"vector": [1.0]}
        )
        assert list(vr.collect_scope_convergence([b])) == ["scope7"]


@pytest.mark.unit
class TestClassification:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("f_cost", (True, False)),
            ("Obj value", (True, False)),
            ("x_traj", (False, True)),
            ("position", (False, True)),
            ("error", (False, False)),
            ("Scope0", (False, False)),
        ],
    )
    def test_auto_uses_name_keywords(self, name, expected):
        assert vr.classify_scope(name, "auto") == expected

    def test_explicit_modes(self):
        assert vr.classify_scope("anything", "objective") == (True, False)
        assert vr.classify_scope("anything", "trajectory") == (False, True)
        assert vr.classify_scope("f_cost", "comparison") == (False, False)
        assert vr.classify_scope("f_cost", "bogus") == (False, False)

    def test_format_value(self):
        assert vr.format_value(None) == "N/A"
        assert vr.format_value(np.array([0.5])) == "0.5"
        assert vr.format_value(np.array([1.0, 2.0])) == "[1. 2.]"
        assert vr.format_value(np.arange(5.0)) == "[0, 1, ...]"


@pytest.mark.unit
class TestDisplayLines:
    def test_plain_value_gets_the_name_prefix(self):
        assert vr.display_lines({"d0": 42}) == ["", "📊 Display Values:", "   d0: 42"]

    def test_labelled_display_is_not_prefixed_twice(self):
        assert vr.display_lines({"Error": "Error: 7.1e-04"})[2] == "   Error: 7.1e-04"


@pytest.mark.unit
class TestJudgement:
    def test_state_variable_converged_to_zero_passes(self):
        lines, ok = vr.state_variable_lines(
            {"x": {"final": np.array([1e-4, 0.0]), "initial": np.array([1.0, 1.0])}}
        )
        assert ok is True
        assert lines[2].startswith("   ✓ x:")
        assert any("reduced by" in line for line in lines)
        assert any("Converged to" in line for line in lines)

    def test_state_variable_that_never_moved_fails(self):
        lines, ok = vr.state_variable_lines(
            {"x": {"final": np.array([1.0]), "initial": np.array([1.0])}}
        )
        assert ok is False
        assert lines[2].startswith("   ✗ x:")

    def test_state_variable_without_initial_passes_on_change(self):
        _lines, ok = vr.state_variable_lines({"x": {"final": np.array([5.0]), "initial": None}})
        assert ok is True

    def test_objective_scope_must_drop_ninety_percent(self):
        good = {"f": {"first": 1.0, "last": 0.05, "samples": 10, "verify_mode": "objective"}}
        bad = {"f": {"first": 1.0, "last": 0.5, "samples": 10, "verify_mode": "objective"}}
        assert vr.scope_lines(good)[1] is True
        lines, ok = vr.scope_lines(bad)
        assert ok is False
        assert "✗ f: 1 → 0.5" in lines[2]
        assert "Reduced by 50.0%" in lines[3]

    def test_trajectory_scope_must_change(self):
        still = {"x": {"first": 1.0, "last": 1.0, "samples": 3, "verify_mode": "trajectory"}}
        moved = {"x": {"first": 1.0, "last": 2.0, "samples": 3, "verify_mode": "trajectory"}}
        assert vr.scope_lines(still)[1] is False
        assert vr.scope_lines(moved)[1] is True

    def test_comparison_scope_has_no_verdict(self):
        lines, ok = vr.scope_lines(
            {"err": {"first": 1.0, "last": 1.0, "samples": 3, "verify_mode": "comparison"}}
        )
        assert ok is True
        assert lines[2] == "   • err (3 pts): 1 → 1"

    def test_none_scope_is_skipped(self):
        lines, ok = vr.scope_lines(
            {"f_cost": {"first": 1.0, "last": 1.0, "samples": 3, "verify_mode": "none"}}
        )
        assert ok is True and lines == ["", "📈 Signal Convergence:"]

    def test_objective_starting_at_zero_falls_through_to_plain_listing(self):
        lines, ok = vr.scope_lines(
            {"f": {"first": 0.0, "last": 0.0, "samples": 3, "verify_mode": "objective"}}
        )
        assert ok is True and lines[2].startswith("   • f")


@pytest.mark.unit
class TestBuildReport:
    def test_no_data(self):
        report = vr.build_verification_report([_other()])
        assert report == vr.VerificationReport(text="", passed=True, has_data=False)

    def test_full_report_passes(self):
        blocks = [
            _display("d0", "42", label="answer"),
            _state_variable("x", [0.0], [1.0]),
            _scope("f_cost", [1.0, 0.01]),
            _scope("Scope0", [1.0, 1.0]),
        ]
        report = vr.build_verification_report(blocks)
        assert report.has_data and report.passed
        text = report.text
        assert text.startswith("\n" + "=" * 60 + "\nVERIFICATION RESULTS\n")
        assert "📊 Display Values:\n   answer: 42" in text
        assert "🎯 Optimization Convergence:\n   ✓ x:" in text
        assert "✓ f_cost: 1 → 0.01" in text
        assert "• Scope0 (2 pts): 1 → 1" in text
        assert text.rstrip().endswith("✓ VERIFICATION PASSED\n" + "=" * 60)

    def test_one_failed_check_fails_the_report(self):
        report = vr.build_verification_report([_display("d", 1), _scope("x_state", [2.0, 2.0])])
        assert report.has_data and not report.passed
        assert "✗ VERIFICATION FAILED - Check values above" in report.text

    def test_sections_only_appear_when_populated(self):
        text = vr.build_verification_report([_display("d", 1)]).text
        assert "Display Values" in text
        assert "Optimization Convergence" not in text
        assert "Signal Convergence" not in text


@pytest.mark.unit
class TestReportBlocks:
    def test_prefers_engine_active_list(self):
        dsim = SimpleNamespace(
            engine=SimpleNamespace(active_blocks_list=["flat"]), blocks_list=["top"]
        )
        assert vr.report_blocks(dsim) == ["flat"]

    def test_falls_back_to_blocks_list(self):
        assert vr.report_blocks(
            SimpleNamespace(engine=SimpleNamespace(active_blocks_list=[]), blocks_list=["top"])
        ) == ["top"]
        assert vr.report_blocks(SimpleNamespace(engine=None, blocks_list=["top"])) == ["top"]
        assert vr.report_blocks(SimpleNamespace(blocks_list=["top"])) == ["top"]
