"""Unit tests for the extracted compiled-solver diagnostics helpers.

These functions used to be private methods on SimulationEngine; extracting them
into ``lib/engine/solver_diagnostics.py`` lets us exercise the dict-building and
formatting logic without standing up an engine or running a simulation.
"""

import types

import pytest

from lib.engine.solver_diagnostics import (
    build_diagnostics,
    format_diagnostics_for_log,
    solver_attr,
)


def _fake_sol(**attrs):
    return types.SimpleNamespace(**attrs)


@pytest.mark.unit
class TestSolverAttr:
    def test_reads_present_attribute(self):
        assert solver_attr(_fake_sol(nfev=42), "nfev") == 42

    def test_missing_attribute_returns_default(self):
        # Fixed-step stand-ins omit nfev/njev/nlu.
        assert solver_attr(_fake_sol(), "nfev", None) is None
        assert solver_attr(_fake_sol(), "nfev", 0) == 0


@pytest.mark.unit
class TestBuildDiagnostics:
    def _build(self, **overrides):
        base = dict(
            sol=_fake_sol(t=[0.0, 0.1, 0.2], message="ok", status=0, nfev=10),
            success=True,
            method_requested="RK45",
            method_used="RK45",
            backend="scipy",
            t_span=(0.0, 1.0),
            dt=0.1,
            rtol=1e-6,
            atol=1e-9,
            n_states=2,
            n_blocks=5,
            n_lines=6,
            compile_cache_hit=True,
            compile_cache_hits_total=3,
            compile_cache_misses_total=1,
            compile_time=0.01,
            solve_time=0.02,
            replay_time=0.03,
            total_time=0.06,
        )
        base.update(overrides)
        return build_diagnostics(**base)

    def test_core_fields_and_derived_counts(self):
        d = self._build()
        assert d["success"] is True
        assert d["n_time_points"] == 3
        assert d["n_output_steps"] == 2  # points - 1
        assert d["nfev"] == 10
        assert d["compile_cache_hit"] is True
        assert d["compile_cache_hits_total"] == 3
        assert d["compile_cache_misses_total"] == 1
        assert d["t_start"] == 0.0 and d["t_end"] == 1.0

    def test_missing_time_array_yields_zero_points(self):
        d = self._build(sol=_fake_sol(message="", status=None))
        assert d["n_time_points"] == 0
        assert d["n_output_steps"] == 0  # clamped, never negative

    def test_missing_solver_stats_are_none(self):
        d = self._build(sol=_fake_sol(t=[0.0]))
        assert d["nfev"] is None and d["njev"] is None and d["nlu"] is None

    def test_optional_failure_fields_default_none(self):
        d = self._build()
        assert d["fallback_reason"] is None
        assert d["failure_stage"] is None
        assert d["output_range"] is None


@pytest.mark.unit
class TestFormatDiagnostics:
    def test_hit_and_present_nfev(self):
        text = format_diagnostics_for_log(
            {
                "method_used": "RK45",
                "backend": "scipy",
                "n_states": 2,
                "n_time_points": 3,
                "nfev": 10,
                "compile_cache_hit": True,
                "compile_wall_time": 0.01,
                "solve_wall_time": 0.02,
                "replay_wall_time": 0.03,
                "total_wall_time": 0.06,
            }
        )
        assert "method=RK45" in text
        assert "cache=hit" in text
        assert "nfev=10" in text

    def test_miss_and_absent_nfev(self):
        text = format_diagnostics_for_log({"compile_cache_hit": False})
        assert "cache=miss" in text
        assert "nfev=n/a" in text
