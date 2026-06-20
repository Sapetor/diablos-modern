"""Golden-master characterization of the compiled (fast-solver) path.

This is the safety net for the engine kernel-dedup initiative: before extracting
a shared block-kernel registry consumed by both SystemCompiler and the
run_compiled_simulation replay loop, we pin the *current* compiled-path output
for a set of example diagrams that collectively exercise every compilable block
family (math / sources / state / routing / logical / 1D PDE / 2D PDE). Any
refactor that changes a block's compiled output makes the corresponding trace
diverge from the committed golden and fails here.

Why golden-master and not compiled-vs-interpreter equivalence: the compiled path
(scipy RK45) and the interpreter (forward-Euler at dt) are different integration
schemes and legitimately diverge for stiff/PDE diagrams, so they cannot be
asserted equal. The compiled path is byte-deterministic across runs, so pinning
its own output is the correct invariant.

Regenerate the golden after an *intended* behavior change. Regeneration runs
under pytest (whose conftest initializes Qt headless reliably) gated behind
GOLDEN_REGEN=1:
    GOLDEN_REGEN=1 .venv-win/Scripts/python.exe -m pytest -o addopts="" \
        tests/regression/test_compiled_golden.py::test_regenerate_golden
(On WSL driving the Windows venv, also prefix `WSLENV=GOLDEN_REGEN/u` so the
flag crosses into the Windows process.) Then review the diff and commit
data/golden_compiled_traces.npz.
"""
import os
from pathlib import Path

import numpy as np
import pytest


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
GOLDEN_PATH = Path(__file__).parent / "data" / "golden_compiled_traces.npz"

# Curated set: every compilable block family is exercised by at least one entry.
# Each must be compilable (so the compiler kernels are actually run) and produce
# a numerically stable compiled trace (no blow-ups -> cross-platform robust).
GOLDEN_EXAMPLES = [
    "c01_tank_feedback.diablos",          # Gain, Step, Sum, TranFn (feedback)
    "c02_vehicle_single_agent.diablos",   # Integrator, Step, TranFn
    "c03_bode_frequency_response.diablos",  # Sine, TranFn
    "c05_mass_spring_state_space.diablos",  # StateSpace, Step
    "c11_opinion_dynamics.diablos",       # Constant, StateSpace
    "test_demux_logic.diablos",           # Constant, Demux, LogicalOperator
    "c06_observer_estimation.diablos",    # Mux, StateSpace, Step, Sum
    "heat_equation_1d_verification.diablos",  # HeatEquation1D, Ramp, MathFunction, Gain, Sum
    "heat_equation_demo.diablos",         # HeatEquation1D, FieldProbe
    "wave_equation_demo.diablos",         # WaveEquation1D
    "advection_equation_demo.diablos",    # AdvectionEquation1D
    "diffusion_reaction_demo.diablos",    # DiffusionReaction1D
    "heat_equation_2d_demo.diablos",      # HeatEquation2D, FieldProbe2D (2D PDE gap)
    "pde_comparison_demo.diablos",        # Heat + Wave + Advection 1D together
]

# Cross-platform tolerance: goldens are generated on the local (Windows) venv and
# also checked on Linux CI; RK45 + numpy differ by <~1e-7 relative across
# platforms, while a real kernel regression is O(1) or larger.
RTOL = 1e-5
ATOL = 1e-7


def _ensure_qapp():
    from PyQt5.QtWidgets import QApplication
    import sys
    return QApplication.instance() or QApplication(sys.argv)


def _load_dsim(filename):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(EXAMPLES_DIR / filename))
    assert data is not None, f"Failed to load {filename}"
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _scope_traces(dsim):
    """Return {scope_block_name: (n_samples, vec_dim) ndarray} for every Scope."""
    traces = {}
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        traces[b.name] = np.asarray(vec, dtype=float).reshape(
            -1, params.get("vec_dim", 1)
        )
    return traces


def _run_compiled(filename):
    """Run filename through the compiled path; return (compilable, {scope: trace})."""
    dsim = _load_dsim(filename)
    compilable = dsim.engine.check_compilability(dsim.blocks_list)
    dsim.use_fast_solver = True
    ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
    assert ok, f"Compiled run of {filename} failed: {err}"
    return compilable, _scope_traces(dsim)


def _golden_key(filename, scope_name):
    return f"{Path(filename).stem}__{scope_name}"


@pytest.mark.regression
@pytest.mark.parametrize("filename", GOLDEN_EXAMPLES)
def test_compiled_trace_matches_golden(filename, qapp):
    assert GOLDEN_PATH.exists(), (
        f"Golden file {GOLDEN_PATH} missing -- regenerate with "
        f"`python {Path(__file__).name}`."
    )
    golden = np.load(GOLDEN_PATH)

    compilable, traces = _run_compiled(filename)
    assert compilable, (
        f"{filename} is no longer compilable -- the golden no longer exercises "
        f"the compiler kernels."
    )
    assert traces, f"{filename} produced no Scope traces to compare."

    for scope_name, actual in traces.items():
        key = _golden_key(filename, scope_name)
        assert key in golden.files, (
            f"No golden for {key}; regenerate the golden file."
        )
        expected = golden[key]
        assert actual.shape == expected.shape, (
            f"{key}: shape {actual.shape} != golden {expected.shape}"
        )
        max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL), (
            f"{key}: compiled trace drifted from golden (max|delta|={max_abs:.3e}). "
            f"If this change is intentional, regenerate the golden."
        )


@pytest.mark.regression
@pytest.mark.skipif(
    os.environ.get("GOLDEN_REGEN") != "1",
    reason="set GOLDEN_REGEN=1 to rewrite the golden traces",
)
def test_regenerate_golden(qapp):
    """Rewrite the golden .npz from the current compiled-path output.

    Gated behind GOLDEN_REGEN=1 so a normal run never overwrites the baseline.
    Runs under the qapp fixture, which initializes Qt headless reliably.
    """
    _regenerate()
    assert GOLDEN_PATH.exists()


def _regenerate():
    """Write the golden .npz from the current compiled-path output."""
    _ensure_qapp()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for filename in GOLDEN_EXAMPLES:
        compilable, traces = _run_compiled(filename)
        if not compilable:
            print(f"  WARNING: {filename} not compilable (skipped)")
            continue
        if not traces:
            print(f"  WARNING: {filename} produced no Scope traces (skipped)")
            continue
        for scope_name, trace in traces.items():
            key = _golden_key(filename, scope_name)
            out[key] = trace
            print(f"  {key}: shape={trace.shape} max|v|={float(np.max(np.abs(trace))):.3e}")
    np.savez_compressed(GOLDEN_PATH, **out)
    print(f"Wrote {len(out)} traces to {GOLDEN_PATH}")


