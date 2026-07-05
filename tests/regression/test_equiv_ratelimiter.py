"""Compiled-vs-interpreted equivalence for the RateLimiter block (a D=0 state
block in the compiled path).

Diagram: Step(0 -> 1 at t=0.5) -> RateLimiter(rising=falling=1.0) -> Scope.
The step drives the limiter into its slew-limited regime, so the block's
dynamics -- not a pass-through -- determine the trace: the output should ramp
at the configured slew rate (1.0 units/s) and reach 1.0 at t=1.5, then hold.

Both engines are run over the same diagram and the two Scope trajectories are
compared at the interpreter's sample times (identical dt, so index-aligned).

STATUS: xfail(strict) -- the two paths genuinely diverge.

  The COMPILED path is correct: its kernel integrates dy/dt = clip((u-y)*K,
  -falling, rising) once per RHS evaluation, so the output ramps at exactly the
  configured slew rate (reaches 1.0 at t=1.5).

  The INTERPRETED path ramps at ~2x the configured slew rate (reaches 1.0 at
  t=1.0), violating the documented slew limit. RateLimiter is registered as a
  memory block (lib/engine/memory_blocks.py), so the interpreter executes it
  twice per step: an output_only pass (simulation_engine.execute_block, which
  only special-cases Integrator via next_add_in_memory=False) and a
  state-updating pass. RateLimiterBlock.execute ignores the output_only kwarg
  and advances params['_prev'] by ~rising*dt on BOTH calls, doubling the slew
  per output step.

When the interpreter double-advance is fixed, the remaining difference is only
the compiled kernel's documented stiff-chase approximation (~1e-3 near the ramp
top), well within the tolerance below, so this test will XPASS and the strict
marker will flag the stale xfail for removal.
"""

import json
from pathlib import Path

import numpy as np
import pytest


SIM_TIME = 3.0
SIM_DT = 0.01

# Loose tolerance appropriate for RK45-vs-fixed-step plus the compiled kernel's
# documented stiff-chase approximation. The current interpreter double-advance
# produces a max|delta| of ~0.5 (the two ramps differ by a factor of two), far
# outside this band, which is why the assertion currently fails.
RTOL = 1e-2
ATOL = 1e-2


def _diagram():
    """Step -> RateLimiter -> Scope as a .diablos-format dict."""
    blocks = [
        {
            "block_fn": "Step", "sid": 0, "username": "u",
            "coords_left": 50, "coords_top": 200, "coords_width": 50,
            "coords_height": 40, "coords_height_base": 40,
            "in_ports": 0, "out_ports": 1, "dragging": False,
            "selected": False, "b_color": "#064e3b", "b_type": 0,
            "io_edit": "none", "fn_name": "step",
            "params": {"value": 1.0, "delay": 0.5, "type": "up"},
            "external": False, "flipped": False,
        },
        {
            "block_fn": "RateLimiter", "sid": 1, "username": "rl",
            "coords_left": 200, "coords_top": 200, "coords_width": 60,
            "coords_height": 50, "coords_height_base": 50,
            "in_ports": 1, "out_ports": 1, "dragging": False,
            "selected": False, "b_color": "#DDA0DD", "b_type": 2,
            "io_edit": "none", "fn_name": "ratelimiter",
            "params": {"rising_slew": 1.0, "falling_slew": 1.0},
            "external": False, "flipped": False,
        },
        {
            "block_fn": "Scope", "sid": 2, "username": "out",
            "coords_left": 350, "coords_top": 200, "coords_width": 60,
            "coords_height": 50, "coords_height_base": 50,
            "in_ports": 1, "out_ports": 0, "dragging": False,
            "selected": False, "b_color": "#FFB6C1", "b_type": 3,
            "io_edit": "in", "fn_name": "scope",
            "params": {"labels": "out"},
            "external": False, "flipped": False,
        },
    ]
    lines = [
        {"name": "Line0", "sid": 0, "srcblock": "step0", "srcport": 0,
         "dstblock": "ratelimiter1", "dstport": 0,
         "points": [[110, 220], [200, 220]], "cptr": 0, "selected": False},
        {"name": "Line1", "sid": 1, "srcblock": "ratelimiter1", "srcport": 0,
         "dstblock": "scope2", "dstport": 0,
         "points": [[260, 220], [350, 220]], "cptr": 0, "selected": False},
    ]
    return {
        "sim_data": {
            "wind_width": 1280, "wind_height": 770, "fps": 60,
            "sim_time": SIM_TIME, "sim_dt": SIM_DT, "sim_trange": 100000,
        },
        "blocks_data": blocks, "lines_data": lines, "version": "2.0",
    }


def _write_diagram(tmp_path: Path) -> str:
    path = tmp_path / "rate_limiter_equiv.diablos"
    path.write_text(json.dumps(_diagram()))
    return str(path)


def _scope_trace(dsim):
    """Return the (n_samples, vec_dim) trajectory captured by the Scope."""
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        return np.asarray(vec, dtype=float).reshape(-1, params.get("vec_dim", 1))
    raise AssertionError("No Scope with a captured vector was found")


def _run(diagram_path: str, use_fast: bool):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=diagram_path)
    assert data is not None, f"Failed to load {diagram_path}"
    dsim.file_service.apply_loaded_data(data)
    dsim.use_fast_solver = use_fast
    # apply_loaded_data returns the sim params but does not set them on the DSim
    # facade, so pin them explicitly for a deterministic run duration.
    dsim.sim_time = SIM_TIME
    dsim.sim_dt = SIM_DT
    ok, err = dsim.run_tuning_simulation(SIM_TIME, SIM_DT)
    assert ok, f"Simulation (use_fast={use_fast}) failed: {err}"
    return _scope_trace(dsim)


@pytest.mark.regression
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Interpreted RateLimiter ramps at ~2x the configured slew rate: as a "
        "memory block it runs execute() twice per step (output_only + state "
        "update) and advances params['_prev'] on both, while the compiled "
        "kernel integrates the slew limit once and is correct."
    ),
)
def test_ratelimiter_compiled_matches_interpreted(qapp, tmp_path):
    diagram_path = _write_diagram(tmp_path)

    compiled = _run(diagram_path, use_fast=True)
    interp = _run(diagram_path, use_fast=False)

    # Both engines step at the same dt from t=0, so samples are index-aligned;
    # the interpreter emits one extra trailing half-step, so compare the common
    # prefix (i.e. at the interpreter's sample times).
    n = min(compiled.shape[0], interp.shape[0])
    compiled, interp = compiled[:n], interp[:n]

    max_abs = float(np.max(np.abs(compiled - interp)))
    assert np.allclose(compiled, interp, rtol=RTOL, atol=ATOL), (
        f"RateLimiter compiled vs interpreted traces diverge "
        f"(max|delta|={max_abs:.3e}). compiled[:3,0]={compiled[:3, 0]}, "
        f"interp[:3,0]={interp[:3, 0]}"
    )
