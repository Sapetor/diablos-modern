"""Compiled-vs-interpreted equivalence for an all-Neumann 2D heat equation.

This pins the flagged corner case from ``tasks/todo.md``: a 2D heat-equation
plate with Neumann (insulated) boundaries on all four edges, which is the one
configuration where the compiled kernel applies its special corner-derivative
fill (``lib/engine/compiler_kernels/pde.py:_fill_neumann_corners``). A sinusoidal
initial field diffuses under all-Neumann BCs, so both the average and the peak
temperature evolve over time -- a meaningful trajectory to compare.

The two execution paths are:
  * compiled  -- ``SystemCompiler`` + ``solve_ivp`` (RK45), and
  * interpreter -- the ``block.execute()`` time-step loop.

FINDING (see ``test_compiled_matches_interpreted`` below): the two paths do NOT
agree for the 2D PDE families, and this is not a tolerance question. Unlike the
1D PDE blocks -- whose ``execute()`` self-integrates with Forward Euler and
persists state in ``params`` (e.g. ``blocks/pde/heat_equation_1d.py`` stores
``params['T']`` and steps every call) -- the 2D blocks' ``execute()`` only
reshapes a compiled-solver-supplied ``state`` kwarg
(``blocks/pde/heat_equation_2d.py:173-194``). The interpreter never passes
``state`` and the block never advances or stores its own field, so under the
interpreter the 2D temperature field stays frozen at the initial condition for
the whole run. The compiled path integrates it normally. The equivalence
assertion is therefore marked ``xfail`` until the 2D PDE blocks grow an
interpreter integration path; ``test_divergence_characterization`` pins the
current behaviour so a fix (or a regression in either path) is noticed.
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.regression


def _build_diagram(nx=9, ny=9, alpha=0.05, sim_time=1.0, sim_dt=0.02):
    """All-Neumann 2D heat plate -> Scope(T_avg, T_max), field -> FieldScope2D.

    The FieldScope2D sink consumes the (required) T_field output so the diagram
    passes the engine's connectivity check; the scalar T_avg / T_max outputs
    feed the Scope whose trajectory the two paths are compared on.
    """
    return {
        "sim_data": {
            "wind_width": 1280,
            "wind_height": 770,
            "fps": 60,
            "sim_time": sim_time,
            "sim_dt": sim_dt,
            "sim_trange": 200,
        },
        "blocks_data": [
            {
                "block_fn": "HeatEquation2D",
                "sid": 1,
                "username": "HeatPlate",
                "coords_left": 230, "coords_top": 140,
                "coords_width": 140, "coords_height": 150, "coords_height_base": 150,
                "in_ports": 5, "out_ports": 3,
                "dragging": False, "selected": False,
                "b_color": "#78350f", "b_type": 1, "io_edit": "none",
                "fn_name": "heatequation2d",
                "params": {
                    "alpha": alpha,
                    "Lx": 1.0, "Ly": 1.0,
                    "Nx": nx, "Ny": ny,
                    "bc_type_left": "Neumann",
                    "bc_type_right": "Neumann",
                    "bc_type_bottom": "Neumann",
                    "bc_type_top": "Neumann",
                    "init_temp": "sinusoidal",
                    "init_amplitude": 1.0,
                },
                "external": False, "flipped": False,
            },
            {
                "block_fn": "Scope",
                "sid": 2,
                "username": "AvgMaxScope",
                "coords_left": 610, "coords_top": 220,
                "coords_width": 80, "coords_height": 64, "coords_height_base": 60,
                "in_ports": 2, "out_ports": 0,
                "dragging": False, "selected": False,
                "b_color": "#7f1d1d", "b_type": 3, "io_edit": "input",
                "fn_name": "scope",
                "params": {"labels": "T_avg, T_max"},
                "external": False, "flipped": False,
            },
            {
                "block_fn": "FieldScope2D",
                "sid": 3,
                "username": "HeatMap2D",
                "coords_left": 440, "coords_top": 290,
                "coords_width": 110, "coords_height": 70, "coords_height_base": 70,
                "in_ports": 1, "out_ports": 0,
                "dragging": False, "selected": False,
                "b_color": "#78350f", "b_type": 1, "io_edit": "none",
                "fn_name": "fieldscope2d",
                "params": {
                    "Lx": 1.0, "Ly": 1.0,
                    "colormap": "hot", "title": "2D Temperature",
                    "sample_interval": 10,
                },
                "external": False, "flipped": False,
            },
        ],
        "lines_data": [
            {
                "name": "Line200", "sid": 200,
                "srcblock": "heatequation2d1", "srcport": 0,
                "dstblock": "fieldscope2d3", "dstport": 0,
                "points": [[370, 170], [440, 320]], "cptr": 0, "selected": False,
            },
            {
                "name": "Line201", "sid": 201,
                "srcblock": "heatequation2d1", "srcport": 1,
                "dstblock": "scope2", "dstport": 0,
                "points": [[370, 200], [610, 240]], "cptr": 0, "selected": False,
            },
            {
                "name": "Line202", "sid": 202,
                "srcblock": "heatequation2d1", "srcport": 2,
                "dstblock": "scope2", "dstport": 1,
                "points": [[370, 230], [610, 260]], "cptr": 0, "selected": False,
            },
        ],
    }


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


def _run(filepath, fast):
    """Load ``filepath`` and run it once; return {scope_name: trace}."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(filepath))
    assert data is not None, f"Failed to load {filepath}"
    dsim.file_service.apply_loaded_data(data)
    dsim.use_fast_solver = fast

    compilable = dsim.engine.check_compilability(dsim.blocks_list)
    assert compilable, "diagram must be compilable so the compiled path runs"

    ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
    assert ok, f"{'compiled' if fast else 'interpreted'} run failed: {err}"
    return _scope_traces(dsim)


@pytest.fixture(scope="module")
def both_paths(tmp_path_factory, qapp):
    """Run the same all-Neumann 2D heat diagram through both paths once."""
    path = tmp_path_factory.mktemp("pde_neumann2d") / "heat2d_neumann.diablos"
    path.write_text(json.dumps(_build_diagram()))
    compiled = _run(path, fast=True)
    interpreted = _run(path, fast=False)
    return compiled, interpreted


def test_both_paths_execute(both_paths):
    """Harness sanity: both paths produce a matching-shape (T_avg, T_max) trace."""
    compiled, interpreted = both_paths
    assert "scope2" in compiled and "scope2" in interpreted
    c, i = compiled["scope2"], interpreted["scope2"]
    assert c.shape == i.shape
    assert c.shape[1] == 2  # T_avg, T_max


@pytest.mark.xfail(
    strict=True,
    reason=(
        "HeatEquation2D has no interpreter-mode time integration: its execute() "
        "only reshapes a compiled-solver-supplied 'state' kwarg "
        "(blocks/pde/heat_equation_2d.py:173-194), which the interpreter never "
        "passes, so the 2D field stays frozen at the initial condition. The 1D "
        "PDE blocks self-integrate via Forward Euler in execute(); the 2D blocks "
        "do not. Compiled and interpreted trajectories cannot match until the 2D "
        "PDE blocks gain an interpreter integration path."
    ),
)
def test_compiled_matches_interpreted(both_paths):
    """Compiled (RK45) vs interpreted trajectories at the interpreter's samples.

    Loose tolerances (rtol=1e-2, atol=1e-3) would comfortably absorb the
    RK45-vs-fixed-step scheme difference; the mismatch here is structural (the
    interpreter never integrates the 2D field), not numerical.
    """
    compiled, interpreted = both_paths
    c, i = compiled["scope2"], interpreted["scope2"]
    n = min(len(c), len(i))
    assert np.allclose(c[:n], i[:n], rtol=1e-2, atol=1e-3)


def test_divergence_characterization(both_paths):
    """Pin the current divergent behaviour so a fix/regression is noticed.

    If the 2D PDE interpreter gap is ever closed (or either path regresses),
    one of these assertions -- and the xfail above -- will flip, flagging that
    this file needs updating.
    """
    compiled, interpreted = both_paths
    c, i = compiled["scope2"], interpreted["scope2"]

    # Interpreter: the 2D field is frozen at the IC, so every sample equals the
    # first for both T_avg (col 0) and T_max (col 1).
    assert np.allclose(i, i[0], atol=1e-9), (
        "interpreter is no longer frozen at the initial condition -- the 2D PDE "
        "block may have gained an interpreter integration path"
    )

    # Compiled: all-Neumann diffusion smooths the sinusoidal bump, so the peak
    # temperature (col 1) decays well below its initial value.
    assert c[-1, 1] < c[0, 1] - 0.1, (
        "compiled path no longer integrates the 2D heat PDE as expected"
    )
