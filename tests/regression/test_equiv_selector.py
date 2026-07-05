"""Compiled-vs-interpreted equivalence regression for the Selector block.

The interpreter path (``block.execute()`` time-step loop) and the compiled path
(``SystemCompiler`` -> ``solve_ivp``) implement Selector element extraction
independently: the interpreter runs ``blocks/selector.py``'s ``_parse_indices``,
while the compiled path runs the pre-parsed kernel in
``lib/engine/compiler_kernels/nonlinear.py`` (``build_selector``). This test pins
that both pick the *same* vector elements and produce matching scope
trajectories.

Diagram (all blocks compilable, so the fast path is genuinely exercised):

    Sine ---> Integrator --.
      |                     \
      '-------------------> Mux[3] --> Selector("1:3") --> Scope
    Constant --------------/

The Mux stacks ``[sin, integral(sin), const]``; ``Selector("1:3")`` keeps
elements 1 and 2 -> ``[integral(sin), const]``. The integrator gives the selected
signal genuine dynamics, so the compiled (adaptive RK45) and interpreted
(fixed-step Euler) schemes legitimately differ by the Euler truncation error;
the tolerance is sized to that, not fudged (measured max|delta| ~= 1e-2 for the
dynamic channel, exact for the constant channel).

Index notation note: Selector indices are given as a colon *range* (``"1:3"``)
rather than a comma list (``"1,2"``) on purpose. ``WorkspaceManager.resolve_params``
runs every string param through ``safe_expr`` before the interpreter sees it, and
``"1,2"`` evaluates to the tuple ``(1, 2)`` -- which ``_parse_indices`` cannot
``.split(',')`` (it raises) while the compiler silently defaults to index 0, so a
comma list diverges for an unrelated reason. ``"1:3"`` is not a valid expression,
so it survives as a string in both paths. See the report notes for this session.
"""
import json

import numpy as np
import pytest


SIM_TIME = 5.0
SIM_DT = 0.01

# Euler (interpreter) vs adaptive RK45 (compiled) on integral(2*sin(3t)); the
# constant channel matches exactly, the integrated channel differs by the Euler
# truncation error (measured ~1e-2 over [0, 5]).
RTOL = 1e-2
ATOL = 5e-2


def _block(block_fn, fn_name, name, left, top, in_ports, out_ports, params, w=50, h=50):
    return {
        "block_fn": block_fn, "sid": 0, "name": name, "username": name,
        "coords_left": left, "coords_top": top, "coords_width": w, "coords_height": h,
        "coords_height_base": h, "in_ports": in_ports, "out_ports": out_ports,
        "dragging": False, "selected": False, "b_color": "#888888", "b_type": 2,
        "io_edit": "none", "fn_name": fn_name, "params": params,
        "external": False, "flipped": False,
    }


def _line(sid, src, sp, dst, dp):
    return {
        "name": "L%d" % sid, "sid": sid, "srcblock": src, "srcport": sp,
        "dstblock": dst, "dstport": dp, "points": [[0, 0], [1, 1]],
        "cptr": 0, "selected": False,
    }


def _diagram_dict():
    return {
        "sim_data": {
            "wind_width": 1280, "wind_height": 770, "fps": 60,
            "sim_time": SIM_TIME, "sim_dt": SIM_DT, "sim_trange": 100,
        },
        "blocks_data": [
            _block("Sine", "sine", "sine0", 50, 50, 0, 1,
                   {"amplitude": 2.0, "omega": 3.0, "init_angle": 0.0}),
            # FWD_EULER (not the default SOLVE_IVP/RK45) so the interpreter records
            # one scope sample per dt, aligning its grid with the compiled output.
            _block("Integrator", "integrator", "integrator0", 180, 50, 1, 1,
                   {"init_conds": 0.0, "method": "FWD_EULER"}),
            _block("Constant", "constant", "constant0", 50, 200, 0, 1,
                   {"value": 5.0}),
            _block("Mux", "mux", "mux0", 320, 80, 3, 1, {}, h=80),
            _block("Selector", "selector", "selector0", 460, 90, 1, 1,
                   {"indices": "1:3"}),
            _block("Scope", "scope", "scope0", 600, 90, 1, 0,
                   {"labels": "integ,const"}),
        ],
        "lines_data": [
            _line(0, "sine0", 0, "integrator0", 0),
            _line(1, "sine0", 0, "mux0", 0),
            _line(2, "integrator0", 0, "mux0", 1),
            _line(3, "constant0", 0, "mux0", 2),
            _line(4, "mux0", 0, "selector0", 0),
            _line(5, "selector0", 0, "scope0", 0),
        ],
        "version": "1.0",
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


def _run(diagram_path, use_fast):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(diagram_path))
    assert data is not None, "Failed to load generated diagram"
    dsim.file_service.apply_loaded_data(data)

    assert dsim.engine.check_compilability(dsim.blocks_list), (
        "diagram must be compilable so the compiled path is actually exercised"
    )
    dsim.use_fast_solver = use_fast
    ok, err = dsim.run_tuning_simulation(SIM_TIME, SIM_DT)
    assert ok, "%s run failed: %s" % ("compiled" if use_fast else "interpreted", err)
    return _scope_traces(dsim)


@pytest.mark.regression
def test_selector_compiled_matches_interpreted(qapp, tmp_path):
    diagram_path = tmp_path / "equiv_selector.diablos"
    diagram_path.write_text(json.dumps(_diagram_dict()))

    compiled = _run(diagram_path, use_fast=True)
    interpreted = _run(diagram_path, use_fast=False)

    assert compiled and interpreted, "both paths must produce a Scope trace"
    assert set(compiled) == set(interpreted)

    for name, c_trace in compiled.items():
        i_trace = interpreted[name]
        # Selector("1:3") extracts 2 of the 3 muxed elements in both paths.
        assert c_trace.shape[1] == 2, "Selector should emit a 2-vector"
        assert i_trace.shape[1] == 2

        n = min(len(c_trace), len(i_trace))
        assert n >= int(SIM_TIME / SIM_DT), "too few aligned samples to compare"
        c, i = c_trace[:n], i_trace[:n]

        # Channel 1 is the passed-through Constant(5.0): selection is exact, so
        # this must match to machine precision regardless of integration scheme.
        assert np.allclose(c[:, 1], 5.0, atol=1e-9)
        assert np.allclose(c[:, 1], i[:, 1], atol=1e-9)

        # Channel 0 is integral(sin): compiled RK45 vs interpreted Euler differ
        # only by the Euler truncation error.
        max_abs = float(np.max(np.abs(c - i)))
        assert np.allclose(c, i, rtol=RTOL, atol=ATOL), (
            "%s: compiled/interpreted Selector trajectories diverge "
            "(max|delta|=%.3e), beyond the Euler-vs-RK45 tolerance" % (name, max_abs)
        )
