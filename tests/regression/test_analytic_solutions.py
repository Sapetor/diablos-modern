"""Compiled-path validation against closed-form analytic solutions.

The compiled fast solver (SystemCompiler + scipy solve_ivp) is the numerically
accurate engine, so we pin it against exact solutions rather than only against
its own golden traces or the (lower-order) interpreter. Each test builds a small
diagram, runs the compiled path, and asserts the Scope trajectory matches the
analytic result. A kernel regression that changes the math will break these with
a clear physical meaning, not just a trace diff.

  * Integrator with constant input:        y(t) = y0 + c*t
  * First-order lag 1/(s+1), unit step:    y(t) = 1 - e^{-t}
  * 1D heat eigenmode, homogeneous BCs:     amplitude ~ e^{-alpha (pi/L)^2 t}
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.regression


def _run_compiled(diagram, tmp_path):
    """Write ``diagram`` to a temp .diablos, run the compiled path, and return
    ``{'timeline', 'signals'}`` from the Scopes."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager
    from lib.analysis.resim import harvest_scope_signals

    path = tmp_path / "analytic.diablos"
    path.write_text(json.dumps(diagram))

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None
    dsim.file_service.apply_loaded_data(data)
    dsim.use_fast_solver = True

    sim = diagram["sim_data"]
    ok, err = dsim.run_tuning_simulation(sim["sim_time"], sim["sim_dt"])
    assert ok, f"compiled run failed: {err}"

    assert dsim.engine.check_compilability(dsim.blocks_list), (
        "diagram must be compilable so the compiled kernels actually run"
    )
    return harvest_scope_signals(dsim)


def _blk(fn, sid, name, x, params, in_ports, out_ports, b_type, io):
    return {
        "block_fn": fn,
        "sid": sid,
        "username": name,
        "coords_left": x,
        "coords_top": 200,
        "coords_width": 60,
        "coords_height": 50,
        "coords_height_base": 50,
        "in_ports": in_ports,
        "out_ports": out_ports,
        "dragging": False,
        "selected": False,
        "b_color": "#333333",
        "b_type": b_type,
        "io_edit": io,
        "fn_name": fn.lower(),
        "params": params,
        "external": False,
        "flipped": False,
    }


def _line(sid, srcblock, srcport, dstblock, dstport):
    return {
        "name": f"L{sid}",
        "sid": sid,
        "srcblock": srcblock,
        "srcport": srcport,
        "dstblock": dstblock,
        "dstport": dstport,
        "points": [[0, 0], [1, 1]],
        "cptr": 0,
        "selected": False,
    }


def _diagram(blocks, lines, sim_time, sim_dt):
    return {
        "sim_data": {
            "wind_width": 1280,
            "wind_height": 770,
            "fps": 60,
            "sim_time": sim_time,
            "sim_dt": sim_dt,
            "sim_trange": 100000,
        },
        "blocks_data": blocks,
        "lines_data": lines,
        "version": "2.0",
    }


def test_integrator_of_constant_is_a_ramp(qapp, tmp_path):
    """Integrator(y0=0) of a constant c=2 gives y(t) = 2t."""
    c = 2.0
    diagram = _diagram(
        [
            _blk("Constant", 0, "c", 50, {"value": c}, 0, 1, 0, "none"),
            _blk(
                "Integrator",
                1,
                "int",
                200,
                {"init_conds": 0.0, "method": "SOLVE_IVP", "ivp_method": "RK45"},
                1,
                1,
                1,
                "none",
            ),
            _blk("Scope", 2, "y", 350, {"labels": "y"}, 1, 0, 3, "in"),
        ],
        [_line(0, "constant0", 0, "integrator1", 0), _line(1, "integrator1", 0, "scope2", 0)],
        sim_time=2.0,
        sim_dt=0.02,
    )

    res = _run_compiled(diagram, tmp_path)
    y = next(iter(res["signals"].values()))
    t = res["timeline"][: len(y)]
    assert np.allclose(y, c * t, rtol=1e-4, atol=1e-4)


def test_first_order_lag_step_response(qapp, tmp_path):
    """Unit step into 1/(s+1) gives y(t) = 1 - e^{-t}."""
    diagram = _diagram(
        [
            _blk("Step", 0, "u", 50, {"value": 1.0, "delay": 0.0, "type": "up"}, 0, 1, 0, "none"),
            _blk(
                "TranFn",
                1,
                "G",
                200,
                {"numerator": [1.0], "denominator": [1.0, 1.0]},
                1,
                1,
                1,
                "none",
            ),
            _blk("Scope", 2, "y", 350, {"labels": "y"}, 1, 0, 3, "in"),
        ],
        [_line(0, "step0", 0, "tranfn1", 0), _line(1, "tranfn1", 0, "scope2", 0)],
        sim_time=5.0,
        sim_dt=0.01,
    )

    res = _run_compiled(diagram, tmp_path)
    y = next(iter(res["signals"].values()))
    t = res["timeline"][: len(y)]
    assert np.allclose(y, 1.0 - np.exp(-t), rtol=1e-3, atol=2e-3)


def test_heat_1d_eigenmode_decays_exponentially(qapp, tmp_path):
    """sin(pi x / L) under homogeneous Dirichlet BCs decays as the fundamental
    eigenmode: amplitude(t) = amplitude(0) * e^{-alpha (pi/L)^2 t}.

    The scalar output is the spatial mean of the field; since the mode shape is
    fixed, the mean decays at the same rate. A fine grid (N=41) keeps the
    discrete FD eigenvalue within ~1e-4 of the continuous alpha (pi/L)^2, so the
    continuous rate is a fair reference at the loose tolerance below.
    """
    alpha, L, N = 0.2, 1.0, 41
    # Homogeneous Dirichlet: feed 0 to q_src (port 0) and both BC ports (1, 2).
    diagram = _diagram(
        [
            _blk("Constant", 0, "zero", 50, {"value": 0.0}, 0, 1, 0, "none"),
            _blk(
                "HeatEquation1D",
                1,
                "heat",
                200,
                {
                    "alpha": alpha,
                    "L": L,
                    "N": N,
                    "init_conds": "sin",
                    "bc_type_left": "Dirichlet",
                    "bc_type_right": "Dirichlet",
                },
                3,
                2,
                1,
                "none",
            ),
            _blk("Scope", 2, "Tavg", 400, {"labels": "T_avg"}, 1, 0, 3, "in"),
            _blk("Scope", 3, "Tfield", 400, {"labels": "T_field"}, 1, 0, 3, "in"),
        ],
        [
            _line(0, "constant0", 0, "heatequation1d1", 0),
            _line(1, "constant0", 0, "heatequation1d1", 1),
            _line(2, "constant0", 0, "heatequation1d1", 2),
            _line(3, "heatequation1d1", 0, "scope3", 0),
            _line(4, "heatequation1d1", 1, "scope2", 0),
        ],
        sim_time=0.5,
        sim_dt=0.001,
    )

    res = _run_compiled(diagram, tmp_path)
    tavg = res["signals"]["T_avg"]
    t = res["timeline"][: len(tavg)]
    tavg = tavg[: len(t)]

    rate = alpha * (np.pi / L) ** 2
    analytic = tavg[0] * np.exp(-rate * t)
    # Compare the normalized decay; absolute values are small as the mode decays.
    assert np.allclose(tavg, analytic, rtol=2e-2, atol=1e-3)
    # Sanity: the mode actually decays substantially over the run.
    assert tavg[-1] < 0.5 * tavg[0]
