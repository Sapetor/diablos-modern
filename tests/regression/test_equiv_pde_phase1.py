"""Compiled-vs-interpreted equivalence for the PDE Phase 1 boundary conditions.

Companion to ``test_equiv_pde_neumann2d.py`` (same harness, same two paths):

  * compiled    -- ``SystemCompiler`` + ``solve_ivp`` (RK45), and
  * interpreter -- the ``block.execute()`` fixed-step loop.

Both paths run the SAME spatial operator out of ``lib/engine/pde_ops.py`` and
differ only in the time integrator, so a divergence here means the periodic /
Robin branches were added to one path but not the other -- exactly the class of
bug that the single-sourced operator module exists to prevent.

Covered: periodic heat 1D, periodic heat 2D, periodic wave 1D, conservation of
total heat on a periodic ring, and a Robin plate driven by a time-varying
ambient temperature.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.regression

# Loose enough to absorb RK45-vs-forward-Euler scheme error, tight enough that a
# genuinely different BC branch (an unwrapped end node, a missing Robin flux)
# shows up as an O(1) failure.
RTOL = 2e-2
ATOL = 5e-3


# --------------------------------------------------------------------------- #
# Diagram builders
# --------------------------------------------------------------------------- #
def _sim_data(sim_time, sim_dt):
    return {
        "wind_width": 1280,
        "wind_height": 770,
        "fps": 60,
        "sim_time": sim_time,
        "sim_dt": sim_dt,
        "sim_trange": 200,
    }


def _scope(sid, labels, n_in, left=610, top=220):
    return {
        "block_fn": "Scope",
        "sid": sid,
        "username": f"Scope{sid}",
        "coords_left": left,
        "coords_top": top,
        "coords_width": 80,
        "coords_height": 64,
        "coords_height_base": 60,
        "in_ports": n_in,
        "out_ports": 0,
        "dragging": False,
        "selected": False,
        "b_color": "#7f1d1d",
        "b_type": 3,
        "io_edit": "input",
        "fn_name": "scope",
        "params": {"labels": labels},
        "external": False,
        "flipped": False,
    }


def _field_sink(sid, fn_name, block_fn, params, left=440, top=290):
    return {
        "block_fn": block_fn,
        "sid": sid,
        "username": f"{block_fn}{sid}",
        "coords_left": left,
        "coords_top": top,
        "coords_width": 110,
        "coords_height": 70,
        "coords_height_base": 70,
        "in_ports": 1,
        "out_ports": 0,
        "dragging": False,
        "selected": False,
        "b_color": "#78350f",
        "b_type": 1,
        "io_edit": "none",
        "fn_name": fn_name,
        "params": params,
        "external": False,
        "flipped": False,
    }


def _line(sid, src, srcport, dst, dstport):
    return {
        "name": f"Line{sid}",
        "sid": sid,
        "srcblock": src,
        "srcport": srcport,
        "dstblock": dst,
        "dstport": dstport,
        "points": [[370, 170], [440, 320]],
        "cptr": 0,
        "selected": False,
    }


def _pde_block(block_fn, fn_name, sid, in_ports, out_ports, params):
    return {
        "block_fn": block_fn,
        "sid": sid,
        "username": f"{block_fn}{sid}",
        "coords_left": 230,
        "coords_top": 140,
        "coords_width": 140,
        "coords_height": 150,
        "coords_height_base": 150,
        "in_ports": in_ports,
        "out_ports": out_ports,
        "dragging": False,
        "selected": False,
        "b_color": "#78350f",
        "b_type": 1,
        "io_edit": "none",
        "fn_name": fn_name,
        "params": params,
        "external": False,
        "flipped": False,
    }


def _heat1d_periodic_diagram():
    """Periodic 1D rod, sine IC -> Scope(T_avg), field -> FieldScope.

    The rod is a ring with no source, so T_avg is a conserved quantity: it also
    doubles as the conservation probe in ``test_periodic_heat_1d_conserves``.
    """
    return {
        "sim_data": _sim_data(0.4, 0.002),
        "blocks_data": [
            _pde_block(
                "HeatEquation1D",
                "heatequation1d",
                1,
                3,
                2,
                {
                    "alpha": 0.02,
                    "L": 1.0,
                    "N": 21,
                    "bc_type_left": "Periodic",
                    "bc_type_right": "Periodic",
                    "init_conds": "sine",
                },
            ),
            _scope(2, "T_avg", 1),
            _field_sink(
                3,
                "fieldscope",
                "FieldScope",
                {"L": 1.0, "colormap": "viridis", "title": "Ring", "display_mode": "heatmap"},
            ),
        ],
        "lines_data": [
            _line(200, "heatequation1d1", 0, "fieldscope3", 0),
            _line(201, "heatequation1d1", 1, "scope2", 0),
        ],
    }


def _heat2d_periodic_diagram():
    """x-periodic / y-periodic 2D plate (a torus), sinusoidal IC."""
    return {
        "sim_data": _sim_data(0.6, 0.004),
        "blocks_data": [
            _pde_block(
                "HeatEquation2D",
                "heatequation2d",
                1,
                5,
                3,
                {
                    "alpha": 0.02,
                    "Lx": 1.0,
                    "Ly": 1.0,
                    "Nx": 9,
                    "Ny": 9,
                    "bc_type_left": "Periodic",
                    "bc_type_right": "Periodic",
                    "bc_type_bottom": "Periodic",
                    "bc_type_top": "Periodic",
                    "init_temp": "gaussian",
                    "init_amplitude": 1.0,
                },
            ),
            _scope(2, "T_avg, T_max", 2),
            _field_sink(
                3,
                "fieldscope2d",
                "FieldScope2D",
                {"Lx": 1.0, "Ly": 1.0, "colormap": "hot", "title": "Torus", "sample_interval": 10},
            ),
        ],
        "lines_data": [
            _line(200, "heatequation2d1", 0, "fieldscope2d3", 0),
            _line(201, "heatequation2d1", 1, "scope2", 0),
            _line(202, "heatequation2d1", 2, "scope2", 1),
        ],
    }


def _wave1d_periodic_diagram():
    """Periodic 1D string, gaussian pulse -> Scope(energy), field -> FieldScope."""
    return {
        "sim_data": _sim_data(0.3, 0.0005),
        "blocks_data": [
            _pde_block(
                "WaveEquation1D",
                "waveequation1d",
                1,
                3,
                3,
                {
                    "c": 0.5,
                    "damping": 0.0,
                    "L": 1.0,
                    "N": 41,
                    "bc_type_left": "Periodic",
                    "bc_type_right": "Periodic",
                    "init_displacement": "sine",
                    "init_velocity": [0.0],
                },
            ),
            _scope(2, "energy", 1),
            _field_sink(
                3,
                "fieldscope",
                "FieldScope",
                {"L": 1.0, "colormap": "viridis", "title": "Ring", "display_mode": "heatmap"},
            ),
        ],
        "lines_data": [
            _line(200, "waveequation1d1", 0, "fieldscope3", 0),
            _line(201, "waveequation1d1", 2, "scope2", 0),
        ],
    }


def _heat2d_dynamic_robin_diagram(ambient_slope=40.0):
    """Robin-cooled plate whose ambient temperature RAMPS during the run.

    The ambient value is the ``bc_*`` input port, so a Ramp source drives it.
    The plate starts at 0 and the ambient climbs, so a working Robin edge drags
    the plate up with it -- if the Robin branch were missing on either path,
    that path's plate would sit at its initial value instead.
    """
    ramp = {
        "block_fn": "Ramp",
        "sid": 4,
        "username": "Ambient",
        "coords_left": 60,
        "coords_top": 160,
        "coords_width": 80,
        "coords_height": 60,
        "coords_height_base": 60,
        "in_ports": 0,
        "out_ports": 1,
        "dragging": False,
        "selected": False,
        "b_color": "#1e3a8a",
        "b_type": 0,
        "io_edit": "none",
        "fn_name": "ramp",
        "params": {"slope": ambient_slope, "delay": 0.0},
        "external": False,
        "flipped": False,
    }
    return {
        "sim_data": _sim_data(0.5, 0.002),
        "blocks_data": [
            _pde_block(
                "HeatEquation2D",
                "heatequation2d",
                1,
                5,
                3,
                {
                    "alpha": 0.05,
                    "Lx": 1.0,
                    "Ly": 1.0,
                    "Nx": 7,
                    "Ny": 7,
                    "bc_type_left": "Robin",
                    "bc_type_right": "Robin",
                    "bc_type_bottom": "Robin",
                    "bc_type_top": "Robin",
                    "h_left": 20.0,
                    "h_right": 20.0,
                    "h_bottom": 20.0,
                    "h_top": 20.0,
                    "k_thermal": 1.0,
                    "init_temp": "0.0",
                },
            ),
            _scope(2, "T_avg, T_max", 2),
            _field_sink(
                3,
                "fieldscope2d",
                "FieldScope2D",
                {
                    "Lx": 1.0,
                    "Ly": 1.0,
                    "colormap": "hot",
                    "title": "Robin plate",
                    "sample_interval": 10,
                },
            ),
            ramp,
        ],
        "lines_data": [
            _line(200, "heatequation2d1", 0, "fieldscope2d3", 0),
            _line(201, "heatequation2d1", 1, "scope2", 0),
            _line(202, "heatequation2d1", 2, "scope2", 1),
            # Ambient temperature into all four Robin edges.
            _line(203, "ramp4", 0, "heatequation2d1", 1),
            _line(204, "ramp4", 0, "heatequation2d1", 2),
            _line(205, "ramp4", 0, "heatequation2d1", 3),
            _line(206, "ramp4", 0, "heatequation2d1", 4),
        ],
    }


# --------------------------------------------------------------------------- #
# Harness (mirrors test_equiv_pde_neumann2d.py)
# --------------------------------------------------------------------------- #
def _scope_traces(dsim):
    traces = {}
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        traces[b.name] = np.asarray(vec, dtype=float).reshape(-1, params.get("vec_dim", 1))
    return traces


def _run(filepath, fast):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(filepath))
    assert data is not None, f"Failed to load {filepath}"
    sim_params = dsim.file_service.apply_loaded_data(data)
    # apply_loaded_data only RETURNS the sim params (the GUI normally applies
    # them), so push them onto the DSim ourselves. Without this every diagram
    # silently runs at the 1.0 s / 0.01 s defaults, and the wave diagrams below
    # would be integrated past the interpreter's Forward-Euler stability limit.
    dsim.sim_time = sim_params["sim_time"]
    dsim.sim_dt = sim_params["sim_dt"]
    dsim.use_fast_solver = fast

    assert dsim.engine.check_compilability(dsim.blocks_list), (
        "diagram must be compilable so the compiled path actually runs"
    )

    ok, err = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
    assert ok, f"{'compiled' if fast else 'interpreted'} run failed: {err}"
    return _scope_traces(dsim)


def _both_paths(tmp_path_factory, name, diagram):
    path = tmp_path_factory.mktemp(name) / f"{name}.diablos"
    path.write_text(json.dumps(diagram))
    return _run(path, fast=True), _run(path, fast=False)


@pytest.fixture(scope="module")
def heat1d_periodic(tmp_path_factory, qapp):
    return _both_paths(tmp_path_factory, "heat1d_periodic", _heat1d_periodic_diagram())


@pytest.fixture(scope="module")
def heat2d_periodic(tmp_path_factory, qapp):
    return _both_paths(tmp_path_factory, "heat2d_periodic", _heat2d_periodic_diagram())


@pytest.fixture(scope="module")
def wave1d_periodic(tmp_path_factory, qapp):
    return _both_paths(tmp_path_factory, "wave1d_periodic", _wave1d_periodic_diagram())


@pytest.fixture(scope="module")
def heat2d_dynamic_robin(tmp_path_factory, qapp):
    return _both_paths(tmp_path_factory, "heat2d_dyn_robin", _heat2d_dynamic_robin_diagram())


def _assert_paths_agree(both, label, rtol=RTOL, atol=ATOL):
    compiled, interpreted = both
    assert "scope2" in compiled and "scope2" in interpreted, f"{label}: no scope trace"
    c, i = compiled["scope2"], interpreted["scope2"]
    n = min(len(c), len(i))
    assert n > 5, f"{label}: too few samples to compare ({n})"
    assert np.allclose(c[:n], i[:n], rtol=rtol, atol=atol), (
        f"{label}: compiled and interpreted diverge; max abs diff {np.max(np.abs(c[:n] - i[:n]))}"
    )
    return c, i


# --------------------------------------------------------------------------- #
# Periodic equivalence
# --------------------------------------------------------------------------- #
def test_periodic_heat_1d_paths_agree(heat1d_periodic):
    _assert_paths_agree(heat1d_periodic, "periodic heat 1D")


def test_periodic_heat_1d_conserves_total_heat(heat1d_periodic):
    """A ring with no source conserves total heat, so the mean temperature must
    hold flat on BOTH paths -- the physical invariant that distinguishes a
    correctly wrapped end node from one that leaks (or is frozen).
    """
    compiled, interpreted = heat1d_periodic
    for label, trace in (("compiled", compiled["scope2"]), ("interpreted", interpreted["scope2"])):
        avg = trace[:, 0]
        drift = np.max(np.abs(avg - avg[0]))
        assert drift < 1e-6 * max(1.0, abs(avg[0])) + 1e-9, (
            f"{label} periodic ring lost heat: T_avg drifted by {drift}"
        )
        # Guard against a frozen field masquerading as conservation.
        assert abs(avg[0]) > 1e-3, "IC is ~0, the conservation check proves nothing"


def test_periodic_heat_2d_paths_agree(heat2d_periodic):
    _assert_paths_agree(heat2d_periodic, "periodic heat 2D")


def test_periodic_heat_2d_diffuses_and_conserves(heat2d_periodic):
    """On a torus the peak must decay (diffusion is doing something) while the
    average holds (nothing escapes)."""
    compiled, interpreted = heat2d_periodic
    for label, trace in (("compiled", compiled["scope2"]), ("interpreted", interpreted["scope2"])):
        avg, peak = trace[:, 0], trace[:, 1]
        assert peak[-1] < peak[0] - 1e-3, f"{label}: torus field is not diffusing"
        assert np.max(np.abs(avg - avg[0])) < 1e-6 * abs(avg[0]) + 1e-9, f"{label}: torus lost heat"


def test_periodic_wave_1d_paths_agree(wave1d_periodic):
    _assert_paths_agree(wave1d_periodic, "periodic wave 1D")


def test_periodic_wave_1d_energy_is_bounded(wave1d_periodic):
    """An undamped periodic string neither gains nor loses much energy. A
    reflecting-boundary regression would show up as an energy jump."""
    compiled, _ = wave1d_periodic
    energy = compiled["scope2"][:, 0]
    assert energy[0] > 0
    assert np.max(energy) < 1.5 * energy[0]
    assert np.min(energy) > 0.5 * energy[0]


# --------------------------------------------------------------------------- #
# Dynamic Robin
# --------------------------------------------------------------------------- #
def test_dynamic_robin_2d_paths_agree(heat2d_dynamic_robin):
    """Wider atol than the periodic cases on purpose: the plate AND the ramped
    ambient both start at exactly 0, so for the first few samples there is no
    signal magnitude for rtol to work against and the one-step lag between
    Forward Euler (which sees the ambient at t_n) and RK45 shows up as a
    factor-of-two ratio between two sub-milli-kelvin numbers. 0.05 is under 0.4%
    of the trace's final value, so a real BC regression still fails loudly.
    """
    _assert_paths_agree(heat2d_dynamic_robin, "dynamic-ambient Robin 2D", atol=0.05)


def test_dynamic_robin_2d_field_follows_the_ambient(heat2d_dynamic_robin):
    """The ambient temperature is driven by a Ramp on the bc_* input ports, so
    the Robin-cooled plate must warm up as the run proceeds. A plate stuck at
    its initial 0 would mean the BC value was read once and cached, or that the
    Robin branch never fired.
    """
    compiled, interpreted = heat2d_dynamic_robin
    for label, trace in (("compiled", compiled["scope2"]), ("interpreted", interpreted["scope2"])):
        avg = trace[:, 0]
        assert avg[0] < 0.5, f"{label}: plate should start cold, got {avg[0]}"
        assert avg[-1] > 1.0, (
            f"{label}: plate did not follow the rising ambient (final T_avg {avg[-1]})"
        )
        assert np.all(np.diff(avg) > -1e-6), f"{label}: plate cooled while ambient rose"
