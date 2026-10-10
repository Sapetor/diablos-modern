"""Sampled-data diagrams on the compiled (hybrid) path.

ZeroOrderHold, DiscreteTranFn and DiscreteStateSpace used to send the whole
diagram to the fixed-step interpreter. They now compile: the solver integrates
adaptively between sample instants and the blocks' own updates run at each
instant (``lib/engine/compiler_kernels/discrete.py``, ``breakpoints`` in
``lib/engine/zero_crossing.solve_with_events``). The interpreter is exact on
these diagrams (cont2discrete for a held input), so it is the reference.
"""

import numpy as np
import pytest


def _run(builder, tmp_path, sim_time, sim_dt, fast=True, solver=None, twice=False):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram

    path = tmp_path / "hybrid.diablos"
    builder.save(str(path))
    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = fast
    if solver is not None:
        dsim.solver_method = solver
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, err
    first = harvest_scope_signals(dsim)
    if twice:
        ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
        assert ok, err
        return dsim, first, harvest_scope_signals(dsim)
    return dsim, first


def _backend(dsim):
    return (dsim.engine.last_solver_diagnostics or {}).get("backend")


_DTF = ("DiscreteTranFn", {"numerator": [1.0], "denominator": [1.0, -0.5]})
_DSS = ("DiscreteStateSpace", {"A": [[0.5]], "B": [[1.0]], "C": [[1.0]], "D": [[0.0]]})


def _staircase(sim_time=1.0, sim_dt=0.1, discrete=_DTF):
    """ramp -> ZOH(0.2) -> y[k] = 0.5 y[k-1] + u[k-1] (inherited rate) -> scope."""
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
    z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2})
    fn, params = discrete
    d = b.add_block(fn, 0, 0, name="d", params={**params, "sampling_time": 0.0})
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
    b.connect(r, 0, z, 0)
    b.connect(z, 0, d, 0)
    b.connect(d, 0, sc, 0)
    return b


def _sampled_loop(saturate=False, sim_time=6.0, sim_dt=0.01):
    """Step -> e -> ZOH(0.1) -> discrete PI -> 2/((s+1)(s+2)) -> feedback.

    With ``saturate`` the plant output goes through a Saturation before the
    feedback and the scope, so located zero-crossing events and sample
    instants share one solve.
    """
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=sim_time, sim_dt=sim_dt)
    r = b.add_block("Step", 0, 0, name="r", params={"value": 1.0, "delay": 0.93, "type": "up"})
    e = b.add_block("Sum", 0, 0, name="e", params={"sign": "+-"}, in_ports=2)
    z = b.add_block("ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.1})
    pi = b.add_block(
        "DiscreteTranFn",
        0,
        0,
        name="pi",
        params={"numerator": [1.05, -1.0], "denominator": [1.0, -1.0], "sampling_time": 0.1},
    )
    g = b.add_block(
        "TranFn", 0, 0, name="g", params={"numerator": [2.0], "denominator": [1.0, 3.0, 2.0]}
    )
    sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y, u"}, in_ports=2, out_ports=0)
    b.connect(r, 0, e, 0)
    b.connect(e, 0, z, 0)
    b.connect(z, 0, pi, 0)
    b.connect(pi, 0, g, 0)
    b.connect(pi, 0, sc, 1)
    out = g
    if saturate:
        out = b.add_block("Saturation", 0, 0, name="sat", params={"min": -0.6, "max": 0.6})
        b.connect(g, 0, out, 0)
    b.connect(out, 0, e, 1)
    b.connect(out, 0, sc, 0)
    return b


@pytest.mark.regression
class TestHybridMatchesInterpreter:
    @pytest.mark.parametrize("discrete", [_DTF, _DSS], ids=["tf", "ss"])
    @pytest.mark.parametrize("sim_dt", [0.1, 0.05])
    def test_staircase_is_reproduced_at_its_sample_instants(self, qapp, tmp_path, sim_dt, discrete):
        """No continuous states: the stateless branch of the hybrid driver.

        Same numbers as test_discrete_block_sample_time.py pins for the
        interpreter, including the ZOH -> DTF ordering at shared instants.
        """
        dsim, data = _run(_staircase(sim_dt=sim_dt, discrete=discrete), tmp_path, 1.0, sim_dt)
        assert _backend(dsim) == "scipy+hybrid"
        t = np.ravel(data["timeline"])
        y = np.ravel(data["signals"]["y"])
        for k, expected in enumerate([0.0, 0.0, 0.2, 0.5, 0.85, 1.225]):
            i = int(np.argmin(np.abs(t - 0.2 * k)))
            assert y[i] == pytest.approx(expected, abs=1e-12), (k, t[i])

    @pytest.mark.parametrize("saturate", [False, True])
    def test_sampled_loop_matches_the_exact_interpreter(self, qapp, tmp_path, saturate):
        dsim, fast = _run(_sampled_loop(saturate), tmp_path, 6.0, 0.01)
        assert _backend(dsim) == "scipy+hybrid"
        zc = dsim.engine.last_solver_diagnostics["zero_crossing"]
        assert zc["n_sample_hits"] == 61  # t = 0, 0.1, ..., 6.0
        if saturate:
            assert zc["n_events"] > 0
        _, slow = _run(_sampled_loop(saturate), tmp_path, 6.0, 0.01, fast=False)
        for label in ("y", "u"):
            a, b = np.ravel(fast["signals"][label]), np.ravel(slow["signals"][label])
            assert a.shape == b.shape
            assert np.max(np.abs(a - b)) < 1e-7, label

    def test_two_rates_share_one_schedule(self, qapp, tmp_path):
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=1.2, sim_dt=0.05)
        s = b.add_block(
            "Sine", 0, 0, name="s", params={"amplitude": 1.0, "omega": 3.0, "init_angle": 0.0}
        )
        z1 = b.add_block("ZeroOrderHold", 0, 0, name="z1", params={"sampling_time": 0.1})
        z2 = b.add_block("ZeroOrderHold", 0, 0, name="z2", params={"sampling_time": 0.15})
        g = b.add_block(
            "TranFn", 0, 0, name="g", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
        )
        sc = b.add_block(
            "Scope", 0, 0, name="sc", params={"labels": "a, b, c"}, in_ports=3, out_ports=0
        )
        b.connect(s, 0, z1, 0)
        b.connect(s, 0, z2, 0)
        b.connect(z2, 0, g, 0)
        b.connect(z1, 0, sc, 0)
        b.connect(z2, 0, sc, 1)
        b.connect(g, 0, sc, 2)
        dsim, fast = _run(b, tmp_path, 1.2, 0.05)
        assert _backend(dsim) == "scipy+hybrid"
        # 0.1 k and 0.15 k in (0, 1.2], merged: 12 + 8 - 4 shared, plus t0.
        assert dsim.engine.last_solver_diagnostics["zero_crossing"]["n_sample_hits"] == 17
        _, slow = _run(b, tmp_path, 1.2, 0.05, fast=False)
        for label in ("a", "b", "c"):
            a, c = np.ravel(fast["signals"][label]), np.ravel(slow["signals"][label])
            assert np.max(np.abs(a - c)) < 1e-7, label

    def test_a_cached_compile_starts_each_run_afresh(self, qapp, tmp_path):
        """The holders survive in the compile cache; their state must not."""
        dsim, first, second = _run(_sampled_loop(), tmp_path, 3.0, 0.01, twice=True)
        assert _backend(dsim) == "scipy+hybrid"
        assert dsim.engine.last_solver_diagnostics.get("compile_cache_hit") is True
        for label in ("y", "u"):
            assert np.array_equal(
                np.ravel(first["signals"][label]), np.ravel(second["signals"][label])
            )


@pytest.mark.regression
class TestHybridFallbacks:
    def test_fixed_step_solver_keeps_the_interpreter(self, qapp, tmp_path):
        dsim, _ = _run(_sampled_loop(), tmp_path, 1.0, 0.01, solver="RK4")
        assert _backend(dsim) is None
        assert "fixed-step 'RK4'" in dsim.engine.get_compile_fallback_reason()

    def test_unresolved_rate_keeps_the_interpreter(self, qapp, tmp_path):
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=0.5, sim_dt=0.1)
        r = b.add_block("Ramp", 0, 0, name="r", params={"slope": 1.0, "delay": 0.0})
        d = b.add_block(
            "DiscreteTranFn",
            0,
            0,
            name="d",
            params={"numerator": [1.0], "denominator": [1.0, -0.5], "sampling_time": 0.0},
        )
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(r, 0, d, 0)
        b.connect(d, 0, sc, 0)
        dsim, _ = _run(b, tmp_path, 0.5, 0.1)
        assert _backend(dsim) is None
        assert "unresolved" in dsim.engine.get_compile_fallback_reason()

    def test_linearizer_still_refuses_a_sampled_loop(self, qapp, tmp_path):
        from lib.analysis.linearizer import Linearizer
        from lib.cli import load_diagram

        path = tmp_path / "loop.diablos"
        _sampled_loop().save(str(path))
        dsim, _ = load_diagram(str(path))
        with pytest.raises(ValueError, match="discrete sample time"):
            Linearizer(dsim)._compile_diagram()
