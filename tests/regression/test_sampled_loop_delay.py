"""
Regression: a sampled controller inside a closed loop acted one period late.

In examples/discrete_pi_zoh.diablos (ZOH Ts=0.1 -> discrete PI -> continuous
plant, plant output fed back to the error Sum) the interpreter applied u[k] on
[(k+1)T, (k+2)T) instead of [kT, (k+1)T): the plant output matched the exact
sampled-data solution *with one extra controller period of delay* to 1e-14.

Cause: the memory pass hands every memory block's stale held output to its
consumers and counts it as delivered. In closed loop the ZOH's own input (the
error Sum) sits at a higher hierarchy level than the ZOH, so the
DiscreteTranFn downstream of it looked ready, ran first and latched the
stale sample. The ZOH's later count=False refresh came too late, since the
DTF does not sample again until the next period. Open loop the ZOH ran
first, which is why tests/regression/test_feedthrough_memory.py never saw it.

Found by scripts/benchmarks/pathsim/s1_sampled_data.py (docs/BENCHMARK_PATHSIM.md).
"""

import json

import numpy as np
import pytest
from scipy import signal
from scipy.linalg import expm

from lib.app_paths import resource_path

T, T_STEP, T_END = 0.1, 0.93, 3.0
NUM_C, DEN_C = [2.0], [1.0, 3.0, 2.0]
NUM_D, DEN_D = [1.2, -1.0], [1.0, -1.0]


def _exact(query_t):
    """Exact response: u[k] from e[kT] held on [kT, (k+1)T)."""
    A, B, C, _ = signal.tf2ss(NUM_C, DEN_C)
    n = A.shape[0]

    def phi_gamma(h):
        M = np.zeros((n + 1, n + 1))
        M[:n, :n], M[:n, n:] = A * h, B * h
        E = expm(M)
        return E[:n, :n], E[:n, n:]

    Phi, Gam = phi_gamma(T)
    x, u_prev, e_prev = np.zeros((n, 1)), 0.0, 0.0
    xs, us = [], []
    for k in range(int(round(T_END / T)) + 1):
        e = (1.0 if k * T >= T_STEP else 0.0) - (C @ x).item()
        u = NUM_D[0] * e + NUM_D[1] * e_prev - DEN_D[1] * u_prev
        u_prev, e_prev = u, e
        xs.append(x.copy())
        us.append(u)
        x = Phi @ x + Gam * u
    out = []
    for t in query_t:
        k = min(int(np.floor(t / T + 1e-9)), len(xs) - 1)
        P, G = phi_gamma(t - k * T)
        out.append((C @ (P @ xs[k] + G * us[k])).item())
    return np.array(out)


def _run(path, sim_dt):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import load_diagram

    dsim, _ = load_diagram(str(path))
    dsim.use_fast_solver = False
    ok, err = dsim.run_tuning_simulation(T_END, sim_dt)
    assert ok, err
    data = harvest_scope_signals(dsim)
    return np.asarray(data["timeline"], float), data["signals"]


@pytest.fixture
def pi_zoh_diagram(tmp_path):
    src = resource_path("examples/discrete_pi_zoh.diablos")
    data = json.loads(open(src).read())
    for b in data["blocks_data"]:
        if b.get("block_fn") == "Step":
            b["params"]["delay"] = T_STEP  # off the sampling grid
    path = tmp_path / "pi_zoh.diablos"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.regression
class TestClosedLoopSampledController:
    def test_control_changes_at_first_sample_after_step(self, qapp, pi_zoh_diagram):
        t, sig = _run(pi_zoh_diagram, 0.01)
        u = sig["u[k]"]
        first = t[np.argmax(np.abs(u) > 0)]
        assert first == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("sim_dt", [0.01, 0.005])
    def test_plant_output_matches_exact_sampled_data_solution(self, qapp, pi_zoh_diagram, sim_dt):
        t, sig = _run(pi_zoh_diagram, sim_dt)
        y = sig["plantoutput"]
        assert np.max(np.abs(y - _exact(t))) < 1e-9


@pytest.mark.regression
class TestAlgebraicLoopThroughZohStillRuns:
    def test_zoh_gain_loop_terminates(self, qapp, tmp_path):
        """ZOH -> Gain -> Sum -> ZOH is algebraic at sample instants. Holding
        consumers back until the ZOH has sampled would deadlock here, so the
        hold must be released and the old (stale-value) loop break kept."""
        from lib.diagram_builder import DiagramBuilder

        b = DiagramBuilder(sim_time=0.5, sim_dt=0.1)
        c = b.add_block("Constant", 0, 0, name="c", params={"value": 1.0})
        s = b.add_block("Sum", 0, 0, name="s", params={"sign": "+-"}, in_ports=2)
        z = b.add_block(
            "ZeroOrderHold", 0, 0, name="z", params={"sampling_time": 0.2}, in_ports=1, out_ports=1
        )
        g = b.add_block("Gain", 0, 0, name="g", params={"gain": 0.5})
        sc = b.add_block("Scope", 0, 0, name="sc", params={"labels": "y"}, in_ports=1, out_ports=0)
        b.connect(c, 0, s, 0)
        b.connect(s, 0, z, 0)
        b.connect(z, 0, g, 0)
        b.connect(g, 0, s, 1)
        b.connect(z, 0, sc, 0)
        path = tmp_path / "zoh_loop.diablos"
        b.save(str(path))
        _, sig = _run(path, 0.1)
        assert np.all(np.isfinite(sig["y"]))
