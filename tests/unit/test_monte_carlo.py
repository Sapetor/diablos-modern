"""Tests for the Monte-Carlo ensemble runner (lib/analysis/monte_carlo.py).

Builds small stochastic diagrams via DiagramBuilder, runs ensembles, and checks
per-run seed injection (runs differ but are reproducible from a master seed),
end-to-end packet loss, parameter sampling, and a random-delay integration.
"""

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from lib.analysis.monte_carlo import MonteCarloRunner, derive_seed


_BLOCK_INSTANCES = None


def _params(block_type, **overrides):
    global _BLOCK_INSTANCES
    if _BLOCK_INSTANCES is None:
        from lib.block_loader import load_blocks
        _BLOCK_INSTANCES = {}
        for cls in load_blocks():
            try:
                inst = cls()
                _BLOCK_INSTANCES[inst.block_name] = inst
            except Exception:
                pass
    inst = _BLOCK_INSTANCES.get(block_type)
    out = {}
    if inst is not None:
        for k, v in inst.params.items():
            out[k] = v['default'] if isinstance(v, dict) and 'default' in v else v
    out.update(overrides)
    return out


def _load(builder, tmp_path, name):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager
    path = tmp_path / name
    builder.save(str(path))
    WorkspaceManager._instance = None
    dsim = DSim()
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _name_of(dsim, block_fn):
    for b in dsim.blocks_list:
        if b.block_fn == block_fn:
            return b.name
    raise AssertionError(f"No block with block_fn={block_fn!r}")


def _only_signal(result):
    assert result["signals"], "ensemble produced no signals"
    return result["signals"][sorted(result["signals"])[0]]


def test_derive_seed_reproducible_and_distinct():
    assert derive_seed(7, 0, "a") == derive_seed(7, 0, "a")       # reproducible
    assert derive_seed(7, 0, "a") != derive_seed(7, 1, "a")       # per run
    assert derive_seed(7, 0, "a") != derive_seed(8, 0, "a")       # per master
    assert derive_seed(7, 0, "a") != 0


@pytest.mark.unit
class TestMonteCarlo:
    def test_noise_runs_differ_but_reproducible(self, qapp, tmp_path):
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 250, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "noise.diablos")

        res = MonteCarloRunner(dsim).run(6, master_seed=7, sim_time=0.5, sim_dt=0.05)
        assert res["n_ok"] == 6
        sig = _only_signal(res)
        for key in ("mean", "std", "p5", "p95", "runs"):
            assert key in sig
        # Per-run seed injection: runs are NOT identical.
        assert not np.allclose(sig["runs"][0], sig["runs"][1])

        # Same master seed -> identical ensemble (reproducible).
        res_same = MonteCarloRunner(dsim).run(6, master_seed=7, sim_time=0.5, sim_dt=0.05)
        assert np.allclose(_only_signal(res_same)["runs"], sig["runs"])
        # Different master seed -> different ensemble.
        res_diff = MonteCarloRunner(dsim).run(6, master_seed=99, sim_time=0.5, sim_dt=0.05)
        assert not np.allclose(_only_signal(res_diff)["mean"], sig["mean"])

    def test_packet_loss_mean_matches_keep_probability(self, qapp, tmp_path):
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        pl = b.add_block("PacketLoss", 200, 100,
                         params=_params("PacketLoss", loss_prob=0.5, drop_mode="zero"))
        s = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, pl, 0)
        b.connect(pl, 0, s, 0)
        dsim = _load(b, tmp_path, "ploss.diablos")

        res = MonteCarloRunner(dsim).run(40, master_seed=123, sim_time=1.0, sim_dt=0.05)
        assert res["n_ok"] == 40
        # Input is 1; kept w.p. 0.5 -> 1, dropped -> 0. Ensemble mean ~ keep prob 0.5.
        m = _only_signal(res)["mean"].mean()
        assert 0.3 < m < 0.7, f"ensemble mean {m} not near keep-probability 0.5"

    def test_parameter_sampler_uniform(self, qapp, tmp_path):
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=1.0))
        s = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, g, 0)
        b.connect(g, 0, s, 0)
        dsim = _load(b, tmp_path, "gain.diablos")

        gain_name = _name_of(dsim, "Gain")
        samplers = {(gain_name, "gain"): (0.0, 2.0)}
        res = MonteCarloRunner(dsim).run(40, master_seed=5, sim_time=0.3, sim_dt=0.05,
                                         samplers=samplers)
        assert res["n_ok"] == 40
        sig = _only_signal(res)
        # output = gain * 1 = gain ~ U[0,2]; ensemble mean ~ 1.0, and runs vary.
        assert 0.7 < sig["mean"].mean() < 1.3
        assert not np.allclose(sig["runs"][0], sig["runs"][1])
        # Reproducible from the master seed.
        res2 = MonteCarloRunner(dsim).run(40, master_seed=5, sim_time=0.3, sim_dt=0.05,
                                          samplers=samplers)
        assert np.allclose(_only_signal(res2)["runs"], sig["runs"])

    def test_random_delay_ensemble_runs(self, qapp, tmp_path):
        """Step -> VariableTransportDelay, with tau driven by a Noise source:
        a random-latency ensemble that runs end-to-end and is reproducible."""
        b = DiagramBuilder()
        step = b.add_block("Step", 50, 80, params=_params("Step", value=1.0))
        tau = b.add_block("Noise", 50, 200, params=_params("Noise", mu=0.1, sigma=0.03))
        vd = b.add_block("VariableTransportDelay", 220, 120,
                         params=_params("VariableTransportDelay", max_delay=0.5),
                         in_ports=2)
        s = b.add_block("Scope", 400, 120, params=_params("Scope"))
        b.connect(step, 0, vd, 0)
        b.connect(tau, 0, vd, 1)
        b.connect(vd, 0, s, 0)
        dsim = _load(b, tmp_path, "rdelay.diablos")

        res = MonteCarloRunner(dsim).run(8, master_seed=11, sim_time=1.0, sim_dt=0.05)
        assert res["n_ok"] == 8
        assert res["signals"], "random-delay ensemble produced no signals"
        sig = _only_signal(res)
        res2 = MonteCarloRunner(dsim).run(8, master_seed=11, sim_time=1.0, sim_dt=0.05)
        assert np.allclose(_only_signal(res2)["runs"], sig["runs"])
