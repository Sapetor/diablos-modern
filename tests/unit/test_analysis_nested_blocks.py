"""Monte-Carlo seeding and sweep targeting must reach blocks inside subsystems.

Both runners used to look only at the top-level ``dsim.blocks_list``, so a
stochastic block (or a sweep target) inside a Subsystem was silently ignored.
Nested blocks are addressed by their flattened, ``/``-qualified name.
"""

import numpy as np
import pytest

from lib.analysis.monte_carlo import MonteCarloRunner
from lib.analysis.parameter_sweep import ParameterSweepRunner
from lib.diagram_builder import DiagramBuilder

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
    out = {}
    inst = _BLOCK_INSTANCES.get(block_type)
    if inst is not None:
        for k, v in inst.params.items():
            out[k] = v["default"] if isinstance(v, dict) and "default" in v else v
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


def _wrap(dsim, block_fn):
    """Move the (single) ``block_fn`` block into a subsystem; return (sub, inner)."""
    blk = next(b for b in dsim.blocks_list if b.block_fn == block_fn)
    sub = dsim.create_subsystem_from_selection([blk])
    assert sub is not None
    inner = next(b for b in sub.sub_blocks if b.block_fn == block_fn)
    return sub, inner


def _only_signal(result):
    assert result["signals"], "no signals produced"
    return result["signals"][sorted(result["signals"])[0]]


@pytest.mark.unit
class TestMonteCarloInsideSubsystem:
    def _nested_noise(self, tmp_path):
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 250, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "noise_sub.diablos")
        _sub, inner = _wrap(dsim, "Noise")
        return dsim, inner

    def test_nested_stochastic_block_gets_per_run_seed(self, qapp, tmp_path):
        dsim, inner = self._nested_noise(tmp_path)
        original_seed = inner.params["seed"]

        res = MonteCarloRunner(dsim).run(4, master_seed=7, sim_time=0.5, sim_dt=0.05)
        assert res["n_ok"] == 4
        sig = _only_signal(res)
        assert not np.allclose(sig["runs"][0], sig["runs"][1])  # runs differ
        res2 = MonteCarloRunner(dsim).run(4, master_seed=7, sim_time=0.5, sim_dt=0.05)
        assert np.allclose(_only_signal(res2)["runs"], sig["runs"])  # reproducible
        # The user's nested block is restored.
        assert inner.params["seed"] == original_seed


@pytest.mark.unit
class TestSweepInsideSubsystem:
    def _nested_gain(self, tmp_path):
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=2.0))
        s = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, g, 0)
        b.connect(g, 0, s, 0)
        dsim = _load(b, tmp_path, "sweep_sub.diablos")
        sub, inner = _wrap(dsim, "Gain")
        return dsim, sub, inner

    def test_axis_targets_nested_block_by_qualified_name(self, qapp, tmp_path):
        dsim, sub, inner = self._nested_gain(tmp_path)
        values = [1.0, 2.0, 3.0]

        res = ParameterSweepRunner(dsim).run(
            axes=[{"block": f"{sub.name}/{inner.name}", "param": "gain", "values": values}],
            sim_time=0.3,
            sim_dt=0.05,
        )

        assert res["n_ok"] == 3
        assert np.allclose(_only_signal(res)["metrics"]["final"], values)
        # The user's nested block is restored, params and exec_params alike.
        assert inner.params["gain"] == 2.0
        if getattr(inner, "exec_params", None):
            assert inner.exec_params.get("gain") == 2.0


@pytest.mark.unit
class TestRootScopeIndependence:
    """Experiments address the whole diagram, whatever scope the user is in."""

    def test_monte_carlo_same_inside_and_outside_subsystem(self, qapp, tmp_path):
        b = DiagramBuilder()
        n = b.add_block("Noise", 50, 100, params=_params("Noise"))
        s = b.add_block("Scope", 250, 100, params=_params("Scope"))
        b.connect(n, 0, s, 0)
        dsim = _load(b, tmp_path, "mc_scope.diablos")
        sub, _inner = _wrap(dsim, "Noise")

        kw = dict(master_seed=5, sim_time=0.5, sim_dt=0.05)
        top = MonteCarloRunner(dsim.clone_for_analysis()).run(3, **kw)
        dsim.enter_subsystem(sub)
        try:
            assert dsim.root_blocks_list is not dsim.blocks_list
            inside = MonteCarloRunner(dsim).run(3, **kw)
            cloned_inside = MonteCarloRunner(dsim.clone_for_analysis()).run(3, **kw)
        finally:
            dsim.exit_subsystem()

        assert top["n_ok"] == inside["n_ok"] == cloned_inside["n_ok"] == 3
        assert sorted(top["signals"]) == sorted(inside["signals"])
        for nm in top["signals"]:
            assert np.allclose(top["signals"][nm]["runs"], inside["signals"][nm]["runs"])
            assert np.allclose(top["signals"][nm]["runs"], cloned_inside["signals"][nm]["runs"])

    def test_sweep_same_inside_and_outside_subsystem(self, qapp, tmp_path):
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=2.0))
        s = b.add_block("Scope", 350, 100, params=_params("Scope"))
        b.connect(c, 0, g, 0)
        b.connect(g, 0, s, 0)
        dsim = _load(b, tmp_path, "sweep_scope.diablos")
        sub, inner = _wrap(dsim, "Gain")
        axes = [{"block": f"{sub.name}/{inner.name}", "param": "gain", "values": [1.0, 3.0]}]
        kw = dict(axes=axes, sim_time=0.3, sim_dt=0.05)

        top = ParameterSweepRunner(dsim).run(**kw)
        dsim.enter_subsystem(sub)
        try:
            inside = ParameterSweepRunner(dsim).run(**kw)
            cloned = ParameterSweepRunner(dsim.clone_for_analysis()).run(**kw)
        finally:
            dsim.exit_subsystem()

        for res in (top, inside, cloned):
            assert res["n_ok"] == 2
            assert np.allclose(_only_signal(res)["metrics"]["final"], [1.0, 3.0])

    def test_clone_inside_subsystem_copies_whole_diagram(self, qapp, tmp_path):
        b = DiagramBuilder()
        c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
        g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=2.0))
        b.connect(c, 0, g, 0)
        dsim = _load(b, tmp_path, "clone_scope.diablos")
        sub, _inner = _wrap(dsim, "Gain")
        names = [x.name for x in dsim.blocks_list]
        dsim.enter_subsystem(sub)
        try:
            live_before = dsim.blocks_list
            clone = dsim.clone_for_analysis()
            assert dsim.blocks_list is live_before  # live scope untouched
        finally:
            dsim.exit_subsystem()
        assert [x.name for x in clone.blocks_list] == names
