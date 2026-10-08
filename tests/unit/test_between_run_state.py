"""Regression tests: no state may leak from one run into the next.

Covers the memory-block ``input_queue`` (cleared at run initialisation), the
Hysteresis latch, and the headless re-sim path the analysis runners share.
"""

import os

import numpy as np
import pytest

from lib.analysis.resim import harvest_scope_signals

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")


def _load_example(name):
    from lib.cli import load_diagram

    dsim, _params = load_diagram(os.path.join(_EXAMPLES, name))
    dsim.use_fast_solver = False  # interpreter path
    return dsim


def _run_heads(dsim, n_runs, sim_time, sim_dt, n=3):
    heads = []
    for _ in range(n_runs):
        ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
        assert ok, err
        sigs = harvest_scope_signals(dsim)["signals"]
        heads.append({k: np.array(v[:n]) for k, v in sigs.items()})
    return heads


@pytest.mark.unit
class TestInputQueueDoesNotLeak:
    def test_discrete_pi_zoh_identical_across_runs(self, qapp):
        dsim = _load_example("discrete_pi_zoh.diablos")
        heads = _run_heads(dsim, 3, 1.0, 0.01)
        for run in heads[1:]:
            for name, first in heads[0].items():
                assert np.allclose(run[name], first), (name, first, run[name])


@pytest.mark.unit
class TestHysteresisLatch:
    def test_output_only_probe_after_run_returns_baseline(self):
        from blocks.hysteresis import HysteresisBlock

        blk = HysteresisBlock()
        params = {"upper": 0.5, "lower": -0.5, "high": 1.0, "low": 0.0, "_init_start_": True}
        blk.execute(0.0, {0: np.array([1.0])}, params)
        assert params["_state"] == 1.0
        params["_init_start_"] = True  # what reset_memblocks does
        out = blk.execute(0.0, {}, params)
        assert float(out[0][0]) == 0.0

    def test_thermostat_example_repeatable(self, qapp):
        dsim = _load_example("relay_thermostat_events.diablos")
        heads = _run_heads(dsim, 3, 3.0, 0.01, n=4)
        for run in heads[1:]:
            for name, first in heads[0].items():
                assert np.allclose(run[name], first), (name, first, run[name])


@pytest.mark.unit
class TestHeldOutputResetOnNewRun:
    """Output-only probes before the first input must not return the last run's output."""

    def test_pid(self):
        from blocks.pid import PIDBlock

        blk = PIDBlock()
        params = {"Kp": 2.0, "Ki": 0.0, "Kd": 0.0, "N": 20.0, "dtime": 0.01, "_init_start_": True}
        blk.execute(0.0, {0: np.array([1.0])}, params)
        assert params["_last_output_"] != 0.0
        params["_init_start_"] = True  # what reset_memblocks does
        out = blk.execute(0.0, {}, params)
        assert float(out[0][0]) == 0.0

    @pytest.mark.parametrize("mod,cls", [("adam", "AdamBlock"), ("momentum", "MomentumBlock")])
    def test_optimizer_primitives(self, mod, cls):
        import importlib

        blk = getattr(importlib.import_module(f"blocks.optimization_primitives.{mod}"), cls)()
        params = {"_init_start_": True}
        blk.execute(0.0, {0: np.array([1.0])}, params)
        assert float(np.ravel(params["_last_update_"])[0]) != 0.0
        params["_init_start_"] = True
        out = blk.execute(0.0, {}, params)
        assert float(np.ravel(out[0])[0]) == 0.0
