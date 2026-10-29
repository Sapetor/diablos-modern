"""Streaming aggregation + process-pool parallelism for Monte-Carlo / sweeps.

Covers ``lib/analysis/streaming.py`` and ``lib/analysis/parallel.py``:

  * the streaming ensemble statistics equal the old batch (full trace matrix)
    statistics, for every ``keep_traces`` mode;
  * ``workers=1`` and ``workers=2`` give bit-identical results;
  * cancelling a ``workers=2`` run stops early;
  * a 2-D sweep never ships or stores a trace (no ``(n_points, L)`` matrix);
  * workspace variables reach the worker processes.

The pool tests spawn real worker processes (~1 s start-up each) and are kept
to a handful of tiny diagrams.
"""

import numpy as np
import pytest

from lib.analysis import parallel as parallel_mod
from lib.analysis.monte_carlo import MonteCarloRunner
from lib.analysis.parameter_sweep import ParameterSweepRunner
from lib.analysis.resim import OUTCOME_METRICS
from lib.analysis.streaming import (
    EnsembleAccumulator,
    SweepAccumulator,
    reduce_run,
    trace_sample_indices,
)
from lib.diagram_builder import DiagramBuilder
from tests.unit.test_monte_carlo import _load, _name_of, _params


def _old_aggregate(runs):
    """The pre-streaming batch reduction (reference implementation)."""
    ok = [r for r in runs if r is not None and r["signals"]]
    result = {"n_runs": len(runs), "n_ok": len(ok), "timeline": None, "signals": {}}
    if not ok:
        return result
    names = set(ok[0]["signals"])
    for r in ok[1:]:
        names &= set(r["signals"])
    if not names:
        return result
    lengths = [len(r["timeline"]) for r in ok]
    for r in ok:
        for nm in names:
            lengths.append(len(r["signals"][nm]))
    L = min(lengths)
    result["timeline"] = np.asarray(ok[0]["timeline"][:L], dtype=float)
    for nm in sorted(names):
        M = np.vstack([r["signals"][nm][:L] for r in ok])
        result["signals"][nm] = {
            "runs": M,
            "mean": M.mean(axis=0),
            "std": M.std(axis=0),
            "p5": np.percentile(M, 5, axis=0),
            "p50": np.percentile(M, 50, axis=0),
            "p95": np.percentile(M, 95, axis=0),
            "min": M.min(axis=0),
            "max": M.max(axis=0),
            "metrics": {k: fn(M) for k, fn in OUTCOME_METRICS.items()},
        }
    return result


def _synthetic_runs(n=25, L=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, L)
    runs = []
    for i in range(n):
        if i == 3:
            runs.append(None)  # a failed run
            continue
        sig = {
            "a": 5.0 + rng.normal(size=L).cumsum(),
            "b": 1e6 + rng.normal(scale=1e-3, size=L),  # large offset: Welford matters
        }
        runs.append({"timeline": t, "signals": sig})
    return runs


def _assert_stats_match(new, ref, keys=("mean", "std", "min", "max")):
    assert new["n_runs"] == ref["n_runs"] and new["n_ok"] == ref["n_ok"]
    np.testing.assert_array_equal(new["timeline"], ref["timeline"])
    assert set(new["signals"]) == set(ref["signals"])
    for nm, rs in ref["signals"].items():
        ns = new["signals"][nm]
        for k in keys:
            np.testing.assert_allclose(ns[k], rs[k], rtol=1e-12, atol=1e-12, err_msg=k)
        for m in OUTCOME_METRICS:
            np.testing.assert_allclose(ns["metrics"][m], rs["metrics"][m], rtol=1e-12, err_msg=m)


def _fold(runs, keep_traces):
    acc = EnsembleAccumulator(len(runs), keep_traces=keep_traces)
    for i, h in enumerate(runs):
        acc.add(i, reduce_run(h, keep_signals=True))
    return acc.result()


@pytest.mark.unit
class TestStreamingAccumulators:
    def test_streaming_equals_batch_all_traces(self):
        runs = _synthetic_runs()
        new, ref = _fold(runs, True), _old_aggregate(runs)
        _assert_stats_match(new, ref, keys=("mean", "std", "min", "max", "p5", "p50", "p95"))
        for nm in ref["signals"]:
            np.testing.assert_array_equal(new["signals"][nm]["runs"], ref["signals"][nm]["runs"])
        assert new["traces"] == "all"

    def test_streaming_without_traces_keeps_exact_moments(self):
        runs = _synthetic_runs()
        new, ref = _fold(runs, False), _old_aggregate(runs)
        _assert_stats_match(new, ref)
        assert new["traces"] == "none"
        for sig in new["signals"].values():
            assert sig["runs"] is None
            assert sig["p5"] is None and sig["p50"] is None and sig["p95"] is None

    def test_trace_sample(self):
        runs = _synthetic_runs(n=25)
        new = _fold(runs, 5)
        assert new["traces"] == "sample"
        keep = trace_sample_indices(25, 5)
        assert len(keep) <= 5
        # Sampled indices that succeeded (run 3 failed).
        expected = sorted(i for i in keep if runs[i] is not None)
        M = new["signals"]["a"]["runs"]
        assert M.shape == (len(expected), 60)
        for row, i in zip(M, expected):
            np.testing.assert_array_equal(row, runs[i]["signals"]["a"])
        _assert_stats_match(new, _old_aggregate(runs))

    def test_ragged_lengths_and_missing_signal_match_batch(self):
        runs = _synthetic_runs(n=8, L=40)
        # A later, shorter run (common length shrinks) and one lacking "b".
        runs[5]["signals"] = {k: v[:30] for k, v in runs[5]["signals"].items()}
        runs[6]["signals"] = {"a": runs[6]["signals"]["a"]}
        new, ref = _fold(runs, True), _old_aggregate(runs)
        assert set(new["signals"]) == {"a"}
        _assert_stats_match(new, ref, keys=("mean", "std", "min", "max", "p5", "p95"))

    def test_sweep_2d_keeps_no_traces(self):
        runs = _synthetic_runs(n=6, L=50)
        acc = SweepAccumulator("2d", (2, 3), keep_traces=True)  # ignored in 2-D
        assert acc.keep_traces is False
        for k, h in enumerate(runs):
            acc.add(k, reduce_run(h, keep_signals=acc.keep_traces))
        assert all(not d for d in acc.traces.values())
        result = acc.fill({"signals": {}})
        for sig in result["signals"].values():
            assert "traces" not in sig
            assert sig["metrics"]["final"].shape == (2, 3)
        # Cell (1, 0) is run 3, which failed -> NaN.
        assert np.isnan(result["signals"]["a"]["metrics"]["final"][1, 0])
        assert result["signals"]["a"]["metrics"]["final"][0, 0] == runs[0]["signals"]["a"][-1]


# ------------------------------------------------------------- diagrams
def _noise_tf_scope(tmp_path, name="noise_tf.diablos"):
    b = DiagramBuilder()
    n = b.add_block("Noise", 50, 100, params=_params("Noise"))
    tf = b.add_block("TranFn", 200, 100, params=_params("TranFn"))
    s = b.add_block("Scope", 350, 100, params=_params("Scope"))
    b.connect(n, 0, tf, 0)
    b.connect(tf, 0, s, 0)
    return _load(b, tmp_path, name)


def _gain_scope(tmp_path, const_value=1.0, name="gain.diablos"):
    b = DiagramBuilder()
    c = b.add_block("Constant", 50, 100, params=_params("Constant", value=const_value))
    g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=1.0))
    s = b.add_block("Scope", 350, 100, params=_params("Scope"))
    b.connect(c, 0, g, 0)
    b.connect(g, 0, s, 0)
    return _load(b, tmp_path, name)


def _assert_results_identical(a, b):
    assert a.keys() == b.keys()
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, dict):
            _assert_results_identical(va, vb)
        elif isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            np.testing.assert_array_equal(va, vb, err_msg=k)
        else:
            assert va == vb, k


@pytest.mark.unit
class TestRunnersStreaming:
    def test_mc_streaming_matches_batch_on_seeded_ensemble(self, qapp, tmp_path):
        dsim = _noise_tf_scope(tmp_path)
        kw = dict(master_seed=21, sim_time=0.5, sim_dt=0.01)
        full = MonteCarloRunner(dsim).run(12, **kw)
        lean = MonteCarloRunner(dsim).run(12, keep_traces=False, **kw)
        assert full["n_ok"] == lean["n_ok"] == 12
        for nm, sig in full["signals"].items():
            M = sig["runs"]
            np.testing.assert_allclose(sig["mean"], M.mean(axis=0), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(sig["std"], M.std(axis=0), rtol=1e-10, atol=1e-15)
            np.testing.assert_array_equal(sig["p95"], np.percentile(M, 95, axis=0))
            ls = lean["signals"][nm]
            for k in ("mean", "std", "min", "max"):
                np.testing.assert_array_equal(ls[k], sig[k])
            for m in OUTCOME_METRICS:
                np.testing.assert_array_equal(ls["metrics"][m], sig["metrics"][m])
                np.testing.assert_allclose(
                    sig["metrics"][m], OUTCOME_METRICS[m](M), rtol=1e-12, atol=1e-15
                )
            assert ls["runs"] is None and ls["p5"] is None

    def test_sweep_2d_never_requests_traces(self, qapp, tmp_path, monkeypatch):
        dsim = _gain_scope(tmp_path)
        gain = _name_of(dsim, "Gain")
        const = _name_of(dsim, "Constant")
        seen = []
        real = parallel_mod.reduce_run

        def spy(harvest, keep_signals):
            rec = real(harvest, keep_signals)
            seen.append((keep_signals, rec["signals"] if rec else None))
            return rec

        monkeypatch.setattr(parallel_mod, "reduce_run", spy)
        res = ParameterSweepRunner(dsim).run(
            axes=[
                {"block": gain, "param": "gain", "values": [1.0, 2.0, 3.0]},
                {"block": const, "param": "value", "values": [1.0, 10.0]},
            ],
            sim_time=0.5,
            sim_dt=0.01,
        )
        assert len(seen) == 6
        assert all(keep is False and sigs is None for keep, sigs in seen)
        assert res["n_ok"] == 6
        for sig in res["signals"].values():
            assert set(sig) == {"metrics"}
            np.testing.assert_allclose(
                sig["metrics"]["final"], np.outer([1.0, 2.0, 3.0], [1.0, 10.0])
            )

    def test_sweep_1d_keep_traces_false(self, qapp, tmp_path):
        dsim = _gain_scope(tmp_path)
        gain = _name_of(dsim, "Gain")
        axes = [{"block": gain, "param": "gain", "values": [1.0, 2.0]}]
        res = ParameterSweepRunner(dsim).run(
            axes=axes, sim_time=0.2, sim_dt=0.05, keep_traces=False
        )
        full = ParameterSweepRunner(dsim).run(axes=axes, sim_time=0.2, sim_dt=0.05)
        for nm, sig in res["signals"].items():
            assert "traces" not in sig
            assert full["signals"][nm]["traces"].shape[0] == 2
            for m in OUTCOME_METRICS:
                np.testing.assert_array_equal(sig["metrics"][m], full["signals"][nm]["metrics"][m])


@pytest.mark.unit
class TestProcessPool:
    def test_mc_workers_bit_identical(self, qapp, tmp_path):
        dsim = _noise_tf_scope(tmp_path)
        noise = next(b for b in dsim.blocks_list if b.block_fn == "Noise")
        mu0, seed0 = noise.params["mu"], noise.params.get("seed")
        # A callable sampler: drawn in the parent, so it need not be picklable.
        samplers = {(noise.name, "mu"): lambda rng: float(rng.normal(2.0, 0.5))}
        kw = dict(master_seed=5, sim_time=0.3, sim_dt=0.01, samplers=samplers)
        one = MonteCarloRunner(dsim).run(6, workers=1, **kw)
        two = MonteCarloRunner(dsim).run(6, workers=2, **kw)
        assert one["n_ok"] == 6
        assert not np.allclose(one["signals"][next(iter(one["signals"]))]["runs"][0], 0.0)
        _assert_results_identical(one, two)
        # Neither path leaves the parent's diagram changed.
        assert noise.params["mu"] == mu0 and noise.params.get("seed") == seed0

    def test_sweep_2d_workers_bit_identical(self, qapp, tmp_path):
        dsim = _gain_scope(tmp_path)
        axes = [
            {"block": _name_of(dsim, "Gain"), "param": "gain", "values": [0.5, 2.0]},
            {"block": _name_of(dsim, "Constant"), "param": "value", "values": [1.0, 3.0]},
        ]
        one = ParameterSweepRunner(dsim).run(axes=axes, sim_time=0.2, sim_dt=0.05)
        two = ParameterSweepRunner(dsim).run(axes=axes, sim_time=0.2, sim_dt=0.05, workers=2)
        _assert_results_identical(one, two)

    def test_cancel_with_workers_stops_early(self, qapp, tmp_path):
        dsim = _noise_tf_scope(tmp_path)
        n = 200
        progress = []

        def cancel():
            return len(progress) >= 2

        res = MonteCarloRunner(dsim).run(
            n,
            master_seed=1,
            sim_time=0.5,
            sim_dt=0.01,
            workers=2,
            progress_cb=lambda d, t: progress.append(d),
            cancel_cb=cancel,
        )
        assert 2 <= res["n_runs"] < n
        assert res["n_ok"] == res["n_runs"]
        assert progress[-1] < n

    def test_gui_worker_falls_back_in_process_on_broken_pool(self):
        from concurrent.futures.process import BrokenProcessPool

        from modern_ui.widgets.monte_carlo_worker import run_with_pool_fallback

        calls = []

        def run(workers):
            calls.append(workers)
            if workers > 1:
                raise BrokenProcessPool("no pool here")
            return "ok"

        assert run_with_pool_fallback(run, 3) == "ok"
        assert calls == [3, 1]

    def test_workspace_variables_reach_workers(self, qapp, tmp_path):
        """Workers see the *diagram's* variables (per-DSim workspace)."""
        dsim = _gain_scope(tmp_path, const_value="K", name="ws.diablos")
        dsim.workspace.variables = {"K": 4.0}
        axes = [{"block": _name_of(dsim, "Gain"), "param": "gain", "values": [1.0, 2.0]}]
        res = ParameterSweepRunner(dsim).run(axes=axes, sim_time=0.2, sim_dt=0.05, workers=2)
        (sig,) = res["signals"].values()
        np.testing.assert_allclose(sig["metrics"]["final"], [4.0, 8.0])
