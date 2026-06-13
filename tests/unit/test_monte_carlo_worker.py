"""Tests for the Monte-Carlo background worker
(modern_ui/widgets/monte_carlo_worker.py).

Builds a tiny stochastic diagram (reusing the _params/_load helper pattern from
tests/unit/test_monte_carlo.py), then drives the worker by calling ``run()``
directly. Because the test never calls ``start()``, ``run()`` executes in the
test thread and its signal handlers fire synchronously during emission -- so we
can assert exactly what progress/finished carry without an event loop.
"""

import pytest

from lib.diagram_builder import DiagramBuilder
from modern_ui.widgets.monte_carlo_worker import MonteCarloWorker


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


def _noise_scope(tmp_path, name):
    b = DiagramBuilder()
    n = b.add_block("Noise", 50, 100, params=_params("Noise"))
    s = b.add_block("Scope", 250, 100, params=_params("Scope"))
    b.connect(n, 0, s, 0)
    return _load(b, tmp_path, name)


@pytest.mark.unit
class TestMonteCarloWorker:
    def test_run_emits_progress_and_finished(self, qapp, tmp_path):
        dsim = _noise_scope(tmp_path, "worker.diablos")
        worker = MonteCarloWorker(
            dsim, {"n_runs": 4, "master_seed": 7, "sim_time": 0.3, "sim_dt": 0.05})

        progress, results = [], []
        worker.progress.connect(lambda d, t: progress.append((d, t)))
        worker.finished.connect(results.append)

        worker.run()  # synchronous (no separate thread)

        assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
        assert len(results) == 1
        assert results[0]["n_ok"] == 4

    def test_cancel_before_run_yields_empty_ensemble(self, qapp, tmp_path):
        dsim = _noise_scope(tmp_path, "worker_cancel.diablos")
        worker = MonteCarloWorker(
            dsim, {"n_runs": 5, "master_seed": 7, "sim_time": 0.3, "sim_dt": 0.05})
        worker.cancel()

        results = []
        worker.finished.connect(results.append)
        worker.run()

        assert len(results) == 1
        assert results[0]["n_ok"] == 0
