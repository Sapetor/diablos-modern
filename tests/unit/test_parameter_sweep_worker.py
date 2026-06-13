"""Tests for the parameter-sweep background worker
(modern_ui/widgets/parameter_sweep_worker.py).

Drives the worker by calling ``run()`` directly (no ``start()``), so it executes
in the test thread and its signal handlers fire synchronously during emission.
"""

import pytest

from lib.diagram_builder import DiagramBuilder
from modern_ui.widgets.parameter_sweep_worker import ParameterSweepWorker


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


def _const_gain_scope(tmp_path, name):
    b = DiagramBuilder()
    c = b.add_block("Constant", 50, 100, params=_params("Constant", value=1.0))
    g = b.add_block("Gain", 200, 100, params=_params("Gain", gain=1.0))
    s = b.add_block("Scope", 350, 100, params=_params("Scope"))
    b.connect(c, 0, g, 0)
    b.connect(g, 0, s, 0)
    return _load(b, tmp_path, name)


@pytest.mark.unit
class TestParameterSweepWorker:
    def test_run_emits_progress_and_finished(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sw_worker.diablos")
        gain = _name_of(dsim, "Gain")
        sel = {"axes": [{"block": gain, "param": "gain", "values": [0.0, 1.0, 2.0, 3.0]}],
               "sim_time": 0.3, "sim_dt": 0.05}
        worker = ParameterSweepWorker(dsim, sel)

        progress, results = [], []
        worker.progress.connect(lambda d, t: progress.append((d, t)))
        worker.finished.connect(results.append)

        worker.run()  # synchronous (no separate thread)

        assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
        assert len(results) == 1
        assert results[0]["mode"] == "1d" and results[0]["n_ok"] == 4

    def test_cancel_before_run_yields_empty_grid(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sw_worker_cancel.diablos")
        gain = _name_of(dsim, "Gain")
        sel = {"axes": [{"block": gain, "param": "gain", "values": [0.0, 1.0, 2.0]}],
               "sim_time": 0.3, "sim_dt": 0.05}
        worker = ParameterSweepWorker(dsim, sel)
        worker.cancel()

        results = []
        worker.finished.connect(results.append)
        worker.run()

        assert len(results) == 1
        assert results[0]["n_ok"] == 0

    def test_bad_selection_emits_failed(self, qapp, tmp_path):
        dsim = _const_gain_scope(tmp_path, "sw_worker_bad.diablos")
        sel = {"axes": [{"block": "Nope", "param": "gain", "values": [0, 1]}]}
        worker = ParameterSweepWorker(dsim, sel)

        failed = []
        worker.failed.connect(failed.append)
        worker.run()

        assert len(failed) == 1 and failed[0]
