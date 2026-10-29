"""Run history keeps pinned runs across a restart, and saves atomically.

``record_run`` caps only the *unpinned* runs, but ``load_history`` kept the
last ``limit`` runs of any kind, so with limit 2 a pinned baseline plus two
newer runs lost the baseline on the next start.
"""

import json

import numpy as np
import pytest

from lib.services.run_history_service import RunHistoryService


def _service(path, limit=2):
    svc = RunHistoryService(limit=limit, persist_path=path)
    svc.persist_enabled = True
    return svc


def _record(svc, name):
    svc.record_run(np.array([0.0, 1.0]), [{"name": "y", "y": np.array([1.0, 2.0])}], 0.1, 1.0, name)


@pytest.mark.unit
class TestPinnedRunsSurviveRestart:
    def test_pinned_baseline_reloads(self, tmp_path):
        path = tmp_path / "history.json"
        svc = _service(path)
        _record(svc, "baseline")
        svc.history[0]["pinned"] = True
        _record(svc, "run 2")
        _record(svc, "run 3")
        assert [r["name"] for r in svc.history] == ["baseline", "run 2", "run 3"]

        again = _service(path)
        again.load_history()
        assert [r["name"] for r in again.history] == ["baseline", "run 2", "run 3"]
        assert again.history[0]["pinned"] is True

    def test_load_caps_unpinned_runs_only(self, tmp_path):
        path = tmp_path / "history.json"
        runs = [
            {"name": f"r{k}", "timeline": [0.0], "traces": [], "pinned": k == 1} for k in range(5)
        ]
        path.write_text(json.dumps({"persist": True, "runs": runs}))
        svc = _service(path)
        svc.load_history()
        assert [r["name"] for r in svc.history] == ["r1", "r3", "r4"]

    def test_failed_save_keeps_the_previous_file(self, tmp_path, monkeypatch):
        path = tmp_path / "history.json"
        svc = _service(path)
        _record(svc, "good")
        before = path.read_text()

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(json, "dump", boom)
        _record(svc, "lost")
        assert path.read_text() == before
        assert [p.name for p in tmp_path.iterdir()] == ["history.json"]
