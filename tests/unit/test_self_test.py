"""``diablos self-test`` -- the installation check the release workflow runs
through every packaged build. Here it runs from source."""

import json

import pytest

from lib import self_test


@pytest.mark.unit
class TestSelfTest:
    def test_all_checks_pass_from_source(self, qapp, tmp_path):
        report = tmp_path / "report.json"
        assert self_test.main(["--report", str(report)]) == 0
        data = json.loads(report.read_text())
        assert data["passed"] is True
        assert [c["name"] for c in data["checks"]] == [
            "resources",
            "simulate-compiled",
            "simulate-interpreter",
            "monte-carlo-workers",
            "gui",
        ]
        assert all(c["ok"] for c in data["checks"])

    def test_a_failing_check_fails_the_run(self, tmp_path, monkeypatch):
        def broken():
            raise AssertionError("missing bundled files: examples/x.diablos")

        monkeypatch.setattr(self_test, "_check_resources", broken)
        monkeypatch.setattr(self_test, "_check_workers", lambda: "skipped here")
        report = tmp_path / "report.json"
        assert self_test.main(["--no-gui", "--report", str(report)]) == 1
        data = json.loads(report.read_text())
        assert data["passed"] is False
        failed = [c for c in data["checks"] if not c["ok"]]
        assert [c["name"] for c in failed] == ["resources"]
        assert "examples/x.diablos" in failed[0]["detail"]

    def test_entry_point_dispatches(self, monkeypatch):
        import diablos_modern

        seen = {}
        monkeypatch.setattr(self_test, "main", lambda argv: seen.setdefault("argv", argv) and 0)
        diablos_modern.entry(["DiaBloS", "self-test", "--no-gui"])
        assert seen["argv"] == ["--no-gui"]
