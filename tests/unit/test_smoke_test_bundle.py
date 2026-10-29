"""tools/smoke_test_bundle.sh rejects a CLI run whose CSV holds NaN.

The closed-form check was ``err > 1e-6``; with NaN output ``err`` is NaN, the
comparison is False, and the bundle passed. A stand-in executable answers
``self-test`` with a passing frozen report and ``run`` with a chosen CSV.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "tools", "smoke_test_bundle.sh")

FAKE_EXE = """#!{python}
import json, sys
import numpy as np

args = sys.argv[1:]
if args[0] == "self-test":
    report = args[args.index("--report") + 1]
    check = {{"name": "x", "ok": True, "detail": "stub"}}
    json.dump({{"passed": True, "frozen": True, "checks": [check]}}, open(report, "w"))
elif args[0] == "run":
    out = args[args.index("--out") + 1]
    t = np.linspace(0.0, 10.0, 201)
    y = np.where(t >= 1.0, 1.0 - np.exp(-(t - 1.0) / 2.0), 0.0)
    if {nan}:
        y[:] = np.nan
    np.savetxt(out, np.column_stack([t, t * 0, y]), delimiter=",",
               header="t,reference,output", comments="")
"""


def _smoke(tmp_path, nan):
    exe = tmp_path / "DiaBloS"
    exe.write_text(FAKE_EXE.format(python=sys.executable, nan=nan))
    exe.chmod(0o755)
    env = dict(os.environ, PYTHON=sys.executable)
    return subprocess.run(
        ["bash", SCRIPT, str(exe), str(tmp_path / "report")],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.mark.unit
@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="POSIX shell script")
class TestSmokeScriptChecks:
    def test_matching_output_passes(self, tmp_path):
        proc = _smoke(tmp_path, nan=False)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "bundle smoke test passed" in proc.stdout

    def test_nan_output_fails(self, tmp_path):
        proc = _smoke(tmp_path, nan=True)
        assert proc.returncode != 0
        assert "non-finite" in proc.stderr
        assert "bundle smoke test passed" not in proc.stdout


@pytest.mark.unit
def test_self_test_rejects_nan_output(monkeypatch):
    from lib import api, self_test

    class _Result:
        time = np.linspace(0.0, 10.0, 201)
        engine = "compiled"
        fallback_reason = None

        def __getitem__(self, name):
            return np.full_like(self.time, np.nan)

    class _Diagram:
        def run(self, solver):
            return _Result()

    monkeypatch.setattr(api, "load", lambda path: _Diagram())
    with pytest.raises(AssertionError, match="non-finite"):
        self_test._check_simulation("compiled")
