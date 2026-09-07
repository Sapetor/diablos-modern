"""scripts/resave_examples.py must keep every saved simulation setting."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "resave_examples", ROOT / "scripts" / "resave_examples.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_resave_keeps_solver_settings(qapp, tmp_path):
    src = ROOT / "examples" / "van_der_pol_stiff.diablos"
    before = json.loads(src.read_text())["sim_data"]
    assert before["solver_method"] == "Radau", "fixture example must be saved with Radau"

    target = tmp_path / "vdp.diablos"
    shutil.copy(src, target)
    _load_script().resave_file(str(target))

    after = json.loads(target.read_text())["sim_data"]
    for key in ("solver_method", "rtol", "atol", "zero_crossing", "sim_time", "sim_dt"):
        assert after[key] == before[key], key
