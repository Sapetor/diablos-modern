"""Integration test: the masked-library example runs on both engines.

``examples/library_block_demo.diablos`` closes a proportional loop around a
masked "Vehicle" subsystem whose inner TranFn stores ``denominator = "[m, b]"``.
Getting the right answer therefore requires the mask scope to reach the block
on *both* execution paths (compiled and interpreted), and the shipped library
copy in ``examples/library/vehicle.diablos`` must describe the same block.
"""

from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).parent.parent.parent / "examples"
DEMO = EXAMPLES / "library_block_demo.diablos"
LIBRARY_DIR = EXAMPLES / "library"

# Unity-feedback steady state for K/(m s + b) with K = 800, b = 50 and a step
# of 20: v_ss = 20 * K / (K + b).
EXPECTED_STEADY_STATE = 20.0 * 800.0 / (800.0 + 50.0)


def _run(use_fast_solver):
    from lib.analysis.resim import harvest_scope_signals
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    previous = WorkspaceManager._instance
    WorkspaceManager._instance = None
    try:
        dsim = DSim()
        data = dsim.file_service.load(filepath=str(DEMO))
        assert data is not None
        dsim.deserialize(data)
        dsim.use_fast_solver = use_fast_solver
        ok, message = dsim.run_tuning_simulation(dsim.sim_time, dsim.sim_dt)
        assert ok, f"run failed (fast_solver={use_fast_solver}): {message}"
        harvested = harvest_scope_signals(dsim)
        assert harvested and harvested["signals"], "no scope trace was recorded"
        return next(iter(harvested["signals"].values()))
    finally:
        WorkspaceManager._instance = previous


@pytest.mark.integration
def test_example_files_exist():
    assert DEMO.is_file()
    assert (LIBRARY_DIR / "vehicle.diablos").is_file()


@pytest.mark.integration
def test_the_shipped_library_block_matches_the_demo(qapp):
    """Pointing DIABLOS_LIBRARY_PATH at examples/library exposes 'Vehicle'."""
    from lib.library import read_library_file

    lib_block = read_library_file(str(LIBRARY_DIR / "vehicle.diablos"))
    assert lib_block is not None
    assert lib_block.name == "Vehicle"
    assert [p["name"] for p in lib_block.mask["parameters"]] == ["m", "b"]
    inner = lib_block.block_data["sub_blocks"]
    tranfns = [b for b in inner if b["block_fn"] == "TranFn"]
    assert tranfns and tranfns[0]["params"]["denominator"] == "[m, b]"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("use_fast_solver", [True, False], ids=["compiled", "interpreted"])
def test_masked_example_runs(qapp, use_fast_solver):
    trace = _run(use_fast_solver)
    assert np.isclose(float(trace[-1]), EXPECTED_STEADY_STATE, rtol=1e-3), (
        f"steady state {trace[-1]} != {EXPECTED_STEADY_STATE}"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_both_engines_agree(qapp):
    compiled = _run(True)
    interpreted = _run(False)
    assert np.isclose(float(compiled[-1]), float(interpreted[-1]), rtol=1e-4)
