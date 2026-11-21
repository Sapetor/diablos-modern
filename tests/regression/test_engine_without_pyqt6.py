"""The simulation engine runs in a process where PyQt6 cannot be imported.

Before 2026-11-21 ``import lib.lib`` required PyQt6: the diagram model kept
its geometry and colours as Qt value types imported at module level, DBlock
and MenuBlocks built a QFont/QPixmap (which need a QGuiApplication), and the
headless CLI always created an offscreen QApplication. ``lib/qt_types.py`` now
supplies pure-Python stand-ins when PyQt6 is missing.

Each test runs a fresh interpreter with an import hook that makes PyQt6 and
pyqtgraph unimportable, and compares the results with a normal run.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_BLOCK_QT = textwrap.dedent(
    """
    import importlib.abc, sys

    class _NoQt(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in ("PyQt6", "PyQt5", "PySide6", "pyqtgraph"):
                raise ImportError("PyQt6 is not installed (blocked by the test)")

    sys.meta_path.insert(0, _NoQt())
    """
)


def _run_without_qt(body, timeout=300):
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("DIABLOS_PURE_TYPES", None)
    out = subprocess.run(
        [sys.executable, "-c", _BLOCK_QT + textwrap.dedent(body)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


EXAMPLES = [
    "first_order_step_response.diablos",
    "pid_second_order.diablos",
    "library_block_demo.diablos",  # a library subsystem with Inport/Outport children
    "optimizer_comparison.diablos",
]


def _traces(path):
    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import run_diagram

    sig = harvest_scope_signals(run_diagram(path))
    return {k: np.asarray(v, dtype=float) for k, v in sig["signals"].items()}


@pytest.mark.regression
@pytest.mark.slow
def test_examples_match_a_qt_run_without_pyqt6():
    paths = [os.path.join("examples", name) for name in EXAMPLES]
    result = _run_without_qt(
        f"""
        import json, sys
        from lib.analysis.resim import harvest_scope_signals
        from lib.cli import run_diagram

        out = {{}}
        for path in {paths!r}:
            sig = harvest_scope_signals(run_diagram(path))
            out[path] = {{k: list(map(float, v)) for k, v in sig["signals"].items()}}
        out["_qt_loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in ("PyQt6", "pyqtgraph"))
        print(json.dumps(out))
        """
    )
    assert result.pop("_qt_loaded") == []
    for path in paths:
        expected = _traces(os.path.join(REPO, path))
        got = result[path]
        assert set(got) == set(expected), path
        for name, trace in expected.items():
            assert np.allclose(got[name], trace, rtol=1e-12, atol=1e-12, equal_nan=True), (
                path,
                name,
            )


@pytest.mark.regression
def test_engine_and_analysis_import_and_build_without_pyqt6():
    result = _run_without_qt(
        """
        import json
        import lib.lib, lib.analysis.monte_carlo, lib.analysis.parameter_sweep
        import lib.analysis.linearizer, lib.engine.system_compiler
        from lib import qt_types
        from lib.cli import load_diagram

        dsim, _ = load_diagram("examples/first_order_step_response.diablos")
        block = dsim.blocks_list[0]
        print(json.dumps({
            "has_qt": qt_types.HAS_QT,
            "blocks": len(dsim.blocks_list),
            "lines": len(dsim.line_list),
            "rect": [block.rect.left(), block.rect.top(), block.rect.width(), block.rect.height()],
            "ports": len(block.out_coords),
        }))
        """
    )
    assert result["has_qt"] is False
    assert result["blocks"] == 3 and result["lines"] == 3
    assert result["rect"][2] > 0 and result["ports"] >= 1


@pytest.mark.regression
@pytest.mark.slow
def test_monte_carlo_runs_without_pyqt6():
    result = _run_without_qt(
        """
        import json
        from lib.analysis.monte_carlo import MonteCarloRunner
        from lib.cli import load_diagram

        dsim, params = load_diagram("examples/first_order_step_response.diablos")
        res = MonteCarloRunner(dsim).run(3, master_seed=1, sim_time=1.0, sim_dt=0.01)
        print(json.dumps({"ok": res is not None}))
        """
    )
    assert result["ok"]
