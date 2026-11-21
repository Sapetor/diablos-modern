"""The simulation engine runs in a process where PyQt6 cannot be imported.

Before 2026-11-21 ``import diablos.core.lib`` required PyQt6: the diagram model kept
its geometry and colours as Qt value types imported at module level, DBlock
and MenuBlocks built a QFont/QPixmap (which need a QGuiApplication), and the
headless CLI always created an offscreen QApplication. ``src/diablos/core/qt_types.py`` now
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


@pytest.fixture(scope="module")
def no_qt_path(tmp_path_factory):
    """A directory whose sitecustomize.py blocks PyQt6. Put on PYTHONPATH, it
    applies to every interpreter started with that environment, including the
    spawned worker processes of a parallel Monte Carlo run."""
    d = tmp_path_factory.mktemp("no_qt")
    (d / "sitecustomize.py").write_text(_BLOCK_QT)
    return str(d)


def _run_without_qt(body=None, timeout=300, argv=None, no_qt=None, expect_ok=True):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=no_qt)
    env.pop("DIABLOS_PURE_TYPES", None)
    cmd = [sys.executable] + (argv if argv else ["-c", textwrap.dedent(body)])
    out = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    if not expect_ok:
        return out
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


EXAMPLES = [
    "first_order_step_response.diablos",
    "pid_second_order.diablos",
    "library_block_demo.diablos",  # a library subsystem with Inport/Outport children
    "optimizer_comparison.diablos",
]


def _traces(path):
    from diablos.core.analysis.resim import harvest_scope_signals
    from diablos.core.cli import run_diagram

    sig = harvest_scope_signals(run_diagram(path))
    return {k: np.asarray(v, dtype=float) for k, v in sig["signals"].items()}


@pytest.mark.regression
@pytest.mark.slow
def test_examples_match_a_qt_run_without_pyqt6(no_qt_path):
    paths = [os.path.join("examples", name) for name in EXAMPLES]
    result = _run_without_qt(
        f"""
        import json, sys
        from diablos.core.analysis.resim import harvest_scope_signals
        from diablos.core.cli import run_diagram

        out = {{}}
        for path in {paths!r}:
            sig = harvest_scope_signals(run_diagram(path))
            out[path] = {{k: list(map(float, v)) for k, v in sig["signals"].items()}}
        out["_qt_loaded"] = sorted(m for m in sys.modules if m.split(".")[0] in ("PyQt6", "pyqtgraph"))
        print(json.dumps(out))
        """,
        no_qt=no_qt_path,
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
def test_engine_and_analysis_import_and_build_without_pyqt6(no_qt_path):
    result = _run_without_qt(
        """
        import json
        import diablos.core.lib, diablos.core.analysis.monte_carlo, diablos.core.analysis.parameter_sweep
        import diablos.core.analysis.linearizer, diablos.core.engine.system_compiler
        from diablos.core import qt_types
        from diablos.core.cli import load_diagram

        dsim, _ = load_diagram("examples/first_order_step_response.diablos")
        block = dsim.blocks_list[0]
        print(json.dumps({
            "has_qt": qt_types.HAS_QT,
            "blocks": len(dsim.blocks_list),
            "lines": len(dsim.line_list),
            "rect": [block.rect.left(), block.rect.top(), block.rect.width(), block.rect.height()],
            "ports": len(block.out_coords),
        }))
        """,
        no_qt=no_qt_path,
    )
    assert result["has_qt"] is False
    assert result["blocks"] == 3 and result["lines"] == 3
    assert result["rect"][2] > 0 and result["ports"] >= 1


@pytest.mark.regression
@pytest.mark.slow
def test_monte_carlo_runs_without_pyqt6(no_qt_path):
    result = _run_without_qt(
        """
        import json
        from diablos.core.analysis.monte_carlo import MonteCarloRunner
        from diablos.core.cli import load_diagram

        dsim, params = load_diagram("examples/first_order_step_response.diablos")
        serial = MonteCarloRunner(dsim).run(3, master_seed=1, sim_time=1.0, sim_dt=0.01)
        pooled = MonteCarloRunner(dsim).run(3, master_seed=1, sim_time=1.0, sim_dt=0.01, workers=2)
        import numpy as np
        same = serial["n_ok"] == pooled["n_ok"] == 3 and all(
            np.array_equal(serial["signals"][k]["mean"], pooled["signals"][k]["mean"])
            for k in serial["signals"]
        )
        print(json.dumps({"ok": bool(serial["signals"]), "same": bool(same)}))
        """,
        no_qt=no_qt_path,
    )
    assert result["ok"] and result["same"]


@pytest.mark.regression
def test_headless_run_creates_no_qapplication_when_pyqt6_is_installed():
    """The CLI used to build an offscreen QApplication for every headless DSim."""
    env = dict(os.environ, MPLBACKEND="Agg")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                from diablos.core.cli import run_diagram
                from PyQt6.QtCore import QCoreApplication

                dsim = run_diagram("examples/first_order_step_response.diablos")
                print(QCoreApplication.instance() is None, len(dsim.timeline) > 0)
                """
            ),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    assert out.stdout.split()[-2:] == ["True", "True"]


@pytest.mark.regression
@pytest.mark.slow
def test_launcher_runs_headless_subcommands_without_pyqt6(no_qt_path, tmp_path):
    """``diablos_modern.py run`` imported the whole GUI before dispatching."""
    out_csv = str(tmp_path / "out.csv")
    proc = _run_without_qt(
        argv=[
            "diablos_modern.py",
            "run",
            "examples/first_order_step_response.diablos",
            "-o",
            out_csv,
        ],
        no_qt=no_qt_path,
        expect_ok=False,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert os.path.getsize(out_csv) > 0
    gui = _run_without_qt(argv=["diablos_modern.py"], no_qt=no_qt_path, expect_ok=False)
    assert gui.returncode == 1
    assert "needs PyQt6" in gui.stderr


@pytest.mark.regression
def test_execution_init_does_not_open_a_dialog_headless(no_qt_path):
    """The first-run settings dialog is for the GUI; a script has nobody to ask."""
    result = _run_without_qt(
        """
        import json
        from diablos.core.cli import load_diagram

        dsim, _ = load_diagram("examples/first_order_step_response.diablos")
        print(json.dumps({"reviewed": dsim.settings_reviewed, "ok": dsim.execution_init()}))
        """,
        no_qt=no_qt_path,
    )
    assert result == {"reviewed": False, "ok": True}


@pytest.mark.regression
def test_forced_stand_ins_work_with_pyqt6_installed():
    """DIABLOS_PURE_TYPES=1 with PyQt6 present used to drop every block: the
    theme manager's real QColor failed the stand-in type check."""
    env = dict(os.environ, MPLBACKEND="Agg", DIABLOS_PURE_TYPES="1")
    code = textwrap.dedent(
        """
        import json
        from diablos.core import qt_types
        from diablos.core.analysis.resim import harvest_scope_signals
        from diablos.core.cli import load_diagram, run_diagram

        dsim, _ = load_diagram("examples/first_order_step_response.diablos")
        sig = harvest_scope_signals(run_diagram("examples/first_order_step_response.diablos"))
        print(json.dumps({"has_qt": qt_types.HAS_QT, "blocks": len(dsim.blocks_list),
                          "signals": sorted(sig["signals"])}))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-4000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["has_qt"] is False and result["blocks"] == 3 and result["signals"]


_MINIMAL = textwrap.dedent(
    """
    import importlib.abc, sys

    _OPTIONAL = ("PyQt6", "PyQt5", "PySide6", "pyqtgraph", "matplotlib", "tqdm", "PIL", "sympy")

    class _NumpyScipyOnly(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in _OPTIONAL:
                raise ImportError(name.split(".")[0] + " is not installed (blocked by the test)")

    sys.meta_path.insert(0, _NumpyScipyOnly())
    """
)


@pytest.mark.regression
@pytest.mark.slow
def test_engine_needs_only_numpy_and_scipy(tmp_path):
    """The engine's dependency contract: numpy + scipy. tqdm (progress bar) and
    matplotlib (FFT block's plot window) used to be imported at module level,
    so DSim failed without tqdm and the FFT block silently disappeared."""
    (tmp_path / "sitecustomize.py").write_text(_MINIMAL)
    paths = [os.path.join("examples", name) for name in EXAMPLES]
    result = _run_without_qt(
        f"""
        import json
        from diablos.core.analysis.monte_carlo import MonteCarloRunner
        from diablos.core.analysis.resim import harvest_scope_signals
        from diablos.core.block_loader import load_blocks
        from diablos.core.cli import load_diagram, run_diagram

        out = {{"blocks": sorted(c().block_name for c in load_blocks())}}
        for path in {paths!r}:
            sig = harvest_scope_signals(run_diagram(path))
            out[path] = {{k: list(map(float, v)) for k, v in sig["signals"].items()}}
        dsim, _ = load_diagram({paths[0]!r})
        mc = MonteCarloRunner(dsim).run(2, master_seed=1, sim_time=1.0, sim_dt=0.01)
        out["mc_ok"] = mc["n_ok"]
        print(json.dumps(out))
        """,
        no_qt=str(tmp_path),
    )
    from diablos.core.block_loader import load_blocks

    assert result.pop("blocks") == sorted(c().block_name for c in load_blocks())
    assert result.pop("mc_ok") == 2
    for path in paths:
        expected = _traces(os.path.join(REPO, path))
        for name, trace in expected.items():
            assert np.allclose(result[path][name], trace, rtol=1e-12, atol=1e-12, equal_nan=True)
