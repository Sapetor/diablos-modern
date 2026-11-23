"""diablos.core does not use PyQt6, pyqtgraph or the GUI package.

Stage 3b of the Qt-free core (2026-11) moved the plot windows, dialogs,
analyzer windows, theme manager, document services and preference store to
``diablos.gui``; where the engine needs them, the main window attaches hooks
to its DSim (``scope_plotter``, ``settings_dialog``, ``category_color``,
``file_service.choose_*_path``). This test walks every module in
``diablos.core`` -- imports inside functions, try/if blocks and class bodies
included -- so nothing creeps back.
"""

import ast
import pathlib

import pytest

pytestmark = pytest.mark.unit

CORE = pathlib.Path(__file__).resolve().parents[2] / "src" / "diablos" / "core"

QT = ("PyQt6", "pyqtgraph")

#: The deliberate bridges: Qt value types (re-exported when PyQt6 is
#: installed) and the IconPath -> QPainterPath conversion the renderer calls.
QT_ALLOWED = {"qt_types.py", "icon_path.py"}

#: Lazy compatibility lookups for names that moved to the GUI package
#: (module-level __getattr__); nothing in the engine calls them.
GUI_ALLOWED = {
    "analysis/__init__.py",  # ControlSystemAnalyzer
    "plotting/__init__.py",  # SignalPlot
    "app_paths.py",  # ui_settings, SETTINGS_ORG, SETTINGS_APP
    "i18n.py",  # stored_language_setting, store_language_setting
}


def _imports(tree):
    """(module name, line) for every import and literal dynamic import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or "", node.lineno
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield arg.value, node.lineno


def _modules():
    return sorted(p for p in CORE.rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.relative_to(CORE)))
def test_core_module_stays_on_its_side_of_the_boundary(path):
    rel = str(path.relative_to(CORE))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []
    for module, line in _imports(tree):
        top = module.split(".")[0]
        if top in QT and rel not in QT_ALLOWED:
            problems.append(f"line {line}: imports {module}")
        if (module == "diablos.gui" or module.startswith("diablos.gui.")) and (
            rel not in GUI_ALLOWED
        ):
            problems.append(f"line {line}: imports {module}")
    assert problems == [], f"{rel}: " + "; ".join(problems)


def test_the_guard_sees_nested_and_dynamic_imports():
    src = (
        "def f():\n"
        "    try:\n"
        "        from PyQt6.QtWidgets import QDialog\n"
        "    except ImportError:\n"
        "        pass\n"
        "class A:\n"
        "    import pyqtgraph\n"
        "importlib.import_module('diablos.gui.main_window')\n"
    )
    found = sorted(m for m, _ in _imports(ast.parse(src)))
    assert found == sorted(["PyQt6.QtWidgets", "pyqtgraph", "diablos.gui.main_window"])


def test_a_headless_run_loads_no_gui_modules(tmp_path):
    """With PyQt6 installed, a CLI run still loads no window code: only the
    Qt value types the engine re-exports (diablos.core.qt_types)."""
    import json
    import os
    import subprocess
    import sys
    import textwrap

    repo = CORE.parents[2]
    code = textwrap.dedent(
        """
        import json, sys
        import diablos.main  # the launcher, as `diablos run` imports it
        from diablos.core.cli import run_diagram
        from diablos.core.analysis.monte_carlo import MonteCarloRunner
        dsim = run_diagram("examples/first_order_step_response.diablos")
        MonteCarloRunner(dsim).run(2, master_seed=1, sim_time=0.5, sim_dt=0.01)
        loaded = sorted(m for m in sys.modules
                        if m.startswith(("diablos.gui", "pyqtgraph", "matplotlib", "PyQt6.QtWidgets")))
        print(json.dumps({"loaded": loaded, "plotter": dsim.scope_plotter is None}))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=dict(os.environ, MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result == {"loaded": [], "plotter": True}
