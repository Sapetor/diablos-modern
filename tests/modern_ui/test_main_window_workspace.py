"""The GUI's single diagram owns the workspace the window edits and loads.

Workspace variables are per-DSim (``lib/workspace.py``). The window's DSim is
built on the process-level default, so ``WorkspaceManager()`` (legacy and
user-block code) still sees the variables loaded in the window, and loading a
workspace file still drives a simulation.
"""

import json

import numpy as np
import pytest

from lib.diagram_builder import DiagramBuilder
from lib.workspace import WorkspaceManager


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_variables(window):
    ws = window.dsim.workspace
    saved_vars, saved_file = dict(ws.variables), ws.workspace_file
    yield
    ws.variables = saved_vars
    ws.workspace_file = saved_file
    # Editors hold the same object; keep them in step with it.
    window.variable_editor.workspace_manager = ws
    window.workspace_editor.workspace_manager = ws


def _load_workspace_file(window, monkeypatch, path):
    monkeypatch.setattr(
        "modern_ui.main_window.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(path), ""),
    )
    window.load_workspace()


def test_window_diagram_is_built_on_the_default_workspace(window):
    ws = window.dsim.workspace
    assert ws is window.dsim.engine.workspace
    # The editors and the property editor all edit the diagram's workspace.
    assert window.variable_editor.workspace_manager is ws
    assert window.workspace_editor.workspace_manager is ws
    assert window.property_editor._workspace() is ws


def test_loaded_workspace_file_is_visible_to_legacy_callers(window, monkeypatch, tmp_path):
    path = tmp_path / "vars.txt"
    path.write_text("K = 3.5\nnum = [1, 2]\n")
    _load_workspace_file(window, monkeypatch, path)

    assert window.dsim.workspace.get_value("K") == 3.5
    assert window.dsim.workspace.workspace_file == str(path)
    # Old-style access resolves to the same variables, as before the change
    # (nothing in this module resets the default after the window is built).
    assert WorkspaceManager() is window.dsim.workspace
    assert WorkspaceManager().get_value("K") == 3.5


def test_loaded_workspace_file_drives_a_simulation(window, monkeypatch, tmp_path):
    path = tmp_path / "vars.txt"
    path.write_text("K = 3.0\n")
    _load_workspace_file(window, monkeypatch, path)

    b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    u = b.add_block("Step", 0, 0, name="u", params={"value": 1.0, "delay": 0.0})
    k = b.add_block("Gain", 100, 0, name="k", params={"gain": "K"})
    x = b.add_block("Integrator", 200, 0, name="x")
    y = b.add_block("Scope", 300, 0, name="y", params={"labels": "x"})
    b.connect(u, 0, k, 0)
    b.connect(k, 0, x, 0)
    b.connect(x, 0, y, 0)

    from lib.api import _complete_builder_blocks

    dsim = window.dsim
    data = json.loads(json.dumps(b.to_dict()))
    _complete_builder_blocks(dsim, data["blocks_data"])
    dsim.file_service.apply_loaded_data(data)
    try:
        ok, msg = dsim.run_tuning_simulation(1.0, 0.01)
        assert ok, msg
        scope = next(blk for blk in dsim.blocks_list if blk.block_fn == "Scope")
        assert float(np.ravel(scope.exec_params["vector"])[-1]) == pytest.approx(3.0, rel=1e-3)
    finally:
        dsim.clear_all()


def test_new_diagram_keeps_workspace_variables(window):
    """New/Open never touched the workspace; variables survive them, as before."""
    window.dsim.workspace.set_variable("K", 4.0)
    window.new_diagram()
    assert window.dsim.workspace.get_value("K") == 4.0
