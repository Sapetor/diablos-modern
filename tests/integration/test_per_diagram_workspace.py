"""Workspace variables belong to a diagram (a DSim), not to the process.

Regression for the old process-wide ``WorkspaceManager`` singleton: building a
second diagram through ``lib.api`` reset the shared store, and setting ``K`` on
the second diagram changed what the *unchanged* first diagram computed.
"""

import numpy as np
import pytest

from lib import cli
from lib.api import from_builder
from lib.diagram_builder import DiagramBuilder
from lib.workspace import Workspace, WorkspaceManager

pytestmark = pytest.mark.integration


def _gain_ramp(gain="K", sim_time=1.0):
    """Step(1) -> Gain(gain) -> Integrator -> Scope: the trace is gain * t."""
    b = DiagramBuilder(sim_time=sim_time, sim_dt=0.01)
    u = b.add_block("Step", 0, 0, name="u", params={"value": 1.0, "delay": 0.0})
    k = b.add_block("Gain", 100, 0, name="k", params={"gain": gain})
    x = b.add_block("Integrator", 200, 0, name="x")
    y = b.add_block("Scope", 300, 0, name="y", params={"labels": "x"})
    b.connect(u, 0, k, 0)
    b.connect(k, 0, x, 0)
    b.connect(x, 0, y, 0)
    return b


def _final(res):
    return float(res["x"][-1])


@pytest.mark.parametrize("solver", ["compiled", "interpreter"])
def test_second_diagram_variables_do_not_leak_into_the_first(solver):
    a = from_builder(_gain_ramp())
    a.set_variable("K", 2.0)
    assert _final(a.run(t=1.0, dt=0.01, solver=solver)) == pytest.approx(2.0, rel=1e-3)

    b = from_builder(_gain_ramp())
    b.set_variable("K", 9.0)
    assert _final(b.run(t=1.0, dt=0.01, solver=solver)) == pytest.approx(9.0, rel=1e-3)

    # Rerunning the unchanged diagram A still uses A's K.
    assert _final(a.run(t=1.0, dt=0.01, solver=solver)) == pytest.approx(2.0, rel=1e-3)
    assert a.variables == {"K": 2.0}
    assert b.variables == {"K": 9.0}


def test_headless_diagrams_do_not_touch_the_default_workspace():
    WorkspaceManager().set_variable("_per_diagram_probe", 1)
    try:
        sim = from_builder(_gain_ramp())
        assert sim.workspace is not WorkspaceManager()
        assert "_per_diagram_probe" not in sim.variables
        sim.set_variable("K", 3.0)
        assert WorkspaceManager().get_value("K") != 3.0
    finally:
        WorkspaceManager().delete_variable("_per_diagram_probe")


def test_engine_shares_the_dsim_workspace():
    dsim = cli.new_headless_dsim()
    assert dsim.engine.workspace is dsim.workspace
    replacement = Workspace({"K": 1.0})
    dsim.workspace = replacement
    assert dsim.engine.workspace is replacement


class TestCloneForAnalysis:
    def test_clone_carries_an_independent_copy(self):
        sim = from_builder(_gain_ramp())
        sim.set_variable("K", 2.0)
        sim.set_variable("v", [1.0, 2.0])
        src = sim.dsim

        clone = src.clone_for_analysis()
        assert clone.workspace is not src.workspace
        assert clone.workspace.variables == {"K": 2.0, "v": [1.0, 2.0]}

        clone.workspace.set_variable("K", 5.0)
        clone.workspace.variables["v"].append(3.0)
        clone.workspace.set_variable("extra", 1)
        assert src.workspace.variables == {"K": 2.0, "v": [1.0, 2.0]}

        # And the other direction.
        src.workspace.set_variable("K", 7.0)
        assert clone.workspace.get_value("K") == 5.0

    def test_clone_runs_with_its_own_variables(self):
        sim = from_builder(_gain_ramp())
        sim.set_variable("K", 2.0)
        clone = sim.dsim.clone_for_analysis()
        clone.workspace.set_variable("K", 4.0)

        ok, msg = clone.run_tuning_simulation(1.0, 0.01)
        assert ok, msg
        scope = next(b for b in clone.blocks_list if b.block_fn == "Scope")
        assert float(np.ravel(scope.exec_params["vector"])[-1]) == pytest.approx(4.0, rel=1e-3)

        # The source is untouched.
        assert _final(sim.run(t=1.0, dt=0.01)) == pytest.approx(2.0, rel=1e-3)


class TestWorkspaceCopy:
    def test_to_dict_from_dict_round_trip_is_deep(self):
        ws = Workspace({"A": [[1.0, 0.0], [0.0, 1.0]]}, workspace_file="vars.txt")
        snap = ws.to_dict()
        assert snap == {"variables": {"A": [[1.0, 0.0], [0.0, 1.0]]}, "workspace_file": "vars.txt"}
        snap["variables"]["A"][0][0] = 99.0
        assert ws.variables["A"][0][0] == 1.0

        back = Workspace.from_dict(ws.to_dict())
        assert type(back) is Workspace
        assert back.variables == ws.variables
        assert back.workspace_file == "vars.txt"

    def test_copy_of_the_default_is_a_plain_workspace(self):
        copied = WorkspaceManager().copy()
        assert type(copied) is Workspace
        assert copied is not WorkspaceManager()

    def test_workspace_instances_are_independent(self):
        a, b = Workspace(), Workspace()
        a.set_variable("K", 1)
        assert b.variables == {}

    def test_workspace_manager_is_still_a_singleton(self):
        wm = WorkspaceManager()
        wm.set_variable("_singleton_probe", 1)
        try:
            # A second call must not re-run __init__ and wipe the variables.
            assert WorkspaceManager() is wm
            assert WorkspaceManager().get_value("_singleton_probe") == 1
        finally:
            wm.delete_variable("_singleton_probe")


def test_mask_scope_uses_the_diagram_workspace():
    """Flattening a masked subsystem reads mask expressions from *this* diagram."""
    from lib.engine.flattener import Flattener
    from lib.masks import resolve_mask_scope

    mask = {"parameters": [{"name": "g", "type": "float", "default": "2*K"}]}
    assert resolve_mask_scope(mask, workspace={"K": 3.0}) == {"g": 6.0}
    with pytest.raises(Exception):
        # No variables -> 'K' is undefined (never read from a process global).
        resolve_mask_scope(mask)
    assert Flattener().flatten([], [], workspace={"K": 1.0}) == ([], [])
