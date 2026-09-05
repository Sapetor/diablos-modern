"""Regression tests for the unsaved-changes ("dirty") flag.

BUG DESCRIPTION
---------------
``DSim.dirty`` was a *copy* of ``SimulationModel.dirty``, re-synced only in
add_block / add_line / remove_block_and_lines / clear_all:

  * ``FileService`` clears ``model.dirty`` on save, but the whole GUI (status
    bar, property controller, clipboard/connection managers) reads and writes
    ``dsim.dirty`` -- so **saving never cleared the flag the GUI showed**.
  * ``execution_init`` did ``self.dirty = False``, so **pressing Run marked
    unsaved edits as saved**, and the window then closed without a prompt.

THE FIX
-------
``DSim.dirty`` is a property over ``self.model.dirty`` (both directions), and
the reset inside ``execution_init`` is gone.
"""

import pytest
from PyQt5.QtCore import QPoint


@pytest.fixture
def dsim(qapp):
    from lib.lib import DSim

    return DSim()


@pytest.mark.unit
class TestDirtyIsALiveView:
    def test_reads_through_to_the_model(self, dsim):
        dsim.model.dirty = True
        assert dsim.dirty is True
        dsim.model.dirty = False
        assert dsim.dirty is False

    def test_writes_through_to_the_model(self, dsim):
        dsim.dirty = True
        assert dsim.model.dirty is True
        dsim.dirty = False
        assert dsim.model.dirty is False

    def test_is_not_a_plain_instance_attribute(self, dsim):
        # A per-instance copy would shadow the property and desync again.
        assert "dirty" not in vars(dsim)

    def test_adding_a_block_marks_the_diagram_dirty(self, dsim):
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.dirty = False
        dsim.add_block(menu["step"], QPoint(100, 100))
        assert dsim.dirty is True


@pytest.mark.unit
class TestSaveClearsTheFlagTheGuiReads:
    def test_file_service_save_clears_dsim_dirty(self, dsim, tmp_path):
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["step"], QPoint(100, 100))
        assert dsim.dirty is True

        target = tmp_path / "diagram.diablos"
        assert dsim.file_service.save_to_file(dsim.serialize(), str(target)) is True

        # The GUI reads dsim.dirty; before the fix this stayed True forever.
        assert dsim.model.dirty is False
        assert dsim.dirty is False


@pytest.mark.unit
class TestPlotGateIsSeparateFromDirty:
    """``plot_again`` must gate on "edited since the last run", not "unsaved".

    ScopePlotter used ``dsim.dirty`` as its stale-data guard, which only worked
    because ``execution_init`` cleared it. With that reset removed, a normal
    (unsaved) diagram would never plot again -- so the two notions are now
    separate flags.
    """

    def test_flag_tracks_edits_and_runs(self, dsim, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
        step = dsim.add_block(menu["step"], QPoint(100, 100))
        scope = dsim.add_block(menu["scope"], QPoint(300, 100))
        dsim.add_line((step.name, 0, step.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
        dsim.sim_time, dsim.sim_dt = 0.05, 0.01
        dsim.execution_init_time = lambda: dsim.sim_time

        assert dsim.diagram_changed_since_run is True
        assert dsim.execution_init() is True
        assert dsim.diagram_changed_since_run is False
        dsim.execution_initialized = False

        dsim.dirty = True  # a param edit through the property editor
        assert dsim.diagram_changed_since_run is True

    def test_plot_again_is_not_blocked_by_unsaved_changes(self, dsim, monkeypatch):
        reached = {"n": 0}
        monkeypatch.setattr(
            dsim.scope_plotter, "_close_open_figures", lambda: reached.__setitem__("n", 1)
        )
        dsim.model.dirty = True  # unsaved...
        dsim.diagram_changed_since_run = False  # ...but not edited since the run
        dsim.plot_again()
        assert reached["n"] == 1

    def test_plot_again_is_blocked_after_an_edit(self, dsim, monkeypatch):
        reached = {"n": 0}
        monkeypatch.setattr(
            dsim.scope_plotter, "_close_open_figures", lambda: reached.__setitem__("n", 1)
        )
        dsim.diagram_changed_since_run = True
        dsim.plot_again()
        assert reached["n"] == 0


@pytest.mark.unit
class TestAutosaveDoesNotClearTheFlag:
    def test_autosave_preserves_dirty(self, dsim, tmp_path, monkeypatch):
        """An autosave is a crash snapshot, not the user saving their file."""
        monkeypatch.chdir(tmp_path)
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["step"], QPoint(100, 100))
        assert dsim.dirty is True

        assert dsim.save(autosave=True) == 0
        assert dsim.dirty is True


@pytest.mark.unit
class TestRunningDoesNotClearTheFlag:
    def test_execution_init_leaves_unsaved_edits_marked(self, dsim, tmp_path, monkeypatch):
        # execution_init force-autosaves into ./saves; keep that out of the repo.
        monkeypatch.chdir(tmp_path)
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
        step = dsim.add_block(menu["step"], QPoint(100, 100))
        scope = dsim.add_block(menu["scope"], QPoint(300, 100))
        dsim.add_line((step.name, 0, step.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
        dsim.sim_time = 0.05
        dsim.sim_dt = 0.01
        dsim.execution_init_time = lambda: dsim.sim_time

        assert dsim.dirty is True
        assert dsim.execution_init() is True
        # Pressing Run is not saving.
        assert dsim.dirty is True

        dsim.execution_initialized = False
