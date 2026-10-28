"""Loading and saving never destroy the user's work.

* Loading is transactional: the incoming diagram is rebuilt off to the side and
  only swapped in once that succeeded. ``"sim_data": null`` used to pass the
  structural check and then crash *after* ``clear_all()``, leaving an empty
  canvas behind.
* Saving is atomic: the JSON goes to a temporary file that replaces the target
  in one ``os.replace``, so a failed write keeps the previous file.
* A failed auto-save recovery keeps the auto-save file instead of deleting it.
"""

import json
import os
import stat
import types

import pytest
from PyQt6.QtCore import QPoint


@pytest.fixture
def dsim(qapp):
    from lib.lib import DSim

    sim = DSim()
    menu = {b.fn_name: b for b in sim.menu_blocks}
    for x in (100, 200, 300, 400):
        sim.add_block(menu["step"], QPoint(x, 100))
    sim.dirty = False
    return sim


def _names(sim):
    return [b.name for b in sim.model.blocks_list]


@pytest.mark.unit
class TestTransactionalLoad:
    def test_null_sim_data_loads_with_defaults(self, dsim):
        data = dsim.serialize()
        data["sim_data"] = None
        params = dsim.deserialize(data)
        assert params["sim_time"] == 1.0
        assert len(dsim.model.blocks_list) == 4

    @pytest.mark.parametrize(
        "patch",
        [
            {"sim_data": 5},
            {"blocks_data": "not a list"},
            {"lines_data": {"a": 1}},
        ],
    )
    def test_a_malformed_file_leaves_the_open_diagram_alone(self, dsim, patch):
        before = _names(dsim)
        data = dsim.serialize()
        data.update(patch)
        with pytest.raises(ValueError):
            dsim.deserialize(data)
        assert _names(dsim) == before

    def test_an_unexpected_failure_mid_build_leaves_the_open_diagram_alone(self, dsim, monkeypatch):
        before = _names(dsim)
        data = dsim.serialize()

        def boom(_blocks):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(dsim.file_service, "_build_name_map", boom)
        with pytest.raises(RuntimeError):
            dsim.deserialize(data)
        assert _names(dsim) == before

    def test_a_good_file_replaces_the_diagram_in_place(self, dsim):
        aliased = dsim.model.blocks_list
        data = dsim.serialize()
        data["blocks_data"] = data["blocks_data"][:1]
        dsim.deserialize(data)
        assert dsim.model.blocks_list is aliased
        assert len(aliased) == 1


@pytest.mark.unit
class TestAtomicSave:
    def test_a_failed_write_keeps_the_previous_file(self, dsim, tmp_path, monkeypatch):
        from lib.services import file_service

        target = tmp_path / "d.diablos"
        assert dsim.save(filepath=str(target)) == 0
        original = target.read_text(encoding="utf-8")

        def half_written(obj, fp, **kw):
            fp.write('{"blocks_data": [')
            raise OSError("disk full")

        monkeypatch.setattr(file_service.json, "dump", half_written)
        assert dsim.save(filepath=str(target)) == 1
        assert target.read_text(encoding="utf-8") == original
        assert os.listdir(tmp_path) == ["d.diablos"]

    def test_the_replacement_keeps_the_file_mode(self, tmp_path):
        from lib.services.file_service import atomic_write_json

        target = tmp_path / "d.diablos"
        target.write_text("{}", encoding="utf-8")
        os.chmod(target, 0o644)
        atomic_write_json(str(target), {"a": 1})
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o644
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


@pytest.mark.unit
class TestFailedRecoveryKeepsTheAutosave:
    def _manager(self, tmp_path, load_result):
        from modern_ui.managers.project_manager import ProjectManager

        pm = ProjectManager.__new__(ProjectManager)
        pm.autosave_path = str(tmp_path / ".autosave.diablos")
        pm.diagram_service = types.SimpleNamespace(load_diagram=lambda path: load_result)
        pm.window = None
        return pm

    def test_unloadable_autosave_is_set_aside_not_deleted(self, qapp, tmp_path):
        pm = self._manager(tmp_path, load_result=None)
        with open(pm.autosave_path, "w", encoding="utf-8") as fp:
            fp.write("precious")
        pm.recover_autosave()
        assert not os.path.exists(pm.autosave_path)
        kept = [n for n in os.listdir(tmp_path) if ".failed-" in n]
        assert len(kept) == 1
        assert (tmp_path / kept[0]).read_text(encoding="utf-8") == "precious"
