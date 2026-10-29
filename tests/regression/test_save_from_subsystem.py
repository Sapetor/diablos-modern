"""Saving while a subsystem is open writes the whole document.

Entering a subsystem points the model's block/line lists at its internals,
and every save route serialized "the current lists" -- so saving from inside
``Subsystem1`` wrote only that subsystem's three child blocks over the file,
reported success and cleared the dirty flag. ``FileService.serialize`` now
serializes the root scope (``root_context``, installed by DSim).
"""

import json
import os

import pytest
from PyQt6.QtCore import QPoint

EXAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "library_block_demo.diablos"
)
ROOT = ["Subsystem1", "gain0", "scope0", "step0", "sum0"]


@pytest.fixture
def dsim(qapp):
    from lib.cli import load_diagram

    sim, _ = load_diagram(EXAMPLE)
    return sim


def _subsystem(blocks, name=None):
    return next(b for b in blocks if b.block_fn == "Subsystem" and (name is None or b.name == name))


def _saved(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _names(records):
    return sorted(r["name"] for r in records)


@pytest.mark.regression
class TestSaveFromInsideASubsystem:
    def test_one_level_keeps_the_root_and_the_edit(self, dsim, tmp_path):
        dsim.enter_subsystem(_subsystem(dsim.model.blocks_list))
        inside = list(dsim.model.blocks_list)
        menu = {b.fn_name: b for b in dsim.menu_blocks}
        dsim.add_block(menu["gain"], QPoint(300, 300))
        added = next(b.name for b in dsim.model.blocks_list if b not in inside)

        out = tmp_path / "one.diablos"
        assert dsim.save(filepath=str(out)) == 0
        data = _saved(out)

        assert _names(data["blocks_data"]) == ROOT
        sub = next(r for r in data["blocks_data"] if r["block_fn"] == "Subsystem")
        assert added in _names(sub["sub_blocks"])
        # Root connections survive too.
        original = _saved(EXAMPLE)["lines_data"]
        pairs = {(r["srcblock"], r["dstblock"]) for r in data["lines_data"]}
        assert pairs == {(r["srcblock"], r["dstblock"]) for r in original}
        # Still inside the subsystem, and the document counts as saved.
        assert dsim.navigation_stack
        assert added in [b.name for b in dsim.model.blocks_list]
        assert dsim.dirty is False

    def test_reopening_the_save_restores_everything(self, dsim, tmp_path):
        from lib.cli import load_diagram

        dsim.enter_subsystem(_subsystem(dsim.model.blocks_list))
        out = tmp_path / "reopen.diablos"
        dsim.save(filepath=str(out))
        again, _ = load_diagram(str(out))
        assert sorted(b.name for b in again.model.blocks_list) == ROOT
        assert len(again.model.line_list) == len(_saved(EXAMPLE)["lines_data"])

    def test_two_levels_deep(self, dsim, tmp_path):
        outer = _subsystem(dsim.model.blocks_list)
        dsim.enter_subsystem(outer)
        tranfn = next(b for b in dsim.model.blocks_list if b.block_fn == "TranFn")
        inner = dsim.create_subsystem_from_selection([tranfn])
        if inner is None:
            inner = _subsystem(dsim.model.blocks_list)
        dsim.enter_subsystem(inner)
        assert len(dsim.navigation_stack) == 2

        out = tmp_path / "two.diablos"
        assert dsim.save(filepath=str(out)) == 0
        data = _saved(out)
        assert _names(data["blocks_data"]) == ROOT
        outer_rec = next(r for r in data["blocks_data"] if r["block_fn"] == "Subsystem")
        inner_rec = next(r for r in outer_rec["sub_blocks"] if r["block_fn"] == "Subsystem")
        assert "TranFn" in [r["block_fn"] for r in inner_rec["sub_blocks"]]
        assert len(dsim.navigation_stack) == 2

    def test_autosave_route_writes_the_root(self, dsim, tmp_path):
        dsim.enter_subsystem(_subsystem(dsim.model.blocks_list))
        out = tmp_path / "auto.diablos"
        assert dsim.file_service.save(autosave=True, filepath=str(out)) == 0
        assert _names(_saved(out)["blocks_data"]) == ROOT

    def test_diagram_service_route_writes_the_root(self, dsim, tmp_path):
        from types import SimpleNamespace

        from lib.services.diagram_service import DiagramService

        window = SimpleNamespace(
            dsim=dsim,
            canvas=SimpleNamespace(zoom_factor=1.0),
            main_splitter=SimpleNamespace(sizes=lambda: [1, 1]),
        )
        service = DiagramService(window)
        service.dsim = dsim
        dsim.enter_subsystem(_subsystem(dsim.model.blocks_list))
        out = tmp_path / "service.diablos"
        assert service.save_diagram(str(out)) is True
        assert _names(_saved(out)["blocks_data"]) == ROOT
