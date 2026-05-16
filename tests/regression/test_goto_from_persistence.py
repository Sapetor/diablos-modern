"""
Regression tests for audit priority #6b: Goto/From virtual lines persisted
to disk.

BUG DESCRIPTION
---------------
SimulationModel.link_goto_from() creates hidden=True "virtual" DLine
objects to wire each From block to the source feeding its matching Goto.
These virtual lines must NOT be persisted to .diablos files — they are
recreated at runtime by link_goto_from on every execution_init.

Before the fix, FileService.serialize() iterated model.line_list without
filtering on the `hidden` flag, so any save that occurred AFTER a
simulation had been initialised wrote the virtual lines to disk.  On
reload they came back as ordinary visible lines (DLine default
hidden=False), and link_goto_from could not remove them because its
cleanup pass only filters hidden=True.  The result: ghost lines visible
on the canvas, accumulating on every save/reload cycle.

THE FIX
-------
1. Serialize-time filter: skip lines with `hidden=True` in both
   `serialize()` and `_serialize_block()`'s sub_lines.
2. Legacy migration: when loading, any line whose dstblock is a From
   block is a virtual line by construction (From has no real input
   ports) — tag it hidden=True so link_goto_from cleans it up.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def file_service_with_lines(qapp, tmp_path):
    """Create a FileService backed by a model with Goto/From + virtual line."""
    from lib.services.file_service import FileService
    from lib.simulation.connection import DLine

    # Minimal model with the relevant attributes.
    blocks = [
        SimpleNamespace(
            name="step__0", block_fn="Step", username="src", sid=0,
            left=0, top=0, width=50, height=40, height_base=40,
            in_ports=0, out_ports=1, dragging=False, selected=False,
            b_color=SimpleNamespace(name=lambda: "#000000"),
            b_type=0, io_edit="none", fn_name="step",
            saving_params=lambda: {},
            external=False, flipped=False, sub_blocks=[], sub_lines=[],
            ports={}, ports_map={},
        ),
        SimpleNamespace(
            name="goto__1", block_fn="Goto", username="g", sid=1,
            left=100, top=0, width=50, height=40, height_base=40,
            in_ports=1, out_ports=0, dragging=False, selected=False,
            b_color=SimpleNamespace(name=lambda: "#000000"),
            b_type=2, io_edit="none", fn_name="goto",
            saving_params=lambda: {"tag": "A"},
            external=False, flipped=False,
        ),
        SimpleNamespace(
            name="from__2", block_fn="From", username="f", sid=2,
            left=200, top=0, width=50, height=40, height_base=40,
            in_ports=0, out_ports=1, dragging=False, selected=False,
            b_color=SimpleNamespace(name=lambda: "#000000"),
            b_type=2, io_edit="none", fn_name="from_block",
            saving_params=lambda: {"tag": "A"},
            external=False, flipped=False,
        ),
    ]

    from PyQt5.QtCore import QPoint
    real_line = DLine(0, "step__0", 0, "goto__1", 0,
                      [QPoint(50, 0), QPoint(100, 0)])
    virtual_line = DLine(1, "step__0", 0, "from__2", 0,
                         [QPoint(50, 0), QPoint(200, 0)],
                         hidden=True)
    lines = [real_line, virtual_line]

    model = SimpleNamespace(
        blocks_list=blocks, line_list=lines, dirty=True,
        update_lines=lambda: None, link_goto_from=lambda: None,
    )

    fs = FileService(model)
    return fs, model, real_line, virtual_line


@pytest.mark.regression
class TestVirtualLineNotPersisted:

    def test_serialize_filters_hidden_lines(self, file_service_with_lines):
        """serialize() must skip lines with hidden=True."""
        fs, model, real_line, virtual_line = file_service_with_lines

        data = fs.serialize(sim_params={"sim_time": 1.0, "sim_dt": 0.01})

        line_sids = {ln["sid"] for ln in data["lines_data"]}
        assert real_line.sid in line_sids, (
            "Visible line must be serialized."
        )
        assert virtual_line.sid not in line_sids, (
            "Hidden virtual line must NOT be serialized — it gets re-created "
            "by link_goto_from at runtime."
        )

    def test_save_to_file_and_reload_omits_virtual_line(
        self, file_service_with_lines, tmp_path
    ):
        """End-to-end: write to disk, read raw JSON, ensure virtual line gone."""
        import json
        fs, model, _, virtual_line = file_service_with_lines
        path = tmp_path / "test.diablos"

        data = fs.serialize(sim_params={"sim_time": 1.0, "sim_dt": 0.01})
        fs.save_to_file(data, str(path))

        with open(path) as fp:
            raw = json.load(fp)

        # No line in the saved file should target the From block (it's the
        # virtual-line signature).
        from_dst_lines = [
            ln for ln in raw["lines_data"]
            if ln["dstblock"] == "from__2"
        ]
        assert not from_dst_lines, (
            f"Expected no lines targeting From block on disk, found "
            f"{from_dst_lines}"
        )

    def test_subsystem_sub_lines_also_filtered(self, qapp):
        """_serialize_block must filter hidden lines in sub_lines too."""
        from lib.services.file_service import FileService
        from lib.simulation.connection import DLine
        from PyQt5.QtCore import QPoint

        hidden_sub = DLine(0, "a", 0, "from_inside", 0,
                           [QPoint(0, 0), QPoint(10, 10)], hidden=True)
        visible_sub = DLine(1, "a", 0, "b", 0,
                            [QPoint(0, 0), QPoint(10, 10)])

        subsystem = SimpleNamespace(
            name="sub__0", block_fn="Subsystem", username="s", sid=0,
            left=0, top=0, width=50, height=40, height_base=40,
            in_ports=1, out_ports=1, dragging=False, selected=False,
            b_color=SimpleNamespace(name=lambda: "#000000"),
            b_type=2, io_edit="none", fn_name="subsystem",
            saving_params=lambda: {},
            external=False, flipped=False,
            sub_blocks=[],
            sub_lines=[hidden_sub, visible_sub],
            ports={}, ports_map={},
        )

        model = SimpleNamespace(
            blocks_list=[subsystem], line_list=[], dirty=False,
            update_lines=lambda: None,
        )
        fs = FileService(model)
        block_dict = fs._serialize_block(subsystem)

        sub_sids = {ln["sid"] for ln in block_dict["sub_lines"]}
        assert visible_sub.sid in sub_sids
        assert hidden_sub.sid not in sub_sids, (
            "Hidden line in sub_lines must NOT be persisted."
        )


@pytest.mark.regression
class TestLegacyGhostLineMigration:
    """
    Pre-fix .diablos files persist virtual Goto/From lines as ordinary
    visible lines.  Loading them must mark those lines hidden=True so
    link_goto_from cleans them up on the next execution_init.
    """

    def test_legacy_line_to_from_block_marked_hidden_on_load(self, qapp, tmp_path):
        """
        Construct a file containing a ghost line (no hidden flag) whose
        dst is a From block.  After load, that line must be hidden=True.
        """
        import json
        from lib.lib import DSim
        from lib.workspace import WorkspaceManager

        # Pre-fix-shaped JSON: virtual line stored as ordinary line.
        legacy = {
            "sim_data": {
                "wind_width": 1280, "wind_height": 770, "fps": 60,
                "sim_time": 1.0, "sim_dt": 0.01, "sim_trange": 100,
            },
            "blocks_data": [
                {
                    "block_fn": "Step", "sid": 0, "username": "src",
                    "coords_left": 50, "coords_top": 200,
                    "coords_width": 50, "coords_height": 40,
                    "coords_height_base": 40,
                    "in_ports": 0, "out_ports": 1,
                    "dragging": False, "selected": False,
                    "b_color": "#000000", "b_type": 0,
                    "io_edit": "none", "fn_name": "step",
                    "params": {"value": 1.0, "delay": 0.0, "type": "up"},
                    "external": False, "flipped": False,
                },
                {
                    "block_fn": "Goto", "sid": 1, "username": "g",
                    "coords_left": 200, "coords_top": 200,
                    "coords_width": 50, "coords_height": 40,
                    "coords_height_base": 40,
                    "in_ports": 1, "out_ports": 0,
                    "dragging": False, "selected": False,
                    "b_color": "#000000", "b_type": 2,
                    "io_edit": "none", "fn_name": "goto",
                    "params": {"tag": "A"},
                    "external": False, "flipped": False,
                },
                {
                    "block_fn": "From", "sid": 2, "username": "f",
                    "coords_left": 400, "coords_top": 200,
                    "coords_width": 50, "coords_height": 40,
                    "coords_height_base": 40,
                    "in_ports": 0, "out_ports": 1,
                    "dragging": False, "selected": False,
                    "b_color": "#000000", "b_type": 2,
                    "io_edit": "none", "fn_name": "from_block",
                    "params": {"tag": "A"},
                    "external": False, "flipped": False,
                },
            ],
            "lines_data": [
                # Real line: Step → Goto
                {
                    "name": "line0", "sid": 0,
                    "srcblock": "step0", "srcport": 0,
                    "dstblock": "goto1", "dstport": 0,
                    "points": [[100, 220], [200, 220]],
                    "cptr": 0, "selected": False,
                },
                # Ghost virtual line: Step → From (this is the bug
                # signature — From has no real input ports).
                {
                    "name": "line1", "sid": 1,
                    "srcblock": "step0", "srcport": 0,
                    "dstblock": "from2", "dstport": 0,
                    "points": [[100, 220], [400, 220]],
                    "cptr": 0, "selected": False,
                },
            ],
            "version": "2.0",
        }
        path = tmp_path / "legacy.diablos"
        with open(path, "w") as fp:
            json.dump(legacy, fp)

        prev = WorkspaceManager._instance
        WorkspaceManager._instance = None
        try:
            dsim = DSim()
            data = dsim.file_service.load(filepath=str(path))
            dsim.file_service.apply_loaded_data(data)

            from_lines = [
                ln for ln in dsim.model.line_list
                if ln.dstblock == "from2"
            ]
            assert from_lines, "Test setup: expected a line targeting From."
            for ln in from_lines:
                assert ln.hidden, (
                    f"Legacy line {ln.name} targeting From block should "
                    f"have been marked hidden=True on load."
                )

            # The real Step→Goto line must NOT be tagged hidden.
            goto_lines = [
                ln for ln in dsim.model.line_list
                if ln.dstblock == "goto1"
            ]
            assert goto_lines, "Test setup: expected a line targeting Goto."
            for ln in goto_lines:
                assert not ln.hidden, (
                    f"Real line {ln.name} targeting Goto block must not "
                    f"be marked hidden — only From-targeted lines are virtual."
                )
        finally:
            WorkspaceManager._instance = prev

    def test_round_trip_strips_legacy_ghost_lines(self, qapp, tmp_path):
        """
        Load a legacy file containing a ghost line, then re-save and
        re-load — the ghost line must be gone.
        """
        import json
        from lib.lib import DSim
        from lib.workspace import WorkspaceManager

        legacy = {
            "sim_data": {
                "wind_width": 1280, "wind_height": 770, "fps": 60,
                "sim_time": 1.0, "sim_dt": 0.01, "sim_trange": 100,
            },
            "blocks_data": [
                {
                    "block_fn": "From", "sid": 0, "username": "f",
                    "coords_left": 0, "coords_top": 0,
                    "coords_width": 50, "coords_height": 40,
                    "coords_height_base": 40,
                    "in_ports": 0, "out_ports": 1,
                    "dragging": False, "selected": False,
                    "b_color": "#000000", "b_type": 2,
                    "io_edit": "none", "fn_name": "from_block",
                    "params": {"tag": "A"},
                    "external": False, "flipped": False,
                },
            ],
            "lines_data": [
                {
                    "name": "line0", "sid": 0,
                    "srcblock": "anywhere", "srcport": 0,
                    "dstblock": "from0", "dstport": 0,
                    "points": [[0, 0], [50, 0]],
                    "cptr": 0, "selected": False,
                },
            ],
            "version": "2.0",
        }
        in_path = tmp_path / "legacy.diablos"
        out_path = tmp_path / "saved.diablos"
        with open(in_path, "w") as fp:
            json.dump(legacy, fp)

        prev = WorkspaceManager._instance
        WorkspaceManager._instance = None
        try:
            dsim = DSim()
            data = dsim.file_service.load(filepath=str(in_path))
            dsim.file_service.apply_loaded_data(data)

            # Re-serialize — ghost line should be filtered out by the
            # serialize-time guard since the legacy migration tagged it.
            out_data = dsim.file_service.serialize(
                sim_params={"sim_time": 1.0, "sim_dt": 0.01}
            )
            assert out_data["lines_data"] == [], (
                f"Expected no lines after round-trip; got "
                f"{out_data['lines_data']}"
            )
        finally:
            WorkspaceManager._instance = prev
