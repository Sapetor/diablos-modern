"""
Pasting a Subsystem must keep the copied connections to and from it.

``_build_subsystem`` rebuilds the Subsystem from the clipboard entry; before the
fix it left the block with 0 in/out ports and empty ``in_coords``/``out_coords``,
so ``_resolve_endpoints`` skipped every line touching it.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QRect

from blocks.subsystem import Subsystem
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from modern_ui.managers.clipboard_manager import ClipboardManager


@pytest.fixture(autouse=True)
def _qt(qapp):
    """DBlock builds a QPixmap, which needs a live QApplication."""
    return qapp


def _dblock(block_fn, sid, x, y, in_ports, out_ports, category):
    block = DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(x, y, 60, 40),
        color="#4CAF50",
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=2,
        io_edit=False,
        fn_name=block_fn,
        params={},
        external=False,
        username="",
        block_class=None,
        colors=None,
        category=category,
    )
    block.selected = True
    return block


def _port(kind, index, total, subsys):
    x = 0 if kind == "input" else subsys.width
    return {"pos": (x, subsys.height / (total + 1) * index), "type": kind, "name": str(index)}


def _subsystem(sid, x, y, n_in=1, n_out=1, sub_blocks=(), sub_lines=()):
    """A Subsystem the way SubsystemManager leaves it: ``ports`` filled, geometry updated."""
    subsys = Subsystem(block_name=f"Subsystem{sid}", sid=sid, coords=QRect(x, y, 100, 80))
    subsys.ports = {
        "in": [_port("input", i + 1, n_in, subsys) for i in range(n_in)],
        "out": [_port("output", i + 1, n_out, subsys) for i in range(n_out)],
    }
    subsys.ports_map = {
        "in": {i: f"inport{i + 1}" for i in range(n_in)},
        "out": {i: f"outport{i + 1}" for i in range(n_out)},
    }
    subsys.sub_blocks = list(sub_blocks)
    subsys.sub_lines = list(sub_lines)
    subsys.update_Block()
    subsys.selected = True
    return subsys


def _line(sid, src, srcport, dst, dstport):
    return DLine(
        sid=sid,
        srcblock=src.name,
        srcport=srcport,
        dstblock=dst.name,
        dstport=dstport,
        points=[src.out_coords[srcport], dst.in_coords[dstport]],
    )


def _manager(blocks, lines):
    dsim = SimpleNamespace(
        blocks_list=list(blocks), line_list=list(lines), menu_blocks=[], colors=None, dirty=False
    )
    canvas = SimpleNamespace(
        dsim=dsim,
        update=lambda: None,
        block_selected=SimpleNamespace(emit=lambda *_: None),
        simulation_status_changed=SimpleNamespace(emitted=[]),
        history_manager=SimpleNamespace(
            capture_snapshot=lambda: {"pre": True}, push_snapshot=lambda *_: None
        ),
    )
    canvas.simulation_status_changed.emit = canvas.simulation_status_changed.emitted.append
    return ClipboardManager(canvas), canvas


def _chain(n_in=1, n_out=1, inner=()):
    """Step -> Subsystem -> Scope with both lines, all selected."""
    step = _dblock("Step", 0, 100, 120, 0, 1, "Sources")
    subsys = _subsystem(0, 250, 100, n_in, n_out, sub_blocks=inner)
    scope = _dblock("Scope", 0, 450, 120, 1, 0, "Sinks")
    lines = [_line(0, step, 0, subsys, 0), _line(1, subsys, 0, scope, 0)]
    return [step, subsys, scope], lines


def _copy_paste(blocks, lines, pos=None):
    manager, canvas = _manager(blocks, lines)
    manager.copy_selected_blocks()
    manager.paste_blocks(pos)
    assert canvas.simulation_status_changed.emitted == []
    pasted_blocks = canvas.dsim.blocks_list[len(blocks) :]
    pasted_lines = canvas.dsim.line_list[len(lines) :]
    return canvas, pasted_blocks, pasted_lines


@pytest.mark.unit
class TestPasteSubsystemConnections:
    def test_both_lines_are_recreated_and_point_at_pasted_blocks(self):
        blocks, lines = _chain()
        _, pasted, new_lines = _copy_paste(blocks, lines)

        assert [b.name for b in pasted] == ["step1", "subsystem1", "scope1"]
        assert len(new_lines) == 2, "lines touching the pasted Subsystem were dropped"
        assert [(ln.srcblock, ln.srcport, ln.dstblock, ln.dstport) for ln in new_lines] == [
            ("step1", 0, "subsystem1", 0),
            ("subsystem1", 0, "scope1", 0),
        ]
        assert [ln.sid for ln in new_lines] == [2, 3]

    def test_pasted_subsystem_has_the_copied_port_geometry(self):
        blocks, lines = _chain(n_in=2, n_out=3)
        original = blocks[1]
        _, pasted, _ = _copy_paste(blocks, lines)
        subsys = pasted[1]

        assert isinstance(subsys, Subsystem)
        assert (subsys.in_ports, subsys.out_ports) == (2, 3)
        assert len(subsys.in_coords) == len(original.in_coords) == 2
        assert len(subsys.out_coords) == len(original.out_coords) == 3
        # Same layout as the original, shifted by the keyboard paste offset.
        shift = QPoint(30, 30)
        assert subsys.in_coords == [p + shift for p in original.in_coords]
        assert subsys.out_coords == [p + shift for p in original.out_coords]
        assert subsys.ports["in"][0]["name"] == original.ports["in"][0]["name"]
        assert subsys.ports_map == original.ports_map
        assert subsys.ports is not original.ports  # deep-copied, not shared

    def test_subsystem_without_ports_dict_falls_back_to_port_counts(self):
        """A Subsystem with counts but no ``ports`` dict still gets DBlock geometry."""
        blocks, lines = _chain()
        blocks[1].ports = {}
        blocks[1].in_ports, blocks[1].out_ports = 1, 1
        blocks[1].update_Block()
        _, pasted, new_lines = _copy_paste(blocks, lines)

        assert (pasted[1].in_ports, pasted[1].out_ports) == (1, 1)
        assert len(pasted[1].in_coords) == 1 and len(pasted[1].out_coords) == 1
        assert len(new_lines) == 2

    def test_line_endpoints_sit_on_the_pasted_ports(self):
        blocks, lines = _chain()
        _, pasted, new_lines = _copy_paste(blocks, lines)
        step, subsys, scope = pasted

        assert new_lines[0].points[0] == step.out_coords[0]
        assert new_lines[0].points[-1] == subsys.in_coords[0]
        assert new_lines[1].points[0] == subsys.out_coords[0]
        assert new_lines[1].points[-1] == scope.in_coords[0]

    def test_nested_subsystem_keeps_inner_blocks_lines_and_ports(self):
        gain = _dblock("Gain", 0, 40, 40, 1, 1, "Math")
        inner_line_src = _dblock("Inport", 0, -60, 40, 0, 1, "Routing")
        inner = _subsystem(5, 300, 300, 1, 1, sub_blocks=[inner_line_src, gain], sub_lines=[])
        inner.sub_lines = [_line(0, inner_line_src, 0, gain, 0)]
        blocks, lines = _chain(inner=[inner])
        _, pasted, new_lines = _copy_paste(blocks, lines)

        outer = pasted[1]
        assert len(new_lines) == 2
        assert len(outer.sub_blocks) == 1
        nested = outer.sub_blocks[0]
        assert isinstance(nested, Subsystem) and nested is not inner
        assert nested.name == "subsystem5"
        assert [b.name for b in nested.sub_blocks] == ["inport0", "gain0"]
        assert len(nested.sub_lines) == 1
        assert (nested.sub_lines[0].srcblock, nested.sub_lines[0].dstblock) == ("inport0", "gain0")
        assert len(nested.in_coords) == 1 and len(nested.out_coords) == 1
        # The original's contents are untouched by the paste.
        assert blocks[1].sub_blocks == [inner]

    def test_paste_at_pos_offsets_blocks_ports_and_lines_together(self):
        blocks, lines = _chain()
        pos = QPoint(600, 400)
        _, pasted, new_lines = _copy_paste(blocks, lines, pos=pos)
        shift = pos - QPoint(blocks[0].left, blocks[0].top)

        assert (pasted[0].left, pasted[0].top) == (600, 400)
        for original, copy_ in zip(blocks, pasted):
            assert (copy_.left, copy_.top) == (original.left + shift.x(), original.top + shift.y())
            assert copy_.in_coords == [p + shift for p in original.in_coords]
            assert copy_.out_coords == [p + shift for p in original.out_coords]
        assert len(new_lines) == 2
        assert new_lines[0].points[-1] == pasted[1].in_coords[0]
        assert new_lines[1].points[0] == pasted[1].out_coords[0]
