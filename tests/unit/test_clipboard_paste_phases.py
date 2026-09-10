"""
Unit tests for the paste phases of ClipboardManager.

``paste_blocks`` is an orchestrator over small helpers; these tests pin each
helper's contract with lightweight stubs so the orchestrator can stay short.
"""

import logging
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QRect

from lib.simulation.block import DBlock
from modern_ui.managers import clipboard_manager as cm
from modern_ui.managers.clipboard_manager import ClipboardManager


@pytest.fixture(autouse=True)
def _qt(qapp):
    """DBlock builds a QPixmap, which needs a live QApplication."""
    return qapp


def _stub_block(block_fn, name, sid=0, out_coords=(), in_coords=()):
    return SimpleNamespace(
        block_fn=block_fn,
        name=name,
        sid=sid,
        out_coords=list(out_coords),
        in_coords=list(in_coords),
        selected=True,
    )


def _real_block(block_fn, sid, x=100, y=100, in_ports=1, out_ports=1):
    return DBlock(
        block_fn=block_fn,
        sid=sid,
        coords=QRect(x, y, 60, 40),
        color="#4CAF50",
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=2,
        io_edit=False,
        fn_name=block_fn,
        params={"gain": 2.0},
        external=False,
        username="",
        block_class=None,
        colors=None,
        category="Math",
    )


def _manager(blocks=(), lines=(), menu_blocks=(), with_history=True):
    dsim = SimpleNamespace(
        blocks_list=list(blocks),
        line_list=list(lines),
        menu_blocks=list(menu_blocks),
        colors=None,
        dirty=False,
    )
    canvas = SimpleNamespace(
        dsim=dsim,
        update_calls=0,
        block_selected=SimpleNamespace(emitted=[]),
        simulation_status_changed=SimpleNamespace(emitted=[]),
    )
    canvas.update = lambda: setattr(canvas, "update_calls", canvas.update_calls + 1)
    canvas.block_selected.emit = canvas.block_selected.emitted.append
    canvas.simulation_status_changed.emit = canvas.simulation_status_changed.emitted.append
    if with_history:
        # Mirrors HistoryManager's capture_snapshot/push_snapshot pair: ``undo``
        # records the description of every entry that actually gets pushed.
        history = SimpleNamespace(undo=[], captured=0)
        history.capture_snapshot = lambda: (
            setattr(history, "captured", history.captured + 1) or {"pre": True}
        )
        history.push_snapshot = lambda state, description="Action": history.undo.append(description)
        canvas.history_manager = history
    return ClipboardManager(canvas), canvas


def _block_data(block_fn="Gain", x=250, y=100, **overrides):
    data = {
        "block_fn": block_fn,
        "coords": QRect(x, y, 60, 40),
        "color": "#2196F3",
        "category": "Math",
        "in_ports": 1,
        "out_ports": 1,
        "b_type": 2,
        "io_edit": False,
        "fn_name": block_fn,
        "params": {"gain": 2.0, "_name_": "gain0"},
        "external": False,
        "flipped": True,
    }
    data.update(overrides)
    return data


@pytest.mark.unit
class TestPasteOffset:
    def test_keyboard_paste_uses_fixed_offset(self):
        offset = cm._paste_offset(None, QRect(100, 100, 60, 40))
        assert offset == QPoint(cm.KEYBOARD_PASTE_OFFSET, cm.KEYBOARD_PASTE_OFFSET)
        assert cm.KEYBOARD_PASTE_OFFSET == 30

    def test_explicit_pos_lands_first_block_on_it(self):
        first = QRect(100, 120, 60, 40)
        offset = cm._paste_offset(QPoint(510, 375), first)
        assert first.translated(offset).topLeft() == QPoint(510, 375)


@pytest.mark.unit
class TestIdAllocation:
    def test_block_sid_is_max_plus_one_per_block_fn(self):
        blocks = [
            _stub_block("Gain", "gain0"),
            _stub_block("Gain", "gain5"),
            _stub_block("Constant", "constant9"),
        ]
        assert cm._next_block_sid(blocks, "Gain") == 6
        assert cm._next_block_sid(blocks, "Constant") == 10

    def test_block_sid_starts_at_zero_when_type_absent(self):
        assert cm._next_block_sid([_stub_block("Gain", "gain0")], "Scope") == 0
        assert cm._next_block_sid([], "Gain") == 0

    def test_line_sid_fills_after_highest_existing(self):
        assert cm._next_line_sid([SimpleNamespace(sid=0), SimpleNamespace(sid=7)]) == 8
        assert cm._next_line_sid([]) == 0


@pytest.mark.unit
class TestFindBlockClass:
    def test_returns_matching_palette_class(self):
        class GainImpl:
            pass

        menu = [
            SimpleNamespace(block_fn="Scope", block_class=None),
            SimpleNamespace(block_fn="Gain", block_class=GainImpl),
        ]
        assert cm._find_block_class(menu, "Gain") is GainImpl
        assert cm._find_block_class(menu, "Scope") is None

    def test_unknown_block_fn_gives_none(self):
        assert cm._find_block_class([], "Gain") is None


@pytest.mark.unit
class TestUserParamKeys:
    def test_drops_dunder_keys_only(self):
        params = {"_name_": "s", "K": 1.0, "_mask": {}, "_library_ref": "x", "_inputs_": 0}
        assert cm._user_param_keys(params) == ["K", "_mask", "_library_ref"]


@pytest.mark.unit
class TestResolveEndpoints:
    def _pasted(self):
        src = _stub_block("Constant", "constant1", out_coords=[QPoint(160, 120)])
        dst = _stub_block("Gain", "gain1", in_coords=[QPoint(250, 120)])
        return [src, dst]

    def test_valid_connection_resolves_to_blocks_and_ports(self):
        pasted = self._pasted()
        conn = {"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 0}
        assert cm._resolve_endpoints(conn, pasted) == (pasted[0], 0, pasted[1], 0)

    @pytest.mark.parametrize(
        "conn, fragment",
        [
            ({"start_index": 5, "start_port": 0, "end_index": 1, "end_port": 0}, "start_index 5"),
            ({"start_index": 0, "start_port": 0, "end_index": 5, "end_port": 0}, "end_index 5"),
            ({"start_index": 0, "start_port": 3, "end_index": 1, "end_port": 0}, "start_port 3"),
            ({"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 3}, "end_port 3"),
        ],
    )
    def test_out_of_range_index_is_skipped_with_warning(self, conn, fragment, caplog):
        with caplog.at_level(logging.WARNING, logger=cm.__name__):
            assert cm._resolve_endpoints(conn, self._pasted()) is None
        assert any(fragment in rec.getMessage() for rec in caplog.records)


@pytest.mark.unit
class TestInstantiateBlock:
    def test_dblock_gets_fresh_sid_new_name_and_copied_params(self):
        manager, canvas = _manager(blocks=[_real_block("Gain", 0), _real_block("Gain", 3)])
        data = _block_data()
        block = manager._instantiate_block(data, QRect(280, 130, 60, 40))
        assert type(block) is DBlock
        assert (block.name, block.sid, block.username) == ("gain4", 4, "gain4")
        assert block.rect == QRect(280, 130, 60, 40)
        assert block.params["gain"] == 2.0 and block.params["_name_"] == "gain4"
        assert block.params is not data["params"]
        assert block.category == "Math"

    def test_dblock_class_comes_from_palette(self):
        class GainImpl:
            pass

        menu = [SimpleNamespace(block_fn="Gain", block_class=GainImpl)]
        manager, _ = _manager(menu_blocks=menu)
        block = manager._instantiate_block(_block_data(), QRect(0, 0, 60, 40))
        assert isinstance(block.block_instance, GainImpl)

    def test_subsystem_restores_params_contents_and_init_params_list(self):
        from blocks.subsystem import Subsystem

        inner = _real_block("Inport", 0, in_ports=0)
        data = _block_data(
            "Subsystem",
            params={"_name_": "subsystem0", "K": 4.0, "_mask": {"name": "M"}},
            sub_blocks=[inner],
            sub_lines=[],
            ports={"in": [{"name": "in1", "pos": (0, 40)}], "out": []},
            ports_map={0: ("Inport0", "in")},
            category="Routing",
        )
        manager, _ = _manager()
        block = manager._instantiate_block(data, QRect(330, 230, 100, 80))
        assert isinstance(block, Subsystem)
        assert (block.name, block.username) == ("subsystem0", "Subsystem0")
        assert block.params["_name_"] == "subsystem0" and block.params["K"] == 4.0
        assert block.init_params_list == ["K", "_mask"]
        assert block.category == "Routing"
        assert [b.name for b in block.sub_blocks] == ["inport0"]
        assert block.sub_blocks[0] is not inner  # deep-copied, not shared
        assert block.ports_map == {0: ("Inport0", "in")}
        assert block.ports["in"][0]["name"] == "in1"

    def test_subsystem_without_contents_keeps_empty_structure(self):
        manager, _ = _manager()
        block = manager._instantiate_block(
            _block_data("Subsystem", params={}), QRect(0, 0, 100, 80)
        )
        assert block.sub_blocks == [] and block.sub_lines == [] and block.ports_map == {}


@pytest.mark.unit
class TestInstantiatePastedBlocks:
    def test_appends_in_order_selected_flipped_and_offset(self):
        manager, canvas = _manager(blocks=[_real_block("Gain", 0)])
        manager.clipboard_blocks = [
            _block_data("Gain", 250, 100, flipped=True),
            _block_data("Constant", 100, 100, flipped=False, in_ports=0),
        ]
        pasted = manager._instantiate_pasted_blocks(QPoint(30, 30))
        assert [b.name for b in pasted] == ["gain1", "constant0"]
        assert canvas.dsim.blocks_list[1:] == pasted
        assert [b.flipped for b in pasted] == [True, False]
        assert all(b.selected for b in pasted)
        assert [(b.left, b.top) for b in pasted] == [(280, 130), (130, 130)]

    def test_same_type_twice_gets_consecutive_sids(self):
        manager, _ = _manager()
        manager.clipboard_blocks = [_block_data("Gain"), _block_data("Gain")]
        pasted = manager._instantiate_pasted_blocks(QPoint(0, 0))
        assert [b.sid for b in pasted] == [0, 1]


@pytest.mark.unit
class TestRecreateConnections:
    def test_creates_lines_repointed_to_pasted_blocks(self):
        src, dst = (
            _real_block("Constant", 1, 130, 130, in_ports=0),
            _real_block("Gain", 1, 280, 130),
        )
        existing = SimpleNamespace(sid=4)
        manager, canvas = _manager(blocks=[src, dst], lines=[existing])
        manager.clipboard_connections = [
            {"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 0}
        ]
        manager._recreate_connections([src, dst])
        assert len(canvas.dsim.line_list) == 2
        line = canvas.dsim.line_list[-1]
        assert (line.sid, line.name) == (5, "Line5")
        assert (line.srcblock, line.srcport, line.dstblock, line.dstport) == (
            "constant1",
            0,
            "gain1",
            0,
        )
        assert line.points[0] == src.out_coords[0] and line.points[-1] == dst.in_coords[0]

    def test_bad_entries_are_skipped_and_never_raise(self):
        src, dst = _real_block("Constant", 1, in_ports=0), _real_block("Gain", 1)
        manager, canvas = _manager(blocks=[src, dst])
        manager.clipboard_connections = [
            {"start_index": 9, "start_port": 0, "end_index": 1, "end_port": 0},
            {"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 9},
            {"start_index": "x", "start_port": 0, "end_index": 1, "end_port": 0},
            {"start_index": 0, "start_port": 0, "end_index": 1, "end_port": 0},
        ]
        manager._recreate_connections([src, dst])
        assert [line.sid for line in canvas.dsim.line_list] == [0]


@pytest.mark.unit
class TestFinishPaste:
    def test_marks_dirty_redraws_and_announces_first_block(self):
        manager, canvas = _manager()
        first, second = _stub_block("Gain", "gain1"), _stub_block("Gain", "gain2")
        manager._finish_paste([first, second])
        assert canvas.dsim.dirty is True
        assert canvas.update_calls == 1
        assert canvas.block_selected.emitted == [first]


@pytest.mark.unit
class TestPasteBlocksOrchestration:
    def test_empty_clipboard_is_a_no_op(self):
        manager, canvas = _manager(blocks=[_real_block("Gain", 0)])
        manager.paste_blocks()
        assert canvas.history_manager.undo == []
        assert canvas.dsim.dirty is False and canvas.update_calls == 0

    def test_full_paste_deselects_originals_and_pushes_undo(self):
        original = _real_block("Gain", 0)
        original.selected = True
        manager, canvas = _manager(blocks=[original])
        manager.clipboard_blocks = [_block_data("Gain")]
        manager.paste_blocks()
        assert canvas.history_manager.undo == ["Paste"]
        assert original.selected is False
        assert [b.name for b in canvas.dsim.blocks_list] == ["gain0", "gain1"]
        assert canvas.dsim.blocks_list[1].selected is True
        assert canvas.dsim.dirty is True

    def test_without_history_manager_paste_still_works(self):
        manager, canvas = _manager(with_history=False)
        manager.clipboard_blocks = [_block_data("Gain")]
        manager.paste_blocks()
        assert [b.name for b in canvas.dsim.blocks_list] == ["gain0"]

    def test_failure_is_reported_on_status_signal(self):
        manager, canvas = _manager()
        manager.clipboard_blocks = [{"block_fn": "Gain"}]  # no coords -> KeyError
        manager.paste_blocks()
        emitted = canvas.simulation_status_changed.emitted
        assert len(emitted) == 1 and "'coords'" in emitted[0]
        assert canvas.dsim.dirty is False
        # The pre-paste snapshot was taken but never pushed: no dangling undo entry.
        assert canvas.history_manager.captured == 1
        assert canvas.history_manager.undo == []
