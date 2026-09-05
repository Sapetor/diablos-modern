"""Tests for ConnectionManager wire relayout scoping and validator failure.

Two behaviours are pinned here:

* ``update_line_positions(block_names)`` must touch only the wires attached to
  those blocks. It runs once per mouse-move during a drag, so relaying out the
  whole diagram made drag cost scale with diagram size. The no-argument form
  (used by load/paste/undo/subsystem edits) must still refresh everything.
* A crash inside ``ValidationHelper`` must REJECT the connection. It used to be
  downgraded to a warning and the unvalidated wire was accepted.
"""

import pytest
from PyQt5.QtCore import QPoint, QRect

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

pytestmark = pytest.mark.qt


def _make_block(sid, x, y, name):
    block = DBlock(
        block_fn="Gain",
        sid=sid,
        coords=QRect(x, y, 100, 80),
        color="#4CAF50",
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit="none",
        fn_name="gain",
        params={"gain": 1.0},
        external=False,
        colors=None,
    )
    block.name = name
    return block


@pytest.fixture
def canvas(qapp):
    from modern_ui.widgets.modern_canvas import ModernCanvas

    dsim = DSim()
    widget = ModernCanvas(dsim)
    widget.resize(800, 600)
    return widget


def _populate(canvas):
    """A -> B and C -> D: two disjoint wires so scoping is observable."""
    a = _make_block(0, 50, 50, "A")
    b = _make_block(1, 300, 50, "B")
    c = _make_block(2, 50, 300, "C")
    d = _make_block(3, 300, 300, "D")
    canvas.dsim.blocks_list.extend([a, b, c, d])
    ab = DLine(0, "A", 0, "B", 0, [QPoint(150, 90), QPoint(300, 90)])
    cd = DLine(1, "C", 0, "D", 0, [QPoint(150, 340), QPoint(300, 340)])
    canvas.dsim.line_list.extend([ab, cd])
    return a, b, c, d, ab, cd


class TestUpdateLinePositionsScoping:
    def test_only_named_blocks_wires_are_relaid(self, canvas):
        _a, _b, _c, _d, ab, cd = _populate(canvas)
        updated = []
        for line in (ab, cd):
            line.update_line = lambda blocks, ln=line: updated.append(ln.name)

        canvas.connection_manager.update_line_positions({"A"})
        assert updated == [ab.name]

    def test_dst_side_also_matches(self, canvas):
        _a, _b, _c, _d, ab, cd = _populate(canvas)
        updated = []
        for line in (ab, cd):
            line.update_line = lambda blocks, ln=line: updated.append(ln.name)

        canvas.connection_manager.update_line_positions({"D"})
        assert updated == [cd.name]

    def test_no_argument_refreshes_every_wire(self, canvas):
        _a, _b, _c, _d, ab, cd = _populate(canvas)
        updated = []
        for line in (ab, cd):
            line.update_line = lambda blocks, ln=line: updated.append(ln.name)

        canvas.connection_manager.update_line_positions()
        assert sorted(updated) == sorted([ab.name, cd.name])

    def test_unknown_name_updates_nothing(self, canvas):
        _a, _b, _c, _d, ab, cd = _populate(canvas)
        updated = []
        for line in (ab, cd):
            line.update_line = lambda blocks, ln=line: updated.append(ln.name)

        canvas.connection_manager.update_line_positions({"nope"})
        assert updated == []

    def test_dragging_a_block_still_moves_its_wire_endpoint(self, canvas):
        """End-to-end: the scoped path must produce the same geometry."""
        a, _b, _c, _d, ab, _cd = _populate(canvas)
        canvas.connection_manager.update_line_positions()
        before = QPoint(ab.points[0])

        a.relocate_Block(QPoint(a.left, a.top + 60))
        canvas.connection_manager.update_line_positions({"A"})
        scoped = QPoint(ab.points[0])
        assert scoped != before

        # A full refresh from the same state must agree with the scoped one.
        canvas.connection_manager.update_line_positions()
        assert QPoint(ab.points[0]) == scoped

    def test_canvas_hook_forwards_the_block_names(self, canvas):
        seen = {}
        canvas.connection_manager.update_line_positions = lambda names=None: seen.setdefault(
            "names", names
        )
        canvas._update_line_positions({"A", "B"})
        assert seen["names"] == {"A", "B"}

    def test_canvas_hook_defaults_to_full_refresh(self, canvas):
        seen = {}
        canvas.connection_manager.update_line_positions = lambda names=None: seen.setdefault(
            "names", names
        )
        canvas._update_line_positions()
        assert seen["names"] is None


class TestValidatorCrashRejectsConnection:
    def test_validator_exception_rejects(self, canvas, monkeypatch):
        a, b, _c, _d, _ab, _cd = _populate(canvas)
        canvas.dsim.line_list.clear()

        import modern_ui.managers.connection_manager as cm

        def boom(blocks, lines):
            raise RuntimeError("validator is broken")

        monkeypatch.setattr(cm.ValidationHelper, "validate_block_connections", staticmethod(boom))

        ok, errors = canvas.connection_manager.validate_connection(a, 0, b, 0)
        assert ok is False
        # Specifically the validator-crash branch, not the outer catch-all.
        assert any(e.startswith("Connection validator failed") for e in errors)

    def test_missing_validator_is_still_tolerated(self, canvas, monkeypatch):
        """An absent helper method is a build difference, not a broken wire."""
        a, b, _c, _d, _ab, _cd = _populate(canvas)
        canvas.dsim.line_list.clear()

        import modern_ui.managers.connection_manager as cm

        def absent(blocks, lines):
            raise AttributeError("validate_block_connections")

        monkeypatch.setattr(cm.ValidationHelper, "validate_block_connections", staticmethod(absent))

        ok, errors = canvas.connection_manager.validate_connection(a, 0, b, 0)
        assert ok is True
        assert errors == []
