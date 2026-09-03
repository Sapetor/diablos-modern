"""Wire creation gestures, mode-aware re-routing and undo of wire edits.

Drives a real ModernCanvas/DSim (as the other tests here do) through the
ConnectionManager API the mouse handlers call, so the gestures are exercised
without synthesising Qt mouse events:

* press on an output port, drag, release on an input port -> wire;
* click-click (press+release on the source, later click on the target);
* reverse creation from a free input port;
* release on empty canvas after a real drag cancels;
* a block move keeps bezier wires curved, re-routes auto-routed orthogonal
  wires and leaves hand-bent wires alone;
* undo/redo restores hand-made bends, and a block move can be undone with a
  single Ctrl+Z.
"""

import pytest
from PyQt5.QtCore import QPoint

from lib.lib import DSim

pytestmark = pytest.mark.qt


@pytest.fixture
def canvas(qapp):
    from modern_ui.widgets.modern_canvas import ModernCanvas

    dsim = DSim()
    c = ModernCanvas(dsim)
    c.resize(800, 600)
    return c


def _menu(dsim):
    return {mb.block_fn: mb for mb in dsim.menu_blocks}


def _add(canvas, fn, x, y):
    return canvas.dsim.add_block(_menu(canvas.dsim)[fn], QPoint(x, y))


def _is_cubic(line):
    return any(line.path.elementAt(i).isCurveTo() for i in range(line.path.elementCount()))


def _is_axis_aligned(points):
    return all(a.x() == b.x() or a.y() == b.y() for a, b in zip(points, points[1:]))


@pytest.fixture
def pair(canvas):
    step = _add(canvas, "Step", 100, 100)
    gain = _add(canvas, "Gain", 400, 150)
    return canvas, step, gain


class TestCreationGestures:
    def test_drag_to_connect(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        assert cm.connection_state.creation_state == "start"
        cm.note_drag(QPoint(250, 130))
        assert cm.try_finish_at(gain.in_coords[0])
        assert cm.connection_state.creation_state is None
        assert len(canvas.dsim.line_list) == 1
        line = canvas.dsim.line_list[0]
        assert (line.srcblock, line.dstblock) == (step.name, gain.name)

    def test_click_click_still_works(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        # Release at the press position: still pending.
        cm.try_finish_at(step.out_coords[0])
        assert cm.connection_state.creation_state == "start"
        cm.note_drag(QPoint(300, 300))
        cm.handle_port_click(gain, "i", 0, gain.in_coords[0])
        assert cm.connection_state.creation_state is None
        assert len(canvas.dsim.line_list) == 1

    def test_release_on_empty_canvas_after_drag_cancels(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        cm.note_drag(QPoint(300, 300))
        cm.try_finish_at(QPoint(300, 300))
        assert cm.connection_state.creation_state is None
        assert canvas.dsim.line_list == []

    def test_reverse_creation_from_free_input(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(gain, "i", 0, gain.in_coords[0])
        assert cm.connection_state.creation_state == "start"
        assert cm.connection_state.reverse
        cm.note_drag(QPoint(200, 120))
        assert cm.try_finish_at(step.out_coords[0])
        line = canvas.dsim.line_list[0]
        assert (line.srcblock, line.dstblock) == (step.name, gain.name)
        assert line.points[0] == step.out_coords[0]
        assert line.points[-1] == gain.in_coords[0]

    def test_connected_input_does_not_start_a_wire(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        cm.try_finish_at(gain.in_coords[0])
        assert len(canvas.dsim.line_list) == 1
        cm.handle_port_click(gain, "i", 0, gain.in_coords[0])
        assert cm.connection_state.creation_state is None

    def test_target_validity(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        assert cm.target_validity(gain, "i", 0) is True
        assert cm.target_validity(gain, "o", 0) is None  # wrong kind
        other = _add(canvas, "Scope", 400, 400)
        cm.try_finish_at(gain.in_coords[0])
        # Hovering an already-wired input is a visible "no".
        cm.handle_port_click(other, "o", 0, other.out_coords[0]) if other.out_coords else None
        cm.cancel_line_creation()
        src2 = _add(canvas, "Sine", 100, 300)
        cm.handle_port_click(src2, "o", 0, src2.out_coords[0])
        assert cm.target_validity(gain, "i", 0) is False

    def test_new_wire_is_routed_for_default_mode(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_state = canvas.connection_manager.connection_state
        cm.default_routing_mode = "orthogonal"
        canvas.connection_manager.handle_port_click(step, "o", 0, step.out_coords[0])
        canvas.connection_manager.try_finish_at(gain.in_coords[0])
        line = canvas.dsim.line_list[0]
        assert line.routing_mode == "orthogonal"
        assert line.auto_routed
        assert _is_axis_aligned(line.points)

        cm.default_routing_mode = "bezier"
        scope = _add(canvas, "Scope", 700, 150)
        canvas.connection_manager.handle_port_click(gain, "o", 0, gain.out_coords[0])
        canvas.connection_manager.try_finish_at(scope.in_coords[0])
        line2 = canvas.dsim.line_list[-1]
        assert line2.routing_mode == "bezier"
        assert _is_cubic(line2)
        assert not line2.modified


class TestModeAwareReroute:
    def _connect(self, canvas, src, dst, mode):
        cm = canvas.connection_manager
        cm.connection_state.default_routing_mode = mode
        cm.handle_port_click(src, "o", 0, src.out_coords[0])
        cm.try_finish_at(dst.in_coords[0])
        return canvas.dsim.line_list[-1]

    def _move(self, canvas, block, dx, dy):
        block.relocate_Block(QPoint(block.left + dx, block.top + dy))
        canvas._update_line_positions()
        canvas._reroute_affected_lines({block.name})

    def test_bezier_wire_stays_curved_after_move(self, pair):
        canvas, step, gain = pair
        line = self._connect(canvas, step, gain, "bezier")
        self._move(canvas, gain, 0, 120)
        assert _is_cubic(line)
        assert line.points == [step.out_coords[0], gain.in_coords[0]]
        assert not line.modified and not line.auto_routed

    def test_auto_routed_orthogonal_wire_follows_block(self, pair):
        canvas, step, gain = pair
        line = self._connect(canvas, step, gain, "orthogonal")
        assert line.auto_routed
        self._move(canvas, gain, 0, 120)
        assert line.auto_routed
        assert line.points[-1] == gain.in_coords[0]
        assert _is_axis_aligned(line.points)

    def test_hand_bent_wire_is_left_alone(self, pair):
        canvas, step, gain = pair
        line = self._connect(canvas, step, gain, "orthogonal")
        a, b = line.points[0], line.points[-1]
        bend = [
            a,
            QPoint(a.x() + 60, a.y()),
            QPoint(a.x() + 60, b.y() + 200),
            QPoint(b.x() - 30, b.y() + 200),
            QPoint(b.x() - 30, b.y()),
            b,
        ]
        line.points = bend
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(
            a, b, canvas.dsim.blocks_list, points=bend
        )
        self._move(canvas, gain, 0, 40)
        assert line.modified and not line.auto_routed
        assert len(line.points) == 6
        assert line.points[-1] == gain.in_coords[0]
        assert _is_axis_aligned(line.points)

    def test_reset_and_auto_route_from_manager(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        line = self._connect(canvas, step, gain, "bezier")
        cm.auto_route_line(line)
        assert line.routing_mode == "orthogonal" and line.auto_routed
        cm.reset_line_routing(line)
        assert not line.modified and not line.auto_routed
        assert _is_axis_aligned(line.points)

    def test_remove_waypoint(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        line = self._connect(canvas, step, gain, "bezier")
        a, b = line.points[0], line.points[-1]
        line.points = [a, QPoint(250, a.y()), QPoint(250, b.y()), b]
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(a, b, [], points=line.points)
        assert cm.remove_waypoint(line, 0) is False
        assert cm.remove_waypoint(line, 1)
        assert len(line.points) == 3
        assert cm.remove_waypoint(line, 1)
        # Back to the plain route for the mode.
        assert line.points == [a, b] and not line.modified


class TestUndo:
    def test_undo_restores_manual_bend(self, pair):
        canvas, step, gain = pair
        cm = canvas.connection_manager
        cm.handle_port_click(step, "o", 0, step.out_coords[0])
        cm.try_finish_at(gain.in_coords[0])
        line = canvas.dsim.line_list[0]
        a, b = line.points[0], line.points[-1]
        bent = [a, QPoint(250, a.y()), QPoint(250, b.y()), b]
        line.points = list(bent)
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(a, b, [], points=line.points)

        canvas._push_undo("Bend wire")
        line.reset_routing(canvas.dsim.blocks_list)
        assert line.points == [a, b]

        canvas.history_manager.undo()
        restored = canvas.dsim.line_list[0]
        assert restored.modified
        assert [(p.x(), p.y()) for p in restored.points] == [(p.x(), p.y()) for p in bent]

    def test_block_move_undoes_in_one_step(self, pair):
        canvas, step, gain = pair
        im = canvas.interaction_manager
        original = (gain.left, gain.top)
        gain.selected = True  # a click selects the block before the drag starts
        im.start_drag(gain, QPoint(gain.left + 10, gain.top + 10))
        gain.relocate_Block(QPoint(gain.left + 100, gain.top + 50))
        im._finish_drag()
        assert (gain.left, gain.top) != original

        canvas.history_manager.undo()
        moved_back = next(b for b in canvas.dsim.blocks_list if b.name == gain.name)
        assert (moved_back.left, moved_back.top) == original


class TestPreviewPath:
    def test_preview_follows_routing_mode(self):
        from modern_ui.renderers.canvas_renderer import CanvasRenderer

        start, end = QPoint(100, 100), QPoint(300, 200)
        bez = CanvasRenderer.preview_path(start, end, "bezier")
        ortho = CanvasRenderer.preview_path(start, end, "orthogonal")
        assert any(bez.elementAt(i).isCurveTo() for i in range(bez.elementCount()))
        assert not any(ortho.elementAt(i).isCurveTo() for i in range(ortho.elementCount()))

    def test_reverse_preview_starts_at_cursor(self):
        from modern_ui.renderers.canvas_renderer import CanvasRenderer

        port, cursor = QPoint(300, 200), QPoint(100, 100)
        path = CanvasRenderer.preview_path(port, cursor, "bezier", reverse=True)
        first = path.elementAt(0)
        assert (int(first.x), int(first.y)) == (100, 100)
