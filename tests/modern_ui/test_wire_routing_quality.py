"""Auto-routed wires on the bundled examples: clearance, port lead-ins,
no shared tracks between different signals, arrowheads and junction dots.

Before 2026-10-30 Auto-route on first_order_step_response ran the
reference -> scope branch 10 px in front of the plant's input, so the
plant wire's arrowhead and the branch corner overlapped into a "/" smudge;
the 20 px port stub left the scope arrowheads sitting on a corner fillet;
the arrowhead angle was a chord that tilted on such curves; and in
pid_second_order the feedback wire shared a track with reference -> scope.
"""

import gc
import math
import os

import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QPainterPath

from lib.simulation import wire_router
from modern_ui.renderers.connection_renderer import ConnectionRenderer

pytestmark = pytest.mark.qt

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.dsim.dirty = False
    w.close()
    gc.collect()


def _routed(window, qapp, name):
    assert window.open_example(os.path.join(EXAMPLES, name)) is not None
    qapp.processEvents()
    window.canvas.auto_route_lines()
    return window.dsim


def _segments(points):
    return list(zip(points, points[1:]))


def _shared_length(seg_a, seg_b):
    """Length two axis-aligned segments run on top of each other."""
    (a0, a1), (b0, b1) = seg_a, seg_b
    if a0.y() == a1.y() == b0.y() == b1.y():
        lo = max(min(a0.x(), a1.x()), min(b0.x(), b1.x()))
        hi = min(max(a0.x(), a1.x()), max(b0.x(), b1.x()))
        return max(0, hi - lo)
    if a0.x() == a1.x() == b0.x() == b1.x():
        lo = max(min(a0.y(), a1.y()), min(b0.y(), b1.y()))
        hi = min(max(a0.y(), a1.y()), max(b0.y(), b1.y()))
        return max(0, hi - lo)
    return 0


@pytest.mark.parametrize("name", ["first_order_step_response.diablos", "pid_second_order.diablos"])
def test_every_wire_enters_its_port_on_a_straight_run(window, qapp, name):
    dsim = _routed(window, qapp, name)
    for line in dsim.line_list:
        a, b = line.points[-2], line.points[-1]
        assert a.y() == b.y(), line.name
        # Room for the corner fillet, then the whole arrowhead in front of the port.
        needed = (
            type(line).ORTHOGONAL_CORNER_RADIUS
            + ConnectionRenderer.ARROW_SIZE
            + ConnectionRenderer.ARROW_PORT_INSET
        )
        assert b.x() - a.x() >= needed, (line.name, line.points)


def test_branch_keeps_clear_of_the_block_it_passes(window, qapp):
    dsim = _routed(window, qapp, "first_order_step_response.diablos")
    plant = next(b for b in dsim.blocks_list if b.block_fn == "TranFn")
    step = next(b for b in dsim.blocks_list if b.block_fn == "Step")
    scope = next(b for b in dsim.blocks_list if b.block_fn == "Scope")
    to_scope = next(
        line
        for line in dsim.line_list
        if line.srcblock == step.name and line.dstblock == scope.name
    )
    verticals = [a.x() for a, b in _segments(to_scope.points) if a.x() == b.x()]
    assert verticals
    for x in verticals:
        gap = min(abs(x - plant.left), abs(x - (plant.left + plant.width)))
        assert gap >= wire_router.BLOCK_MARGIN + wire_router.NEAR_BLOCK, (x, plant.left)


@pytest.mark.parametrize("name", ["first_order_step_response.diablos", "pid_second_order.diablos"])
def test_different_signals_never_share_a_track(window, qapp, name):
    dsim = _routed(window, qapp, name)
    lines = dsim.line_list
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            if (a.srcblock, a.srcport) == (b.srcblock, b.srcport):
                continue
            for sa in _segments(a.points):
                for sb in _segments(b.points):
                    assert _shared_length(sa, sb) == 0, (a.name, b.name, sa, sb)


@pytest.mark.parametrize(
    "name, n_dots",
    [("first_order_step_response.diablos", 1), ("pid_second_order.diablos", 3)],
)
def test_one_junction_dot_per_fan_out(window, qapp, name, n_dots):
    dsim = _routed(window, qapp, name)
    dots = ConnectionRenderer.junction_points(dsim.line_list)
    assert len(dots) == n_dots
    # Each dot sits on every wire of its fan-out.
    for dot in dots:
        owners = [
            line
            for line in dsim.line_list
            if ConnectionRenderer._on_polyline(dot.toPoint(), line.points)
        ]
        assert len({(line.srcblock, line.srcport) for line in owners}) == 1
        assert len(owners) >= 2


def test_bezier_fan_out_from_the_port_gets_no_dot():
    from types import SimpleNamespace

    from PyQt6.QtCore import QPoint

    def wire(end):
        return SimpleNamespace(
            srcblock="s", srcport=0, routing_mode="bezier", points=[QPoint(0, 0), end]
        )

    assert ConnectionRenderer.junction_points([wire(QPoint(100, 40)), wire(QPoint(100, -40))]) == []


class _Painter:
    def __init__(self):
        self.polygons = []

    def setBrush(self, *_):
        pass

    def setPen(self, *_):
        pass

    def drawPolygon(self, poly):
        self.polygons.append([poly.at(i) for i in range(poly.count())])


def test_arrowhead_follows_the_final_tangent_not_a_chord():
    """A fillet ending 8 px before the port used to tilt the arrow."""
    from types import SimpleNamespace

    from PyQt6.QtGui import QColor

    path = QPainterPath(QPointF(0, 0))
    path.lineTo(QPointF(0, 32))
    path.quadTo(QPointF(0, 40), QPointF(8, 40))  # corner fillet
    path.lineTo(QPointF(16, 40))  # 8 px straight into the port
    painter = _Painter()
    ConnectionRenderer()._draw_arrowhead(SimpleNamespace(path=path), painter, QColor("white"))
    tip, p1, p2 = painter.polygons[0]
    # Symmetric about the horizontal: the base is vertical, centred on the tip.
    assert math.isclose(p1.x(), p2.x(), abs_tol=1e-6)
    assert math.isclose((p1.y() + p2.y()) / 2, tip.y(), abs_tol=1e-6)
    assert tip.x() > p1.x()


@pytest.mark.parametrize("name", ["optimizer_comparison.diablos", "pid_second_order.diablos"])
def test_no_wire_passes_through_a_block(window, qapp, name):
    """A port stub longer than the gap to the next block once made A* fail,
    and the straight fallback cut through the blocks of the densest example."""
    from PyQt6.QtCore import QRect

    dsim = _routed(window, qapp, name)
    for line in dsim.line_list:
        for block in dsim.blocks_list:
            if block.name in (line.srcblock, line.dstblock):
                continue
            body = QRect(block.left + 1, block.top + 1, block.width - 2, block.height - 2)
            for a, b in _segments(line.points):
                seg = QRect(a, b).normalized()
                seg = QRect(seg.left(), seg.top(), max(seg.width(), 1), max(seg.height(), 1))
                assert not body.intersects(seg), (line.name, block.name, a, b)
