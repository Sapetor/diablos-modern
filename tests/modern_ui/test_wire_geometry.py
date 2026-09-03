"""DLine geometry: hit-testing on curves, forced re-routing, manual bends.

Covers the wire-model half of the wiring overhaul:

* ``collision``/``distance_to`` test the *drawn* wire (a sampled cubic for
  bezier routes), not the straight chord between the ports;
* the port endpoints are never reported as draggable waypoints;
* ``reroute`` always rebuilds the path from block positions (``update_line``
  short-circuits when the endpoints did not move);
* manual Manhattan bends stay axis-aligned when a block is nudged, and
  ``reset_routing`` drops them again.
"""

import copy

import pytest
from PyQt5.QtCore import QPoint, QRect

from lib.simulation.connection import DLine


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


class _StubBlock:
    """Minimal block exposing what DLine's routing reads."""

    def __init__(self, name, left, top, width=80, height=60, flipped=False):
        self.name = name
        self.left, self.top, self.width, self.height = left, top, width, height
        self.flipped = flipped
        self.rect = QRect(left, top, width, height)
        self.in_ports = 1
        self.out_ports = 1
        in_x = left if not flipped else left + width
        out_x = left + width if not flipped else left
        self.in_coords = [QPoint(in_x, top + height // 2)]
        self.out_coords = [QPoint(out_x, top + height // 2)]


def _line(start, finish, sid=0):
    return DLine(sid, "src", 0, "dst", 0, [start, finish])


def _is_axis_aligned(points):
    return all(a.x() == b.x() or a.y() == b.y() for a, b in zip(points, points[1:]))


# ---------------------------------------------------------------------------
# Hit-testing
# ---------------------------------------------------------------------------


class TestCurveHitTesting:
    def test_click_on_curve_registers(self):
        start, finish = QPoint(100, 100), QPoint(300, 400)
        line = _line(start, finish)
        on_curve = line.path.pointAtPercent(0.25)
        hit = line.collision(QPoint(int(on_curve.x()), int(on_curve.y())))
        assert hit == ("segment", 0)

    def test_click_on_chord_away_from_curve_misses(self):
        """The straight chord between the ports is NOT the wire."""
        start, finish = QPoint(100, 100), QPoint(300, 400)
        line = _line(start, finish)
        chord_quarter = QPoint(150, 175)
        assert line.distance_to(chord_quarter) > 5
        assert line.collision(chord_quarter) is None

    def test_endpoints_are_not_draggable_points(self):
        start, finish = QPoint(100, 100), QPoint(300, 160)
        line = _line(start, finish)
        assert line.collision(start) != ("point", 0)
        assert line.collision(finish) != ("point", 1)

    def test_interior_waypoint_is_reported(self):
        start, finish = QPoint(100, 100), QPoint(300, 300)
        line = _line(start, finish)
        line.points = [start, QPoint(200, 100), QPoint(200, 300), finish]
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(
            start, finish, [], points=line.points
        )
        assert line.collision(QPoint(200, 100)) == ("point", 1)
        assert line.collision(QPoint(200, 200)) == ("segment", 1)

    def test_hit_cache_tracks_geometry_changes(self):
        start, finish = QPoint(100, 100), QPoint(300, 400)
        line = _line(start, finish)
        first = line.path.pointAtPercent(0.25)
        assert line.collision(QPoint(int(first.x()), int(first.y()))) is not None
        # Re-route somewhere else: the old curve point must no longer hit.
        new_finish = QPoint(300, 100)
        line.path, line.points, line.segments = line.create_trajectory(start, new_finish, [])
        assert line.collision(QPoint(int(first.x()), int(first.y()))) is None

    def test_deepcopy_still_hit_tests(self):
        start, finish = QPoint(100, 100), QPoint(300, 100)
        line = _line(start, finish)
        clone = copy.deepcopy(line)
        # The copied path is empty, so the copy falls back to its waypoints.
        assert clone.collision(QPoint(200, 100)) == ("segment", 0)


# ---------------------------------------------------------------------------
# Re-routing
# ---------------------------------------------------------------------------


class TestReroute:
    def test_update_line_short_circuits_but_reroute_does_not(self):
        src = _StubBlock("src", 300, 100)
        dst = _StubBlock("dst", 100, 100)
        line = _line(src.out_coords[0], dst.in_coords[0])
        # Built without block knowledge: fallback mid-x route, collinear -> 2 pts
        assert len(line.points) == 2
        line.update_line([src, dst])
        assert len(line.points) == 2, "update_line must not touch unchanged endpoints"
        assert line.reroute([src, dst])
        # With blocks known the feedback wire drops below the source block.
        assert len(line.points) > 2
        assert max(p.y() for p in line.points) > src.top + src.height

    def test_reroute_returns_false_when_block_missing(self):
        src = _StubBlock("src", 100, 100)
        line = _line(src.out_coords[0], QPoint(300, 130))
        assert line.reroute([src]) is False

    def test_reset_routing_drops_manual_bends(self):
        src = _StubBlock("src", 100, 100)
        dst = _StubBlock("dst", 400, 100)
        line = _line(src.out_coords[0], dst.in_coords[0])
        a, b = line.points[0], line.points[-1]
        line.points = [
            a,
            QPoint(250, a.y()),
            QPoint(250, b.y() + 80),
            QPoint(300, b.y() + 80),
            QPoint(300, b.y()),
            b,
        ]
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(
            a, b, [src, dst], points=line.points
        )
        assert line.modified and len(line.points) == 6

        line.reset_routing([src, dst])

        assert not line.modified
        assert not line.auto_routed
        assert line.points == [a, b]

    def test_manual_bend_stays_manhattan_when_block_moves(self):
        src = _StubBlock("src", 100, 100)
        dst = _StubBlock("dst", 400, 200)
        line = _line(src.out_coords[0], dst.in_coords[0])
        a, b = line.points[0], line.points[-1]
        line.points = [a, QPoint(250, a.y()), QPoint(250, b.y()), b]
        line.mark_manual_edit()
        line.path, line.points, line.segments = line.create_trajectory(
            a, b, [src, dst], points=line.points
        )

        # Nudge the destination block down; its input port moves with it.
        dst.top += 90
        dst.in_coords = [QPoint(dst.left, dst.top + dst.height // 2)]
        line.update_line([src, dst])

        assert line.modified
        assert line.points[-1] == dst.in_coords[0]
        assert _is_axis_aligned(line.points)

    def test_set_routing_mode_clears_auto_routed(self):
        line = _line(QPoint(0, 0), QPoint(100, 0))
        line.auto_routed = True
        line.modified = True
        line.set_routing_mode("bezier")
        assert not line.auto_routed and not line.modified

    def test_orthogonal_mode_is_axis_aligned_with_obstacle(self):
        src = _StubBlock("src", 100, 100)
        mid = _StubBlock("mid", 250, 90)
        dst = _StubBlock("dst", 400, 100)
        line = _line(src.out_coords[0], dst.in_coords[0])
        line.routing_mode = "orthogonal"
        line.reroute([src, mid, dst])
        assert _is_axis_aligned(line.points)
        assert len(line.points) >= 4, "the route must detour around the middle block"
