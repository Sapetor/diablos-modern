"""
Tests for DLine.create_trajectory bezier routing.

A committed connection wire in ``routing_mode == "bezier"`` must render as a
true cubic bezier curve that matches the live drag preview built in
``CanvasRenderer.draw_temp_line`` -- not a rounded-Manhattan stair-step. These
tests pin that contract: the forward bezier route contains genuine cubic
(cubicTo) segments, its control points equal the preview's, and it is distinct
from the orthogonal route.
"""

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QPainterPath

from lib.simulation.connection import DLine


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind to the session QApplication; never create our own (keeps the
    global theme_manager singleton alive across modules)."""
    return qapp


def _count_curve_segments(path: QPainterPath) -> int:
    """Number of cubic-curve segments in a QPainterPath.

    Qt represents a cubicTo as one ``CurveToElement`` followed by two
    ``CurveToDataElement`` entries, so counting ``isCurveTo()`` elements yields
    the number of cubic segments in the path.
    """
    return sum(1 for i in range(path.elementCount()) if path.elementAt(i).isCurveTo())


def _make_line(sid: int, start: QPoint, finish: QPoint) -> DLine:
    return DLine(sid, 'src', 0, 'dst', 0, [start, finish])


def test_default_routing_mode_is_bezier():
    """The change must not alter the default routing mode."""
    line = _make_line(0, QPoint(100, 100), QPoint(300, 160))
    assert line.routing_mode == "bezier"


def test_bezier_route_contains_cubic_segments():
    """A forward bezier route is a real cubic (contains cubicTo elements)."""
    start, finish = QPoint(100, 100), QPoint(300, 160)
    line = _make_line(0, start, finish)

    path, _points, _segments = line.create_trajectory(start, finish, [])

    assert _count_curve_segments(path) >= 1


def test_orthogonal_route_has_no_cubic_segments():
    """The orthogonal route stays purely straight-segment (no cubicTo)."""
    start, finish = QPoint(100, 100), QPoint(300, 160)
    line = _make_line(0, start, finish)
    line.routing_mode = "orthogonal"

    path, _points, _segments = line.create_trajectory(start, finish, [])

    assert _count_curve_segments(path) == 0


def test_bezier_distinct_from_orthogonal():
    """Bezier and orthogonal routes between the same endpoints differ."""
    start, finish = QPoint(100, 100), QPoint(300, 160)

    bezier_line = _make_line(0, start, finish)
    bezier_path, _, _ = bezier_line.create_trajectory(start, finish, [])

    ortho_line = _make_line(1, start, finish)
    ortho_line.routing_mode = "orthogonal"
    ortho_path, _, _ = ortho_line.create_trajectory(start, finish, [])

    # The bezier route curves; the orthogonal route does not.
    assert _count_curve_segments(bezier_path) >= 1
    assert _count_curve_segments(ortho_path) == 0
    assert bezier_path != ortho_path


def test_bezier_control_points_match_preview_math():
    """The committed cubic must use the SAME control-point math as the live
    drag preview in CanvasRenderer.draw_temp_line so the wire does not snap on
    mouse-release.

    Preview (unflipped layout):
        offset = min(distance * 0.5, 100)
        cp1 = (start.x + offset, start.y)
        cp2 = (end.x - offset, end.y)
    """
    start, finish = QPoint(100, 100), QPoint(300, 160)
    line = _make_line(0, start, finish)

    path, _, _ = line.create_trajectory(start, finish, [])

    # Reproduce the preview's control-point math.
    dx = finish.x() - start.x()
    dy = finish.y() - start.y()
    distance = (dx * dx + dy * dy) ** 0.5
    offset = min(distance * 0.5, 100)
    expected_cp1 = (int(start.x() + offset), start.y())
    expected_cp2 = (int(finish.x() - offset), finish.y())

    # Element layout: [moveTo(start), curveTo(cp1), curveToData(cp2), curveToData(finish)]
    assert path.elementCount() == 4
    e0, e1, e2, e3 = (path.elementAt(i) for i in range(4))

    assert e0.isMoveTo()
    assert (int(e0.x), int(e0.y)) == (start.x(), start.y())

    assert e1.isCurveTo()
    assert (int(e1.x), int(e1.y)) == expected_cp1
    assert (int(e2.x), int(e2.y)) == expected_cp2
    assert (int(e3.x), int(e3.y)) == (finish.x(), finish.y())


def test_bezier_offset_is_clamped_for_long_runs():
    """For long horizontal runs the offset clamps at 100 px (preview parity)."""
    start, finish = QPoint(0, 0), QPoint(1000, 0)
    line = _make_line(0, start, finish)

    path, _, _ = line.create_trajectory(start, finish, [])

    cp1 = path.elementAt(1)
    cp2 = path.elementAt(2)
    # offset = min(500, 100) = 100
    assert int(cp1.x) == 100
    assert int(cp2.x) == 900


def test_bezier_route_returns_two_endpoint_waypoints():
    """A plain cubic exposes only its two endpoints as collision waypoints."""
    start, finish = QPoint(100, 100), QPoint(300, 160)
    line = _make_line(0, start, finish)

    _path, points, segments = line.create_trajectory(start, finish, [])

    assert points == [start, finish]
    assert len(segments) == 1
