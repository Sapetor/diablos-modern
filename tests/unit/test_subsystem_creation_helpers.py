"""Unit tests for the helpers behind SubsystemManager.create_subsystem_from_selection.

The end-to-end behaviour (ports created, lines re-attached) is pinned by
tests/integration/test_subsystems.py; these cover the pure helpers the method
is now composed of.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QRect

from lib.managers import subsystem_manager as sm


def _blk(name, left, top, w=50, h=40, in_ports=1, out_ports=1, sid=0):
    return SimpleNamespace(
        name=name,
        sid=sid,
        rect=QRect(left, top, w, h),
        in_ports=in_ports,
        out_ports=out_ports,
        in_coords=[QPoint(left, top + 10 * (i + 1)) for i in range(in_ports)],
        out_coords=[QPoint(left + w, top + 10 * (i + 1)) for i in range(out_ports)],
    )


def _line(src, dst, srcport=0, dstport=0):
    return SimpleNamespace(srcblock=src, srcport=srcport, dstblock=dst, dstport=dstport)


@pytest.mark.unit
class TestPureHelpers:
    def test_next_sid_is_max_plus_one_or_one(self):
        assert sm._next_sid([]) == 1
        assert sm._next_sid([SimpleNamespace(sid=3), SimpleNamespace(sid=7)]) == 8

    def test_bounding_box(self):
        blocks = [_blk("a", 10, 20), _blk("b", 100, 5, w=30, h=10)]
        assert sm._bounding_box(blocks) == (10, 5, 129, 59)

    def test_classify_lines(self):
        lines = [
            _line("ext", "a"),  # in
            _line("a", "b"),  # internal
            _line("b", "ext2"),  # out
            _line("ext", "ext2"),  # unrelated, dropped
        ]
        internal, boundary = sm._classify_lines(lines, {"a", "b"})
        assert internal == [lines[1]]
        assert boundary == [(lines[0], "in"), (lines[2], "out")]

    def test_unconnected_ports(self):
        a = _blk("a", 0, 0, in_ports=2, out_ports=1)
        b = _blk("b", 0, 0, in_ports=1, out_ports=2)
        lines = [_line("a", "b", srcport=0, dstport=0), _line("x", "a", dstport=1)]
        inputs, outputs = sm._unconnected_ports([a, b], lines)
        assert inputs == [(a, 0)]
        assert outputs == [(b, 0), (b, 1)]

    def test_port_point(self):
        b = _blk("a", 10, 20, in_ports=2)
        assert sm._port_point(b, "in_coords", 1) == QPoint(10, 40)
        assert sm._port_point(b, "in_coords", 5) is None
        assert sm._port_point(None, "in_coords", 0) is None


@pytest.mark.unit
class TestRerouteInternalLines:
    def test_points_shift_and_snap_to_moved_ports(self, qapp):
        from lib.simulation.connection import DLine

        a = _blk("a", 100, 100)
        b = _blk("b", 300, 100)
        line = DLine(
            sid=1,
            srcblock="a",
            srcport=0,
            dstblock="b",
            dstport=0,
            points=(a.out_coords[0], b.in_coords[0]),
        )
        offset = QPoint(-50, 20)
        # Simulate the blocks having been moved by the offset already.
        for blk in (a, b):
            blk.rect.translate(offset)
            blk.in_coords = [p + offset for p in blk.in_coords]
            blk.out_coords = [p + offset for p in blk.out_coords]
        sm._reroute_internal_lines([line], {"a": a, "b": b}, offset, [a, b])
        assert line.points[0] == a.out_coords[0]
        assert line.points[-1] == b.in_coords[0]

    def test_lines_without_points_are_left_alone(self):
        bare = SimpleNamespace(srcblock="a", dstblock="b", srcport=0, dstport=0)
        sm._reroute_internal_lines([bare], {}, QPoint(1, 1), [])
        assert not hasattr(bare, "points")


@pytest.mark.unit
class TestAddInternalLine:
    def test_appends_a_routed_line_with_next_sid(self, qapp):
        subsys = SimpleNamespace(sub_blocks=[], sub_lines=[SimpleNamespace(sid=4)])
        line = sm._add_internal_line(subsys, "src", 0, "dst", 1, QPoint(0, 0), QPoint(100, 0))
        assert subsys.sub_lines[-1] is line
        assert (line.sid, line.srcblock, line.srcport, line.dstblock, line.dstport) == (
            5,
            "src",
            0,
            "dst",
            1,
        )
        assert line.points[0] == QPoint(0, 0) and line.points[-1] == QPoint(100, 0)
