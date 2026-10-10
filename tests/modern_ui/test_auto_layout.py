"""One-click auto-layout: the pure layered layout and the canvas action."""

import itertools

import pytest

from modern_ui.tools.auto_layout import layered_layout


def _nodes(*names, w=60, h=40):
    return [(n, w, h) for n in names]


def _overlap(pos, size, a, b, pad=0):
    (ax, ay), (bx, by) = pos[a], pos[b]
    (aw, ah), (bw, bh) = size[a], size[b]
    return ax < bx + bw + pad and bx < ax + aw + pad and ay < by + bh + pad and by < ay + ah + pad


class TestLayeredLayout:
    def test_chain_reads_left_to_right(self):
        pos = layered_layout(_nodes("a", "b", "c"), [("a", "b", 0), ("b", "c", 0)])
        assert pos["a"][0] < pos["b"][0] < pos["c"][0]

    def test_fan_in_sources_share_a_column_in_port_order(self):
        pos = layered_layout(_nodes("scope", "s2", "s1"), [("s1", "scope", 0), ("s2", "scope", 1)])
        assert pos["s1"][0] == pos["s2"][0] < pos["scope"][0]
        assert pos["s1"][1] < pos["s2"][1]

    def test_feedback_loop_terminates_and_keeps_the_forward_path(self):
        nodes = _nodes("step", "sum", "ctrl", "plant")
        edges = [("step", "sum", 0), ("sum", "ctrl", 0), ("ctrl", "plant", 0), ("plant", "sum", 1)]
        pos = layered_layout(nodes, edges)
        assert pos["step"][0] < pos["sum"][0] < pos["ctrl"][0] < pos["plant"][0]

    def test_two_node_cycle(self):
        pos = layered_layout(_nodes("a", "b"), [("a", "b", 0), ("b", "a", 0)])
        assert pos["a"][0] < pos["b"][0]

    def test_no_overlaps_with_mixed_sizes_and_components(self):
        nodes = [
            ("a", 60, 40),
            ("b", 120, 90),
            ("c", 60, 40),
            ("d", 80, 60),
            ("x", 60, 40),
            ("y", 60, 40),
        ]
        edges = [("a", "b", 0), ("a", "c", 0), ("b", "d", 0), ("c", "d", 1), ("x", "y", 0)]
        pos = layered_layout(nodes, edges, grid=10)
        size = {n: (w, h) for n, w, h in nodes}
        assert set(pos) == set(size)
        for p, q in itertools.combinations(size, 2):
            assert not _overlap(pos, size, p, q), (p, q)
        # The second component sits below the first.
        assert min(pos["x"][1], pos["y"][1]) > max(pos[n][1] + size[n][1] for n in "abcd")

    def test_snaps_to_grid_and_respects_origin(self):
        pos = layered_layout(_nodes("a", "b"), [("a", "b", 0)], origin=(103, 47), grid=10)
        assert all(x % 10 == 0 and y % 10 == 0 for x, y in pos.values())
        assert min(x for x, _ in pos.values()) >= 100

    def test_deterministic(self):
        nodes = _nodes(*"abcdefg")
        edges = [
            ("a", "c", 0),
            ("b", "c", 1),
            ("c", "d", 0),
            ("c", "e", 0),
            ("e", "a", 0),
            ("f", "g", 0),
        ]
        assert layered_layout(nodes, edges) == layered_layout(list(nodes), list(edges))

    def test_long_chain_does_not_recurse(self):
        names = [f"n{i}" for i in range(3000)]
        edges = [(a, b, 0) for a, b in zip(names, names[1:])]
        pos = layered_layout(_nodes(*names), edges)
        assert pos["n2999"][0] > pos["n0"][0]


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _snapshot(dsim):
    return {b.name: (b.left, b.top) for b in dsim.blocks_list}


@pytest.mark.qt
@pytest.mark.parametrize("example", ["multirate_demo", "discrete_pi_zoh"])
def test_canvas_auto_layout_and_undo(window, qapp, example):
    import os

    window.open_example(os.path.abspath(f"examples/{example}.diablos"))
    qapp.processEvents()
    dsim, canvas = window.dsim, window.canvas
    before = _snapshot(dsim)
    assert len(before) >= 4

    canvas.auto_layout()
    qapp.processEvents()

    after = _snapshot(dsim)
    assert after != before
    blocks = {b.name: b for b in dsim.blocks_list}
    for a, b in itertools.combinations(blocks.values(), 2):
        assert not a.rect.intersects(b.rect), (a.name, b.name)
    for line in dsim.line_list:
        src, dst = blocks[line.srcblock], blocks[line.dstblock]
        assert line.points[0] == src.out_coords[line.srcport]
        assert line.points[-1] == dst.in_coords[line.dstport]
    assert all(x % canvas.grid_size == 0 and y % canvas.grid_size == 0 for x, y in after.values())

    canvas.undo()
    qapp.processEvents()
    assert _snapshot(dsim) == before
