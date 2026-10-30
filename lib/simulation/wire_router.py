"""Auto-route DLine connections around block obstacles using A* on a grid."""

import heapq
import logging
from typing import Iterable, List, Optional, Set, Tuple

from PyQt6.QtCore import QPoint, QRect

logger = logging.getLogger(__name__)

GRID_SIZE = 10
BLOCK_MARGIN = 8
# Extra clearance below each block so routes do not run through the block
# name drawn under it (see BlockRenderer.draw_block: 28 px text band).
LABEL_CLEARANCE = 20
TURN_PENALTY = 4
# Straight run out of / into a port. It must outlast the orthogonal corner
# fillet (ORTHOGONAL_CORNER_RADIUS = 10) plus the arrowhead and its inset in
# front of the port disc (~17 px), or the arrow sits on the curve. Shortened
# where a neighbouring block leaves less room (see _stub_length).
PORT_STUB = 30
# Cells this close (px) to a block's obstacle rect cost NEAR_BLOCK_PENALTY
# extra, so a route uses free space instead of hugging a block edge. Without
# it every turn position costs the same and A* happily runs a leg right
# along a block (e.g. 10 px in front of an input it does not even enter).
NEAR_BLOCK = 20
NEAR_BLOCK_PENALTY = 2
# Cost per cell of running on top of another source's wire. Without it two
# unrelated signals could share a track and read as one wire. A crossing
# touches one cell, so it stays cheap; wires from the same output port may
# share freely (they carry the same signal and get a junction dot).
OVERLAP_PENALTY = 8

_Point = Tuple[int, int]
_Dir = Optional[Tuple[int, int]]
_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _inflate(rect: QRect, margin: int) -> QRect:
    return QRect(
        rect.left() - margin,
        rect.top() - margin,
        rect.width() + 2 * margin,
        rect.height() + 2 * margin,
    )


def _block_obstacles(blocks) -> List[QRect]:
    rects: List[QRect] = []
    for b in blocks:
        rects.append(
            _inflate(QRect(b.left, b.top, b.width, b.height + LABEL_CLEARANCE), BLOCK_MARGIN)
        )
    return rects


def _rect_cells(rects: Iterable[QRect]) -> Set[_Point]:
    """Grid cells whose corner point (cell * GRID_SIZE) lies in any of
    ``rects``, as a set so A* tests membership in O(1)."""
    cells: Set[_Point] = set()
    for r in rects:
        x0, x1 = -(-r.left() // GRID_SIZE), r.right() // GRID_SIZE
        y0, y1 = -(-r.top() // GRID_SIZE), r.bottom() // GRID_SIZE
        cells.update((x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1))
    return cells


def _grids(blocks) -> Tuple[Set[_Point], Set[_Point]]:
    """(blocked cells, near-a-block cells) for ``blocks``."""
    obstacles = _block_obstacles(blocks)
    return _rect_cells(obstacles), _rect_cells(_inflate(r, NEAR_BLOCK) for r in obstacles)


def _astar(
    start: _Point,
    goal: _Point,
    blocked: Set[_Point],
    bounds: Tuple[int, int, int, int],
    initial_dir: _Dir = None,
    near: Set[_Point] = frozenset(),
    taken: Optional[dict] = None,
    source=None,
) -> Optional[List[_Point]]:
    if start == goal:
        return [start]

    minx, miny, maxx, maxy = bounds
    counter = 0
    open_heap: List[Tuple[int, int, _Point, _Dir]] = []
    heapq.heappush(open_heap, (0, counter, start, initial_dir))
    g_score = {(start, initial_dir): 0}
    came_from = {(start, initial_dir): None}

    while open_heap:
        _, _, cur, last_dir = heapq.heappop(open_heap)
        if cur == goal:
            path: List[_Point] = []
            key = (cur, last_dir)
            while key is not None:
                path.append(key[0])
                key = came_from.get(key)
            path.reverse()
            return path

        cur_cost = g_score[(cur, last_dir)]
        for dx, dy in _DIRS:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (minx <= nx <= maxx and miny <= ny <= maxy):
                continue
            new_pt = (nx, ny)
            if new_pt != goal and new_pt in blocked:
                continue
            this_dir = (dx, dy)
            turn_cost = 0 if last_dir is None or last_dir == this_dir else TURN_PENALTY
            near_cost = NEAR_BLOCK_PENALTY if new_pt in near else 0
            owners = taken.get(new_pt) if taken else None
            overlap_cost = OVERLAP_PENALTY if owners and owners - {source} else 0
            new_g = cur_cost + 1 + turn_cost + near_cost + overlap_cost
            new_key = (new_pt, this_dir)
            if new_g < g_score.get(new_key, 10**9):
                g_score[new_key] = new_g
                came_from[new_key] = (cur, last_dir)
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                counter += 1
                heapq.heappush(open_heap, (new_g + h, counter, new_pt, this_dir))

    return None


def _simplify(points: List[QPoint]) -> List[QPoint]:
    # Drop consecutive duplicates first so the collinearity test below isn't
    # confused by zero-length segments.
    deduped: List[QPoint] = []
    for p in points:
        if not deduped or p != deduped[-1]:
            deduped.append(p)
    if len(deduped) <= 2:
        return deduped
    out = [deduped[0]]
    for i in range(1, len(deduped) - 1):
        p, c, n = out[-1], deduped[i], deduped[i + 1]
        dx1, dy1 = c.x() - p.x(), c.y() - p.y()
        dx2, dy2 = n.x() - c.x(), n.y() - c.y()
        if dx1 * dy2 == dx2 * dy1:
            continue
        out.append(c)
    out.append(deduped[-1])
    return out


def _stub_length(port: QPoint, step: int, blocked: Set[_Point]) -> int:
    """Longest of PORT_STUB, 20, 10 px whose straight run out of ``port``
    (``step`` = +1 rightwards, -1 leftwards) stays on free cells.

    A long stub that ends inside a neighbouring block makes A* fail and the
    wire fall back to a straight line through the blocks, so tight gaps get
    the shorter stub the router always used.
    """
    y = port.y() // GRID_SIZE
    for length in (PORT_STUB, 20, GRID_SIZE):
        cells = range(GRID_SIZE, length + 1, GRID_SIZE)
        if all((port.x() // GRID_SIZE + step * c // GRID_SIZE, y) not in blocked for c in cells):
            return length
    return GRID_SIZE


def _stubbed_endpoints(
    start: QPoint, end: QPoint, blocked: Set[_Point] = frozenset()
) -> Tuple[QPoint, QPoint]:
    return (
        QPoint(start.x() + _stub_length(start, 1, blocked), start.y()),
        QPoint(end.x() - _stub_length(end, -1, blocked), end.y()),
    )


def _scene_bounds(blocks, extra_points: List[QPoint]) -> Tuple[int, int, int, int]:
    xs = [b.left for b in blocks] + [b.left + b.width for b in blocks]
    ys = [b.top for b in blocks] + [b.top + b.height for b in blocks]
    xs.extend(p.x() for p in extra_points)
    ys.extend(p.y() for p in extra_points)
    pad = 200
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _source(line):
    return (line.srcblock, line.srcport)


def _mark_taken(taken: dict, line, points) -> None:
    """Record the grid cells ``points`` (a polyline) passes through."""
    src = _source(line)
    for a, b in zip(points, points[1:]):
        ax, ay = round(a.x() / GRID_SIZE), round(a.y() / GRID_SIZE)
        bx, by = round(b.x() / GRID_SIZE), round(b.y() / GRID_SIZE)
        n = max(abs(bx - ax), abs(by - ay), 1)
        for i in range(n + 1):
            cell = (ax + round((bx - ax) * i / n), ay + round((by - ay) * i / n))
            taken.setdefault(cell, set()).add(src)


def route_line(line, blocks, block_index=None, taken=None, grids=None) -> Optional[List[QPoint]]:
    """Compute orthogonal waypoints for one DLine. Returns None on failure.

    `block_index` is an optional {block.name: block} mapping; when provided it
    avoids re-scanning `blocks` to resolve the endpoints. `taken` maps grid
    cells to the sources of wires already laid there (see OVERLAP_PENALTY);
    `grids` is ``_grids(blocks)``, passed in when routing many wires.
    """
    if block_index is None:
        block_index = {b.name: b for b in blocks}
    src = block_index.get(line.srcblock)
    dst = block_index.get(line.dstblock)
    if src is None or dst is None:
        return None
    if line.srcport >= len(src.out_coords) or line.dstport >= len(dst.in_coords):
        return None

    start = src.out_coords[line.srcport]
    end = dst.in_coords[line.dstport]
    blocked, near = grids if grids is not None else _grids(blocks)
    stub_start, stub_end = _stubbed_endpoints(start, end, blocked)
    minx, miny, maxx, maxy = _scene_bounds(blocks, [start, end, stub_start, stub_end])
    bounds = (minx // GRID_SIZE, miny // GRID_SIZE, maxx // GRID_SIZE, maxy // GRID_SIZE)

    s_cell = (stub_start.x() // GRID_SIZE, stub_start.y() // GRID_SIZE)
    e_cell = (stub_end.x() // GRID_SIZE, stub_end.y() // GRID_SIZE)

    grid_path = _astar(
        s_cell,
        e_cell,
        blocked,
        bounds,
        initial_dir=(1, 0),
        near=near,
        taken=taken,
        source=_source(line),
    )
    if grid_path is None:
        logger.debug(f"route_line: A* failed for {line.name}, falling back to L-route")
        return [start, QPoint(end.x(), start.y()), end]

    waypoints = [start, stub_start]
    waypoints.extend(QPoint(g[0] * GRID_SIZE, g[1] * GRID_SIZE) for g in grid_path)
    waypoints.append(stub_end)
    waypoints.append(end)
    return _simplify(waypoints)


def route_all_lines(lines, blocks, others=()) -> int:
    """Recompute routing for every line in `lines` in place. Returns count routed.

    Each routed wire is avoided by the ones routed after it, and so are the
    wires in `others` (left as they are): pass the rest of the diagram when
    re-routing only some wires.
    """
    rerouted = 0
    block_index = {b.name: b for b in blocks}
    taken: dict = {}
    grids = _grids(blocks)
    for other in others:
        if not getattr(other, "hidden", False) and getattr(other, "points", None):
            _mark_taken(taken, other, other.points)
    for line in lines:
        if getattr(line, "hidden", False):
            continue
        new_pts = route_line(line, blocks, block_index=block_index, taken=taken, grids=grids)
        if not new_pts or len(new_pts) < 2:
            continue
        src = block_index.get(line.srcblock)
        dst = block_index.get(line.dstblock)
        if src is None or dst is None:
            continue
        start = src.out_coords[line.srcport]
        end = dst.in_coords[line.dstport]
        # The result is a Manhattan route, so the wire becomes an orthogonal,
        # auto-routed one: it is re-routed when its blocks move and its
        # context-menu routing radio reflects what is drawn.
        line.routing_mode = "orthogonal"
        line.modified = True
        line.auto_routed = True
        line.selected_segment = -1
        path, pts, segs = line.create_trajectory(start, end, blocks, points=new_pts)
        line.path = path
        line.points = pts
        line.segments = segs
        _mark_taken(taken, line, new_pts)
        rerouted += 1
    return rerouted
