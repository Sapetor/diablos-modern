"""Auto-route DLine connections around block obstacles using A* on a grid."""

import heapq
import logging
from typing import Iterable, List, Optional, Tuple

from PyQt5.QtCore import QPoint, QRect

logger = logging.getLogger(__name__)

GRID_SIZE = 10
BLOCK_MARGIN = 8
TURN_PENALTY = 4
PORT_STUB = 20

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
        rects.append(_inflate(QRect(b.left, b.top, b.width, b.height), BLOCK_MARGIN))
    return rects


def _cell_blocked(cell: _Point, obstacles: Iterable[QRect]) -> bool:
    px = cell[0] * GRID_SIZE
    py = cell[1] * GRID_SIZE
    for r in obstacles:
        if r.left() <= px <= r.right() and r.top() <= py <= r.bottom():
            return True
    return False


def _astar(
    start: _Point,
    goal: _Point,
    obstacles: List[QRect],
    bounds: Tuple[int, int, int, int],
    initial_dir: _Dir = None,
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
            if new_pt != goal and _cell_blocked(new_pt, obstacles):
                continue
            this_dir = (dx, dy)
            turn_cost = 0 if last_dir is None or last_dir == this_dir else TURN_PENALTY
            new_g = cur_cost + 1 + turn_cost
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


def _stubbed_endpoints(start: QPoint, end: QPoint) -> Tuple[QPoint, QPoint]:
    return (
        QPoint(start.x() + PORT_STUB, start.y()),
        QPoint(end.x() - PORT_STUB, end.y()),
    )


def _scene_bounds(blocks, extra_points: List[QPoint]) -> Tuple[int, int, int, int]:
    xs = [b.left for b in blocks] + [b.left + b.width for b in blocks]
    ys = [b.top for b in blocks] + [b.top + b.height for b in blocks]
    xs.extend(p.x() for p in extra_points)
    ys.extend(p.y() for p in extra_points)
    pad = 200
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def route_line(line, blocks) -> Optional[List[QPoint]]:
    """Compute orthogonal waypoints for one DLine. Returns None on failure."""
    src = next((b for b in blocks if b.name == line.srcblock), None)
    dst = next((b for b in blocks if b.name == line.dstblock), None)
    if src is None or dst is None:
        return None
    if line.srcport >= len(src.out_coords) or line.dstport >= len(dst.in_coords):
        return None

    start = src.out_coords[line.srcport]
    end = dst.in_coords[line.dstport]
    stub_start, stub_end = _stubbed_endpoints(start, end)

    obstacles = _block_obstacles(blocks)
    minx, miny, maxx, maxy = _scene_bounds(blocks, [start, end, stub_start, stub_end])
    bounds = (minx // GRID_SIZE, miny // GRID_SIZE, maxx // GRID_SIZE, maxy // GRID_SIZE)

    s_cell = (stub_start.x() // GRID_SIZE, stub_start.y() // GRID_SIZE)
    e_cell = (stub_end.x() // GRID_SIZE, stub_end.y() // GRID_SIZE)

    grid_path = _astar(s_cell, e_cell, obstacles, bounds, initial_dir=(1, 0))
    if grid_path is None:
        logger.debug(f"route_line: A* failed for {line.name}, falling back to L-route")
        return [start, QPoint(end.x(), start.y()), end]

    waypoints = [start, stub_start]
    waypoints.extend(QPoint(g[0] * GRID_SIZE, g[1] * GRID_SIZE) for g in grid_path)
    waypoints.append(stub_end)
    waypoints.append(end)
    return _simplify(waypoints)


def route_all_lines(lines, blocks) -> int:
    """Recompute routing for every line in `lines` in place. Returns count routed."""
    rerouted = 0
    for line in lines:
        if getattr(line, 'hidden', False):
            continue
        new_pts = route_line(line, blocks)
        if not new_pts or len(new_pts) < 2:
            continue
        src = next((b for b in blocks if b.name == line.srcblock), None)
        dst = next((b for b in blocks if b.name == line.dstblock), None)
        if src is None or dst is None:
            continue
        start = src.out_coords[line.srcport]
        end = dst.in_coords[line.dstport]
        line.modified = True
        path, pts, segs = line.create_trajectory(start, end, blocks, points=new_pts)
        line.path = path
        line.points = pts
        line.segments = segs
        rerouted += 1
    return rerouted
