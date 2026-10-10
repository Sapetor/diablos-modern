"""Layered (Sugiyama-style) left-to-right layout for block diagrams.

Pure and deterministic: no Qt, and every tie is broken by the input order of
``nodes`` (never by set or dict iteration), so Windows and Linux agree.

Steps:
1. Break cycles: edges that close a feedback loop (DFS back edges, visited
   in input order) are ignored for layering, so the loop's forward path
   still reads left to right.
2. Layers: longest path from the sources.
3. Order inside each layer: barycenter sweeps (down, then up) using the
   neighbors' positions; the destination port breaks ties so a Scope's
   inputs come in port order.
4. Coordinates: columns as wide as their widest block plus ``col_gap``;
   blocks stacked with ``row_gap`` and each column centered vertically.
   Disconnected parts of the diagram are laid out one below the other.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Node = Tuple[str, int, int]  # name, width, height
Edge = Tuple[str, str, int]  # src, dst, dst_port


def _components(names: List[str], adj: Dict[str, List[str]]) -> List[List[str]]:
    seen, comps = set(), []
    for start in names:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        order = {n: i for i, n in enumerate(names)}
        comps.append(sorted(comp, key=order.__getitem__))
    return comps


def _acyclic_edges(names: List[str], edges: List[Edge]) -> List[Edge]:
    """Drop DFS back edges (feedback), starting from sources in input order."""
    out: Dict[str, List[Edge]] = {n: [] for n in names}
    indeg = {n: 0 for n in names}
    for e in edges:
        out[e[0]].append(e)
        indeg[e[1]] += 1
    roots = [n for n in names if indeg[n] == 0] + [n for n in names if indeg[n] > 0]
    state: Dict[str, int] = {}  # 1 = on stack, 2 = done
    back = set()
    for root in roots:
        if root in state:
            continue
        stack = [(root, iter(out[root]))]
        state[root] = 1
        while stack:
            node, it = stack[-1]
            e = next(it, None)
            if e is None:
                state[node] = 2
                stack.pop()
                continue
            dst = e[1]
            if state.get(dst) == 1:
                back.add(id(e))
            elif dst not in state:
                state[dst] = 1
                stack.append((dst, iter(out[dst])))
    return [e for e in edges if id(e) not in back]


def _layers(names: List[str], edges: List[Edge]) -> Dict[str, int]:
    """Longest path from the sources (Kahn's order; edges are acyclic)."""
    succs: Dict[str, List[str]] = {n: [] for n in names}
    indeg = {n: 0 for n in names}
    for src, dst, _ in edges:
        succs[src].append(dst)
        indeg[dst] += 1
    layer = {n: 0 for n in names}
    ready = [n for n in names if indeg[n] == 0]
    while ready:
        n = ready.pop(0)
        for m in succs[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return layer


def _order(names, edges, layer, sweeps=4) -> List[List[str]]:
    n_layers = max(layer.values()) + 1
    rows: List[List[str]] = [[] for _ in range(n_layers)]
    for n in names:
        rows[layer[n]].append(n)
    ups: Dict[str, List[Tuple[str, int]]] = {n: [] for n in names}
    downs: Dict[str, List[Tuple[str, int]]] = {n: [] for n in names}
    for src, dst, port in edges:
        ups[dst].append((src, port))
        downs[src].append((dst, port))

    pos = {n: i for row in rows for i, n in enumerate(row)}

    def resort(row, neighbors, side):
        # side: +1 = neighbors in later layers, -1 = earlier layers.
        lay = layer[row[0]]

        def key(n):
            linked = [(pos[m], p) for m, p in neighbors[n] if (layer[m] - lay) * side > 0]
            if not linked:
                return (pos[n], 0, pos[n])
            bary = sum(i for i, _ in linked) / len(linked)
            return (bary, min(p for _, p in linked), pos[n])

        row.sort(key=key)
        for i, n in enumerate(row):
            pos[n] = i

    for _ in range(sweeps):
        for i in range(1, n_layers):
            resort(rows[i], ups, -1)
        for i in range(n_layers - 2, -1, -1):
            resort(rows[i], downs, +1)
    return rows


def layered_layout(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    origin: Tuple[int, int] = (0, 0),
    col_gap: int = 80,
    row_gap: int = 50,
    grid: int = 0,
) -> Dict[str, Tuple[int, int]]:
    """Top-left position for every node name."""
    names = [n for n, _, _ in nodes]
    size = {n: (w, h) for n, w, h in nodes}
    known = set(names)
    edges = [(s, d, int(p)) for s, d, p in edges if s in known and d in known and s != d]
    adj: Dict[str, List[str]] = {n: [] for n in names}
    for s, d, _ in edges:
        adj[s].append(d)
        adj[d].append(s)

    def snap(v):
        return int(round(v / grid) * grid) if grid else int(round(v))

    result: Dict[str, Tuple[int, int]] = {}
    top = origin[1]
    for comp in _components(names, adj):
        members = set(comp)
        comp_edges = [e for e in edges if e[0] in members]
        dag = _acyclic_edges(comp, comp_edges)
        layer = _layers(comp, dag)
        rows = _order(comp, dag, layer)
        heights = [sum(size[n][1] for n in row) + row_gap * (len(row) - 1) for row in rows]
        comp_h = max(heights)
        x = origin[0]
        for row, row_h in zip(rows, heights):
            col_w = max(size[n][0] for n in row)
            y = top + (comp_h - row_h) / 2
            for n in row:
                w, h = size[n]
                result[n] = (snap(x + (col_w - w) / 2), snap(y))
                y += h + row_gap
            x += col_w + col_gap
        top = snap(top + comp_h + row_gap * 2)
    return result
