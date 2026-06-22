"""Kahn's-algorithm topological ordering, shared by the engine's two graph passes.

Both ``SimulationEngine.detect_algebraic_loops`` (feedback-cycle detection over
all connections) and ``run_compiled_simulation``'s replay pass (ordering blocks
by direct feedthrough before recomputing their outputs) ran their own copy of
Kahn's algorithm. This is the single implementation they both call.
"""
from collections import deque


def kahn_topological_order(nodes, adjacency, key=None):
    """Topologically order a directed graph with Kahn's algorithm.

    Args:
        nodes: iterable of node keys (hashable, unique).
        adjacency: mapping ``node -> iterable of successor nodes`` (edge
            ``node -> successor``). Successors not in ``nodes`` are ignored;
            parallel edges are counted as-is.
        key: optional sort key applied to the initially-ready nodes so the
            output is deterministic (e.g. ``key=lambda n: n`` for name order).
            Nodes that become ready later follow adjacency order, as in plain
            Kahn.

    Returns:
        ``(order, remaining)`` where ``order`` is a topological ordering of the
        acyclic part and ``remaining`` lists the nodes still inside a cycle, in
        the original ``nodes`` order. ``remaining`` is empty iff the graph is
        acyclic; ``order + remaining`` always covers every node exactly once.
    """
    nodes = list(nodes)
    in_deg = {n: 0 for n in nodes}
    for n in nodes:
        for m in adjacency.get(n, ()):
            if m in in_deg:
                in_deg[m] += 1

    ready = [n for n in nodes if in_deg[n] == 0]
    if key is not None:
        ready.sort(key=key)
    queue = deque(ready)

    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adjacency.get(u, ()):
            if v in in_deg:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

    remaining = [n for n in nodes if in_deg[n] > 0]
    return order, remaining
