"""Unit tests for the shared Kahn topological-order helper (lib.engine.topo).

Used by SimulationEngine.detect_algebraic_loops (cycle detection) and the
compiled replay sort (feedthrough ordering).
"""
import pytest

from lib.engine.topo import kahn_topological_order


@pytest.mark.unit
class TestKahnTopologicalOrder:
    def test_linear_chain(self):
        order, remaining = kahn_topological_order(
            ["a", "b", "c"], {"a": ["b"], "b": ["c"], "c": []})
        assert order == ["a", "b", "c"]
        assert remaining == []

    def test_respects_dependencies(self):
        # d depends on b and c; b and c depend on a.
        adj = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
        order, remaining = kahn_topological_order(["a", "b", "c", "d"], adj)
        assert remaining == []
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_detected_in_remaining(self):
        # b -> c -> b is a cycle; a is acyclic.
        adj = {"a": ["b"], "b": ["c"], "c": ["b"]}
        order, remaining = kahn_topological_order(["a", "b", "c"], adj)
        assert order == ["a"]
        assert set(remaining) == {"b", "c"}

    def test_remaining_in_node_order(self):
        adj = {"x": ["y"], "y": ["x"]}
        order, remaining = kahn_topological_order(["x", "y"], adj)
        assert order == []
        assert remaining == ["x", "y"]  # original node order preserved

    def test_order_plus_remaining_covers_all(self):
        adj = {"a": ["b"], "b": ["a"], "c": []}
        order, remaining = kahn_topological_order(["a", "b", "c"], adj)
        assert sorted(order + remaining) == ["a", "b", "c"]

    def test_key_orders_initial_ready_set(self):
        # Three independent roots; key sorts the initially-ready set.
        adj = {"c": [], "a": [], "b": []}
        order, _ = kahn_topological_order(["c", "a", "b"], adj, key=lambda n: n)
        assert order == ["a", "b", "c"]

    def test_unknown_successors_ignored(self):
        # 'b' lists a successor not in nodes; it must not raise or skew degrees.
        adj = {"a": ["b"], "b": ["zzz"]}
        order, remaining = kahn_topological_order(["a", "b"], adj)
        assert order == ["a", "b"]
        assert remaining == []

    def test_parallel_edges_counted(self):
        # Two a->b edges: b's in-degree is 2; still resolves once a is emitted.
        adj = {"a": ["b", "b"], "b": []}
        order, remaining = kahn_topological_order(["a", "b"], adj)
        assert order == ["a", "b"]
        assert remaining == []

    def test_empty(self):
        order, remaining = kahn_topological_order([], {})
        assert order == [] and remaining == []
