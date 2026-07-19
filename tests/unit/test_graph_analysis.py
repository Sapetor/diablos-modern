"""Unit tests for the extracted diagram graph-analysis helpers.

These were private methods on SimulationEngine; extracting them into
``lib/engine/graph_analysis.py`` lets us exercise the connectivity/topology
logic against plain stand-in blocks and lines, with no engine or simulation.
"""

import types

import pytest

from lib.engine import graph_analysis as ga


def _line(src, dst, srcport=0, dstport=0):
    return types.SimpleNamespace(
        srcblock=src, srcport=srcport, dstblock=dst, dstport=dstport, hidden=False
    )


def _block(name, in_ports=1, out_ports=1, hierarchy=0, instance=None):
    return types.SimpleNamespace(
        name=name,
        in_ports=in_ports,
        out_ports=out_ports,
        hierarchy=hierarchy,
        block_instance=instance,
    )


def _instance(optional_inputs=(), optional_outputs=(), requires_outputs=True):
    return types.SimpleNamespace(
        optional_inputs=set(optional_inputs),
        optional_outputs=set(optional_outputs),
        requires_outputs=requires_outputs,
    )


@pytest.mark.unit
class TestNeighborsAndOutputs:
    def test_get_neighbors_splits_in_and_out(self):
        lines = [_line("a", "b"), _line("b", "c"), _line("x", "b", dstport=1)]
        inputs, outputs = ga.get_neighbors("b", lines)
        assert {i["srcblock"] for i in inputs} == {"a", "x"}
        assert [o["dstblock"] for o in outputs] == ["c"]

    def test_get_outputs_only_outgoing(self):
        lines = [_line("a", "b"), _line("a", "c"), _line("z", "a")]
        outs = ga.get_outputs("a", lines)
        assert {o["dstblock"] for o in outs} == {"b", "c"}

    def test_no_connections(self):
        assert ga.get_neighbors("lonely", []) == ([], [])
        assert ga.get_outputs("lonely", []) == []


@pytest.mark.unit
class TestMaxHierarchy:
    def test_max(self):
        blocks = [_block("a", hierarchy=0), _block("b", hierarchy=3), _block("c", hierarchy=1)]
        assert ga.get_max_hierarchy(blocks) == 3

    def test_empty_is_minus_one(self):
        assert ga.get_max_hierarchy([]) == -1


@pytest.mark.unit
class TestChildrenRecognition:
    def test_recursive_downstream(self):
        lines = [_line("a", "b"), _line("b", "c"), _line("c", "d")]
        assert ga.children_recognition("a", [], lines) == ["b", "c", "d"]

    def test_cycle_terminates(self):
        # a -> b -> a should not infinite-loop (already-seen guard).
        lines = [_line("a", "b"), _line("b", "a")]
        assert ga.children_recognition("a", [], lines) == ["b", "a"]

    def test_is_child_of(self):
        children = [{"dstblock": "b", "srcport": 0, "dstport": 0}, {"dstblock": "c"}]
        assert ga.is_child_of("b", children) == (True, [children[0]])
        assert ga.is_child_of("z", children) == (False, [])


@pytest.mark.unit
class TestDetectAlgebraicLoops:
    def test_no_blocks(self):
        assert ga.detect_algebraic_loops([], [], set()) == (False, [])

    def test_acyclic_is_clear(self):
        blocks = [_block("a"), _block("b")]
        lines = [_line("a", "b")]
        assert ga.detect_algebraic_loops(blocks, lines, set()) == (False, [])

    def test_pure_cycle_is_algebraic_loop(self):
        blocks = [_block("a"), _block("b")]
        lines = [_line("a", "b"), _line("b", "a")]
        is_loop, cycle = ga.detect_algebraic_loops(blocks, lines, set())
        assert is_loop is True
        assert set(cycle) == {"a", "b"}

    def test_memory_block_breaks_the_loop(self):
        # Same cycle, but 'b' is a memory block (e.g. a transfer function) -> not algebraic.
        blocks = [_block("a"), _block("b")]
        lines = [_line("a", "b"), _line("b", "a")]
        assert ga.detect_algebraic_loops(blocks, lines, {"b"}) == (False, [])


@pytest.mark.unit
class TestCheckDiagramIntegrity:
    def test_fully_connected_is_valid(self):
        blocks = [_block("src", in_ports=0, out_ports=1), _block("sink", in_ports=1, out_ports=0)]
        lines = [_line("src", "sink")]
        assert ga.check_diagram_integrity(blocks, lines) is True

    def test_unlinked_required_input_fails(self):
        blocks = [_block("src", in_ports=0, out_ports=1), _block("sink", in_ports=1, out_ports=0)]
        assert ga.check_diagram_integrity(blocks, []) is False

    def test_unlinked_required_output_fails(self):
        blocks = [_block("src", in_ports=0, out_ports=1), _block("sink", in_ports=1, out_ports=0)]
        # src output goes nowhere.
        lines = [_line("other", "sink")]
        assert ga.check_diagram_integrity(blocks, lines) is False

    def test_optional_input_exempt(self):
        # 2 input ports, port 1 optional and unconnected -> still valid.
        sink = _block("sink", in_ports=2, out_ports=0, instance=_instance(optional_inputs=(1,)))
        blocks = [_block("src", in_ports=0, out_ports=1), sink]
        lines = [_line("src", "sink", dstport=0)]
        assert ga.check_diagram_integrity(blocks, lines) is True

    def test_requires_outputs_false_exempts_output(self):
        # A sink whose output is not required (requires_outputs False).
        src = _block("src", in_ports=0, out_ports=1, instance=_instance(requires_outputs=False))
        blocks = [src]
        assert ga.check_diagram_integrity(blocks, []) is True

    def test_multiple_wires_into_one_input_fails(self):
        sink = _block("sink", in_ports=1, out_ports=0)
        blocks = [_block("a", in_ports=0, out_ports=1), _block("b", in_ports=0, out_ports=1), sink]
        lines = [_line("a", "sink", dstport=0), _line("b", "sink", dstport=0)]
        assert ga.check_diagram_integrity(blocks, lines) is False
