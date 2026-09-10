"""Unit tests for the pre-flight checks in lib/diagram_validator.py.

``validate_block_connections`` / ``detect_algebraic_loops`` gate every proposed
wire (ConnectionManager) and the Run action (SimulationController);
``check_simulation_state`` / ``check_block_integrity`` gate Run and the
main-window tick. They work on the raw diagram, before flattening or parameter
resolution, so the stubs here carry only the attributes they read.
"""

from types import SimpleNamespace

import pytest

from lib.diagram_validator import (
    DiagramValidator,
    ErrorSeverity,
    check_block_integrity,
    check_simulation_state,
    detect_algebraic_loops,
    find_duplicate_input_connections,
    validate_block_connections,
)


def _blk(name, block_fn="Gain", b_type=2, in_ports=1, out_ports=1, params=None, sub_blocks=None):
    return SimpleNamespace(
        name=name,
        sid=0,
        fn_name=block_fn,
        block_fn=block_fn,
        b_type=b_type,
        in_ports=in_ports,
        out_ports=out_ports,
        params=dict(params or {}),
        exec_params={},
        sub_blocks=sub_blocks or [],
    )


def _line(src, dst, srcport=0, dstport=0):
    return SimpleNamespace(srcblock=src, srcport=srcport, dstblock=dst, dstport=dstport)


@pytest.mark.unit
class TestValidateBlockConnections:
    def test_clean_chain_is_ok(self):
        blocks = [_blk("step", "Step", 0, 0, 1), _blk("gain"), _blk("scope", "Scope", 3, 1, 0)]
        lines = [_line("step", "gain"), _line("gain", "scope")]
        assert validate_block_connections(blocks, lines) == (True, [])

    def test_unconnected_block_is_a_warning_not_an_error(self):
        blocks = [_blk("step", "Step", 0, 0, 1), _blk("gain"), _blk("lonely")]
        ok, messages = validate_block_connections(blocks, [_line("step", "gain")])
        assert ok is True
        assert messages == ["Block 'lonely' has no connections"]

    def test_duplicate_input_port_is_an_error(self):
        blocks = [_blk("a", "Step", 0, 0, 1), _blk("b", "Step", 0, 0, 1), _blk("gain")]
        lines = [_line("a", "gain"), _line("b", "gain")]
        ok, messages = validate_block_connections(blocks, lines)
        assert ok is False
        assert messages == ["Multiple connections to same input port: block gain, port 0"]

    def test_algebraic_loop_is_an_error_listed_before_warnings(self):
        blocks = [_blk("g1"), _blk("g2"), _blk("lonely")]
        ok, messages = validate_block_connections(blocks, [_line("g1", "g2"), _line("g2", "g1")])
        assert ok is False
        assert messages[0].startswith("Algebraic loop detected")
        assert messages[1] == "Block 'lonely' has no connections"

    def test_three_lines_into_one_port_report_each_extra_line_in_order(self):
        blocks = [_blk(n, "Step", 0, 0, 1) for n in "abc"] + [_blk("gain")]
        lines = [_line("a", "gain"), _line("b", "gain"), _line("c", "gain")]
        ok, messages = validate_block_connections(blocks, lines)
        assert ok is False
        assert messages == ["Multiple connections to same input port: block gain, port 0"] * 2


@pytest.mark.unit
class TestDuplicateInputCheckIsShared:
    """Both entry points read the duplicated pair off find_duplicate_input_connections."""

    def _diagram(self):
        blocks = [_blk("a", "Step", 0, 0, 1), _blk("b", "Step", 0, 0, 1), _blk("gain")]
        for b in blocks:
            b.username = None
        lines = [_line("a", "gain"), _line("b", "gain"), _line("a", "gain", dstport=1)]
        return blocks, lines

    def test_helper_finds_the_pair_with_its_lines(self):
        _, lines = self._diagram()
        assert find_duplicate_input_connections(lines) == {("gain", 0): lines[:2]}

    def test_both_entry_points_report_the_same_pair(self):
        blocks, lines = self._diagram()

        ok, messages = validate_block_connections(blocks, lines)
        assert ok is False
        assert messages == ["Multiple connections to same input port: block gain, port 0"]

        validator = DiagramValidator(SimpleNamespace(blocks_list=blocks, line_list=lines))
        validator._check_duplicate_connections()
        assert len(validator.errors) == 1
        err = validator.errors[0]
        assert err.severity is ErrorSeverity.ERROR
        assert err.message == "Block 'gain' input port 1 has 2 connections"
        assert err.connections == lines[:2]
        assert err.blocks == [blocks[2]]

    def test_hidden_lines_are_ignored_by_the_validator_only(self):
        blocks, lines = self._diagram()
        lines[1].hidden = True
        validator = DiagramValidator(SimpleNamespace(blocks_list=blocks, line_list=lines))
        validator._check_duplicate_connections()
        assert validator.errors == []
        # The pre-flight gate works on the raw line list and keeps flagging it.
        assert validate_block_connections(blocks, lines)[0] is False


@pytest.mark.unit
class TestDetectAlgebraicLoops:
    def test_pure_gain_loop_is_reported_with_its_blocks(self):
        blocks = [_blk("g1"), _blk("g2"), _blk("g3")]
        lines = [_line("g1", "g2"), _line("g2", "g3"), _line("g3", "g1")]
        no_loops, errors = detect_algebraic_loops(blocks, lines)
        assert no_loops is False
        assert errors == ["Algebraic loop detected involving blocks: ['g1', 'g2', 'g3']"]

    def test_b_type_1_memory_block_breaks_the_loop(self):
        blocks = [_blk("sum", "Sum"), _blk("int", "Integrator", b_type=1)]
        assert detect_algebraic_loops(blocks, [_line("sum", "int"), _line("int", "sum")]) == (
            True,
            [],
        )

    def test_strictly_proper_tf_breaks_the_loop_from_raw_params(self):
        tf = _blk("tf", "TranFn", params={"numerator": [1.0], "denominator": [1.0, 1.0]})
        blocks = [_blk("sum", "Sum"), tf]
        assert detect_algebraic_loops(blocks, [_line("sum", "tf"), _line("tf", "sum")])[0] is True

    def test_feedthrough_tf_does_not_break_the_loop(self):
        tf = _blk("tf", "TranFn", params={"numerator": [1.0, 1.0], "denominator": [1.0, 2.0]})
        blocks = [_blk("sum", "Sum"), tf]
        assert detect_algebraic_loops(blocks, [_line("sum", "tf"), _line("tf", "sum")])[0] is False

    def test_subsystem_with_memory_inside_breaks_the_loop(self):
        inner = _blk("inner_int", "Integrator", b_type=1)
        sub = _blk("sub", "Subsystem", sub_blocks=[_blk("gain"), inner])
        blocks = [_blk("sum", "Sum"), sub]
        assert detect_algebraic_loops(blocks, [_line("sum", "sub"), _line("sub", "sum")])[0] is True

    def test_nested_subsystem_memory_is_found(self):
        deep = _blk("deep", "Subsystem", sub_blocks=[_blk("d_int", "Integrator", b_type=1)])
        sub = _blk("sub", "Subsystem", sub_blocks=[deep])
        blocks = [_blk("sum", "Sum"), sub]
        assert detect_algebraic_loops(blocks, [_line("sum", "sub"), _line("sub", "sum")])[0] is True

    def test_algebraic_subsystem_does_not_break_the_loop(self):
        sub = _blk("sub", "Subsystem", sub_blocks=[_blk("gain")])
        blocks = [_blk("sum", "Sum"), sub]
        assert (
            detect_algebraic_loops(blocks, [_line("sum", "sub"), _line("sub", "sum")])[0] is False
        )

    def test_dangling_lines_are_ignored(self):
        blocks = [_blk("g1"), _blk("g2")]
        lines = [_line("g1", "g2"), _line("g2", "ghost"), _line("ghost", "g1")]
        assert detect_algebraic_loops(blocks, lines) == (True, [])

    def test_empty_diagram(self):
        assert detect_algebraic_loops([], []) == (True, [])


@pytest.mark.unit
class TestSafetyChecks:
    def test_well_formed_block_passes(self):
        assert check_block_integrity(_blk("g")) == (True, [])

    def test_missing_attributes_and_bad_values(self):
        ok, errors = check_block_integrity(
            SimpleNamespace(name="x", in_ports=-1, out_ports=1, b_type=7)
        )
        assert ok is False
        assert "Block missing required attribute: sid" in errors
        assert "Block missing required attribute: fn_name" in errors
        assert "Block has negative input ports" in errors
        assert "Invalid block type: 7" in errors

    def test_runnable_dsim_passes(self):
        dsim = SimpleNamespace(
            blocks_list=[_blk("g")],
            line_list=[],
            execution_initialized=False,
            sim_time=1.0,
            sim_dt=0.1,
        )
        assert check_simulation_state(dsim) == (True, [])

    def test_empty_diagram_and_bad_parameters(self):
        dsim = SimpleNamespace(
            blocks_list=[], line_list=[], execution_initialized=False, sim_time=0.0, sim_dt=-1.0
        )
        ok, errors = check_simulation_state(dsim)
        assert ok is False
        assert errors == [
            "No blocks in simulation",
            "Invalid simulation time",
            "Invalid simulation time step",
        ]

    def test_bad_block_is_reported_with_its_index(self):
        dsim = SimpleNamespace(
            blocks_list=[_blk("g"), SimpleNamespace(name="bad")],
            line_list=[],
            execution_initialized=False,
        )
        ok, errors = check_simulation_state(dsim)
        assert ok is False
        assert all(e.startswith("Block 1: ") for e in errors)

    def test_missing_dsim_attributes(self):
        ok, errors = check_simulation_state(SimpleNamespace())
        assert ok is False
        assert errors == [
            "DSim missing required attribute: blocks_list",
            "DSim missing required attribute: line_list",
            "DSim missing required attribute: execution_initialized",
        ]
