"""Regression tests for mask resolution inside the Flattener.

Two guarantees:

  1. A diagram with **no** masks flattens exactly as it did before masks
     existed -- every inner parameter string is passed through untouched, so
     workspace variables are still resolved later by ``WorkspaceManager``.
  2. A masked subsystem's inner parameters are resolved from the mask scope
     *into the flattener's clones only*; the user's stored parameter strings
     are never mutated, so the expression survives save/load and re-runs.
"""

import copy

import pytest
from PyQt5.QtCore import QPoint, QRect

from lib.engine.flattener import Flattener
from lib.masks import set_mask
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


def _primitive(block_fn, name, params, in_ports=1, out_ports=1, b_type=2):
    block = DBlock(
        block_fn=block_fn,
        sid=0,
        coords=QRect(10, 10, 80, 60),
        color="#4CAF50",
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=b_type,
        io_edit="none",
        fn_name=block_fn.lower(),
        params=dict(params),
    )
    block.name = name
    block.params["_name_"] = name
    return block


def _line(sid, src, sp, dst, dp):
    return DLine(sid, src, sp, dst, dp, [QPoint(0, 0), QPoint(10, 10)])


def _subsystem(qapp, name, inner_blocks, inner_lines, mask=None):
    from blocks.subsystem import Subsystem

    subsys = Subsystem(block_name=name, sid=1, coords=QRect(200, 100, 120, 90))
    subsys.name = name
    subsys.sub_blocks.extend(inner_blocks)
    subsys.sub_lines.extend(inner_lines)
    if mask is not None:
        set_mask(subsys, mask)
    return subsys


def _plant(qapp, name="Plant", mask=None, gain="K", labels="speed"):
    """A subsystem: In1 -> Gain -> Scope, with an unconnected Outport."""
    from blocks.inport import Inport
    from blocks.outport import Outport

    inport = Inport(block_name="In1", sid=1, coords=QRect(10, 10, 40, 40))
    inport.name = "inport1"
    gain_block = _primitive("Gain", "gain0", {"gain": gain})
    scope = _primitive("Scope", "scope0", {"labels": labels}, out_ports=0, b_type=3)
    outport = Outport(block_name="Out1", sid=1, coords=QRect(300, 10, 40, 40))
    outport.name = "outport1"

    lines = [
        _line(0, "inport1", 0, "gain0", 0),
        _line(1, "gain0", 0, "scope0", 0),
        _line(2, "gain0", 0, "outport1", 0),
    ]
    return _subsystem(qapp, name, [inport, gain_block, scope, outport], lines, mask=mask)


GAIN_MASK = {
    "name": "Plant",
    "category": "User Library",
    "parameters": [{"name": "K", "type": "float", "default": 2.5, "doc": "Loop gain"}],
}


@pytest.mark.regression
class TestNoMaskIsUnchanged:
    def test_inner_param_strings_pass_through_untouched(self, qapp):
        """Without a mask the flattener must not evaluate anything."""
        source = _primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0)
        subsys = _plant(qapp)
        top_lines = [_line(0, "step0", 0, "Plant", 0)]

        before = copy.deepcopy(subsys.sub_blocks[1].params)
        blocks, _lines = Flattener().flatten([source, subsys], top_lines)

        flat = {b.name: b for b in blocks}
        assert flat["Plant/gain0"].params["gain"] == "K"  # still the raw string
        assert flat["Plant/scope0"].params["labels"] == "speed"
        # ...and the original block was not touched either.
        assert subsys.sub_blocks[1].params == before

    def test_topology_matches_the_unmasked_baseline(self, qapp):
        """Masking must not change which primitives and wires come out."""
        plain = _plant(qapp)
        masked = _plant(qapp, mask=GAIN_MASK)

        def signature(subsys):
            blocks, lines = Flattener().flatten(
                [
                    _primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0),
                    subsys,
                ],
                [_line(0, "step0", 0, subsys.name, 0)],
            )
            return (
                sorted(b.name for b in blocks),
                sorted((ln.srcblock, ln.srcport, ln.dstblock, ln.dstport) for ln in lines),
            )

        assert signature(plain) == signature(masked)


@pytest.mark.regression
class TestMaskedResolution:
    def test_mask_parameter_reaches_the_inner_block(self, qapp):
        subsys = _plant(qapp, mask=GAIN_MASK)
        blocks, _lines = Flattener().flatten(
            [_primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0), subsys],
            [_line(0, "step0", 0, "Plant", 0)],
        )
        flat = {b.name: b for b in blocks}
        assert flat["Plant/gain0"].params["gain"] == 2.5
        # A non-expression string is still left alone.
        assert flat["Plant/scope0"].params["labels"] == "speed"

    def test_instance_value_overrides_the_mask_default(self, qapp):
        subsys = _plant(qapp, mask=GAIN_MASK)
        subsys.params["K"] = 7.0
        blocks, _lines = Flattener().flatten(
            [_primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0), subsys],
            [_line(0, "step0", 0, "Plant", 0)],
        )
        assert {b.name: b for b in blocks}["Plant/gain0"].params["gain"] == 7.0

    def test_stored_params_are_never_mutated(self, qapp):
        """Re-running must keep resolving from the original expression."""
        subsys = _plant(qapp, mask=GAIN_MASK)
        top = [_primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0), subsys]
        top_lines = [_line(0, "step0", 0, "Plant", 0)]

        Flattener().flatten(top, top_lines)
        assert subsys.sub_blocks[1].params["gain"] == "K"

        subsys.params["K"] = 9.0
        blocks, _lines = Flattener().flatten(top, top_lines)
        assert {b.name: b for b in blocks}["Plant/gain0"].params["gain"] == 9.0

    def test_nested_masks_resolve_outer_to_inner(self, qapp):
        """Inner mask values are evaluated in the outer mask's scope."""
        from blocks.inport import Inport
        from blocks.outport import Outport

        inner = _plant(qapp, name="Inner", mask=GAIN_MASK)
        inner.params["K"] = "2 * Kout"

        outer_in = Inport(block_name="In1", sid=1, coords=QRect(10, 10, 40, 40))
        outer_in.name = "inport1"
        outer_out = Outport(block_name="Out1", sid=1, coords=QRect(400, 10, 40, 40))
        outer_out.name = "outport1"
        outer = _subsystem(
            qapp,
            "Outer",
            [outer_in, inner, outer_out],
            [_line(0, "inport1", 0, "Inner", 0), _line(1, "Inner", 0, "outport1", 0)],
            mask={
                "name": "Outer",
                "parameters": [{"name": "Kout", "type": "float", "default": 3.0}],
            },
        )

        blocks, _lines = Flattener().flatten(
            [_primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0), outer],
            [_line(0, "step0", 0, "Outer", 0)],
        )
        flat = {b.name: b for b in blocks}
        assert flat["Outer/Inner/gain0"].params["gain"] == 6.0

    def test_unmasked_subsystem_inherits_the_enclosing_mask_scope(self, qapp):
        from blocks.inport import Inport
        from blocks.outport import Outport

        inner = _plant(qapp, name="Inner")  # no mask of its own

        outer_in = Inport(block_name="In1", sid=1, coords=QRect(10, 10, 40, 40))
        outer_in.name = "inport1"
        outer_out = Outport(block_name="Out1", sid=1, coords=QRect(400, 10, 40, 40))
        outer_out.name = "outport1"
        outer = _subsystem(
            qapp,
            "Outer",
            [outer_in, inner, outer_out],
            [_line(0, "inport1", 0, "Inner", 0), _line(1, "Inner", 0, "outport1", 0)],
            mask={"name": "Outer", "parameters": [{"name": "K", "type": "float", "default": 4.0}]},
        )

        blocks, _lines = Flattener().flatten(
            [_primitive("Step", "step0", {"value": 1.0}, in_ports=0, b_type=0), outer],
            [_line(0, "step0", 0, "Outer", 0)],
        )
        assert {b.name: b for b in blocks}["Outer/Inner/gain0"].params["gain"] == 4.0
