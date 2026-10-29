"""A subsystem's external ports map to the boundary blocks the user placed.

``SubsystemManager._sync_subsystem_ports`` rebuilt the parent's numbered
external ports from the Inport/Outport blocks (sorted top to bottom) but left
``ports_map`` alone, so the flattener fell back to the boundary block's name:
an Inport added as ``inport99`` resolved to input 98 and the run failed with
"Could not map Inport 'Subsystem1/inport99' ...". The sync now writes
``ports_map`` from the same ordering, and the parent's wires follow their
boundary block when ports are renumbered, reordered or deleted.

Every test edits a diagram the way the GUI does, saves it, reopens the file
and runs it, asserting the values that reach the Scope -- a mis-routed port
shows up as a wrong number, not just a wrong count.
"""

import json

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QRect

A, B = 1.0, 10.0  # the two root sources; a Sum "+-" of them gives A - B = -9


@pytest.fixture
def dsim(qapp):
    from lib.cli import new_headless_dsim

    return new_headless_dsim()


def _add(dsim, fn_name, x, y, **params):
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    block = dsim.add_block(menu[fn_name], QPoint(x, y))
    block.params.update(params)
    return block


def _boundary(dsim, kind, name, top):
    """Add an Inport/Outport called ``name`` to the current scope, ``top`` px down."""
    from blocks.inport import Inport
    from blocks.outport import Outport

    cls, x = (Inport, 20) if kind == "in" else (Outport, 600)
    sid = max([b.sid for b in dsim.model.blocks_list] + [0]) + 1
    port = cls(block_name=name, sid=sid, coords=QRect(x, top, 40, 30), color="#888888")
    port.name = name
    dsim.model.blocks_list.append(port)
    return port


def _wire(dsim, src, srcport, dst, dstport):
    line = dsim.add_line(
        (src.name, srcport, src.out_coords[srcport]), (dst.name, dstport, dst.in_coords[dstport])
    )
    assert line is not None
    return line


def _terminate(dsim, block):
    """Sink ``block``'s output so the integrity check does not flag it unused."""
    _wire(dsim, block, 0, _add(dsim, "term", block.left + 150, block.top + 50), 0)


def _named(blocks, name):
    return next(b for b in blocks if b.name == name)


def _lines_into(lines, block_name):
    return sorted((l.srcblock, l.dstport) for l in lines if l.dstblock == block_name)


def _root_sources(dsim):
    """Constants ``a`` (=A) and ``b`` (=B) plus a Scope labelled ``y``."""
    a = _add(dsim, "constant", 50, 100, value=A)
    b = _add(dsim, "constant", 50, 300, value=B)
    scope = _add(dsim, "scope", 700, 150, labels="y")
    return a, b, scope


def _save_reopen_run(dsim, path):
    """Save ``dsim`` to ``path``, reopen the file and simulate it; return the reopened DSim."""
    from lib.cli import load_diagram

    assert dsim.save(filepath=str(path)) == 0
    again, _ = load_diagram(str(path))
    ok, err = again.run_tuning_simulation(0.1, 0.01)
    assert ok, err
    return again


def _final(dsim, label):
    from lib.analysis.resim import harvest_scope_signals

    signals = harvest_scope_signals(dsim)["signals"]
    return float(np.atleast_1d(signals[label])[-1])


def _saved_subsystem(path, name):
    with open(path, encoding="utf-8") as fp:
        return next(r for r in json.load(fp)["blocks_data"] if r["name"] == name)


def _gain_subsystem(dsim):
    """Root ``a -> [inport1 -> gain -> outport1] -> scope``; ``b`` unconnected.

    Returns ``(a, b, subsystem)``.
    """
    a, b, scope = _root_sources(dsim)
    gain = _add(dsim, "gain", 300, 100, gain=1.0)
    _wire(dsim, a, 0, gain, 0)
    _wire(dsim, gain, 0, scope, 0)
    sub = dsim.create_subsystem_from_selection([gain])
    assert sub.in_ports == 1 and sub.out_ports == 1
    return a, b, sub


def _difference_inside(dsim, extra_inport):
    """Inside a ``_gain_subsystem``: ``outport1 = gain - extra_inport``."""
    lines = dsim.model.line_list
    gain = next(b for b in dsim.model.blocks_list if b.block_fn == "Gain")
    outport = next(b for b in dsim.model.blocks_list if b.block_fn == "Outport")
    for line in [l for l in lines if l.srcblock == gain.name]:
        dsim.model.remove_line(line)
    diff = _add(dsim, "sum", 400, gain.top, sign="+-")
    _wire(dsim, gain, 0, diff, 0)
    _wire(dsim, extra_inport, 0, diff, 1)
    _wire(dsim, diff, 0, outport, 0)


@pytest.mark.regression
class TestAddedPortsAreMapped:
    def test_added_inport_is_saved_in_ports_map_and_routes(self, dsim, tmp_path):
        """The review's repro: ``inport99`` added as the second input."""
        a, b, sub = _gain_subsystem(dsim)
        dsim.enter_subsystem(sub)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        inport99 = _boundary(dsim, "in", "inport99", inport1.top + 120)
        _difference_inside(dsim, inport99)

        # Saving from inside syncs the parent, ports_map included.
        inside = tmp_path / "inside.diablos"
        assert dsim.save(filepath=str(inside)) == 0
        rec = _saved_subsystem(inside, sub.name)
        assert len(rec["ports"]["in"]) == 2
        assert rec["ports_map"]["input"] == {"0": "inport1", "1": "inport99"}

        dsim.exit_subsystem()
        assert sub.ports_map["input"] == {0: "inport1", 1: "inport99"}
        _wire(dsim, b, 0, sub, 1)

        again = _save_reopen_run(dsim, tmp_path / "added.diablos")
        assert np.isclose(_final(again, "y"), A - B)

    def test_inport_added_above_takes_the_top_port(self, dsim, tmp_path):
        """A new Inport placed above ``inport1`` becomes input 0; ``a``'s wire follows inport1."""
        a, b, sub = _gain_subsystem(dsim)
        dsim.enter_subsystem(sub)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        inport99 = _boundary(dsim, "in", "inport99", inport1.top - 120)
        _difference_inside(dsim, inport99)
        dsim.exit_subsystem()

        assert sub.ports_map["input"] == {0: "inport99", 1: "inport1"}
        assert _lines_into(dsim.model.line_list, sub.name) == [(a.name, 1)]
        _wire(dsim, b, 0, sub, 0)

        again = _save_reopen_run(dsim, tmp_path / "above.diablos")
        assert np.isclose(_final(again, "y"), A - B)

    def test_added_outport_routes(self, dsim, tmp_path):
        """``outport99`` (= 5 * input) becomes output 1 and drives a second Scope."""
        a, b, sub = _gain_subsystem(dsim)
        dsim.enter_subsystem(sub)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        outport1 = _named(dsim.model.blocks_list, "outport1")
        five = _add(dsim, "gain", 300, outport1.top + 150, gain=5.0)
        outport99 = _boundary(dsim, "out", "outport99", outport1.top + 150)
        _wire(dsim, inport1, 0, five, 0)
        _wire(dsim, five, 0, outport99, 0)
        dsim.exit_subsystem()

        assert sub.ports_map["output"] == {0: "outport1", 1: "outport99"}
        scope_z = _add(dsim, "scope", 700, 400, labels="z")
        _wire(dsim, sub, 1, scope_z, 0)
        _terminate(dsim, b)

        again = _save_reopen_run(dsim, tmp_path / "outport.diablos")
        assert np.isclose(_final(again, "y"), A)
        assert np.isclose(_final(again, "z"), 5 * A)


def _difference_subsystem(dsim):
    """Root ``a, b -> [inport1, inport2 -> sum(+-) -> outport1] -> scope``."""
    a, b, scope = _root_sources(dsim)
    diff = _add(dsim, "sum", 300, 150, sign="+-")
    _wire(dsim, a, 0, diff, 0)
    _wire(dsim, b, 0, diff, 1)
    _wire(dsim, diff, 0, scope, 0)
    sub = dsim.create_subsystem_from_selection([diff])
    assert sub.ports_map["input"] == {0: "inport1", 1: "inport2"}
    assert _lines_into(dsim.model.line_list, sub.name) == [(a.name, 0), (b.name, 1)]
    return a, b, sub


@pytest.mark.regression
class TestReorderedAndRemovedPorts:
    def test_moving_inports_reorders_ports_and_wires_follow(self, dsim, tmp_path):
        a, b, sub = _difference_subsystem(dsim)
        dsim.enter_subsystem(sub)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        inport2 = _named(dsim.model.blocks_list, "inport2")
        assert inport1.top < inport2.top
        top1, top2 = inport1.top, inport2.top
        inport1.relocate_Block(QPoint(inport1.left, top2 + 60))
        inport2.relocate_Block(QPoint(inport2.left, top1))
        dsim.exit_subsystem()

        # The top external port now leads to inport2, and each root wire still
        # feeds the Inport it fed before the move.
        assert sub.ports_map["input"] == {0: "inport2", 1: "inport1"}
        assert _lines_into(dsim.model.line_list, sub.name) == [(a.name, 1), (b.name, 0)]

        again = _save_reopen_run(dsim, tmp_path / "reordered.diablos")
        reopened = _named(again.model.blocks_list, sub.name)
        assert reopened.ports_map["input"] == {0: "inport2", 1: "inport1"}
        assert _lines_into(again.model.line_list, sub.name) == [(a.name, 1), (b.name, 0)]
        assert np.isclose(_final(again, "y"), A - B)

    def test_deleted_inport_drops_its_wire_and_renumbers_the_rest(self, dsim, tmp_path):
        a, b, sub = _difference_subsystem(dsim)
        dsim.enter_subsystem(sub)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        diff = next(b for b in dsim.model.blocks_list if b.block_fn == "Sum")
        dsim.remove_block_and_lines(inport1)
        hundred = _add(dsim, "constant", 100, 40, value=100.0)
        _wire(dsim, hundred, 0, diff, 0)
        dsim.exit_subsystem()

        assert sub.in_ports == 1
        assert sub.ports_map["input"] == {0: "inport2"}
        # a fed the deleted inport1: its wire is gone. b's wire moves to port 0.
        assert _lines_into(dsim.model.line_list, sub.name) == [(b.name, 0)]
        _terminate(dsim, a)

        again = _save_reopen_run(dsim, tmp_path / "removed.diablos")
        assert np.isclose(_final(again, "y"), 100.0 - B)

    def test_resync_without_edits_changes_nothing(self, dsim):
        a, b, sub = _difference_subsystem(dsim)
        before = [(l.srcblock, l.srcport, l.dstblock, l.dstport) for l in dsim.model.line_list]
        for _ in range(2):
            dsim.enter_subsystem(sub)
            dsim.exit_subsystem()
        after = [(l.srcblock, l.srcport, l.dstblock, l.dstport) for l in dsim.model.line_list]
        assert after == before
        assert sub.ports_map == {"input": {0: "inport1", 1: "inport2"}, "output": {0: "outport1"}}


@pytest.mark.regression
class TestLegacySubsystemWithoutPortsMap:
    def test_first_sync_attributes_wires_by_boundary_name(self, qapp, tmp_path):
        """A file saved before ``ports_map`` was written keeps its wiring on resync."""
        import os

        from lib.cli import load_diagram

        example = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "library_block_demo.diablos"
        )
        dsim, _ = load_diagram(example)
        sub = next(b for b in dsim.model.blocks_list if b.block_fn == "Subsystem")
        assert not sub.ports_map
        before = sorted(
            (l.srcblock, l.srcport, l.dstblock, l.dstport) for l in dsim.model.line_list
        )

        dsim.enter_subsystem(sub)
        dsim.exit_subsystem()

        assert sub.ports_map == {"input": {0: "inport1"}, "output": {0: "outport1"}}
        after = sorted((l.srcblock, l.srcport, l.dstblock, l.dstport) for l in dsim.model.line_list)
        assert after == before
        again = _save_reopen_run(dsim, tmp_path / "legacy.diablos")
        assert again.timeline is not None and len(again.timeline) > 1


@pytest.mark.regression
class TestNestedSubsystems:
    def test_added_ports_route_through_two_levels(self, dsim, tmp_path):
        """``b`` reaches an Inport added two levels down via an Inport added one level down."""
        a, b, outer = _gain_subsystem(dsim)
        dsim.enter_subsystem(outer)
        gain = next(x for x in dsim.model.blocks_list if x.block_fn == "Gain")
        inner = dsim.create_subsystem_from_selection([gain])
        dsim.enter_subsystem(inner)
        inport1 = _named(dsim.model.blocks_list, "inport1")
        inport99 = _boundary(dsim, "in", "inport99", inport1.top + 120)
        _difference_inside(dsim, inport99)

        # Saving two levels deep maps the inner parent's ports too.
        deep = tmp_path / "deep.diablos"
        assert dsim.save(filepath=str(deep)) == 0
        inner_rec = next(
            r for r in _saved_subsystem(deep, outer.name)["sub_blocks"] if r["name"] == inner.name
        )
        assert inner_rec["ports_map"]["input"] == {"0": "inport1", "1": "inport99"}

        dsim.exit_subsystem()  # back in outer
        outer_in1 = _named(dsim.model.blocks_list, "inport1")
        inport7 = _boundary(dsim, "in", "inport7", outer_in1.top + 200)
        assert inner.ports_map["input"] == {0: "inport1", 1: "inport99"}
        _wire(dsim, inport7, 0, inner, 1)
        dsim.exit_subsystem()  # back at the root

        assert outer.ports_map["input"] == {0: "inport1", 1: "inport7"}
        _wire(dsim, b, 0, outer, 1)

        again = _save_reopen_run(dsim, tmp_path / "nested.diablos")
        assert np.isclose(_final(again, "y"), A - B)
