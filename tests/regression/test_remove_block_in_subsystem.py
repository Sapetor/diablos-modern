"""Deleting a block inside a subsystem must drop its wires from sub_lines.

SimulationModel.remove_block used to rebind ``line_list`` to a new list. Inside
a subsystem ``line_list`` *is* the subsystem's ``sub_lines`` (enter_subsystem
aliases it and exit_subsystem never writes back), so the rebinding left the
deleted block's wires in ``sub_lines``: dangling wires that reappeared on
re-entry, save and simulation.
"""

import pytest
from PyQt6.QtCore import QRect

from blocks.inport import Inport
from blocks.outport import Outport
from blocks.subsystem import Subsystem
from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


def _subsystem():
    sub = Subsystem()
    sub.name = "Sub1"
    inport, outport = Inport("In1"), Outport("Out1")
    inport.name, outport.name = "In1", "Out1"
    gain = DBlock(
        "Gain", 3, QRect(0, 0, 50, 50), None, 1, 1, 2, "both", "gain_fn", {"val": 5}, False
    )
    gain.name = "Gain1"
    sub.sub_blocks.extend([inport, gain, outport])
    sub.sub_lines.extend(
        [
            DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
            DLine(2, "Gain1", 0, "Out1", 0, [(20, 20), (30, 30)]),
        ]
    )
    return sub, gain


@pytest.mark.regression
def test_removed_block_wires_leave_sub_lines(qapp):
    dsim = DSim()
    sub, gain = _subsystem()
    dsim.blocks_list.append(sub)

    dsim.enter_subsystem(sub)
    dsim.model.remove_block(gain)
    assert dsim.line_list == []  # the active view is right either way
    dsim.exit_subsystem()

    assert [b.name for b in sub.sub_blocks] == ["In1", "Out1"]
    assert sub.sub_lines == [], "deleted block's wires must not survive in sub_lines"


@pytest.mark.regression
def test_remove_block_keeps_the_root_list_object(qapp):
    # Anything holding the active list (DSim, the navigation stack) must keep
    # seeing the same object after a delete.
    dsim = DSim()
    sub, gain = _subsystem()
    dsim.model.blocks_list.extend(sub.sub_blocks)
    dsim.model.line_list.extend(sub.sub_lines)
    lines = dsim.model.line_list

    dsim.model.remove_block(gain)
    assert dsim.model.line_list is lines
    assert lines == []
