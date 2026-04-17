"""
Unit tests for Flattener subsystem boundary-connection resolution.

Regression coverage for the silent-drop bug: when the flattener cannot
resolve a connection that crosses a Subsystem boundary (either an Inport
inside a nested subsystem or an Outport driven from a Subsystem output),
it used to log a warning and return None. That silently dropped the
connection and downstream blocks read zeros with no user-visible error.

The fix: when BOTH ports_map and the naming-convention fallbacks fail,
the flattener raises RuntimeError. The conventional-name path (Out1 /
outport1 / In1 / inport1) is preserved for backward compatibility with
older .diablos files.
"""

import pytest
from PyQt5.QtCore import QRect

from lib.engine.flattener import Flattener
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


def _make_gain(name, sid):
    """Create a minimal gain-like primitive block used as internal logic."""
    g = DBlock(
        "Gain", sid, QRect(0, 0, 50, 50), None,
        1, 1, 2, 'both', "gain_fn", {'val': 1}, False,
    )
    g.name = name
    return g


def _make_const(name, sid):
    """Create a minimal source primitive block."""
    c = DBlock(
        "Constant", sid, QRect(0, 0, 50, 50), None,
        0, 1, 0, 'output', "const_fn", {'val': 1}, False,
    )
    c.name = name
    return c


def _make_sink(name, sid):
    """Create a minimal sink primitive block."""
    s = DBlock(
        "Scope", sid, QRect(0, 0, 50, 50), None,
        1, 0, 2, 'input', "scope_fn", {}, False,
    )
    s.name = name
    return s


@pytest.mark.unit
class TestFlattenerSubsystemResolution:
    """Regression coverage for the silent-drop resolution bug."""

    def test_conventional_outport_name_resolves(self, qapp):
        """
        Sanity: a subsystem whose internal Outport uses the conventional
        name 'Out1' (with no ports_map entry) must still flatten
        correctly. This exercises the fallback path that historically
        worked and MUST keep working for legacy .diablos files.
        """
        from blocks.subsystem import Subsystem
        from blocks.inport import Inport
        from blocks.outport import Outport

        # Build subsystem: In1 -> Gain1 -> Out1
        subsys = Subsystem()
        subsys.name = "Sub1"
        subsys.sid = 1

        inport = Inport("In1")
        inport.name = "In1"
        inport.sid = 1

        gain = _make_gain("Gain1", 3)

        outport = Outport("Out1")
        outport.name = "Out1"
        outport.sid = 2

        subsys.sub_blocks.extend([inport, gain, outport])
        subsys.sub_lines.extend([
            DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
            DLine(2, "Gain1", 0, "Out1", 0, [(0, 0), (10, 10)]),
        ])

        # Top level: Const -> Sub1 -> Scope
        const = _make_const("Const1", 4)
        scope = _make_sink("Scope1", 5)

        top_blocks = [const, subsys, scope]
        top_lines = [
            DLine(3, "Const1", 0, "Sub1", 0, [(0, 0), (10, 10)]),
            DLine(4, "Sub1", 0, "Scope1", 0, [(0, 0), (10, 10)]),
        ]

        flattener = Flattener()
        flat_blocks, flat_lines = flattener.flatten(top_blocks, top_lines)

        block_names = {b.name for b in flat_blocks}
        assert "Const1" in block_names
        assert "Sub1/Gain1" in block_names
        assert "Scope1" in block_names

        connections = {(l.srcblock, l.dstblock) for l in flat_lines}
        assert ("Const1", "Sub1/Gain1") in connections, (
            "Outport Out1 should resolve via the conventional-name fallback"
        )
        assert ("Sub1/Gain1", "Scope1") in connections, (
            "Outport Out1 should resolve via the conventional-name fallback"
        )

    def test_unresolvable_outport_raises_runtime_error(self, qapp):
        """
        The bug: a subsystem whose internal Outport has a non-conventional
        name (e.g. 'y_out') and no ports_map entry used to be silently
        dropped. After the fix, the flattener must raise RuntimeError
        when ALL resolution paths (ports_map + conventional name +
        case-insensitive scan) fail.
        """
        from blocks.subsystem import Subsystem
        from blocks.inport import Inport
        from blocks.outport import Outport

        # Build subsystem: In1 -> Gain1 -> y_out
        # (Outport has a non-convention name and no ports_map entry.)
        subsys = Subsystem()
        subsys.name = "Sub1"
        subsys.sid = 1
        # Explicitly leave ports_map empty to exercise the failure path.
        subsys.ports_map = {}

        inport = Inport("In1")
        inport.name = "In1"
        inport.sid = 1

        gain = _make_gain("Gain1", 3)

        outport = Outport("y_out")
        outport.name = "y_out"  # non-convention name
        outport.sid = 2

        subsys.sub_blocks.extend([inport, gain, outport])
        subsys.sub_lines.extend([
            DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
            DLine(2, "Gain1", 0, "y_out", 0, [(0, 0), (10, 10)]),
        ])

        # Top level: Const -> Sub1 -> Scope (drives through Sub1 output 0)
        const = _make_const("Const1", 4)
        scope = _make_sink("Scope1", 5)

        top_blocks = [const, subsys, scope]
        top_lines = [
            DLine(3, "Const1", 0, "Sub1", 0, [(0, 0), (10, 10)]),
            DLine(4, "Sub1", 0, "Scope1", 0, [(0, 0), (10, 10)]),
        ]

        flattener = Flattener()
        with pytest.raises(RuntimeError, match=r"Could not resolve Subsystem"):
            flattener.flatten(top_blocks, top_lines)

    def test_ports_map_resolves_non_convention_outport(self, qapp):
        """
        When ports_map IS populated, a non-convention Outport name should
        still resolve — documenting the authoritative-map path so future
        callers (UI builders) can rely on it.
        """
        from blocks.subsystem import Subsystem
        from blocks.inport import Inport
        from blocks.outport import Outport

        subsys = Subsystem()
        subsys.name = "Sub1"
        subsys.sid = 1
        # Authoritative map: external output port 0 -> internal 'y_out'
        subsys.ports_map = {'output': {0: 'y_out'}, 'input': {0: 'In1'}}

        inport = Inport("In1")
        inport.name = "In1"
        inport.sid = 1

        gain = _make_gain("Gain1", 3)

        outport = Outport("y_out")
        outport.name = "y_out"
        outport.sid = 2

        subsys.sub_blocks.extend([inport, gain, outport])
        subsys.sub_lines.extend([
            DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
            DLine(2, "Gain1", 0, "y_out", 0, [(0, 0), (10, 10)]),
        ])

        const = _make_const("Const1", 4)
        scope = _make_sink("Scope1", 5)

        top_blocks = [const, subsys, scope]
        top_lines = [
            DLine(3, "Const1", 0, "Sub1", 0, [(0, 0), (10, 10)]),
            DLine(4, "Sub1", 0, "Scope1", 0, [(0, 0), (10, 10)]),
        ]

        flattener = Flattener()
        flat_blocks, flat_lines = flattener.flatten(top_blocks, top_lines)

        connections = {(l.srcblock, l.dstblock) for l in flat_lines}
        assert ("Sub1/Gain1", "Scope1") in connections, (
            "ports_map should authoritatively resolve the non-convention Outport"
        )
