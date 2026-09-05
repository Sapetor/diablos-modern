"""Subsystem support in ``BaseAnalyzer._extract_system_model``.

The Subsystem branch used to be a bare ``pass``: analysis fell through to the
parameter-based extraction, found no numerator/denominator on the container,
and returned ``None`` -- so Bode/Nyquist/root-locus silently reported "no
linear system found" for any diagram whose plant lived inside a subsystem.

It now flattens the subsystem (with the engine's own Flattener) and composes
the enclosed series chain, or refuses with a specific reason in
``analyzer.last_error``.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QRect

from lib.analysis.analyzers.base_analyzer import BaseAnalyzer
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine


@pytest.fixture(autouse=True)
def _require_qapp(qapp):
    """DBlock builds Qt objects in its constructor."""
    return qapp


def _primitive(block_fn, name, sid, in_ports, out_ports, params):
    block = DBlock(
        block_fn,
        sid,
        QRect(0, 0, 50, 50),
        None,
        in_ports,
        out_ports,
        2,
        "both",
        f"{block_fn.lower()}_fn",
        params,
        False,
    )
    block.name = name
    return block


def _subsystem(name, sub_blocks, sub_lines):
    from blocks.subsystem import Subsystem

    subsys = Subsystem()
    subsys.name = name
    subsys.sid = 1
    subsys.sub_blocks.extend(sub_blocks)
    subsys.sub_lines.extend(sub_lines)
    return subsys


def _io_ports(sub_name="Sub1"):
    from blocks.inport import Inport
    from blocks.outport import Outport

    inport = Inport("In1")
    inport.name = "In1"
    inport.sid = 1
    outport = Outport("Out1")
    outport.name = "Out1"
    outport.sid = 2
    return inport, outport


@pytest.mark.unit
class TestSubsystemModelExtraction:
    def test_series_chain_is_composed(self):
        """In1 -> Gain(3) -> TranFn 1/(s+2) -> Out1  =>  3 / (s + 2)."""
        inport, outport = _io_ports()
        gain = _primitive("Gain", "Gain1", 3, 1, 1, {"gain": 3.0})
        tf = _primitive("TranFn", "Tf1", 4, 1, 1, {"numerator": [1.0], "denominator": [1.0, 2.0]})
        subsys = _subsystem(
            "Sub1",
            [inport, gain, tf, outport],
            [
                DLine(1, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
                DLine(2, "Gain1", 0, "Tf1", 0, [(0, 0), (10, 10)]),
                DLine(3, "Tf1", 0, "Out1", 0, [(0, 0), (10, 10)]),
            ],
        )

        analyzer = BaseAnalyzer()
        model = analyzer._extract_system_model(subsys, canvas=None)

        assert model is not None, analyzer.last_error
        num, den, dt = model
        np.testing.assert_allclose(np.asarray(num, dtype=float), [3.0])
        np.testing.assert_allclose(np.asarray(den, dtype=float), [1.0, 2.0])
        assert dt == 0.0
        assert analyzer.last_error is None

    def test_two_transfer_functions_multiply(self):
        """1/(s+1) * 1/(s+2) = 1 / (s^2 + 3s + 2)."""
        inport, outport = _io_ports()
        tf1 = _primitive("TranFn", "Tf1", 3, 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]})
        tf2 = _primitive("TranFn", "Tf2", 4, 1, 1, {"numerator": [1.0], "denominator": [1.0, 2.0]})
        subsys = _subsystem(
            "Sub1",
            [inport, tf1, tf2, outport],
            [
                DLine(1, "In1", 0, "Tf1", 0, [(0, 0), (10, 10)]),
                DLine(2, "Tf1", 0, "Tf2", 0, [(0, 0), (10, 10)]),
                DLine(3, "Tf2", 0, "Out1", 0, [(0, 0), (10, 10)]),
            ],
        )

        model = BaseAnalyzer()._extract_system_model(subsys, canvas=None)
        assert model is not None
        num, den, _dt = model
        np.testing.assert_allclose(np.asarray(num, dtype=float), [1.0])
        np.testing.assert_allclose(np.asarray(den, dtype=float), [1.0, 3.0, 2.0])

    def test_branching_subsystem_is_reported_not_ignored(self):
        """Two parallel sinks: not a single series path -> explicit refusal."""
        inport, outport = _io_ports()
        tf = _primitive("TranFn", "Tf1", 3, 1, 1, {"numerator": [1.0], "denominator": [1.0, 1.0]})
        gain = _primitive("Gain", "Gain1", 4, 1, 1, {"gain": 2.0})
        subsys = _subsystem(
            "Sub1",
            [inport, tf, gain, outport],
            [
                DLine(1, "In1", 0, "Tf1", 0, [(0, 0), (10, 10)]),
                DLine(2, "In1", 0, "Gain1", 0, [(0, 0), (10, 10)]),
                DLine(3, "Tf1", 0, "Out1", 0, [(0, 0), (10, 10)]),
            ],
        )

        analyzer = BaseAnalyzer()
        assert analyzer._extract_system_model(subsys, canvas=None) is None
        assert "single signal path" in analyzer.last_error
        assert "Sub1" in analyzer.last_error

    def test_non_linear_member_is_reported(self):
        inport, outport = _io_ports()
        sat = _primitive("Saturation", "Sat1", 3, 1, 1, {"upper": 1.0, "lower": -1.0})
        subsys = _subsystem(
            "Sub1",
            [inport, sat, outport],
            [
                DLine(1, "In1", 0, "Sat1", 0, [(0, 0), (10, 10)]),
                DLine(2, "Sat1", 0, "Out1", 0, [(0, 0), (10, 10)]),
            ],
        )

        analyzer = BaseAnalyzer()
        assert analyzer._extract_system_model(subsys, canvas=None) is None
        assert "no linear model" in analyzer.last_error
        assert "Sat1" in analyzer.last_error

    def test_empty_subsystem_is_reported(self):
        analyzer = BaseAnalyzer()
        assert analyzer._extract_system_model(_subsystem("Sub1", [], []), canvas=None) is None
        assert "no contents" in analyzer.last_error
