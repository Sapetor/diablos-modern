"""Compiled-solver coverage for the Sine source block.

Ports the uniquely valuable assertions from the retired legacy test
``tests/test_sine_params.py``: the compiler accepts a lowercase ``sine``
block_fn (case-insensitivity) and the compiled fast solver reproduces
``amplitude * sin(omega * t)`` for a Sine block parameterised with ``omega``.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor

from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.engine.simulation_engine import SimulationEngine
from lib.engine.system_compiler import SystemCompiler
from lib.engine.block_names import canonical_fn
from blocks.sine import SineBlock
from blocks.scope import ScopeBlock


class _MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}


def _sine(block_fn="sine"):
    sine = DBlock(block_fn, 1, QRect(0, 0, 50, 50), QColor("blue"),
                  in_ports=0, out_ports=1, b_type=0, block_class=SineBlock)
    sine.params = {"amplitude": 1.0, "omega": 2.0, "init_angle": 0.0}
    sine.hierarchy = 0
    return sine


@pytest.mark.unit
class TestCompiledSine:
    def test_lowercase_sine_is_canonicalized(self):
        assert canonical_fn("sine") == "Sine"

    def test_lowercase_sine_is_compilable(self, qapp):
        compiler = SystemCompiler()
        assert compiler.check_compilability([_sine("sine")])

    def test_compiled_sine_matches_analytic(self, qapp):
        engine = SimulationEngine(_MockModel())
        T, dt = 10.0, 0.01

        sine = _sine("sine")  # lowercase block_fn exercises the case fix
        scope = DBlock("Scope", 2, QRect(100, 0, 50, 50), QColor("black"),
                       in_ports=1, out_ports=0, b_type=1, block_class=ScopeBlock)
        scope.hierarchy = 1
        line = DLine(1, sine.name, 0, scope.name, 0, [QPoint(0, 0), QPoint(100, 0)])

        blocks = [sine, scope]
        lines = [line]

        assert engine.check_compilability(blocks)
        assert engine.run_compiled_simulation(blocks, lines, (0.0, T), dt)

        data = np.asarray(scope.exec_params["vector"], dtype=float).flatten()
        t = np.arange(0, T + dt, dt)[:len(data)]
        expected = np.sin(2.0 * t)
        n = min(len(data), len(expected))
        assert n > 0
        assert np.max(np.abs(data[:n] - expected[:n])) < 1e-2
