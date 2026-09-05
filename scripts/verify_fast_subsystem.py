"""Manual check that a Subsystem flattens and runs through the compiled solver.

Run headlessly with::

    QT_QPA_PLATFORM=offscreen python scripts/verify_fast_subsystem.py

Builds Step -> [Inport -> Gain -> Outport] -> Scope and asserts the Scope sees
``step_value * gain``.
"""

import logging
import os
import sys
import unittest

import numpy as np

# Run from anywhere: put the repo root on sys.path before importing lib/blocks.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DBlock builds QPixmap-backed icons, which needs a live QApplication even
# headless; without this the script aborts with "Must construct a
# QGuiApplication before a QPixmap".
from PyQt5.QtWidgets import QApplication

if not QApplication.instance():
    _app = QApplication(sys.argv)

from PyQt5.QtCore import QRect

from blocks.gain import GainBlock
from blocks.inport import Inport
from blocks.outport import Outport
from blocks.scope import ScopeBlock
from blocks.step import StepBlock
from blocks.subsystem import Subsystem
from lib.engine.simulation_engine import SimulationEngine
from lib.models.simulation_model import SimulationModel
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def default_params(block_class):
    """Flatten a block class's params spec into the {name: default} map DBlock wants."""
    spec = block_class().params or {}
    return {
        name: (entry.get("default") if isinstance(entry, dict) else entry)
        for name, entry in spec.items()
    }


def make_block(block_class, block_fn, sid, overrides=None, **kwargs):
    """Build a DBlock backed by ``block_class`` with its spec defaults applied."""
    params = default_params(block_class)
    params.update(overrides or {})
    return DBlock(
        block_fn,
        sid,
        QRect(0, 0, 50, 50),
        "white",
        block_class=block_class,
        params=params,
        **kwargs,
    )


class TestFastSubsystem(unittest.TestCase):
    def setUp(self):
        self.model = SimulationModel()
        self.engine = SimulationEngine(self.model)

    def _build(self, step_value=2.0, gain_value=5.0):
        """Step -> Subsystem(Inport -> Gain -> Outport) -> Scope.

        Blocks are wired by ``DBlock.name`` (``block_fn.lower() + sid``), which
        is what the Flattener and the compiler key on: "inport1"/"outport1" also
        satisfy the Flattener's conventional Inport/Outport naming, so the
        subsystem boundary resolves without an explicit ports_map.
        """
        sub = Subsystem(block_name="Sub1", sid=1)
        # Subsystem defaults to 0/0 ports; the boundary ports are what the top
        # level connects to, so declare them.
        sub.in_ports = 1
        sub.out_ports = 1

        inport = Inport(block_name="In1", sid=1)
        outport = Outport(block_name="Out1", sid=1)
        gain = make_block(GainBlock, "Gain", 1, {"gain": gain_value})

        sub.sub_blocks = [inport, gain, outport]
        sub.sub_lines = [
            DLine(1, inport.name, 0, gain.name, 0, [(0, 0), (0, 0)]),
            DLine(2, gain.name, 0, outport.name, 0, [(0, 0), (0, 0)]),
        ]

        step = make_block(StepBlock, "Step", 1, {"value": step_value, "delay": 0.0}, in_ports=0)
        scope = make_block(ScopeBlock, "Scope", 1, out_ports=0)

        self.model.blocks_list = [step, sub, scope]
        self.model.line_list = [
            DLine(10, step.name, 0, sub.name, 0, [(0, 0), (0, 0)]),
            DLine(11, sub.name, 0, scope.name, 0, [(0, 0), (0, 0)]),
        ]
        return step, sub, scope

    def test_simple_subsystem_compilability(self):
        """A subsystem containing compilable blocks must be accepted as compilable."""
        self._build()
        self.assertTrue(
            self.engine.check_compilability(self.model.blocks_list),
            "System with Subsystem should be compilable",
        )

    def test_run_fast_subsystem(self):
        """The flattened subsystem must run and deliver gain * step to the Scope."""
        self._build(step_value=2.0, gain_value=5.0)

        self.assertTrue(
            self.engine.run_compiled_simulation(
                self.model.blocks_list, self.model.line_list, (0.0, 1.0), 0.1
            ),
            "Simulation should succeed",
        )

        # The replay updates the *flattened* block copies the engine owns, not
        # the originals in model.blocks_list.
        found_scope = None
        for block in self.engine.active_blocks_list:
            if block.name.endswith("scope1"):
                found_scope = block
                break

        self.assertIsNotNone(found_scope, "Scope missing from the flattened block list")
        vec = np.asarray(found_scope.exec_params.get("vector", []), dtype=float)
        print(f"Scope samples: {vec.shape}, last = {vec[-1] if vec.size else 'n/a'}")
        self.assertTrue(vec.size > 0, "Scope collected no data")
        self.assertAlmostEqual(float(np.ravel(vec[-1])[0]), 10.0, places=5)


if __name__ == "__main__":
    unittest.main()
