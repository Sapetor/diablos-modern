"""Manual check of the compiled (fast) solver on a Sine -> Integrator -> Scope chain.

Run headlessly with::

    QT_QPA_PLATFORM=offscreen python scripts/verify_fast_solver.py

Reports whether the diagram is accepted by ``check_compilability`` and whether
the compiled run reproduces the analytic integral of ``sin(t)``.
"""

import logging
import os
import sys

import numpy as np

# Run from anywhere: put the repo root on sys.path before importing lib.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DBlock builds QPixmap-backed icons, which needs a live QApplication even
# headless; without this the script aborts with "Must construct a
# QGuiApplication before a QPixmap".
from PyQt5.QtWidgets import QApplication

if not QApplication.instance():
    _app = QApplication(sys.argv)

from PyQt5.QtCore import QRect
from blocks.integrator import IntegratorBlock
from blocks.scope import ScopeBlock
from blocks.sine import SineBlock
from lib.engine.simulation_engine import SimulationEngine
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel:
    """Minimal stand-in for SimulationModel (blocks + lines + goto linking)."""

    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}

    def link_goto_from(self):
        pass


def default_params(block_class):
    """Flatten a block class's params spec into the {name: default} map DBlock wants."""
    spec = block_class().params or {}
    return {
        name: (entry.get("default") if isinstance(entry, dict) else entry)
        for name, entry in spec.items()
    }


def create_simple_system():
    """Sine -> Integrator -> Scope, wired by DBlock.name (block_fn.lower() + sid)."""
    model = MockModel()
    engine = SimulationEngine(model)

    sine = DBlock(
        "Sine",
        1,
        QRect(0, 0, 50, 50),
        "blue",
        in_ports=0,
        block_class=SineBlock,
        params=default_params(SineBlock),
    )
    sine.params["amplitude"] = 1.0
    sine.params["omega"] = 1.0
    sine.hierarchy = 0

    integ = DBlock(
        "Integrator",
        1,
        QRect(100, 0, 50, 50),
        "green",
        block_class=IntegratorBlock,
        params=default_params(IntegratorBlock),
    )
    integ.params["init_conds"] = 0.0
    integ.hierarchy = 1

    # Scope buffers are set up by DBlock.__init__ these days; the old
    # scope.get_id() bootstrap no longer exists.
    scope = DBlock(
        "Scope",
        1,
        QRect(200, 0, 50, 50),
        "black",
        out_ports=0,
        block_class=ScopeBlock,
        params=default_params(ScopeBlock),
    )
    scope.hierarchy = 2

    # DLine(sid, srcblock_name, srcport, dstblock_name, dstport, points)
    line1 = DLine(1, sine.name, 0, integ.name, 0, [(0, 0), (0, 0)])
    line2 = DLine(2, integ.name, 0, scope.name, 0, [(0, 0), (0, 0)])

    model.blocks_list = [sine, integ, scope]
    model.line_list = [line1, line2]

    return engine, model, [sine, integ, scope]


def verify_fast_solver():
    engine, model, blocks = create_simple_system()
    sine, integ, scope = blocks

    logger.info("--- Check compilability (whole diagram, Scope included) ---")
    logger.info("Compilable? %s", engine.check_compilability(model.blocks_list))

    logger.info("--- Run compiled simulation ---")
    t_span = (0.0, 10.0)
    dt = 0.01

    if not engine.run_compiled_simulation(model.blocks_list, model.line_list, t_span, dt):
        logger.error("Compiled simulation returned False")
        return False

    logger.info("Timeline steps: %s", len(engine.timeline))

    # Integral of sin(t) over [0, 10] is 1 - cos(10).
    expected = 1.0 - np.cos(10.0)
    vector = np.asarray(scope.exec_params.get("vector", []), dtype=float).ravel()
    if vector.size == 0:
        logger.error("Scope collected no data")
        return False

    final = float(vector[-1])
    logger.info("Final integrator value: %.6f (expected %.6f)", final, expected)
    if abs(final - expected) < 1e-2:
        logger.info("Fast solver verified.")
        return True
    logger.error("Fast solver value off by %.3e", abs(final - expected))
    return False


if __name__ == "__main__":
    try:
        sys.exit(0 if verify_fast_solver() else 1)
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
