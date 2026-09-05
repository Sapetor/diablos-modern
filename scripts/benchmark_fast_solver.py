import sys
import os
import time
import logging

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from lib.simulation.connection import DLine as Line

# Setup logging
logging.basicConfig(level=logging.ERROR)  # Silence info for benchmark
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)


class MockModel:
    def __init__(self):
        self.blocks_list = []
        self.line_list = []
        self.variables = {}
        self.dirty = False

    def link_goto_from(self):
        pass


def default_params(block_class):
    """Flatten a block class's params spec into the {name: default} map DBlock wants."""
    spec = block_class().params or {}
    return {
        name: (entry.get("default") if isinstance(entry, dict) else entry)
        for name, entry in spec.items()
    }


def create_benchmark_system(n_integrators=10):
    """Create a chain of integrators to stress the solver."""
    model = MockModel()
    engine = SimulationEngine(model)
    engine.sim_dt = 0.001  # Small step for load
    engine.sim_time = 10.0  # 10 seconds = 10,000 steps
    engine.execution_time = 10.0

    blocks = []
    lines = []

    # Source: Sine
    sine = DBlock(
        "Sine",
        0,
        QRect(0, 0, 50, 50),
        "blue",
        in_ports=0,
        block_class=SineBlock,
        params=default_params(SineBlock),
    )
    sine.hierarchy = 0
    blocks.append(sine)

    prev_block = sine
    prev_port = 0

    for i in range(n_integrators):
        integ = DBlock(
            "Integrator",
            i + 1,
            QRect(100 + i * 60, 0, 50, 50),
            "green",
            block_class=IntegratorBlock,
            params=default_params(IntegratorBlock),
        )
        integ.hierarchy = i + 1
        blocks.append(integ)

        line = Line(i, prev_block.name, prev_port, integ.name, 0, [(0, 0), (0, 0)])
        lines.append(line)

        prev_block = integ

    # Scope
    scope = DBlock(
        "Scope",
        n_integrators + 1,
        QRect(100 + n_integrators * 60, 0, 50, 50),
        "black",
        out_ports=0,
        block_class=ScopeBlock,
        params=default_params(ScopeBlock),
    )
    scope.hierarchy = n_integrators + 1
    blocks.append(scope)

    line = Line(n_integrators, prev_block.name, 0, scope.name, 0, [(0, 0), (0, 0)])
    lines.append(line)

    model.blocks_list = blocks
    model.line_list = lines

    return engine, model, blocks


def run_benchmark():
    engine, model, blocks = create_benchmark_system(n_integrators=10)  # 10 serial integrators

    logger.info("System: 1 Sine -> 10 Integrators -> Scope (dt=0.001, T=10.0s, Steps=10,000)")

    # 1. Fast Solver
    start_time = time.time()
    if engine.check_compilability(blocks):
        t_span = (0.0, engine.execution_time)
        engine.run_compiled_simulation(blocks, model.line_list, t_span, engine.sim_dt)
    fast_duration = time.time() - start_time
    logger.info(f"Fast Solver Time: {fast_duration:.4f}s")

    # 2. Interpreter
    # Reset engine state
    engine.execution_initialized = False

    # NOTE: this is a *proxy*, not a real interpreter run -- it times the
    # per-block execute() cost over the same number of steps but skips output
    # propagation, so it under-counts the interpreter. Treat the ratio below as
    # a lower bound on the speedup, and tests/regression/ for correctness.
    start_time = time.time()

    engine.initialize_execution(blocks, model.line_list)
    steps = int(engine.execution_time / engine.sim_dt)

    # Simple Interpreter Loop equivalent
    for step in range(steps):
        # Update time
        engine.time_step += engine.sim_dt

        # Execute blocks in order
        for block in blocks:
            engine.execute_block(block)

    std_duration = time.time() - start_time
    logger.info(f"Per-block execute() loop (proxy, no propagation): {std_duration:.4f}s")

    speedup = std_duration / fast_duration if fast_duration > 0 else 0
    logger.info(f"Speedup (lower bound): {speedup:.2f}x")


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
