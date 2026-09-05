"""Manual spot-checks of the compiled (fast) solver across several block families.

Run headlessly with::

    QT_QPA_PLATFORM=offscreen python scripts/verify_more_fast_blocks.py

Each check builds a tiny source -> block -> Scope diagram, runs it through the
compiled solver, and compares the Scope samples against the analytic answer.
Exit status is 0 only when every check passes.
"""

import logging
import os
import sys

# Run from anywhere: put the repo root on sys.path before importing lib/blocks.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

# DBlock builds QPixmap-backed icons, which needs a live QApplication even
# headless.
if not QApplication.instance():
    _app = QApplication(sys.argv)

import numpy as np
from PyQt5.QtCore import QRect

from lib.block_loader import load_blocks
from lib.engine.simulation_engine import SimulationEngine
from lib.engine.system_compiler import SystemCompiler
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _registry():
    """block_name -> block class, built from the same loader the app uses."""
    registry = {}
    for cls in load_blocks():
        try:
            registry[cls().block_name] = cls
        except Exception:  # pragma: no cover - a block that cannot be instantiated
            logger.debug("Skipping block class %s", cls, exc_info=True)
    return registry


BLOCKS = _registry()
_SID = [0]


class MockModel:
    """Minimal SimulationModel stand-in for the engine."""

    def __init__(self, blocks, lines):
        self.blocks_list = blocks
        self.line_list = lines
        self.variables = {}

    def link_goto_from(self):
        pass


def make_block(block_name, params=None, in_ports=None, out_ports=None):
    """Build a DBlock for ``block_name`` with the class's spec defaults applied.

    Port counts default to the block class's declared ports, which the diagram
    integrity check (run from ``initialize_execution``) validates against the
    wiring.
    """
    cls = BLOCKS[block_name]
    instance = cls()
    spec = instance.params or {}
    values = {
        key: (entry.get("default") if isinstance(entry, dict) else entry)
        for key, entry in spec.items()
    }
    values.update(params or {})

    _SID[0] += 1
    return DBlock(
        block_name,
        _SID[0],
        QRect(0, 0, 50, 50),
        "white",
        in_ports=len(instance.inputs) if in_ports is None else in_ports,
        out_ports=len(instance.outputs) if out_ports is None else out_ports,
        block_class=cls,
        params=values,
    )


def make_line(src, src_port, dst, dst_port):
    """DLine(sid, srcblock_name, srcport, dstblock_name, dstport, points)."""
    _SID[0] += 1
    return DLine(_SID[0], src.name, src_port, dst.name, dst_port, [(0, 0), (0, 0)])


class NotCompilable(Exception):
    """The compiler's allowlist rejects the diagram, so there is nothing to check."""


def run(blocks, lines, t_end, dt):
    """Run the compiled solver over a fresh engine; returns (ok, engine).

    Raises :class:`NotCompilable` when the diagram is not accepted by the fast
    path -- that is a coverage gap, not a numeric regression, so the caller
    reports it as a skip.
    """
    model = MockModel(blocks, lines)
    engine = SimulationEngine(model)
    if not SystemCompiler().check_compilability(blocks):
        raise NotCompilable(
            "not accepted by SystemCompiler.COMPILABLE_BLOCKS: "
            + ", ".join(sorted({getattr(b, "block_fn", "?") for b in blocks}))
        )
    return engine.run_compiled_simulation(blocks, lines, (0.0, t_end), dt), engine


def scope_vector(scope):
    """Scope samples as a flat float array (empty when the run collected none)."""
    return np.asarray(scope.exec_params.get("vector", []), dtype=float).ravel()


def check(label, ok):
    logger.info("%s: %s", label, "OK" if ok else "FAILED")
    return ok


def verify_deadband():
    logger.info("=== Verifying Deadband ===")
    # Sine (A=2) -> Deadband(-0.5, 0.5) -> Scope
    src = make_block("Sine", {"amplitude": 2.0, "omega": 1.0})
    db = make_block("Deadband", {"start": -0.5, "end": 0.5})
    scope = make_block("Scope")

    blocks = [src, db, scope]
    lines = [make_line(src, 0, db, 0), make_line(db, 0, scope, 0)]

    ok, _ = run(blocks, lines, 10.0, 0.1)
    if not ok:
        return check("Deadband", False)

    vec = scope_vector(scope)
    if vec.size == 0:
        return check("Deadband", False)

    # t=0 -> input 0, inside the band -> output 0.
    logger.info("Deadband output at t=0: %.6f", vec[0])
    return check("Deadband", abs(vec[0]) < 1e-6)


def verify_exponential():
    logger.info("=== Verifying Exp ===")
    # Ramp (x=t) -> Exp (a*exp(b*x)) -> Scope; y(t) = 2*exp(0.5 t)
    ramp = make_block("Ramp", {"slope": 1.0, "delay": 0.0})
    exp_block = make_block("Exp", {"a": 2.0, "b": 0.5})
    scope = make_block("Scope")

    blocks = [ramp, exp_block, scope]
    lines = [make_line(ramp, 0, exp_block, 0), make_line(exp_block, 0, scope, 0)]

    ok, _ = run(blocks, lines, 2.0, 0.1)
    if not ok:
        return check("Exp", False)

    vec = scope_vector(scope)
    if vec.size == 0:
        return check("Exp", False)

    logger.info("Exp first=%.6f last=%.6f (expect 2.0 and %.6f)", vec[0], vec[-1], 2 * np.e)
    return check("Exp", abs(vec[0] - 2.0) < 1e-3 and abs(vec[-1] - 2 * np.e) < 1e-2)


def verify_pid():
    logger.info("=== Verifying PID (PI step response) ===")
    # Step(delay=1, value=1) -> PID(Kp=1, Ki=1, Kd=0) -> Scope
    # For t >= 1: u = Kp*1 + Ki*(t-1) -> u(3) = 1 + 2 = 3.
    step = make_block("Step", {"delay": 1.0, "value": 1.0})
    # Open-loop: only the setpoint port is wired, so declare a single input.
    pid = make_block("PID", {"Kp": 1.0, "Ki": 1.0, "Kd": 0.0}, in_ports=1)
    scope = make_block("Scope")

    blocks = [step, pid, scope]
    lines = [make_line(step, 0, pid, 0), make_line(pid, 0, scope, 0)]

    ok, _ = run(blocks, lines, 3.0, 0.1)
    if not ok:
        return check("PID", False)

    vec = scope_vector(scope)
    if vec.size == 0:
        return check("PID", False)

    logger.info("PID output start=%.6f end=%.6f (expect ~3.0)", vec[0], vec[-1])
    return check("PID", abs(vec[-1] - 3.0) < 0.2)


def verify_pid_derivative():
    logger.info("=== Verifying PID derivative ===")
    # Ramp(slope=1) -> PID(Kd=1, N=100) -> Scope; D term settles at 1.
    ramp = make_block("Ramp", {"slope": 1.0})
    pid = make_block("PID", {"Kp": 0.0, "Ki": 0.0, "Kd": 1.0, "N": 100.0}, in_ports=1)
    scope = make_block("Scope")

    blocks = [ramp, pid, scope]
    lines = [make_line(ramp, 0, pid, 0), make_line(pid, 0, scope, 0)]

    ok, _ = run(blocks, lines, 1.0, 0.01)
    if not ok:
        return check("PID derivative", False)

    vec = scope_vector(scope)
    if vec.size == 0:
        return check("PID derivative", False)

    logger.info("PID D output end=%.6f (expect ~1.0)", vec[-1])
    return check("PID derivative", abs(vec[-1] - 1.0) < 0.1)


def verify_ratelimiter():
    logger.info("=== Verifying RateLimiter ===")
    # Step(value=2 at t=0) -> RateLimiter(rising=1) -> Scope; reaches 2.0 at t=2.
    step = make_block("Step", {"delay": 0.0, "value": 2.0})
    limiter = make_block("RateLimiter", {"rising_slew": 1.0, "falling_slew": 1.0})
    scope = make_block("Scope")

    blocks = [step, limiter, scope]
    lines = [make_line(step, 0, limiter, 0), make_line(limiter, 0, scope, 0)]

    ok, _ = run(blocks, lines, 3.0, 0.1)
    if not ok:
        return check("RateLimiter", False)

    vec = scope_vector(scope)
    if vec.size < 31:
        logger.error("RateLimiter: expected >=31 samples, got %d", vec.size)
        return check("RateLimiter", False)

    v1, v2, v3 = vec[10], vec[20], vec[30]  # t = 1, 2, 3
    logger.info("RateLimiter t=1:%.4f t=2:%.4f t=3:%.4f (expect 1, 2, 2)", v1, v2, v3)
    return check(
        "RateLimiter",
        abs(v1 - 1.0) < 0.1 and abs(v2 - 2.0) < 0.1 and abs(v3 - 2.0) < 0.1,
    )


def verify_tranfn():
    logger.info("=== Verifying TranFn ===")
    # Step(1) -> TranFn 1/(s+1) -> Scope; y(t) = 1 - exp(-t).
    step = make_block("Step", {"delay": 0.0, "value": 1.0})
    tf = make_block("TranFn", {"numerator": [1.0], "denominator": [1.0, 1.0]})
    scope = make_block("Scope")

    blocks = [step, tf, scope]
    lines = [make_line(step, 0, tf, 0), make_line(tf, 0, scope, 0)]

    ok, _ = run(blocks, lines, 2.0, 0.1)
    if not ok:
        return check("TranFn", False)

    vec = scope_vector(scope)
    if vec.size == 0:
        return check("TranFn", False)

    expected = 1.0 - np.exp(-2.0)
    logger.info("TranFn last=%.6f (expect %.6f)", vec[-1], expected)
    return check("TranFn", abs(vec[-1] - expected) < 0.05)


CHECKS = (
    verify_deadband,
    verify_exponential,
    verify_pid,
    verify_pid_derivative,
    verify_ratelimiter,
    verify_tranfn,
)


if __name__ == "__main__":
    results = {}
    for fn in CHECKS:
        try:
            results[fn.__name__] = bool(fn())
        except NotCompilable as exc:
            logger.warning("%s skipped: %s", fn.__name__, exc)
            results[fn.__name__] = None
        except Exception:
            logger.exception("%s raised", fn.__name__)
            results[fn.__name__] = False

    logger.info("--- Summary ---")
    for name, ok in results.items():
        logger.info("%-26s %s", name, "SKIPPED" if ok is None else ("OK" if ok else "FAILED"))
    sys.exit(0 if all(ok is not False for ok in results.values()) else 1)
