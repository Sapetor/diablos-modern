"""
Regression tests for DSim integrity checks.

Ensures diagram validation handles DLine connections without treating them
as subscriptable objects (previously caused false algebraic-loop errors).
"""

import pytest
from PyQt5.QtCore import QRect, QPoint

from lib.lib import DSim
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from blocks.step import StepBlock
from blocks.sum import SumBlock
from blocks.integrator import IntegratorBlock
from blocks.scope import ScopeBlock


@pytest.mark.unit
@pytest.mark.qt
def test_check_diagram_integrity_handles_dline_connections(qapp):
    """DSim integrity check should accept a simple connected diagram."""
    sim = DSim()

    source = DBlock(
        block_fn="Source",
        sid=0,
        coords=QRect(0, 0, 100, 80),
        color="red",
        in_ports=0,
        out_ports=1,
        b_type=2,
        io_edit="none",
        fn_name="source",
        params={},
        external=False,
        colors=sim.colors,
    )
    sink = DBlock(
        block_fn="Sink",
        sid=0,
        coords=QRect(200, 0, 100, 80),
        color="blue",
        in_ports=1,
        out_ports=0,
        b_type=2,
        io_edit="none",
        fn_name="sink",
        params={},
        external=False,
        colors=sim.colors,
    )

    sim.blocks_list.extend([source, sink])

    line = DLine(
        sid=0,
        srcblock=source.name,
        srcport=0,
        dstblock=sink.name,
        dstport=0,
        points=[source.out_coords[0], sink.in_coords[0]],
    )
    sim.line_list.append(line)

    # get_neighbors should yield dicts, not DLine objects
    inputs, outputs = sim.get_neighbors(sink.name)
    assert inputs == [
        {"srcblock": source.name, "srcport": 0, "dstport": 0},
    ]
    assert outputs == []

    assert sim.check_diagram_integrity() is True


@pytest.mark.unit
@pytest.mark.qt
def test_execution_init_runs_without_algebraic_loop(monkeypatch, qapp):
    """
    End-to-end smoke test: a simple feedforward diagram should initialize
    without triggering algebraic loop detection.
    """
    sim = DSim()
    sim.main_buttons_init()  # needed for execution_init scope button toggle

    # Skip UI dialogs and disk writes during the test
    monkeypatch.setattr(sim, "execution_init_time", lambda: 1.0)
    monkeypatch.setattr(sim, "save", lambda *args, **kwargs: 0)

    # Blocks with real block classes so execution functions exist
    step = DBlock(
        block_fn="Step",
        sid=0,
        coords=QRect(0, 0, 100, 80),
        color="blue",
        in_ports=0,
        out_ports=1,
        b_type=0,
        io_edit="none",
        fn_name="step",
        params={"value": 1.0, "delay": 0.0, "type": "up", "pulse_start_up": True, "_init_start_": True},
        external=False,
        colors=sim.colors,
        block_class=StepBlock,
    )

    summation = DBlock(
        block_fn="Sum",
        sid=0,
        coords=QRect(200, 0, 100, 80),
        color="lime_green",
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit="none",
        fn_name="sum",
        params={"sign": "+"},
        external=False,
        colors=sim.colors,
        block_class=SumBlock,
    )

    integrator = DBlock(
        block_fn="Integrator",
        sid=0,
        coords=QRect(400, 0, 100, 80),
        color="magenta",
        in_ports=1,
        out_ports=1,
        b_type=1,
        io_edit="none",
        fn_name="integrator",
        params={"init_conds": 0.0, "method": "SOLVE_IVP", "_init_start_": True},
        external=False,
        colors=sim.colors,
        block_class=IntegratorBlock,
    )

    scope = DBlock(
        block_fn="Scope",
        sid=0,
        coords=QRect(600, 0, 100, 80),
        color="red",
        in_ports=1,
        out_ports=0,
        b_type=2,
        io_edit="none",
        fn_name="scope",
        params={"labels": "default", "_init_start_": True},
        external=False,
        colors=sim.colors,
        block_class=ScopeBlock,
    )

    sim.blocks_list.extend([step, summation, integrator, scope])

    sim.line_list.append(
        DLine(0, step.name, 0, summation.name, 0, [step.out_coords[0], summation.in_coords[0]])
    )
    sim.line_list.append(
        DLine(1, summation.name, 0, integrator.name, 0, [summation.out_coords[0], integrator.in_coords[0]])
    )
    sim.line_list.append(
        DLine(2, integrator.name, 0, scope.name, 0, [integrator.out_coords[0], scope.in_coords[0]])
    )

    try:
        assert sim.execution_init() is True
        assert sim.error_msg == ""
        assert sim.execution_initialized is True
    finally:
        # Make sure tqdm resources are released even if assertion fails
        if hasattr(sim, "pbar"):
            sim.pbar.close()
