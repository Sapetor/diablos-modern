"""Unit tests for the LQR design block (blocks/lqr.py).

LQR is a *design* block, not a simulation block: it carries the A/B/Q/R spec
that the right-click "Compute LQR Gain" action reads, and its ``execute()`` is
deliberately a no-op. The contract that actually matters is therefore the
declared one -- parameter names and defaults, port shape, and the "this block
never blocks a diagram from running" flags -- so that is what is asserted here.
Before this file the block had no test at all.
"""

import ast

import numpy as np
import pytest

from blocks.lqr import LQRBlock


@pytest.mark.unit
class TestLQRBlockContract:
    def test_identity(self):
        block = LQRBlock()
        assert block.block_name == "LQR"
        assert block.category == "Control"

    def test_declares_the_four_riccati_matrices(self):
        params = LQRBlock().params
        assert set(params) == {"A", "B", "Q", "R"}
        for name, spec in params.items():
            # Strings, not floats: every one accepts a workspace variable name
            # as well as a literal matrix.
            assert spec["type"] == "string", name
            assert "doc" in spec, name

    def test_default_matrices_are_a_consistent_double_integrator(self):
        """The defaults must parse and have compatible shapes, or the very
        first "Compute LQR Gain" click on a fresh block fails."""
        params = LQRBlock().params
        A, B, Q, R = (np.array(ast.literal_eval(params[k]["default"]), dtype=float) for k in "ABQR")

        n = A.shape[0]
        m = B.shape[1]
        assert A.shape == (n, n)
        assert B.shape == (n, m)
        assert Q.shape == (n, n)
        assert R.shape == (m, m)

        # Q positive semidefinite, R positive definite -- the CARE solvability
        # preconditions.
        assert np.all(np.linalg.eigvalsh(Q) >= 0)
        assert np.all(np.linalg.eigvalsh(R) > 0)

        # (A, B) controllable: [B AB] full rank for the default double integrator.
        ctrb = np.hstack([B, A @ B])
        assert np.linalg.matrix_rank(ctrb) == n

    def test_ports(self):
        block = LQRBlock()
        assert [p["name"] for p in block.inputs] == ["plant"]
        assert block.outputs == []
        # The single input is optional: A/B can be typed in by hand instead of
        # read off a connected StateSpace block.
        assert 0 in block.optional_inputs

    def test_never_blocks_diagram_validation(self):
        """A design block with nothing wired to it must not fail validation."""
        block = LQRBlock()
        assert block.requires_inputs is False
        assert block.requires_outputs is False


@pytest.mark.unit
class TestLQRBlockExecution:
    def test_execute_is_a_no_op(self):
        """It produces no signal, so a diagram containing it still simulates."""
        block = LQRBlock()
        params = {k: v["default"] for k, v in block.params.items()}

        assert block.execute(time=0.0, inputs={}, params=params, dtime=0.01) == {}
        assert block.execute(time=1.5, inputs={0: np.array([3.0])}, params=params, dtime=0.01) == {}

    def test_execute_does_not_mutate_params(self):
        block = LQRBlock()
        params = {k: v["default"] for k, v in block.params.items()}
        before = dict(params)

        block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=0.01)

        assert params == before

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = LQRBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()
