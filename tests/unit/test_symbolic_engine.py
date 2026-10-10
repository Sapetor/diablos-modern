"""SymPy equation extraction (lib/engine/symbolic_engine.py) and the blocks'
symbolic_execute() hooks. SymPy is a dev requirement, so these tests import it
outright instead of skipping when it is missing."""

import json
import os
import types

import numpy as np
import pytest
import sympy
from scipy import signal

from blocks.derivative import DerivativeBlock
from blocks.gain import GainBlock
from blocks.integrator import IntegratorBlock
from blocks.pid import PIDBlock
from blocks.product import ProductBlock
from blocks.sum import SumBlock
from blocks.transfer_function import TransferFunctionBlock
from lib.analysis.block_algebra import sink_input_signal, transfer_function
from lib.engine.symbolic_engine import SymbolicEngine

s = sympy.Symbol("s")

HOOKS = {
    "Gain": GainBlock,
    "Sum": SumBlock,
    "Product": ProductBlock,
    "Integrator": IntegratorBlock,
    "Deriv": DerivativeBlock,
    "PID": PIDBlock,
    "TranFn": TransferFunctionBlock,
}


def B(name, fn, hook=True, **params):
    """A canvas-like block; with ``hook`` it carries the real block instance, so
    the engine goes through its symbolic_execute() instead of the fallback."""
    instance = HOOKS[fn]() if hook and fn in HOOKS else None
    return types.SimpleNamespace(name=name, block_fn=fn, params=params, block_instance=instance)


def L(src, dst, dport=0, sport=0):
    return types.SimpleNamespace(srcblock=src, srcport=sport, dstblock=dst, dstport=dport)


def engine(blocks, lines):
    eng = SymbolicEngine()
    eng.build_graph(blocks, lines)
    return eng


def coeffs(G):
    """Monic (num, den) coefficient arrays of a rational G(s)."""
    num, den = sympy.fraction(sympy.cancel(G))
    num = np.array([float(c) for c in sympy.Poly(num, s).all_coeffs()])
    den = np.array([float(c) for c in sympy.Poly(den, s).all_coeffs()])
    return num / den[0], den / den[0]


def assert_tf(G, num, den):
    n, d = coeffs(G)
    np.testing.assert_allclose(n, np.asarray(num, float) / den[0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(d, np.asarray(den, float) / den[0], rtol=1e-9, atol=1e-12)


def feedback_diagram(hook):
    blocks = [
        B("r", "Step", value=1.0),
        B("e", "Sum", hook, sign="+-"),
        B("k", "Gain", hook, gain=5.0),
        B("p", "TranFn", hook, numerator=[1.0], denominator=[1.0, 1.0]),
        B("scope", "Scope"),
    ]
    lines = [L("r", "e", 0), L("p", "e", 1), L("e", "k"), L("k", "p"), L("p", "scope")]
    return blocks, lines


@pytest.mark.unit
class TestTransferFunction:
    @pytest.mark.parametrize("hook", [True, False], ids=["block-hooks", "fallback"])
    def test_unity_feedback_loop_is_closed(self, hook):
        G = engine(*feedback_diagram(hook)).extract_transfer_function("r", "scope")
        assert G.free_symbols == {s}  # no dangling feedback symbol
        assert_tf(G, [5.0], [1.0, 6.0])  # K / (s + 1 + K)

    def test_nested_loops_match_block_algebra(self):
        # Inner rate loop around 1/s, outer position loop: x/r = 2 / (s^2 + 3 s + 2)
        blocks = [
            B("r", "Step"),
            B("e1", "Sum", sign="+-"),
            B("k1", "Gain", gain=2.0),
            B("e2", "Sum", sign="+-"),
            B("p", "TranFn", numerator=[1.0], denominator=[1.0, 1.0]),
            B("i", "Integrator"),
            B("scope", "Scope"),
        ]
        lines = [
            L("r", "e1", 0),
            L("i", "e1", 1),
            L("e1", "k1"),
            L("k1", "e2", 0),
            L("p", "e2", 1),
            L("e2", "p"),
            L("p", "i"),
            L("i", "scope"),
        ]
        G = engine(blocks, lines).extract_transfer_function("r", "scope")
        ref = transfer_function(blocks, lines, "r", sink_input_signal(lines, "scope")).tf
        assert_tf(G, ref.num, ref.den)

    def test_other_sources_are_zeroed(self):
        blocks, lines = feedback_diagram(True)
        blocks.append(B("d", "Step", value=3.0))
        blocks[1].params["sign"] = "+-+"
        lines.append(L("d", "e", 2))
        G = engine(blocks, lines).extract_transfer_function("r", "scope")
        assert_tf(G, [5.0], [1.0, 6.0])

    def test_pid_example_matches_block_algebra(self):
        """The welcome screen's PID loop: same closed-loop G(s) as the numeric
        Mason's-rule engine, which is itself checked against the simulator."""
        src = os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "pid_second_order.diablos"
        )
        data = json.load(open(src, encoding="utf-8"))
        blocks = [B(b["name"], b["block_fn"], **b["params"]) for b in data["blocks_data"]]
        lines = [
            L(ln["srcblock"], ln["dstblock"], ln["dstport"], ln["srcport"])
            for ln in data["lines_data"]
        ]
        G = engine(blocks, lines).extract_transfer_function("step0", "scope5", to_port=1)
        ref = transfer_function(blocks, lines, "step0", sink_input_signal(lines, "scope5", 1)).tf
        assert_tf(G, ref.num, ref.den)

    def test_state_space_fallback_matches_scipy(self):
        A, Bm, C, D = [[0.0, 1.0], [-2.0, -3.0]], [[0.0], [1.0]], [[1.0, 0.0]], [[0.5]]
        blocks = [
            B("u", "Step"),
            types.SimpleNamespace(
                name="ss",
                block_fn="StateSpace",
                params={"A": A, "B": Bm, "C": C, "D": D},
                block_instance=None,
            ),
            B("scope", "Scope"),
        ]
        lines = [L("u", "ss"), L("ss", "scope")]
        G = engine(blocks, lines).extract_transfer_function("u", "scope")
        num, den = signal.ss2tf(np.array(A), np.array(Bm), np.array(C), np.array(D))
        assert_tf(G, num[0], den)

    def test_linearize_at_point_returns_state_space(self):
        eng = engine(*feedback_diagram(True))
        A, Bm, C, D = eng.linearize_at_point({}, "r", "scope")
        num, den = signal.ss2tf(A, Bm, C, D)
        np.testing.assert_allclose(np.trim_zeros(num[0], "f"), [5.0], atol=1e-12)
        np.testing.assert_allclose(den, [1.0, 6.0], atol=1e-12)

    def test_nonlinear_loop_warns_instead_of_failing(self, caplog):
        # y = r - y*y has two solutions, so the loop cannot be closed uniquely.
        blocks = [
            B("r", "Step"),
            B("e", "Sum", sign="+-"),
            B("sq", "Product", ops="**"),
            B("scope", "Scope"),
        ]
        lines = [L("r", "e", 0), L("sq", "e", 1), L("e", "sq", 0), L("e", "sq", 1)]
        lines.append(L("e", "scope"))
        eqs = engine(blocks, lines).get_all_equations()
        assert "feedback loop" in caplog.text
        assert eqs["scope"].free_symbols & set(sympy.symbols("e_0 sq_0"))  # loop left open


@pytest.mark.unit
class TestEquations:
    def test_every_entry_is_an_expression_and_sinks_show_their_input(self):
        blocks, lines = feedback_diagram(True)
        eqs = engine(blocks, lines).get_all_equations()
        assert set(eqs) == {b.name for b in blocks}
        assert all(isinstance(v, sympy.Expr) for v in eqs.values()), eqs
        r = sympy.Symbol("r")
        assert sympy.simplify(eqs["scope"] - 5 * r / (s + 6)) == 0
        assert sympy.simplify(eqs["e"] - r * (s + 1) / (s + 6)) == 0

    def test_export_latex_document(self, tmp_path):
        eng = engine(*feedback_diagram(True))
        out = tmp_path / "eq.tex"
        doc = eng.export_equations_latex(filename=str(out))
        assert out.read_text(encoding="utf-8") == doc
        assert doc.startswith(r"\documentclass{article}") and doc.endswith(r"\end{document}")
        assert r"\subsection{scope}" in doc and r"\frac" in doc

    def test_a_failing_hook_falls_back(self, caplog):
        blocks, lines = feedback_diagram(True)

        def boom(inputs, params):
            raise ValueError("bad param")

        blocks[2].block_instance.symbolic_execute = boom  # Gain: fallback is K*u too
        G = engine(blocks, lines).extract_transfer_function("r", "scope")
        assert_tf(G, [5.0], [1.0, 6.0])
        assert "bad param" in caplog.text


u, v = sympy.symbols("u v")


@pytest.mark.unit
@pytest.mark.parametrize(
    "block, params, inputs, expected",
    [
        (GainBlock, {"gain": 2.5}, {0: u}, 2.5 * u),
        (GainBlock, {"gain": "3"}, {0: u}, 3 * u),
        (GainBlock, {"gain": [4.0]}, {0: u}, 4 * u),
        (GainBlock, {"gain": [[1.0, 2.0], [3.0, 4.0]]}, {0: u}, sympy.Matrix([[1, 2], [3, 4]]) * u),
        (SumBlock, {"sign": "+-"}, {0: u, 1: v}, u - v),
        (SumBlock, {"sign": "-+-"}, {0: u, 1: v}, -u + v - sympy.Symbol("u2")),
        (ProductBlock, {"ops": "*/"}, {0: u, 1: v}, u / v),
        (IntegratorBlock, {}, {0: u}, u / s),
        (DerivativeBlock, {}, {0: u}, s * u),
        (
            TransferFunctionBlock,
            {"numerator": [2.0], "denominator": [1.0, 3.0]},
            {0: u},
            2 * u / (s + 3),
        ),
        (PIDBlock, {"Kp": 2.0, "Ki": 1.0, "Kd": 0.0}, {0: u, 1: v}, (2 + 1 / s) * (u - v)),
        (
            PIDBlock,
            {"Kp": 1.0, "Ki": 0.0, "Kd": 0.5, "N": 10.0},
            {0: u, 1: v},
            (1 + 5 * s / (s + 10)) * (u - v),
        ),
    ],
)
def test_block_symbolic_execute(block, params, inputs, expected):
    out = block().symbolic_execute(inputs, params)
    assert set(out) == {0}
    diff = out[0] - expected
    if isinstance(diff, sympy.MatrixBase):
        assert diff.applyfunc(sympy.simplify).is_zero_matrix
    else:
        assert sympy.simplify(diff) == 0


def test_engine_does_not_import_sympy_eagerly():
    """SymPy takes ~0.2 s to import; app startup must not pay for it."""
    import subprocess
    import sys

    code = "import sys, lib.engine; print('sympy' in sys.modules)"
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
