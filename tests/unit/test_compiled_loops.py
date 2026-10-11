"""Unit tests for lib/engine/compiled_loops.py (algebraic loops in the compiled RHS)."""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.engine.compiled_loops import (
    CompiledLoopError,
    _flush_tiny,
    make_loop_executor,
    order_units,
)


def _blocks(*names):
    return [SimpleNamespace(name=n) for n in names]


@pytest.mark.unit
class TestOrderUnits:
    def test_loop_is_one_unit_between_its_feeder_and_its_consumer(self):
        # pre -> s <-> g -> post ; listed in an order that puts post first.
        post, g, s, pre = _blocks("post", "g", "s", "pre")
        input_map = {
            "pre": {},
            "s": {0: ("pre", 0), 1: ("g", 0)},
            "g": {0: ("s", 0)},
            "post": {0: ("g", 0)},
        }
        units = order_units([post, g, s, pre], input_map)
        names = [[b.name for b in u] if isinstance(u, list) else u.name for u in units]
        assert names == ["pre", ["g", "s"], "post"]

    def test_acyclic_middle_has_no_loop_units(self):
        a, b = _blocks("a", "b")
        units = order_units([b, a], {"a": {}, "b": {0: ("a", 0)}})
        assert [u.name for u in units] == ["a", "b"]


def _loop(u, k, f=lambda v: v):
    """s = u - k * f(g_in), g = s: one Sum and one Gain-like member."""
    s, g = _blocks("s", "g")
    input_map = {"s": {0: ("src", 0), 1: ("g", 0)}, "g": {0: ("s", 0)}}

    def ex_s(t, y, dy, sig):
        sig["s"] = u - k * f(float(np.ravel(sig.get("g", 0.0))[0]))

    def ex_g(t, y, dy, sig):
        sig["g"] = sig["s"]

    return make_loop_executor([s, g], [ex_s, ex_g], input_map)


@pytest.mark.unit
class TestLoopExecutor:
    @pytest.mark.parametrize("k", [0.5, 3.0])
    def test_solves_a_linear_loop(self, k):
        signals = {}
        _loop(1.0, k)(0.0, np.zeros(0), np.zeros(0), signals)
        assert signals["s"] == pytest.approx(1.0 / (1.0 + k), abs=1e-12)

    def test_warm_start_from_a_zero_solution(self):
        """The second call starts from ~0; a subnormal guess used to stall hybr."""
        ex = _loop(0.0, 0.5)
        signals = {}
        ex(0.0, np.zeros(0), np.zeros(0), signals)
        ex = _loop(1.0, 0.5)
        ex(0.0, np.zeros(0), np.zeros(0), {})  # shapes
        signals = {}
        ex(0.1, np.zeros(0), np.zeros(0), signals)
        assert signals["s"] == pytest.approx(1.0 / 1.5, abs=1e-12)

    def test_no_solution_raises(self):
        # s = 1 + (s + 1) has no solution.
        ex = _loop(1.0, -1.0, f=lambda v: v + 1.0)
        with pytest.raises(CompiledLoopError, match="did not converge"):
            ex(0.0, np.zeros(0), np.zeros(0), {})


@pytest.mark.unit
def test_flush_tiny_zeroes_subnormal_leftovers():
    z = _flush_tiny([-5e-324, 1e-323, 1e-3, -2.0])
    assert z.tolist() == [0.0, 0.0, 1e-3, -2.0]
