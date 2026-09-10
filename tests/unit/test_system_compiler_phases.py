"""Unit tests for the compile_system phases in lib.engine.system_compiler.

``compile_system`` is a short sequence over module-level helpers: wiring
(``_build_input_map``), state layout (``STATE_ALLOCATORS`` / ``_allocate_states``),
execution grouping (``_execution_groups`` / ``_is_d0_state_block``), the D=0
pre-population list (``_state_output_preloads``) and the RHS closure
(``_make_model_func``). The end-to-end contract is pinned by the compiled golden
and equivalence suites; these tests cover each phase in isolation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.engine import system_compiler as sc


def _block(name, fn, **params):
    return SimpleNamespace(name=name, block_fn=fn, params=dict(params), exec_params=dict(params))


def _line(src, dst, srcport=0, dstport=0):
    return SimpleNamespace(srcblock=src, srcport=srcport, dstblock=dst, dstport=dstport)


@pytest.mark.unit
class TestInputMap:
    def test_maps_destination_ports_to_sources(self):
        blocks = [_block("a", "Step"), _block("b", "Sum"), _block("c", "Scope")]
        lines = [_line("a", "b", 0, 1), _line("a", "c"), _line("b", "c", 0, 1)]
        m = sc._build_input_map(blocks, lines)
        assert m == {"a": {}, "b": {1: ("a", 0)}, "c": {0: ("a", 0), 1: ("b", 0)}}


@pytest.mark.unit
class TestStateAllocators:
    def test_pad_initial_conditions(self):
        assert np.array_equal(sc._pad_initial_conditions([1.0], 3), [1.0, 0.0, 0.0])
        assert np.array_equal(sc._pad_initial_conditions([1.0, 2.0, 3.0], 2), [1.0, 2.0])
        assert np.array_equal(sc._pad_initial_conditions(5.0, 1), [5.0])

    def test_integrator_vector_initial_condition(self):
        n, y0, mats = sc.STATE_ALLOCATORS["Integrator"](None, {"init_conds": [1.0, 2.0]})
        assert n == 2 and np.array_equal(y0, [1.0, 2.0]) and mats is None

    def test_state_space_reshapes_and_pads(self):
        p = {
            "A": [[0, 1], [-1, -1]],
            "B": [[0], [1]],
            "C": [[1, 0]],
            "D": [[0]],
            "init_conds": [3.0],
        }
        n, y0, (A, B, C, D) = sc.STATE_ALLOCATORS["StateSpace"](None, p)
        assert n == 2 and A.shape == (2, 2) and np.array_equal(y0, [3.0, 0.0])
        assert not np.any(D)

    def test_transfer_fcn_uses_denominator_order(self):
        n, y0, (A, B, C, D) = sc.STATE_ALLOCATORS["TransferFcn"](
            None, {"numerator": [1.0], "denominator": [1.0, 3.0, 2.0]}
        )
        assert n == 2 and A.shape == (2, 2) and y0.shape == (2,) and D.item() == 0.0

    def test_proper_transfer_fcn_has_nonzero_D(self):
        _n, _y0, (_A, _B, _C, D) = sc.STATE_ALLOCATORS["TransferFcn"](
            None, {"numerator": [1.0, 1.0], "denominator": [1.0, 2.0]}
        )
        assert np.any(D != 0)

    def test_pid_and_rate_limiter_start_at_zero(self):
        assert sc.STATE_ALLOCATORS["PID"](None, {})[0] == 2
        assert np.array_equal(sc.STATE_ALLOCATORS["PID"](None, {})[1], [0.0, 0.0])
        assert sc.STATE_ALLOCATORS["RateLimiter"](None, {})[0] == 1

    def test_heat_1d_and_wave_1d_sizes(self):
        n, y0, _ = sc.STATE_ALLOCATORS["Heatequation1D"](None, {"N": 7, "init_conds": [0.0]})
        assert n == 7 and len(y0) == 7
        n, y0, _ = sc.STATE_ALLOCATORS["Waveequation1D"](
            None, {"N": 5, "init_displacement": [0.0], "init_velocity": [0.0]}
        )
        assert n == 10 and len(y0) == 10

    def test_heat_2d_flattens(self):
        n, y0, _ = sc.STATE_ALLOCATORS["Heatequation2D"](
            None, {"Nx": 3, "Ny": 4, "init_temp": "0.0"}
        )
        assert n == 12 and y0.shape == (12,)

    def test_state_variable_has_no_ode_state(self):
        assert "StateVariable" not in sc.STATE_ALLOCATORS
        assert "Statevariable" not in sc.STATE_ALLOCATORS


@pytest.mark.unit
class TestAllocateStates:
    def test_layout_is_contiguous_in_block_order(self):
        blocks = [
            _block("g", "Gain", gain=2.0),
            _block("i", "Integrator", init_conds=[1.0, 2.0]),
            _block("tf", "TranFn", numerator=[1.0], denominator=[1.0, 1.0]),
            _block("p", "PID"),
        ]
        state_map, mats, y0 = sc._allocate_states(blocks)
        assert state_map == {"i": (0, 2), "tf": (2, 1), "p": (3, 2)}
        assert set(mats) == {"tf"}
        assert np.array_equal(y0, [1.0, 2.0, 0.0, 0.0, 0.0])

    def test_no_state_blocks_gives_empty_vector(self):
        state_map, mats, y0 = sc._allocate_states([_block("g", "Gain")])
        assert state_map == {} and mats == {} and y0.shape == (0,)

    def test_failure_is_logged_and_reraised(self, caplog):
        blocks = [_block("bad", "StateSpace", A="not a matrix", B=1, C=1, D=0)]
        with caplog.at_level("ERROR", logger="lib.engine.system_compiler"):
            with pytest.raises(Exception):
                sc._allocate_states(blocks)
        assert any("Failed to compile StateSpace bad" in r.message for r in caplog.records)


@pytest.mark.unit
class TestExecutionGroups:
    def _mats(self, d):
        return (np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.array([[d]]))

    def test_d0_classification(self):
        mats = {"tf0": self._mats(0.0), "tf1": self._mats(2.0)}
        assert sc._is_d0_state_block(_block("tf0", "TranFn"), "TransferFcn", mats)
        assert not sc._is_d0_state_block(_block("tf1", "TranFn"), "TransferFcn", mats)
        assert sc._is_d0_state_block(_block("i", "Integrator"), "Integrator", {})
        assert not sc._is_d0_state_block(_block("p", "PID"), "PID", {})
        assert not sc._is_d0_state_block(_block("g", "Gain"), "Gain", {})

    def test_three_groups_keep_topological_order(self):
        order = [
            _block("step", "Step"),
            _block("sum", "Sum"),
            _block("pid", "PID"),
            _block("plant", "TranFn"),
            _block("sine", "Sine"),
            _block("gain", "Gain"),
            _block("int", "Integrator"),
            _block("scope", "Scope"),
        ]
        mats = {"plant": self._mats(0.0)}
        sources, middle, d0 = sc._execution_groups(order, mats)
        assert [b.name for b in sources] == ["step", "sine"]
        assert [b.name for b in middle] == ["sum", "pid", "gain", "scope"]
        assert [b.name for b in d0] == ["plant", "int"]

    def test_feedthrough_tf_runs_in_middle(self):
        order = [_block("tf", "TranFn"), _block("g", "Gain")]
        _s, middle, d0 = sc._execution_groups(order, {"tf": self._mats(1.0)})
        assert [b.name for b in middle] == ["tf", "g"] and d0 == []


@pytest.mark.unit
class TestPreloadsAndModelFunc:
    def test_preloads_only_for_d0_blocks(self):
        C = np.array([[1.0, 0.0]])
        state_map = {"tf": (0, 2), "i": (2, 1), "pid": (3, 2)}
        mats = {"tf": (None, None, C, np.zeros((1, 1)))}
        pre = sc._state_output_preloads(state_map, mats, {"tf", "i"})
        assert pre == [("tf", 0, 2, C), ("i", 2, 1, None)]

    def test_model_func_prepopulates_then_runs_sources_overrides_rest(self):
        calls = []

        def src(t, y, dy, signals):
            calls.append(("src", dict(signals)))
            signals["src"] = 10.0

        def sink(t, y, dy, signals):
            calls.append(("sink", dict(signals)))
            dy[0] = signals["src"] + signals["tf"] + signals["i"]

        C = np.array([[2.0, 0.0]])
        preloads = [("tf", 0, 2, C), ("i", 2, 1, None)]
        f = sc._make_model_func([src, sink], 1, preloads)
        y = np.array([1.0, 5.0, 7.0])

        dy = f(0.0, y)
        assert dy[0] == 10.0 + 2.0 + 7.0
        assert calls[0][1] == {"tf": 2.0, "i": 7.0}  # preloaded before the sources
        assert "src" in calls[1][1]  # source output visible to the sink

        dy2, signals = f.evaluate(0.0, y, input_overrides={"src": -1.0})
        assert dy2[0] == -1.0 + 2.0 + 7.0  # override applied after the source ran
        assert signals["src"] == -1.0

    def test_vector_state_outputs_keep_shape(self):
        C = np.eye(2)
        f = sc._make_model_func([], 0, [("ss", 0, 2, C), ("iv", 2, 2, None)])
        _dy, signals = f.evaluate(0.0, np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.array_equal(signals["ss"], [1.0, 2.0])
        assert np.array_equal(signals["iv"], [3.0, 4.0])
