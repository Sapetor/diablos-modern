"""Unit tests for lib.engine.replay_handlers -- the per-block replay dispatch.

The end-to-end invariant (compiled replay == the traces it produced before the
handlers were extracted from ``replay_compiled_signals``) is pinned by
``tests/regression/test_compiled_golden.py``; these tests cover the handler
semantics in isolation so a future edit to one handler fails here, by name.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lib.engine import replay_handlers as rh


def _block(name="b", fn="Test", params=None, exec_params=None, **attrs):
    blk = SimpleNamespace(
        name=name,
        block_fn=fn,
        params=dict(params or {}),
        block_instance=None,
        in_ports=1,
        out_ports=1,
    )
    if exec_params is not None:
        blk.exec_params = exec_params
    for k, v in attrs.items():
        setattr(blk, k, v)
    return blk


def _step(t=0.0, states=None):
    return rh.ReplayStep(t, states or {}, {})


@pytest.mark.unit
class TestDispatchTables:
    def test_every_handler_and_recorder_is_callable(self):
        assert all(callable(h) for h in rh.REPLAY_HANDLERS.values())
        assert all(callable(r) for r in rh.RECORDERS.values())

    def test_sinks_do_not_compute(self):
        for fn in ("Scope", "Display", "Terminator"):
            assert rh.REPLAY_HANDLERS[fn](_step(), _block(fn=fn), {0: 5.0}) == 0.0

    def test_both_statevariable_spellings_share_a_handler(self):
        assert rh.REPLAY_HANDLERS["Statevariable"] is rh.REPLAY_HANDLERS["StateVariable"]


@pytest.mark.unit
class TestStateHandlers:
    def test_integrator_scalar_and_vector(self):
        h = rh.REPLAY_HANDLERS["Integrator"]
        assert h(_step(states={"i": np.array([2.5])}), _block("i"), {}) == 2.5
        vec = h(_step(states={"i": np.array([1.0, 2.0])}), _block("i"), {})
        assert np.array_equal(vec, [1.0, 2.0])
        assert h(_step(), _block("i"), {}) == 0.0

    def test_heat_1d_publishes_mean(self):
        step = _step(states={"h": np.array([1.0, 3.0])})
        out = rh.REPLAY_HANDLERS["Heatequation1D"](step, _block("h"), {})
        assert np.array_equal(out, [1.0, 3.0])
        assert step.signals["h_out1"] == 2.0

    def test_heat_1d_without_state_is_zeros_of_N(self):
        out = rh.REPLAY_HANDLERS["Heatequation1D"](_step(), _block("h", params={"N": 4}), {})
        assert out.shape == (4,) and not out.any()

    def test_wave_1d_splits_u_v_and_publishes_energy(self):
        blk = _block("w", params={"N": 2, "L": 1.0, "c": 1.0})
        step = _step(states={"w": np.array([0.1, 0.2, 5.0, 6.0])})
        out = rh.REPLAY_HANDLERS["Waveequation1D"](step, blk, {})
        assert np.array_equal(out, [0.1, 0.2])
        assert np.array_equal(step.signals["w_out1"], [5.0, 6.0])
        assert isinstance(step.signals["w_out2"], float)

    def test_diffusion_reaction_publishes_total_and_rate(self):
        blk = _block("d", params={"L": 1.0, "k": 2.0, "n": 1})
        step = _step(states={"d": np.array([1.0, 1.0, 1.0])})  # dx = 0.5
        rh.REPLAY_HANDLERS["Diffusionreaction1D"](step, blk, {})
        assert step.signals["d_out1"] == pytest.approx(1.5)
        assert step.signals["d_out2"] == pytest.approx(3.0)

    def test_heat_2d_reshapes_and_publishes_mean_max(self):
        blk = _block("h2", params={"Nx": 2, "Ny": 2})
        step = _step(states={"h2": np.array([1.0, 2.0, 3.0, 4.0])})
        out = rh.REPLAY_HANDLERS["Heatequation2D"](step, blk, {})
        assert out.shape == (2, 2)
        assert step.signals["h2_out1"] == 2.5 and step.signals["h2_out2"] == 4.0


@pytest.mark.unit
class TestFieldHandlers:
    def test_probe_2d_bilinear_center(self):
        field = np.array([[0.0, 1.0], [2.0, 3.0]])
        blk = _block(params={"x_position": 0.5, "y_position": 0.5})
        assert rh.REPLAY_HANDLERS["Fieldprobe2D"](_step(), blk, {0: field}) == pytest.approx(1.5)

    def test_probe_2d_rejects_non_2d(self):
        assert rh.REPLAY_HANDLERS["Fieldprobe2D"](_step(), _block(), {0: np.array([1.0])}) == 0.0

    def test_slice_x_and_y(self):
        field = np.array([[0.0, 1.0], [2.0, 3.0]])
        h = rh.REPLAY_HANDLERS["Fieldslice"]
        row = h(_step(), _block(params={"slice_direction": "x", "slice_position": 1.0}), {0: field})
        col = h(_step(), _block(params={"slice_direction": "y", "slice_position": 0.0}), {0: field})
        assert np.array_equal(row, [2.0, 3.0]) and np.array_equal(col, [0.0, 2.0])

    def test_probe_1d_interpolates_normalized_and_absolute(self):
        field = np.array([0.0, 10.0, 20.0])
        h = rh.REPLAY_HANDLERS["Fieldprobe"]
        assert h(_step(), _block(params={"position": 0.25}), {0: field}) == pytest.approx(5.0)
        blk = _block(params={"position": 1.0, "position_mode": "absolute", "L": 2.0})
        assert h(_step(), blk, {0: field}) == pytest.approx(10.0)
        assert h(_step(), _block(), {0: np.array([])}) == 0.0

    def test_fieldscope_handlers_pass_through(self):
        assert np.array_equal(
            rh.REPLAY_HANDLERS["Fieldscope"](_step(), _block(), {0: [[1.0, 2.0]]}), [1.0, 2.0]
        )
        assert rh.REPLAY_HANDLERS["Fieldscope2D"](_step(), _block(), {0: [1.0, 2.0]}).shape == (
            1,
            2,
        )


@pytest.mark.unit
class TestDiscreteHandlers:
    def test_state_variable_one_step_delay_and_reset(self):
        h = rh.REPLAY_HANDLERS["StateVariable"]
        blk = _block(params={"initial_value": "[1.0, 2.0]", "_init_start_": True})
        assert np.array_equal(h(_step(), blk, {0: np.array([7.0, 8.0])}), [1.0, 2.0])
        assert np.array_equal(h(_step(), blk, {0: np.array([9.0, 9.0])}), [7.0, 8.0])
        # reset_memblocks sets _init_start_ again -> back to the initial value
        blk.params["_init_start_"] = True
        assert np.array_equal(h(_step(), blk, {}), [1.0, 2.0])

    def test_state_variable_scalar_output_and_bad_initial(self):
        h = rh.REPLAY_HANDLERS["StateVariable"]
        blk = _block(params={"initial_value": "not a literal"})
        assert h(_step(), blk, {}) == 1.0

    def test_hysteresis_latches_in_exec_params(self):
        h = rh.REPLAY_HANDLERS["Hysteresis"]
        blk = _block(params={"upper": 1.0, "lower": -1.0, "high": 5.0, "low": -5.0}, exec_params={})
        assert h(_step(), blk, {0: 0.0}) == -5.0
        assert h(_step(), blk, {0: 2.0}) == 5.0
        assert h(_step(), blk, {0: 0.0}) == 5.0  # holds
        assert h(_step(), blk, {0: -2.0}) == -5.0
        assert blk.exec_params["_init_start_"] is False

    def test_demux_primary_and_secondary_ports(self):
        blk = _block("m", params={"output_shape": 2, "_outputs_": 3})
        step = _step()
        out = rh.REPLAY_HANDLERS["Demux"](step, blk, {0: np.arange(6.0)})
        assert np.array_equal(out, [0.0, 1.0])
        assert np.array_equal(step.signals["m_out1"], [2.0, 3.0])
        assert np.array_equal(step.signals["m_out2"], [4.0, 5.0])


@pytest.mark.unit
class TestMathFunction:
    h = staticmethod(rh.REPLAY_HANDLERS["Mathfunction"])

    @pytest.mark.parametrize(
        "func, value, expected",
        [
            ("sin", 0.0, 0.0),
            ("square", 3.0, 9.0),
            ("cube", 2.0, 8.0),
            ("sqrt", -4.0, 0.0),
            ("log", -1.0, 0.0),
            ("log10", 100.0, 2.0),
            ("asin", 2.0, 0.0),
            ("acos", 1.0, 0.0),
            ("reciprocal", 0.0, 0.0),
            ("reciprocal", 4.0, 0.25),
            ("SIGN", -3.0, -1.0),
        ],
    )
    def test_named_functions_are_domain_guarded(self, func, value, expected):
        out = self.h(_step(), _block(params={"function": func}), {0: value})
        assert np.asarray(out, dtype=float) == pytest.approx(expected)

    def test_every_table_entry_handles_a_vector(self):
        v = np.array([-1.0, 0.0, 0.5, 2.0])
        for name, op in rh.MATHFUNCTION_OPS.items():
            out = np.asarray(op(v), dtype=float)
            assert out.shape == v.shape, name
            assert np.all(np.isfinite(out)), name

    def test_expression_fallback_sees_u_and_t(self):
        out = self.h(_step(t=2.0), _block(params={"expression": "u * t + 1"}), {0: 3.0})
        assert float(out) == pytest.approx(7.0)

    def test_expression_value_error_yields_zero(self, monkeypatch):
        def boom(*a, **k):
            raise ValueError("bad")

        monkeypatch.setattr(rh, "safe_expr", boom)
        assert self.h(_step(), _block(params={"function": "u + 1"}), {0: 0.0}) == 0.0


@pytest.mark.unit
class TestFallback:
    def test_no_instance_is_zero(self):
        assert rh.replay_fallback(_block(), {}, 0.0, 0.01, set()) == 0.0

    def test_execute_result_port0(self):
        inst = SimpleNamespace(execute=lambda **kw: {0: kw["inputs"][0] * 2, "E": False})
        blk = _block(block_instance=inst)
        assert rh.replay_fallback(blk, {0: 4.0}, 0.0, 0.01, set()) == 8.0

    def test_failure_logged_once_and_stays_zero(self, caplog):
        def boom(**kw):
            raise RuntimeError("nope")

        blk = _block("bad", block_instance=SimpleNamespace(execute=boom))
        failed = set()
        with caplog.at_level("WARNING", logger="lib.engine.replay_handlers"):
            assert rh.replay_fallback(blk, {}, 0.0, 0.01, failed) == 0.0
            assert rh.replay_fallback(blk, {}, 0.0, 0.01, failed) == 0.0
        assert failed == {"bad"}
        assert sum("Replay fallback execute() failed" in r.message for r in caplog.records) == 1


@pytest.mark.unit
class TestRecorders:
    def test_scope_preallocates_and_labels(self):
        run = rh.ReplayRun(num_steps=3)
        blk = _block("s", fn="Scope", params={"labels": "a, b"}, exec_params={}, in_ports=2)
        rec = rh.RECORDERS["Scope"]
        rec(run, blk, {0: 1.0, 1: np.array([2.0, 3.0])}, 0, 0.0)
        assert blk.exec_params["vector"].shape == (3, 3)
        assert blk.exec_params["vec_dim"] == 3
        assert blk.exec_params["vec_labels"] == ["a", "b", "s-2"]
        rec(run, blk, {0: 4.0, 1: np.array([5.0, 6.0])}, 1, 0.1)
        assert np.array_equal(blk.exec_params["vector"][1], [4.0, 5.0, 6.0])

    def test_scope_width_change_pads_and_warns_once(self, caplog):
        run = rh.ReplayRun(num_steps=3)
        blk = _block("s", fn="Scope", exec_params={})
        rec = rh.RECORDERS["Scope"]
        rec(run, blk, {0: np.array([1.0, 2.0])}, 0, 0.0)
        with caplog.at_level("WARNING", logger="lib.engine.replay_handlers"):
            rec(run, blk, {0: np.array([9.0])}, 1, 0.1)
            rec(run, blk, {0: np.array([1.0, 2.0, 3.0])}, 2, 0.2)
        vec = blk.exec_params["vector"]
        assert np.array_equal(vec[1], [9.0, 0.0]) and np.array_equal(vec[2], [1.0, 2.0])
        assert sum("width changed" in r.message for r in caplog.records) == 1

    def test_scope_default_labels_and_missing_exec_params(self):
        run = rh.ReplayRun(num_steps=1)
        blk = _block("s", fn="Scope", params={"labels": "default"})
        rh.RECORDERS["Scope"](run, blk, {0: 1.0}, 0, 0.0)
        assert blk.exec_params["vec_labels"] == ["s-0"]

    def test_field_scopes_record_and_finalize(self):
        run = rh.ReplayRun(num_steps=4)
        fs = _block("f", fn="FieldScope", exec_params={})
        fs2 = _block("g", fn="FieldScope2D", params={"sample_interval": 2}, exec_params={})
        for i in range(4):
            rh.RECORDERS["Fieldscope"](run, fs, {0: np.array([i, i])}, i, i * 0.1)
            rh.RECORDERS["Fieldscope2D"](run, fs2, {0: np.full((2, 2), i)}, i, i * 0.1)
        assert len(fs.exec_params["_field_history_"]) == 4
        assert len(fs2.exec_params["_field_history_2d_"]) == 2  # every 2nd frame
        rh.finalize_recorders([fs, fs2])
        assert fs.exec_params["_field_history_"].shape == (4, 2)
        assert fs.exec_params["_time_history_"].shape == (4,)
        assert fs2.exec_params["_field_history_2d_"].shape == (2, 2, 2)
        assert np.array_equal(fs2.exec_params["_time_history_"], [0.0, 0.2])

    def test_finalize_skips_blocks_without_history(self):
        blk = _block("s", fn="Scope", exec_params={"vector": [[1.0], [2.0]]})
        other = _block("o", fn="Gain")
        rh.finalize_recorders([blk, other])
        assert isinstance(blk.exec_params["vector"], np.ndarray)
