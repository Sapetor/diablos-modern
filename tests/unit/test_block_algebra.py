"""Closed-loop transfer functions by Mason's rule (lib/analysis/block_algebra.py)."""

import types

import numpy as np
import pytest

from lib.analysis.block_algebra import AlgebraError, Rational, sink_input_signal, transfer_function


def B(name, fn, **params):
    return types.SimpleNamespace(name=name, username=name, block_fn=fn, params=params)


def L(src, dst, dport=0, sport=0):
    return types.SimpleNamespace(srcblock=src, srcport=sport, dstblock=dst, dstport=dport)


def _tf(result):
    return np.asarray(result.tf.num), np.asarray(result.tf.den)


def _close(result, num, den):
    n, d = _tf(result)
    np.testing.assert_allclose(n, num, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(d, den, rtol=1e-9, atol=1e-12)


@pytest.mark.unit
class TestBlockAlgebra:
    def test_open_loop_chain(self):
        blocks = [
            B("u", "Step"),
            B("k", "Gain", gain=3.0),
            B("g", "TranFn", numerator=[2.0], denominator=[1.0, 4.0]),
        ]
        lines = [L("u", "k"), L("k", "g")]
        r = transfer_function(blocks, lines, "u", ("g", 0))
        _close(r, [6.0], [1.0, 4.0])
        assert len(r.steps.paths) == 1 and r.steps.loops == []

    def test_unity_feedback(self):
        blocks = [
            B("r", "Step"),
            B("e", "Sum", sign="+-"),
            B("k", "Gain", gain=5.0),
            B("p", "TranFn", numerator=[1.0], denominator=[1.0, 1.0]),
            B("scope", "Scope"),
        ]
        lines = [L("r", "e", 0), L("p", "e", 1), L("e", "k"), L("k", "p"), L("p", "scope")]
        r = transfer_function(blocks, lines, "r", sink_input_signal(lines, "scope"))
        _close(r, [5.0], [1.0, 6.0])  # K/(s+1+K)
        assert len(r.steps.loops) == 1
        _close(types.SimpleNamespace(tf=r.steps.loops[0][1]), [-5.0], [1.0, 1.0])
        assert r.tf.dc_gain() == pytest.approx(5.0 / 6.0)

    def test_two_nested_loops_and_integrator(self):
        # Inner rate loop around 1/s, outer position loop: x/r = 2 / (s^2 + 3 s + 2)
        blocks = [
            B("r", "Step"),
            B("e1", "Sum", sign="+-"),
            B("k1", "Gain", gain=2.0),
            B("e2", "Sum", sign="+-"),
            B("int1", "Integrator"),
            B("k2", "Gain", gain=3.0),
            B("int2", "Integrator"),
        ]
        lines = [
            L("r", "e1"),
            L("int2", "e1", 1),
            L("e1", "k1"),
            L("k1", "e2"),
            L("k2", "e2", 1),
            L("e2", "int1"),
            L("int1", "k2"),
            L("int1", "int2"),
        ]
        r = transfer_function(blocks, lines, "r", ("int2", 0))
        _close(r, [2.0], [1.0, 3.0, 2.0])
        assert len(r.steps.loops) == 2

    def test_non_touching_loops(self):
        # Two separate feedback loops in series: (1/(s+2)) * (1/(s+3)).
        blocks = [
            B("u", "Step"),
            B("a", "Sum", sign="+-"),
            B("i1", "Integrator"),
            B("g1", "Gain", gain=2.0),
            B("b", "Sum", sign="+-"),
            B("i2", "Integrator"),
            B("g2", "Gain", gain=3.0),
        ]
        lines = [
            L("u", "a"),
            L("g1", "a", 1),
            L("a", "i1"),
            L("i1", "g1"),
            L("i1", "b"),
            L("g2", "b", 1),
            L("b", "i2"),
            L("i2", "g2"),
        ]
        r = transfer_function(blocks, lines, "u", ("i2", 0))
        _close(r, [1.0], [1.0, 5.0, 6.0])
        assert len(r.steps.loops) == 2

    def test_pid_setpoint_measurement_form(self):
        blocks = [
            B("r", "Step"),
            B("c", "PID", Kp=2.0, Ki=1.0, Kd=0.0, N=20.0),
            B("p", "TranFn", numerator=[1.0], denominator=[1.0, 1.0]),
        ]
        lines = [L("r", "c", 0), L("p", "c", 1), L("c", "p")]
        r = transfer_function(blocks, lines, "r", ("p", 0))
        # C = (2s+1)/s, P = 1/(s+1): CP/(1+CP) = (2s+1)/(s^2+3s+1)
        _close(r, [2.0, 1.0], [1.0, 3.0, 1.0])

    def test_other_sources_are_zeroed_and_off_path_blocks_ignored(self):
        blocks = [
            B("r", "Step"),
            B("d", "Constant"),
            B("s", "Sum", sign="++"),
            B("g", "Gain", gain=4.0),
            B("sat", "Saturation"),
            B("scope", "Scope"),
        ]
        lines = [L("r", "s", 0), L("d", "s", 1), L("s", "g"), L("g", "sat"), L("g", "scope")]
        r = transfer_function(blocks, lines, "r", ("g", 0))
        _close(r, [4.0], [1.0])

    def test_unsupported_block_on_the_path(self):
        blocks = [B("r", "Step"), B("sat", "Saturation"), B("g", "Gain", gain=1.0)]
        lines = [L("r", "sat"), L("sat", "g")]
        with pytest.raises(AlgebraError, match='Saturation "sat"'):
            transfer_function(blocks, lines, "r", ("g", 0))

    def test_sampled_block_and_no_path(self):
        blocks = [
            B("r", "Step"),
            B("g", "TranFn", numerator=[1.0], denominator=[1.0, 1.0], sampling_time=0.1),
        ]
        with pytest.raises(AlgebraError, match="sampled"):
            transfer_function(blocks, [L("r", "g")], "r", ("g", 0))
        with pytest.raises(AlgebraError, match="no signal path"):
            transfer_function(blocks, [], "r", ("g", 0))

    def test_minreal_cancels_common_roots(self):
        g = Rational.of([1.0, 1.0], [1.0, 3.0, 2.0]).minreal()  # (s+1)/((s+1)(s+2))
        np.testing.assert_allclose(g.num, [1.0])
        np.testing.assert_allclose(g.den, [1.0, 2.0])


@pytest.mark.integration
def test_matches_the_engine_on_the_pid_example(tmp_path):
    """G(s) from the reference Step to the plant output of the welcome screen's
    PID example, simulated with scipy, must match the engine's Scope trace.
    Checks sign conventions and the PID form against the real simulator."""
    import json
    import os

    from scipy import signal

    from lib.analysis.resim import harvest_scope_signals
    from lib.cli import run_diagram

    src = os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "pid_second_order.diablos"
    )
    data = json.load(open(src, encoding="utf-8"))
    for b in data["blocks_data"]:
        if b["block_fn"] == "PID":  # a linear TF cannot model output saturation
            b["params"]["u_min"], b["params"]["u_max"] = -1e9, 1e9
    path = tmp_path / "pid_linear.diablos"
    path.write_text(json.dumps(data), encoding="utf-8")

    blocks = [
        types.SimpleNamespace(
            name=b["name"], username=b["username"], block_fn=b["block_fn"], params=b["params"]
        )
        for b in data["blocks_data"]
    ]
    lines = [
        types.SimpleNamespace(**{k: ln[k] for k in ("srcblock", "srcport", "dstblock", "dstport")})
        for ln in data["lines_data"]
    ]
    r = transfer_function(blocks, lines, "step0", sink_input_signal(lines, "scope5", 1))

    result = harvest_scope_signals(run_diagram(str(path)))
    t = np.ravel(result["timeline"])
    y_engine = np.ravel(list(result["signals"].values())[1])
    u = np.where(t >= 1.0, 1.0, 0.0)  # reference Step: value 1, delay 1 s
    # interp=False holds each input sample, so the step lands exactly at 1 s
    # (linear interpolation would ramp it over one dt).
    _, y_tf, _ = signal.lsim(signal.lti(r.tf.num, r.tf.den), u, t, interp=False)
    window = t < 15.0  # the load disturbance (another source) starts at 15 s
    assert list(result["signals"])[1] == "output"
    assert np.max(np.abs(y_tf[window] - y_engine[window])) < 1e-6  # measured: ~7e-11


@pytest.mark.unit
def test_output_is_the_source_itself():
    r = transfer_function([B("u", "Step")], [], "u", ("u", 0))
    _close(r, [1.0], [1.0])
