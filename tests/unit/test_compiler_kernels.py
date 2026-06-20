"""Unit tests for the compiled-path kernel registry (lib.engine.compiler_kernels).

The golden harness (tests/regression/test_compiled_golden.py) exercises the
kernels end-to-end; these pin the registry mechanism and a couple of builders
directly so the infrastructure has its own fast coverage.
"""
import numpy as np
import pytest

from lib.engine.compiler_kernels import (
    BuildContext,
    KERNEL_BUILDERS,
    get_kernel_builder,
    kernel,
)


def _ctx(b_name="blk0", **params):
    return BuildContext(
        block=None, b_name=b_name, fn="?", params=params,
        input_sources=[], deps={}, state_map={}, block_matrices={},
    )


def _run_at(executor, t, b_name="blk0"):
    """Run a source executor at time t and return its scalar output."""
    sig = {}
    executor(t, None, None, sig)
    return sig[b_name]


@pytest.mark.unit
class TestKernelRegistry:
    def test_source_family_registered(self):
        for name in ("Constant", "Step", "Ramp", "Sine"):
            assert name in KERNEL_BUILDERS
            assert get_kernel_builder(name) is KERNEL_BUILDERS[name]

    def test_unknown_fn_returns_none(self):
        assert get_kernel_builder("NotABlock") is None
        # Not-yet-migrated blocks must fall through (None) to the legacy if/elif.
        assert get_kernel_builder("Integrator") is None

    def test_kernel_decorator_registers_all_names(self):
        @kernel("AaaTmp1", "AaaTmp2")
        def _b(ctx):
            return None
        try:
            assert get_kernel_builder("AaaTmp1") is _b
            assert get_kernel_builder("AaaTmp2") is _b
        finally:
            KERNEL_BUILDERS.pop("AaaTmp1", None)
            KERNEL_BUILDERS.pop("AaaTmp2", None)


@pytest.mark.unit
class TestSourceKernels:
    def test_constant_scalar(self):
        ex = get_kernel_builder("Constant")(_ctx(value=2.5))
        sig = {}
        ex(0.0, None, None, sig)
        assert sig["blk0"] == 2.5

    def test_constant_vector(self):
        ex = get_kernel_builder("Constant")(_ctx(value=[1.0, 2.0, 3.0]))
        sig = {}
        ex(0.0, None, None, sig)
        assert np.array_equal(sig["blk0"], np.array([1.0, 2.0, 3.0]))

    def test_step_up_before_and_after_delay(self):
        ex = get_kernel_builder("Step")(_ctx(value=1.0, delay=0.5, type="up"))
        sig = {}
        ex(0.4, None, None, sig)
        assert sig["blk0"] == 0.0
        ex(0.6, None, None, sig)
        assert sig["blk0"] == 1.0

    def test_ramp_clamps_at_delay(self):
        ex = get_kernel_builder("Ramp")(_ctx(slope=2.0, delay=1.0))
        sig = {}
        ex(0.5, None, None, sig)   # before delay -> 0
        assert sig["blk0"] == 0.0
        ex(3.0, None, None, sig)   # 2*(3-1) = 4
        assert sig["blk0"] == 4.0

    def test_sine_value(self):
        ex = get_kernel_builder("Sine")(_ctx(amplitude=2.0, frequency=1.0, phase=0.0, bias=0.5))
        sig = {}
        ex(np.pi / 2, None, None, sig)
        assert np.isclose(sig["blk0"], 2.0 * 1.0 + 0.5)

    def test_wavegenerator_sine(self):
        ex = get_kernel_builder("Wavegenerator")(
            _ctx(waveform="Sine", amplitude=2.0, frequency=1.0, phase=0.0, bias=0.5))
        sig = {}
        ex(0.25, None, None, sig)  # 2*pi*1*0.25 = pi/2 -> sin = 1
        assert np.isclose(sig["blk0"], 0.5 + 2.0 * 1.0)

    def test_wavegenerator_square_sign(self):
        ex = get_kernel_builder("Wavegenerator")(
            _ctx(waveform="Square", amplitude=1.0, frequency=1.0, phase=0.0, bias=0.0))
        sig = {}
        ex(0.1, None, None, sig)   # first half period -> +1
        assert sig["blk0"] == 1.0

    def test_impulse_window(self):
        ex = get_kernel_builder("Impulse")(_ctx(delay=1.0, value=1.0, dtime=0.01))
        sig = {}
        ex(0.5, None, None, sig)            # before delay -> 0
        assert sig["blk0"] == 0.0
        ex(1.0, None, None, sig)            # inside narrow pulse -> large height
        assert sig["blk0"] > 0.0

    def test_prbs_deterministic_from_seed(self):
        ex = get_kernel_builder("Prbs")(_ctx(high=1.0, low=0.0, bit_time=1.0, order=3, seed=1))
        # order=3 -> period 7; sample two whole periods and assert repetition.
        first = [(_run_at(ex, float(t))) for t in range(7)]
        second = [(_run_at(ex, float(t + 7))) for t in range(7)]
        assert first == second
        assert set(first) <= {0.0, 1.0}

    def test_noise_zero_sigma_is_mu(self):
        ex = get_kernel_builder("Noise")(_ctx(mu=3.0, sigma=0.0))
        sig = {}
        ex(0.0, None, None, sig)
        assert sig["blk0"] == 3.0


def _ctx_io(b_name, params, input_sources):
    return BuildContext(
        block=None, b_name=b_name, fn="?", params=params,
        input_sources=input_sources, deps={}, state_map={}, block_matrices={},
    )


@pytest.mark.unit
class TestMathKernels:
    def test_gain_scales_input(self):
        ex = get_kernel_builder("Gain")(_ctx_io("g0", {"gain": 3.0}, ["src"]))
        sig = {"src": 4.0}
        ex(0.0, None, None, sig)
        assert sig["g0"] == 12.0

    def test_gain_no_input_is_zero(self):
        ex = get_kernel_builder("Gain")(_ctx_io("g0", {"gain": 3.0}, []))
        sig = {}
        ex(0.0, None, None, sig)
        assert sig["g0"] == 0.0

    def test_sum_applies_signs(self):
        ex = get_kernel_builder("Sum")(_ctx_io("s0", {"sign": "+-"}, ["a", "b"]))
        sig = {"a": 5.0, "b": 2.0}
        ex(0.0, None, None, sig)
        assert sig["s0"] == 3.0

    def test_sum_extra_wired_input_defaults_plus(self):
        # 3 wired inputs but only 2 sign chars -> 3rd defaults to '+'.
        ex = get_kernel_builder("Sum")(_ctx_io("s0", {"sign": "+-"}, ["a", "b", "c"]))
        sig = {"a": 5.0, "b": 2.0, "c": 1.0}
        ex(0.0, None, None, sig)
        assert sig["s0"] == 4.0


@pytest.mark.unit
class TestNonlinearKernels:
    def test_saturation_clips(self):
        ex = get_kernel_builder("Saturation")(_ctx_io("s0", {"min": -1.0, "max": 1.0}, ["x"]))
        for v, expected in [(2.0, 1.0), (-2.0, -1.0), (0.3, 0.3)]:
            sig = {"x": v}
            ex(0.0, None, None, sig)
            assert np.isclose(sig["s0"], expected)

    def test_deadband_zero_inside_offset_outside(self):
        ex = get_kernel_builder("Deadband")(_ctx_io("d0", {"start": -0.5, "end": 0.5}, ["x"]))
        sig = {"x": 0.2}
        ex(0.0, None, None, sig)
        assert np.isclose(np.asarray(sig["d0"]).item(), 0.0)
        sig = {"x": 1.5}
        ex(0.0, None, None, sig)
        assert np.isclose(np.asarray(sig["d0"]).item(), 1.0)  # 1.5 - end(0.5)

    def test_switch_threshold_selects(self):
        # ctrl >= threshold -> data input 0 (port index 1); else input 1 (port 2)
        ex = get_kernel_builder("Switch")(
            _ctx_io("sw0", {"mode": "threshold", "n_inputs": 2, "threshold": 0.0},
                    ["ctrl", "a", "b"]))
        sig = {"ctrl": 1.0, "a": 10.0, "b": 20.0}
        ex(0.0, None, None, sig)
        assert sig["sw0"] == 10.0
        sig = {"ctrl": -1.0, "a": 10.0, "b": 20.0}
        ex(0.0, None, None, sig)
        assert sig["sw0"] == 20.0

    def test_selector_index_and_range(self):
        ex = get_kernel_builder("Selector")(_ctx_io("se0", {"indices": "1,3"}, ["x"]))
        sig = {"x": np.array([10.0, 11.0, 12.0, 13.0])}
        ex(0.0, None, None, sig)
        assert np.allclose(sig["se0"], np.array([11.0, 13.0]))
        ex = get_kernel_builder("Selector")(_ctx_io("se0", {"indices": "1:3"}, ["x"]))
        sig = {"x": np.array([10.0, 11.0, 12.0, 13.0])}
        ex(0.0, None, None, sig)
        assert np.allclose(sig["se0"], np.array([11.0, 12.0]))

    def test_hysteresis_latches(self):
        ex = get_kernel_builder("Hysteresis")(
            _ctx_io("h0", {"upper": 0.5, "lower": -0.5, "high": 1.0, "low": 0.0}, ["x"]))
        sig = {"x": 0.0}
        ex(0.0, None, None, sig)
        assert sig["h0"] == 0.0          # starts low, stays in deadband
        sig = {"x": 1.0}
        ex(0.0, None, None, sig)
        assert sig["h0"] == 1.0          # crosses upper -> high
        sig = {"x": 0.0}
        ex(0.0, None, None, sig)
        assert sig["h0"] == 1.0          # in deadband -> retains high
        sig = {"x": -1.0}
        ex(0.0, None, None, sig)
        assert sig["h0"] == 0.0          # crosses lower -> low

    def test_exponential(self):
        ex = get_kernel_builder("Exponential")(_ctx_io("e0", {"a": 2.0, "b": 0.5}, ["x"]))
        sig = {"x": 4.0}
        ex(0.0, None, None, sig)
        assert np.isclose(sig["e0"], 2.0 * np.exp(0.5 * 4.0))

    def test_abs(self):
        ex = get_kernel_builder("Abs")(_ctx_io("a0", {}, ["x"]))
        sig = {"x": -3.5}
        ex(0.0, None, None, sig)
        assert sig["a0"] == 3.5

    def test_sgprod_multiplies_inputs(self):
        ex = get_kernel_builder("SgProd")(_ctx_io("p0", {}, ["a", "b", "c"]))
        sig = {"a": 2.0, "b": 3.0, "c": 4.0}
        ex(0.0, None, None, sig)
        assert sig["p0"] == 24.0

    def test_product_multiply_and_divide(self):
        ex = get_kernel_builder("Product")(_ctx_io("pr0", {"ops": "*/"}, ["a", "b"]))
        sig = {"a": 12.0, "b": 3.0}
        ex(0.0, None, None, sig)
        assert np.isclose(np.asarray(sig["pr0"]).item(), 4.0)

    def test_product_divide_by_zero_is_finite(self):
        ex = get_kernel_builder("Product")(_ctx_io("pr0", {"ops": "*/"}, ["a", "b"]))
        sig = {"a": 1.0, "b": 0.0}
        ex(0.0, None, None, sig)
        # divide-by-zero -> large finite (sign-preserving), never inf/nan.
        out = np.asarray(sig["pr0"])
        assert np.all(np.isfinite(out)) and np.all(out > 0)

    def test_matrixgain_scalar(self):
        ex = get_kernel_builder("MatrixGain")(_ctx_io("m0", {"gain": "3.0"}, ["x"]))
        sig = {"x": 2.0}
        ex(0.0, None, None, sig)
        assert np.allclose(sig["m0"], 6.0)

    def test_matrixgain_matrix(self):
        ex = get_kernel_builder("MatrixGain")(_ctx_io("m0", {"gain": "[[1, 2], [0, 1]]"}, ["x"]))
        sig = {"x": np.array([1.0, 2.0])}
        ex(0.0, None, None, sig)
        assert np.allclose(sig["m0"], np.array([5.0, 2.0]))

    def test_mathfunction_named(self):
        ex = get_kernel_builder("Mathfunction")(_ctx_io("f0", {"function": "sqrt"}, ["x"]))
        sig = {"x": 9.0}
        ex(0.0, None, None, sig)
        assert np.isclose(sig["f0"], 3.0)

    def test_mathfunction_expression_fallback(self):
        ex = get_kernel_builder("Mathfunction")(_ctx_io("f0", {"function": "u*u + 1"}, ["x"]))
        sig = {"x": 3.0}
        ex(0.0, None, None, sig)
        assert np.isclose(sig["f0"], 10.0)
