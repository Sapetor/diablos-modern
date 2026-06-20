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


@pytest.mark.unit
class TestKernelRegistry:
    def test_source_family_registered(self):
        for name in ("Constant", "Step", "Ramp", "Sine"):
            assert name in KERNEL_BUILDERS
            assert get_kernel_builder(name) is KERNEL_BUILDERS[name]

    def test_unknown_fn_returns_none(self):
        assert get_kernel_builder("NotABlock") is None
        # Not-yet-migrated blocks must fall through (None) to the legacy if/elif.
        assert get_kernel_builder("Gain") is None

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
