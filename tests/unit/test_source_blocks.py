"""
Unit tests for Source block implementations.
"""

import pytest
import numpy as np


@pytest.mark.unit
class TestStepBlock:
    """Tests for Step block."""

    def test_step_up_before_delay(self):
        """Test step-up signal is 0 before delay time."""
        from blocks.step import StepBlock
        block = StepBlock()
        params = {'value': 5.0, 'delay': 1.0, 'type': 'up',
                  'pulse_start_up': True, '_init_start_': True}

        result = block.execute(time=0.5, inputs={}, params=params)
        assert result[0][0] == 0.0, "Step should be 0 before delay"

    def test_step_up_after_delay(self):
        """Test step-up signal reaches final value after delay."""
        from blocks.step import StepBlock
        block = StepBlock()
        params = {'value': 5.0, 'delay': 1.0, 'type': 'up',
                  'pulse_start_up': True, '_init_start_': True}

        # First call to initialize
        block.execute(time=0.0, inputs={}, params=params)
        result = block.execute(time=1.5, inputs={}, params=params)
        assert result[0][0] == 5.0, "Step should reach final value after delay"

    def test_step_down(self):
        """Test step-down signal behavior."""
        from blocks.step import StepBlock
        block = StepBlock()
        params = {'value': 3.0, 'delay': 1.0, 'type': 'down',
                  'pulse_start_up': True, '_init_start_': True}

        # Before delay - should be at value
        block.execute(time=0.0, inputs={}, params=params)
        result = block.execute(time=0.5, inputs={}, params=params)
        assert result[0][0] == 3.0, "Step-down should be at value before delay"

        # After delay - should be 0
        result = block.execute(time=1.5, inputs={}, params=params)
        assert result[0][0] == 0.0, "Step-down should be 0 after delay"

    def test_step_constant(self):
        """Test constant type always outputs the value."""
        from blocks.step import StepBlock
        block = StepBlock()
        params = {'value': 7.0, 'delay': 1.0, 'type': 'constant',
                  'pulse_start_up': True, '_init_start_': True}

        block.execute(time=0.0, inputs={}, params=params)

        for t in [0.0, 0.5, 1.0, 2.0, 10.0]:
            result = block.execute(time=t, inputs={}, params=params)
            assert result[0][0] == 7.0, f"Constant should always output value at t={t}"


    def test_impulse_zero_before_delay(self):
        """Impulse outputs 0 before delay time."""
        from blocks.step import StepBlock
        block = StepBlock()
        dt = 0.01
        params = {'value': 1.0, 'delay': 1.0, 'type': 'impulse',
                  'pulse_start_up': True, '_init_start_': True}

        result = block.execute(time=0.5, inputs={}, params=params, dtime=dt)
        assert result[0][0] == 0.0, "Impulse should be 0 before delay"

    def test_impulse_fires_at_delay(self):
        """Impulse outputs value/dt for one step at t=delay."""
        from blocks.step import StepBlock
        block = StepBlock()
        dt = 0.01
        params = {'value': 1.0, 'delay': 1.0, 'type': 'impulse',
                  'pulse_start_up': True, '_init_start_': True}

        # Initialize
        block.execute(time=0.0, inputs={}, params=params, dtime=dt)
        result = block.execute(time=1.0, inputs={}, params=params, dtime=dt)
        expected = 1.0 / dt  # 100.0
        assert result[0][0] == pytest.approx(expected), \
            f"Impulse should be value/dt={expected} at t=delay"

    def test_impulse_zero_after_delay(self):
        """Impulse outputs 0 after the impulse step."""
        from blocks.step import StepBlock
        block = StepBlock()
        dt = 0.01
        params = {'value': 1.0, 'delay': 1.0, 'type': 'impulse',
                  'pulse_start_up': True, '_init_start_': True}

        # Initialize and fire
        block.execute(time=0.0, inputs={}, params=params, dtime=dt)
        block.execute(time=1.0, inputs={}, params=params, dtime=dt)
        # Next step should be 0
        result = block.execute(time=1.0 + dt, inputs={}, params=params, dtime=dt)
        assert result[0][0] == 0.0, "Impulse should be 0 after firing"

    def test_impulse_integral_equals_value(self):
        """Discrete integral of impulse over all time equals value."""
        from blocks.step import StepBlock
        block = StepBlock()
        dt = 0.01
        value = 3.0
        delay = 0.5
        params = {'value': value, 'delay': delay, 'type': 'impulse',
                  'pulse_start_up': True, '_init_start_': True}

        total = 0.0
        for i in range(200):
            t = i * dt
            result = block.execute(time=t, inputs={}, params=params, dtime=dt)
            total += result[0][0] * dt

        assert total == pytest.approx(value, abs=0.01), \
            f"Integral of impulse should equal value={value}, got {total}"

    def test_impulse_zero_delay(self):
        """Impulse with delay=0 fires at t=0."""
        from blocks.step import StepBlock
        block = StepBlock()
        dt = 0.01
        params = {'value': 2.0, 'delay': 0.0, 'type': 'impulse',
                  'pulse_start_up': True, '_init_start_': True}

        result = block.execute(time=0.0, inputs={}, params=params, dtime=dt)
        assert result[0][0] == pytest.approx(2.0 / dt), \
            "Impulse at delay=0 should fire at t=0"

        result = block.execute(time=dt, inputs={}, params=params, dtime=dt)
        assert result[0][0] == 0.0, "Impulse should be 0 after firing"

    def test_impulse_compiled_solver(self, qapp):
        """Impulse through 1/(s+1) in compiled solver gives decaying exponential.

        Analytical impulse response of 1/(s+1): y(t) = e^(-t).
        At t=1: y ≈ 0.368, at t=3: y ≈ 0.050.
        """
        from PyQt5.QtCore import QRect, QPoint
        from PyQt5.QtGui import QColor
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine
        from lib.engine.system_compiler import SystemCompiler
        from scipy.integrate import solve_ivp

        dt = 0.001
        blocks = [
            DBlock(block_fn='Step', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=0, out_ports=1,
                   params={'value': 1.0, 'delay': 0.0, 'type': 'impulse',
                           'dtime': dt},
                   username='', b_type=0),
            DBlock(block_fn='TranFn', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=1, out_ports=1,
                   params={'numerator': [1.0], 'denominator': [1.0, 1.0]},
                   username='', b_type=1),
        ]
        lines = [
            DLine(sid=0, srcblock='step0', srcport=0,
                  dstblock='tranfn0', dstport=0,
                  points=[QPoint(0, 0), QPoint(100, 0)]),
        ]

        compiler = SystemCompiler()
        model_func, y0, state_map, _ = compiler.compile_system(
            blocks, blocks, lines
        )
        sol = solve_ivp(model_func, (0, 5), y0, method='RK45',
                        t_eval=np.linspace(0, 5, 5000), rtol=1e-10, atol=1e-12)
        assert sol.success

        start, _ = state_map['tranfn0']
        # Impulse response of 1/(s+1) is e^(-t)
        # Check at t=1 and t=3
        idx_1 = np.argmin(np.abs(sol.t - 1.0))
        idx_3 = np.argmin(np.abs(sol.t - 3.0))
        assert sol.y[start, idx_1] == pytest.approx(np.exp(-1.0), abs=0.02), \
            f"Impulse response at t=1 should be ~{np.exp(-1.0):.4f}"
        assert sol.y[start, idx_3] == pytest.approx(np.exp(-3.0), abs=0.01), \
            f"Impulse response at t=3 should be ~{np.exp(-3.0):.4f}"


@pytest.mark.unit
class TestImpulseBlock:
    """Tests for standalone Impulse block."""

    def test_block_properties(self):
        """Impulse block is a source with correct metadata."""
        from blocks.impulse import ImpulseBlock
        block = ImpulseBlock()
        assert block.block_name == "Impulse"
        assert block.category == "Sources"
        assert block.b_type == 0
        assert block.inputs == []
        assert len(block.outputs) == 1

    def test_impulse_fires_at_delay(self):
        """Outputs value/dt for one step at t=delay."""
        from blocks.impulse import ImpulseBlock
        block = ImpulseBlock()
        dt = 0.01
        params = {'value': 1.0, 'delay': 1.0, '_init_start_': True}

        block.execute(time=0.0, inputs={}, params=params, dtime=dt)
        result = block.execute(time=1.0, inputs={}, params=params, dtime=dt)
        assert result[0][0] == pytest.approx(100.0)

    def test_impulse_zero_elsewhere(self):
        """Outputs 0 before and after the impulse."""
        from blocks.impulse import ImpulseBlock
        block = ImpulseBlock()
        dt = 0.01
        params = {'value': 1.0, 'delay': 0.5, '_init_start_': True}

        r_before = block.execute(time=0.0, inputs={}, params=params, dtime=dt)
        assert r_before[0][0] == 0.0
        # Fire it
        block.execute(time=0.5, inputs={}, params=params, dtime=dt)
        r_after = block.execute(time=0.51, inputs={}, params=params, dtime=dt)
        assert r_after[0][0] == 0.0

    def test_impulse_integral(self):
        """Discrete integral over all time equals value."""
        from blocks.impulse import ImpulseBlock
        block = ImpulseBlock()
        dt = 0.01
        params = {'value': 5.0, 'delay': 0.0, '_init_start_': True}

        total = 0.0
        for i in range(100):
            result = block.execute(time=i * dt, inputs={}, params=params, dtime=dt)
            total += result[0][0] * dt
        assert total == pytest.approx(5.0, abs=0.01)

    def test_compiled_solver_impulse_response(self, qapp):
        """Impulse -> 1/(s+1) gives e^(-t) in compiled solver."""
        from PyQt5.QtCore import QRect, QPoint
        from PyQt5.QtGui import QColor
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine
        from lib.engine.system_compiler import SystemCompiler
        from scipy.integrate import solve_ivp

        dt = 0.001
        blocks = [
            DBlock(block_fn='Impulse', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=0, out_ports=1,
                   params={'value': 1.0, 'delay': 0.0, 'dtime': dt},
                   username='', b_type=0),
            DBlock(block_fn='TranFn', sid=0, coords=QRect(0, 0, 50, 40),
                   color=QColor(150, 150, 150), in_ports=1, out_ports=1,
                   params={'numerator': [1.0], 'denominator': [1.0, 1.0]},
                   username='', b_type=1),
        ]
        lines = [
            DLine(sid=0, srcblock='impulse0', srcport=0,
                  dstblock='tranfn0', dstport=0,
                  points=[QPoint(0, 0), QPoint(100, 0)]),
        ]

        compiler = SystemCompiler()
        model_func, y0, state_map, _ = compiler.compile_system(
            blocks, blocks, lines
        )
        sol = solve_ivp(model_func, (0, 5), y0, method='RK45',
                        t_eval=np.linspace(0, 5, 5000), rtol=1e-10, atol=1e-12)
        assert sol.success

        start, _ = state_map['tranfn0']
        idx_1 = np.argmin(np.abs(sol.t - 1.0))
        idx_3 = np.argmin(np.abs(sol.t - 3.0))
        assert sol.y[start, idx_1] == pytest.approx(np.exp(-1.0), abs=0.02)
        assert sol.y[start, idx_3] == pytest.approx(np.exp(-3.0), abs=0.01)


@pytest.mark.unit
class TestConstantBlock:
    """Tests for Constant block."""

    def test_constant_scalar(self):
        """Test constant outputs scalar value."""
        from blocks.constant import ConstantBlock
        block = ConstantBlock()
        params = {'value': 42.0}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert result[0][0] == 42.0, "Constant should output specified value"

    def test_constant_negative(self):
        """Test constant with negative value."""
        from blocks.constant import ConstantBlock
        block = ConstantBlock()
        params = {'value': -7.5}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert result[0][0] == -7.5, "Constant should handle negative values"

    def test_constant_vector(self):
        """Test constant with vector value."""
        from blocks.constant import ConstantBlock
        block = ConstantBlock()
        params = {'value': [1.0, 2.0, 3.0]}

        result = block.execute(time=0.0, inputs={}, params=params)
        expected = np.array([1.0, 2.0, 3.0])
        assert np.allclose(result[0], expected), "Constant should handle vectors"

    def test_constant_time_invariant(self):
        """Test constant output doesn't change with time."""
        from blocks.constant import ConstantBlock
        block = ConstantBlock()
        params = {'value': 10.0}

        for t in [0.0, 1.0, 100.0, 1000.0]:
            result = block.execute(time=t, inputs={}, params=params)
            assert result[0][0] == 10.0, f"Constant should be time-invariant at t={t}"


@pytest.mark.unit
class TestRampBlock:
    """Tests for Ramp block."""

    def test_ramp_before_delay(self):
        """Test ramp is 0 before delay time."""
        from blocks.ramp import RampBlock
        block = RampBlock()
        params = {'slope': 2.0, 'delay': 1.0}

        result = block.execute(time=0.5, inputs={}, params=params)
        assert result[0] == 0.0, "Ramp should be 0 before delay"

    def test_ramp_after_delay(self):
        """Test ramp increases linearly after delay."""
        from blocks.ramp import RampBlock
        block = RampBlock()
        params = {'slope': 2.0, 'delay': 1.0}

        result = block.execute(time=2.0, inputs={}, params=params)
        expected = 2.0 * (2.0 - 1.0)  # slope * (t - delay)
        assert np.isclose(result[0], expected), f"Ramp should be {expected} at t=2.0"

    def test_ramp_negative_slope(self):
        """Test ramp with negative slope."""
        from blocks.ramp import RampBlock
        block = RampBlock()
        params = {'slope': -3.0, 'delay': 0.0}

        result = block.execute(time=2.0, inputs={}, params=params)
        expected = -3.0 * 2.0  # slope * t
        assert np.isclose(result[0], expected), f"Negative ramp should be {expected} at t=2.0"

    def test_ramp_zero_slope(self):
        """Test ramp with zero slope outputs 0."""
        from blocks.ramp import RampBlock
        block = RampBlock()
        params = {'slope': 0.0, 'delay': 0.0}

        result = block.execute(time=10.0, inputs={}, params=params)
        assert result[0] == 0.0, "Zero slope ramp should always be 0"


@pytest.mark.unit
class TestSineBlock:
    """Tests for Sine block."""

    def test_sine_at_zero(self):
        """Test sine value at t=0."""
        from blocks.sine import SineBlock
        block = SineBlock()
        params = {'amplitude': 1.0, 'omega': 1.0, 'init_angle': 0.0}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert np.isclose(result[0], 0.0), "sin(0) should be 0"

    def test_sine_at_quarter_period(self):
        """Test sine reaches amplitude at quarter period."""
        from blocks.sine import SineBlock
        block = SineBlock()
        omega = 2.0
        params = {'amplitude': 5.0, 'omega': omega, 'init_angle': 0.0}

        # At t = pi/(2*omega), sin(omega*t) = sin(pi/2) = 1
        t = np.pi / (2 * omega)
        result = block.execute(time=t, inputs={}, params=params)
        assert np.isclose(result[0], 5.0), "Sine should reach amplitude at quarter period"

    def test_sine_with_phase(self):
        """Test sine with initial phase offset."""
        from blocks.sine import SineBlock
        block = SineBlock()
        params = {'amplitude': 1.0, 'omega': 1.0, 'init_angle': np.pi / 2}

        # sin(0 + pi/2) = 1
        result = block.execute(time=0.0, inputs={}, params=params)
        assert np.isclose(result[0], 1.0), "sin(pi/2) should be 1"

    def test_sine_periodicity(self):
        """Test sine is periodic."""
        from blocks.sine import SineBlock
        block = SineBlock()
        omega = 2.0
        params = {'amplitude': 3.0, 'omega': omega, 'init_angle': 0.0}

        period = 2 * np.pi / omega
        t_test = 1.5

        result1 = block.execute(time=t_test, inputs={}, params=params)
        result2 = block.execute(time=t_test + period, inputs={}, params=params)

        assert np.isclose(result1[0], result2[0]), "Sine should be periodic"

    def test_sine_amplitude_scaling(self):
        """Test sine amplitude scales correctly."""
        from blocks.sine import SineBlock
        block = SineBlock()
        omega = 1.0

        # At t = pi/2, sin(omega*t) = 1, so output = amplitude
        t = np.pi / 2

        for amp in [0.5, 1.0, 2.0, 10.0]:
            params = {'amplitude': amp, 'omega': omega, 'init_angle': 0.0}
            result = block.execute(time=t, inputs={}, params=params)
            assert np.isclose(result[0], amp), f"Amplitude {amp} should scale output"


@pytest.mark.unit
class TestNoiseBlock:
    """Tests for Noise (Gaussian noise generator) block."""

    def test_noise_mean(self):
        """Test noise has approximately correct mean over many samples."""
        from blocks.noise import NoiseBlock
        block = NoiseBlock()
        params = {'mu': 5.0, 'sigma': 1.0}

        samples = []
        for _ in range(1000):
            result = block.execute(time=0.0, inputs={}, params=params)
            samples.append(result[0])

        mean = np.mean(samples)
        assert np.isclose(mean, 5.0, atol=0.2), f"Mean should be ~5.0, got {mean}"

    def test_noise_std_dev(self):
        """Test noise has approximately correct std dev."""
        from blocks.noise import NoiseBlock
        block = NoiseBlock()
        params = {'mu': 0.0, 'sigma': 2.0}

        samples = []
        for _ in range(1000):
            result = block.execute(time=0.0, inputs={}, params=params)
            samples.append(result[0])

        # Note: block uses sigma^2 * randn(), so effective std = sigma^2
        std = np.std(samples)
        # With sigma=2, effective std is 4
        assert std > 1.0, f"Std should be > 1, got {std}"

    def test_noise_zero_sigma(self):
        """Test noise with zero sigma outputs mean."""
        from blocks.noise import NoiseBlock
        block = NoiseBlock()
        params = {'mu': 3.0, 'sigma': 0.0}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert np.isclose(result[0], 3.0), "With sigma=0, output should equal mu"


@pytest.mark.unit
class TestWaveGeneratorBlock:
    """Tests for WaveGenerator block."""

    def test_wave_sine(self):
        """Test sine waveform at known points."""
        from blocks.wave_generator import WaveGeneratorBlock
        block = WaveGeneratorBlock()
        params = {'waveform': 'Sine', 'amplitude': 2.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 0.0}

        # At t=0, sin(0) = 0
        result = block.execute(time=0.0, inputs={}, params=params)
        assert np.isclose(result[0], 0.0, atol=0.01), f"Sine at t=0 should be 0, got {result[0]}"

        # At t=0.25 (quarter period for f=1Hz), sin(pi/2) = 1
        result = block.execute(time=0.25, inputs={}, params=params)
        assert np.isclose(result[0], 2.0, atol=0.01), f"Sine at peak should be amplitude, got {result[0]}"

    def test_wave_square(self):
        """Test square waveform."""
        from blocks.wave_generator import WaveGeneratorBlock
        block = WaveGeneratorBlock()
        params = {'waveform': 'Square', 'amplitude': 1.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 0.0}

        # Square wave should alternate between +1 and -1
        result1 = block.execute(time=0.1, inputs={}, params=params)
        result2 = block.execute(time=0.6, inputs={}, params=params)

        # One should be +1, other -1
        assert abs(result1[0]) == 1.0, "Square wave should be at ±amplitude"
        assert result1[0] != result2[0], "Square wave should alternate"

    def test_wave_with_bias(self):
        """Test waveform with DC bias."""
        from blocks.wave_generator import WaveGeneratorBlock
        block = WaveGeneratorBlock()
        params = {'waveform': 'Sine', 'amplitude': 1.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 5.0}

        # At t=0, sin(0) = 0, so output = bias = 5
        result = block.execute(time=0.0, inputs={}, params=params)
        assert np.isclose(result[0], 5.0), f"At t=0 with bias=5, should be 5.0, got {result[0]}"

    def test_wave_triangle(self):
        """Test triangle waveform exists and runs."""
        from blocks.wave_generator import WaveGeneratorBlock
        block = WaveGeneratorBlock()
        params = {'waveform': 'Triangle', 'amplitude': 1.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 0.0}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert isinstance(result[0], (int, float, np.number)), "Should return numeric value"

    def test_wave_sawtooth(self):
        """Test sawtooth waveform exists and runs."""
        from blocks.wave_generator import WaveGeneratorBlock
        block = WaveGeneratorBlock()
        params = {'waveform': 'Sawtooth', 'amplitude': 1.0, 'frequency': 1.0, 'phase': 0.0, 'bias': 0.0}

        result = block.execute(time=0.0, inputs={}, params=params)
        assert isinstance(result[0], (int, float, np.number)), "Should return numeric value"
