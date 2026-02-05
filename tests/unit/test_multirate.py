"""
Unit tests for multi-rate simulation support.

Tests the core multi-rate infrastructure:
- DBlock sample time properties and methods
- Sample time propagation in SimulationEngine
- RateTransition and FirstOrderHold blocks
- Rate mismatch validation
"""

import pytest
import numpy as np
from unittest.mock import MagicMock


class TestDBlockSampleTime:
    """Tests for DBlock sample time properties and methods."""

    def test_default_sample_time_is_continuous(self, qapp, sample_block):
        """Default effective_sample_time should be -1.0 (continuous)."""
        assert sample_block.effective_sample_time == -1.0

    def test_should_execute_always_true_for_continuous(self, qapp, sample_block):
        """Continuous blocks should always execute."""
        sample_block.effective_sample_time = -1.0
        assert sample_block.should_execute(0.0) is True
        assert sample_block.should_execute(0.5) is True
        assert sample_block.should_execute(1.0) is True

    def test_should_execute_at_sample_times(self, qapp, sample_block):
        """Discrete blocks should execute at sample times."""
        sample_block.effective_sample_time = 0.1  # 10Hz
        sample_block._next_execution_time = 0.0

        # Should execute at t=0
        assert sample_block.should_execute(0.0) is True

        # After scheduling, should not execute before next time
        sample_block.schedule_next_execution(0.0)
        assert sample_block._next_execution_time == 0.1

        # Should not execute between samples
        assert sample_block.should_execute(0.05) is False
        assert sample_block.should_execute(0.09) is False

        # Should execute at next sample time
        assert sample_block.should_execute(0.1) is True

    def test_schedule_next_execution(self, qapp, sample_block):
        """Test scheduling of next execution time."""
        sample_block.effective_sample_time = 0.1

        sample_block._next_execution_time = 0.0
        sample_block.schedule_next_execution(0.0)
        assert sample_block._next_execution_time == 0.1

        sample_block.schedule_next_execution(0.1)
        assert sample_block._next_execution_time == 0.2

    def test_held_outputs(self, qapp, sample_block):
        """Test held output storage and retrieval."""
        sample_block.set_held_output(0, 5.0)
        assert sample_block.get_held_output(0) == 5.0

        sample_block.set_held_output(1, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(
            sample_block.get_held_output(1), np.array([1.0, 2.0])
        )

        # Non-existent port returns 0.0
        assert sample_block.get_held_output(99) == 0.0

    def test_reset_sample_time_state(self, qapp, sample_block):
        """Test resetting sample time state."""
        sample_block._next_execution_time = 0.5
        sample_block._last_execution_time = 0.4
        sample_block._held_outputs = {0: 5.0}

        sample_block.reset_sample_time_state()

        assert sample_block._next_execution_time == 0.0
        assert sample_block._last_execution_time == -1.0
        assert sample_block._held_outputs == {}

    def test_resolve_sample_time_from_params(self, qapp, sample_block):
        """Test resolving sample time from block params."""
        # Default (no param)
        assert sample_block.resolve_sample_time() == -1.0

        # Explicit sampling_time
        sample_block.params["sampling_time"] = 0.05
        assert sample_block.resolve_sample_time() == 0.05

        # sample_time alias
        del sample_block.params["sampling_time"]
        sample_block.params["sample_time"] = 0.02
        assert sample_block.resolve_sample_time() == 0.02

        # String value
        sample_block.params["sample_time"] = "0.1"
        assert sample_block.resolve_sample_time() == 0.1


class TestRateTransitionBlock:
    """Tests for RateTransition block."""

    @pytest.fixture
    def rate_transition(self):
        """Create a RateTransition block instance."""
        from blocks.rate_transition import RateTransitionBlock

        return RateTransitionBlock()

    def test_block_properties(self, rate_transition):
        """Test basic block properties."""
        assert rate_transition.block_name == "RateTransition"
        assert rate_transition.category == "Control"
        assert "output_sample_time" in rate_transition.params
        assert "transition_mode" in rate_transition.params

    def test_zoh_mode(self, rate_transition):
        """Test Zero-Order Hold mode."""
        params = {
            "output_sample_time": 0.1,
            "transition_mode": "ZOH",
            "_init_start_": True,
        }

        # First execution - initializes
        result = rate_transition.execute(time=0.0, inputs={0: 5.0}, params=params)
        assert result["E"] is False
        assert result[0] == 5.0

        # Second execution - holds value
        result = rate_transition.execute(time=0.05, inputs={0: 10.0}, params=params)
        assert result[0] == 5.0  # Held value

        # At next sample time - updates
        result = rate_transition.execute(time=0.1, inputs={0: 10.0}, params=params)
        assert result[0] == 10.0

    def test_continuous_passthrough(self, rate_transition):
        """Test continuous mode (output_sample_time <= 0)."""
        params = {
            "output_sample_time": -1.0,
            "transition_mode": "ZOH",
            "_init_start_": True,
        }

        result = rate_transition.execute(time=0.0, inputs={0: 5.0}, params=params)
        assert result[0] == 5.0

        result = rate_transition.execute(time=0.05, inputs={0: 10.0}, params=params)
        assert result[0] == 10.0  # Passes through immediately

    def test_linear_mode_ramping(self, rate_transition):
        """Test Linear mode produces smooth ramps between input changes."""
        params = {
            "output_sample_time": 0.01,
            "transition_mode": "Linear",
            "sampling_time": -1.0,  # Continuous execution
            "_init_start_": True,
        }

        # Initialize at t=0 with input=0
        result = rate_transition.execute(time=0.0, inputs={0: 0.0}, params=params)
        assert result[0] == 0.0

        # At t=0.05, input still 0, output should still be 0
        result = rate_transition.execute(time=0.05, inputs={0: 0.0}, params=params)
        assert result[0] == 0.0

        # At t=0.1, input changes to 1.0 - start of ramp
        result = rate_transition.execute(time=0.1, inputs={0: 1.0}, params=params)
        assert abs(result[0] - 0.0) < 0.01  # Just started ramping, still near 0

        # At t=0.15, halfway through ramp (0.1s ramp duration)
        result = rate_transition.execute(time=0.15, inputs={0: 1.0}, params=params)
        assert 0.4 < result[0] < 0.6  # Should be around 0.5 (halfway)

        # At t=0.2, ramp complete, should be at 1.0
        result = rate_transition.execute(time=0.2, inputs={0: 1.0}, params=params)
        assert abs(result[0] - 1.0) < 0.01  # Ramp complete

        # Now input changes to 0.5 at t=0.2
        result = rate_transition.execute(time=0.2, inputs={0: 0.5}, params=params)
        # Output should be near 1.0 (start of new ramp)

        # At t=0.25, should be ramping down towards 0.5
        result = rate_transition.execute(time=0.25, inputs={0: 0.5}, params=params)
        assert 0.5 < result[0] < 1.0  # Should be between 0.5 and 1.0


class TestFirstOrderHoldBlock:
    """Tests for FirstOrderHold block."""

    @pytest.fixture
    def foh(self):
        """Create a FirstOrderHold block instance."""
        from blocks.first_order_hold import FirstOrderHoldBlock

        return FirstOrderHoldBlock()

    def test_block_properties(self, foh):
        """Test basic block properties."""
        assert foh.block_name == "FirstOrderHold"
        assert foh.category == "Control"
        assert "sampling_time" in foh.params

    def test_initialization(self, foh):
        """Test FOH initializes correctly."""
        params = {"sampling_time": 0.1, "_init_start_": True}

        # First sample at t=0
        result = foh.execute(time=0.0, inputs={0: 5.0}, params=params)
        assert result["E"] is False
        assert result[0] == 5.0


class TestRateMismatchValidation:
    """Tests for rate mismatch validation in DiagramValidator."""

    def test_discrete_to_continuous_info(self):
        """Test that discrete→continuous generates INFO level message."""
        from lib.diagram_validator import DiagramValidator, ErrorSeverity

        # Create mock DSim with a discrete source and continuous destination
        mock_dsim = MagicMock()

        # Discrete source block (ZOH at 0.1s)
        src_block = MagicMock()
        src_block.name = "zoh1"
        src_block.username = "ZOH"
        src_block.block_fn = "ZeroOrderHold"
        src_block.params = {"sampling_time": 0.1}
        src_block.in_ports = 1
        src_block.out_ports = 1
        src_block.block_instance = None
        src_block.category = "Control"

        # Continuous destination block
        dst_block = MagicMock()
        dst_block.name = "gain1"
        dst_block.username = "Gain"
        dst_block.block_fn = "Gain"
        dst_block.params = {}  # No sampling_time = continuous
        dst_block.in_ports = 1
        dst_block.out_ports = 1
        dst_block.block_instance = None
        dst_block.category = "Math"

        # Connection
        line = MagicMock()
        line.srcblock = "zoh1"
        line.dstblock = "gain1"
        line.hidden = False

        mock_dsim.blocks_list = [src_block, dst_block]
        mock_dsim.line_list = [line]

        validator = DiagramValidator(mock_dsim)
        errors = validator.validate()

        # Should have INFO about discrete→continuous
        info_errors = [e for e in errors if e.severity == ErrorSeverity.INFO]
        assert any("Discrete signal" in str(e.message) for e in info_errors)

    def test_non_integer_rate_ratio_warning(self):
        """Test that non-integer rate ratios generate WARNING."""
        from lib.diagram_validator import DiagramValidator, ErrorSeverity

        mock_dsim = MagicMock()

        # Source at 0.1s (10Hz)
        src_block = MagicMock()
        src_block.name = "block1"
        src_block.username = "Block1"
        src_block.block_fn = "ZeroOrderHold"
        src_block.params = {"sampling_time": 0.1}
        src_block.in_ports = 1
        src_block.out_ports = 1
        src_block.block_instance = None
        src_block.category = "Control"

        # Destination at 0.03s (~33Hz) - non-integer ratio
        dst_block = MagicMock()
        dst_block.name = "block2"
        dst_block.username = "Block2"
        dst_block.block_fn = "Gain"
        dst_block.params = {"sampling_time": 0.03}
        dst_block.in_ports = 1
        dst_block.out_ports = 1
        dst_block.block_instance = None
        dst_block.category = "Math"

        line = MagicMock()
        line.srcblock = "block1"
        line.dstblock = "block2"
        line.hidden = False

        mock_dsim.blocks_list = [src_block, dst_block]
        mock_dsim.line_list = [line]

        validator = DiagramValidator(mock_dsim)
        errors = validator.validate()

        # Should have WARNING about non-integer ratio
        warnings = [e for e in errors if e.severity == ErrorSeverity.WARNING]
        assert any("Non-integer" in str(e.message) for e in warnings)


class TestSampleTimePropagation:
    """Tests for sample time propagation in SimulationEngine."""

    def test_explicit_sample_times_resolved(self, qapp, sample_block):
        """Test that explicit sample times are resolved from params."""
        sample_block.params["sampling_time"] = 0.05

        resolved = sample_block.resolve_sample_time()
        assert resolved == 0.05

    def test_continuous_default(self, qapp, sample_block):
        """Test that blocks without sample_time are continuous."""
        # Remove any sample_time params
        sample_block.params.pop("sampling_time", None)
        sample_block.params.pop("sample_time", None)

        resolved = sample_block.resolve_sample_time()
        assert resolved == -1.0


class TestBlockSamplingTimeParam:
    """Test that control blocks have sampling_time parameter."""

    def test_integrator_has_sampling_time(self):
        """Test Integrator block has sampling_time param."""
        from blocks.integrator import IntegratorBlock

        block = IntegratorBlock()
        assert "sampling_time" in block.params

    def test_pid_has_sampling_time(self):
        """Test PID block has sampling_time param."""
        from blocks.pid import PIDBlock

        block = PIDBlock()
        assert "sampling_time" in block.params

    def test_transfer_function_has_sampling_time(self):
        """Test TransferFunction block has sampling_time param."""
        from blocks.transfer_function import TransferFunctionBlock

        block = TransferFunctionBlock()
        assert "sampling_time" in block.params

    def test_statespace_has_sampling_time(self):
        """Test StateSpace block has sampling_time param."""
        from blocks.statespace import StateSpaceBlock

        block = StateSpaceBlock()
        assert "sampling_time" in block.params

    def test_derivative_has_sampling_time(self):
        """Test Derivative block has sampling_time param."""
        from blocks.derivative import DerivativeBlock

        block = DerivativeBlock()
        assert "sampling_time" in block.params

    def test_rate_transition_has_output_sample_time(self):
        """Test RateTransition block has output_sample_time param."""
        from blocks.rate_transition import RateTransitionBlock

        block = RateTransitionBlock()
        assert "output_sample_time" in block.params

    def test_first_order_hold_has_sampling_time(self):
        """Test FirstOrderHold block has input_sample_time param."""
        from blocks.first_order_hold import FirstOrderHoldBlock

        block = FirstOrderHoldBlock()
        assert "input_sample_time" in block.params
        # Also has sampling_time=-1 for continuous execution
        assert "sampling_time" in block.params
        assert block.params["sampling_time"]["default"] == -1.0
