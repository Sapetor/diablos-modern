"""
Automated Block Testing Framework for DiaBloS

Uses DiagramBuilder to create test diagrams programmatically,
run simulations, and validate expected behavior.

Usage:
    python -m pytest tests/test_blocks.py -v

NOTE: These tests are currently skipped because DSim requires GUI initialization
that is not available in headless test mode. The DSim class needs refactoring
to support headless operation.

For block testing, see:
- tests/unit/blocks/ - Unit tests for individual blocks
- tests/integration/ - Integration tests for diagram loading
"""

import sys
import os
import numpy as np
import pytest

# Skip all tests in this module - DSim requires GUI components
pytestmark = pytest.mark.skip(
    reason="DSim requires GUI initialization (buttons_list, etc). "
    "Use tests/unit/blocks/ for block testing."
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.diagram_builder import DiagramBuilder


class SimulationRunner:
    """Runs simulations on diagrams and extracts results."""

    def __init__(self, diagram_path: str):
        """Load a diagram and prepare for simulation."""
        self.diagram_path = diagram_path
        self.dsim = None
        self.results = {}

    def run(self, sim_time: float = 1.0, sim_dt: float = 0.01) -> dict:
        """
        Run simulation and return scope data.

        Returns:
            Dictionary with scope names as keys and numpy arrays as values.
        """
        from lib.lib import DSim

        # Create DSim instance
        self.dsim = DSim()

        # Load diagram using FileService (modern pattern)
        data = self.dsim.file_service.load(filepath=self.diagram_path)
        if data is None:
            return {}

        # Apply loaded data using FileService
        sim_params = self.dsim.file_service.apply_loaded_data(data)

        # Sync simulation parameters to DSim
        self.dsim.sim_time = sim_params.get('sim_time', sim_time)
        self.dsim.sim_dt = sim_params.get('sim_dt', sim_dt)
        self.dsim.ss_count = 0

        # Override with test parameters
        self.dsim.sim_time = sim_time
        self.dsim.sim_dt = sim_dt

        # Run simulation
        try:
            result = self.dsim.execution_init()
            if result == -1:
                return {}

            while self.dsim.execution_initialized:
                self.dsim.execution_loop()
        except Exception as e:
            print(f"Simulation error: {e}")
            return {}

        # Extract scope data
        for block in self.dsim.blocks_list:
            if block.block_fn == 'Scope':
                params = getattr(block, 'exec_params', block.params)
                if 'vector' in params:
                    self.results[block.username or block.name] = {
                        'time': self.dsim.timeline,
                        'data': np.array(params['vector'])
                    }

        return self.results


@pytest.fixture
def temp_save_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


class TestBlockIntegration:
    """Integration tests for DiaBloS blocks using full simulation."""

    def test_step_response(self, qapp, temp_save_dir):
        """Test that Step block produces correct output."""
        builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)

        step = builder.add_block("Step", x=50, y=100, name="step",
                                 params={"delay": 0.5, "value": 1.0})
        scope = builder.add_block("Scope", x=200, y=100, name="scope")
        builder.connect(step, 0, scope, 0)

        test_path = str(temp_save_dir / "_test_step.dat")
        builder.save(test_path)

        runner = SimulationRunner(test_path)
        results = runner.run(sim_time=1.0, sim_dt=0.01)

        assert 'scope' in results, "No scope data found"
        data = results['scope']['data']
        time = results['scope']['time']

        # Before step (t < 0.5): should be 0
        before_step = data[time < 0.5]
        # After step (t >= 0.5): should be 1
        after_step = data[time >= 0.5]

        assert np.allclose(before_step, 0.0, atol=0.01), f"Before step should be 0, got {before_step[:5]}"
        assert np.allclose(after_step, 1.0, atol=0.01), f"After step should be 1, got {after_step[:5]}"

    def test_gain(self, qapp, temp_save_dir):
        """Test that Gain block multiplies correctly."""
        builder = DiagramBuilder(sim_time=0.5, sim_dt=0.01)

        step = builder.add_block("Step", x=50, y=100, name="step",
                                 params={"delay": 0.0, "value": 2.0})
        gain = builder.add_block("Gain", x=150, y=100, name="gain",
                                 params={"gain": 3.0})
        scope = builder.add_block("Scope", x=250, y=100, name="scope")

        builder.connect(step, 0, gain, 0)
        builder.connect(gain, 0, scope, 0)

        test_path = str(temp_save_dir / "_test_gain.dat")
        builder.save(test_path)

        runner = SimulationRunner(test_path)
        results = runner.run(sim_time=0.5, sim_dt=0.01)

        assert 'scope' in results, "No scope data found"
        data = results['scope']['data']
        # After step: should be 2.0 * 3.0 = 6.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 6.0, atol=0.1), \
            f"Gain output should be 6.0, got {final_value}"

    def test_integrator(self, qapp, temp_save_dir):
        """Test that Integrator block integrates correctly."""
        builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)

        # Constant input of 1.0 should integrate to t
        step = builder.add_block("Step", x=50, y=100, name="step",
                                 params={"delay": 0.0, "value": 1.0})
        integ = builder.add_block("Integrator", x=150, y=100, name="integ",
                                  params={"init_conds": 0.0, "method": "FWD_EULER"})
        scope = builder.add_block("Scope", x=250, y=100, name="scope")

        builder.connect(step, 0, integ, 0)
        builder.connect(integ, 0, scope, 0)

        test_path = str(temp_save_dir / "_test_integrator.dat")
        builder.save(test_path)

        runner = SimulationRunner(test_path)
        results = runner.run(sim_time=1.0, sim_dt=0.01)

        assert 'scope' in results, "No scope data found"
        data = results['scope']['data']
        time = results['scope']['time']

        # Integrating 1.0 from t=0 to t=1 should give approximately t
        final_value = data[-1] if len(data) > 0 else None
        final_time = time[-1] if len(time) > 0 else None

        # Allow 10% error due to numerical integration
        assert final_value is not None and np.isclose(final_value, final_time, rtol=0.1), \
            f"Integrator output at t={final_time} should be ~{final_time}, got {final_value}"

    def test_sum(self, qapp, temp_save_dir):
        """Test that Sum block adds/subtracts correctly."""
        builder = DiagramBuilder(sim_time=0.5, sim_dt=0.01)

        step1 = builder.add_block("Step", x=50, y=50, name="step1",
                                  params={"delay": 0.0, "value": 5.0})
        step2 = builder.add_block("Step", x=50, y=150, name="step2",
                                  params={"delay": 0.0, "value": 3.0})
        sumblock = builder.add_block("Sum", x=150, y=100, name="sum",
                                     params={"sign": "+-"}, in_ports=2)
        scope = builder.add_block("Scope", x=250, y=100, name="scope")

        builder.connect(step1, 0, sumblock, 0)
        builder.connect(step2, 0, sumblock, 1)
        builder.connect(sumblock, 0, scope, 0)

        test_path = str(temp_save_dir / "_test_sum.dat")
        builder.save(test_path)

        runner = SimulationRunner(test_path)
        results = runner.run(sim_time=0.5, sim_dt=0.01)

        assert 'scope' in results, "No scope data found"
        data = results['scope']['data']
        # 5.0 - 3.0 = 2.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 2.0, atol=0.1), \
            f"Sum output should be 2.0, got {final_value}"

    def test_feedback_loop(self, qapp, temp_save_dir):
        """Test a simple feedback system (integrator with negative feedback)."""
        builder = DiagramBuilder(sim_time=5.0, sim_dt=0.01)

        # Step input → Sum → Gain → Integrator → output
        #                ↑________________________↓ (negative feedback)

        step = builder.add_block("Step", x=50, y=100, name="step",
                                 params={"delay": 0.0, "value": 1.0})
        sumblock = builder.add_block("Sum", x=150, y=100, name="sum",
                                     params={"sign": "+-"}, in_ports=2)
        gain = builder.add_block("Gain", x=230, y=100, name="gain",
                                 params={"gain": 2.0})
        integ = builder.add_block("Integrator", x=310, y=100, name="integ",
                                  params={"init_conds": 0.0, "method": "FWD_EULER"})
        scope = builder.add_block("Scope", x=400, y=100, name="scope")

        builder.connect(step, 0, sumblock, 0)
        builder.connect(sumblock, 0, gain, 0)
        builder.connect(gain, 0, integ, 0)
        builder.connect(integ, 0, scope, 0)
        builder.connect(integ, 0, sumblock, 1)  # feedback

        test_path = str(temp_save_dir / "_test_feedback.dat")
        builder.save(test_path)

        runner = SimulationRunner(test_path)
        results = runner.run(sim_time=5.0, sim_dt=0.01)

        assert 'scope' in results, "No scope data found"
        data = results['scope']['data']
        # First-order system with step input should approach 1.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 1.0, atol=0.1), \
            f"Feedback system should settle to 1.0, got {final_value}"
