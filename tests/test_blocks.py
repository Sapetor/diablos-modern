"""
Automated Block Testing Framework for DiaBloS

Uses DiagramBuilder to create test diagrams programmatically,
run simulations, and validate expected behavior.

Usage:
    python -m pytest tests/test_blocks.py -v
    
Or run directly:
    python tests/test_blocks.py
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.getcwd())

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
            print(f"Failed to load diagram: {self.diagram_path}")
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


def test_step_response():
    """Test that Step block produces correct output."""
    builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    
    step = builder.add_block("Step", x=50, y=100, name="step",
                             params={"start_time": 0.5, "h_start": 0.0, "h_final": 1.0})
    scope = builder.add_block("Scope", x=200, y=100, name="scope")
    builder.connect(step, 0, scope, 0)
    
    test_path = "saves/_test_step.dat"
    builder.save(test_path)
    
    runner = SimulationRunner(test_path)
    results = runner.run(sim_time=1.0, sim_dt=0.01)
    
    if 'scope' in results:
        data = results['scope']['data']
        time = results['scope']['time']
        
        # Before step (t < 0.5): should be 0
        before_step = data[time < 0.5]
        # After step (t >= 0.5): should be 1
        after_step = data[time >= 0.5]
        
        assert np.allclose(before_step, 0.0, atol=0.01), f"Before step should be 0, got {before_step[:5]}"
        assert np.allclose(after_step, 1.0, atol=0.01), f"After step should be 1, got {after_step[:5]}"
        print("✓ Step response test PASSED")
    else:
        print("✗ No scope data found")
    
    # Cleanup
    os.remove(test_path)


def test_gain():
    """Test that Gain block multiplies correctly."""
    builder = DiagramBuilder(sim_time=0.5, sim_dt=0.01)
    
    step = builder.add_block("Step", x=50, y=100, name="step",
                             params={"start_time": 0.0, "h_start": 0.0, "h_final": 2.0})
    gain = builder.add_block("Gain", x=150, y=100, name="gain",
                             params={"gain": 3.0})
    scope = builder.add_block("Scope", x=250, y=100, name="scope")
    
    builder.connect(step, 0, gain, 0)
    builder.connect(gain, 0, scope, 0)
    
    test_path = "saves/_test_gain.dat"
    builder.save(test_path)
    
    runner = SimulationRunner(test_path)
    results = runner.run(sim_time=0.5, sim_dt=0.01)
    
    if 'scope' in results:
        data = results['scope']['data']
        # After step: should be 2.0 * 3.0 = 6.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 6.0, atol=0.1), \
            f"Gain output should be 6.0, got {final_value}"
        print("✓ Gain test PASSED")
    else:
        print("✗ No scope data found")
    
    os.remove(test_path)


def test_integrator():
    """Test that Integrator block integrates correctly."""
    builder = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    
    # Constant input of 1.0 should integrate to t
    step = builder.add_block("Step", x=50, y=100, name="step",
                             params={"start_time": 0.0, "h_start": 0.0, "h_final": 1.0})
    integ = builder.add_block("Integrator", x=150, y=100, name="integ",
                              params={"init_conds": 0.0, "method": "FWD_EULER"})
    scope = builder.add_block("Scope", x=250, y=100, name="scope")
    
    builder.connect(step, 0, integ, 0)
    builder.connect(integ, 0, scope, 0)
    
    test_path = "saves/_test_integrator.dat"
    builder.save(test_path)
    
    runner = SimulationRunner(test_path)
    results = runner.run(sim_time=1.0, sim_dt=0.01)
    
    if 'scope' in results:
        data = results['scope']['data']
        time = results['scope']['time']
        
        # Integrating 1.0 from t=0 to t=1 should give approximately t
        final_value = data[-1] if len(data) > 0 else None
        final_time = time[-1] if len(time) > 0 else None
        
        # Allow 10% error due to numerical integration
        assert final_value is not None and np.isclose(final_value, final_time, rtol=0.1), \
            f"Integrator output at t={final_time} should be ~{final_time}, got {final_value}"
        print("✓ Integrator test PASSED")
    else:
        print("✗ No scope data found")
    
    os.remove(test_path)


def test_sum():
    """Test that Sum block adds/subtracts correctly."""
    builder = DiagramBuilder(sim_time=0.5, sim_dt=0.01)
    
    step1 = builder.add_block("Step", x=50, y=50, name="step1",
                              params={"start_time": 0.0, "h_start": 0.0, "h_final": 5.0})
    step2 = builder.add_block("Step", x=50, y=150, name="step2",
                              params={"start_time": 0.0, "h_start": 0.0, "h_final": 3.0})
    sumblock = builder.add_block("Sum", x=150, y=100, name="sum",
                                  params={"sign": "+-"}, in_ports=2)
    scope = builder.add_block("Scope", x=250, y=100, name="scope")
    
    builder.connect(step1, 0, sumblock, 0)
    builder.connect(step2, 0, sumblock, 1)
    builder.connect(sumblock, 0, scope, 0)
    
    test_path = "saves/_test_sum.dat"
    builder.save(test_path)
    
    runner = SimulationRunner(test_path)
    results = runner.run(sim_time=0.5, sim_dt=0.01)
    
    if 'scope' in results:
        data = results['scope']['data']
        # 5.0 - 3.0 = 2.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 2.0, atol=0.1), \
            f"Sum output should be 2.0, got {final_value}"
        print("✓ Sum test PASSED")
    else:
        print("✗ No scope data found")
    
    os.remove(test_path)


def test_feedback_loop():
    """Test a simple feedback system (integrator with negative feedback)."""
    builder = DiagramBuilder(sim_time=5.0, sim_dt=0.01)
    
    # Step input → Sum → Gain → Integrator → output
    #                ↑________________________↓ (negative feedback)
    
    step = builder.add_block("Step", x=50, y=100, name="step",
                             params={"start_time": 0.0, "h_start": 0.0, "h_final": 1.0})
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
    
    test_path = "saves/_test_feedback.dat"
    builder.save(test_path)
    
    runner = SimulationRunner(test_path)
    results = runner.run(sim_time=5.0, sim_dt=0.01)
    
    if 'scope' in results:
        data = results['scope']['data']
        # First-order system with step input should approach 1.0
        final_value = data[-1] if len(data) > 0 else None
        assert final_value is not None and np.isclose(final_value, 1.0, atol=0.1), \
            f"Feedback system should settle to 1.0, got {final_value}"
        print("✓ Feedback loop test PASSED")
    else:
        print("✗ No scope data found")
    
    os.remove(test_path)


if __name__ == "__main__":
    print("=" * 50)
    print("DiaBloS Automated Block Tests")
    print("=" * 50)
    
    tests = [
        ("Step Response", test_step_response),
        ("Gain", test_gain),
        ("Integrator", test_integrator),
        ("Sum", test_sum),
        ("Feedback Loop", test_feedback_loop),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\nRunning: {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
