"""
Example of how to use the incremental improvements with existing DiaBloS code.

This demonstrates practical ways to enhance the existing codebase without
breaking changes, by adding helper functions and better practices.
"""

import json
import os
import sys

# Run from anywhere: put the repo root (this file's parent) on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.lib import DSim
from lib.improvements import (
    ValidationHelper,
    PerformanceHelper,
    SafetyChecks,
    LoggingHelper,
    validate_simulation_parameters,
)


def load_default_config():
    """Read config/default_config.json -- the app's only configuration file.

    This used to go through a ``lib.config_manager.ConfigManager`` whose
    hard-coded defaults had drifted away from the JSON it was supposed to be
    loading; the module was dead except for this example and has been deleted.
    Reading the JSON directly is what the application itself does (see
    ``diablos_modern.setup_application``).
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "default_config.json",
    )
    with open(path, "r") as fh:
        return json.load(fh)


def demonstrate_improved_simulation():
    """Demonstrate how to run a simulation with improvements."""

    # Setup enhanced logging
    LoggingHelper.setup_logging(level="INFO")

    # Load configuration
    config = load_default_config()
    simulation_cfg = config.get("simulation", {})
    display_cfg = config.get("display", {})
    print("\\n=== Configuration Demo ===")
    print(f"Default simulation time: {simulation_cfg.get('default_time')}")
    print(f"Default timestep: {simulation_cfg.get('default_timestep')}")
    print(f"Window size: {display_cfg.get('window_width')}x{display_cfg.get('window_height')}")

    # Create DSim instance
    print("\\n=== Creating DSim Instance ===")
    dsim = DSim()
    print(f"Screen size: {dsim.SCREEN_WIDTH}x{dsim.SCREEN_HEIGHT}")

    # Create performance helper
    perf_helper = PerformanceHelper()

    # Load a simple example (you would typically load from file)
    print("\\n=== Setting up Simple Test ===")
    # This would normally be done through the GUI or file loading
    # For demo purposes, we'll just set some basic properties
    dsim.sim_time = simulation_cfg.get("default_time", 10.0)
    dsim.sim_dt = simulation_cfg.get("default_timestep", 0.01)

    # Validate simulation parameters
    print("\\n=== Parameter Validation ===")
    is_valid, errors = validate_simulation_parameters(dsim.sim_time, dsim.sim_dt)
    if is_valid:
        print("✓ Simulation parameters are valid")
    else:
        print("✗ Simulation parameter errors:")
        for error in errors:
            print(f"  - {error}")
        return

    # Check DSim state safety
    print("\\n=== Safety Checks ===")
    is_safe, safety_errors = SafetyChecks.check_simulation_state(dsim)
    if is_safe:
        print("✓ DSim state is safe")
    else:
        print("✗ DSim safety issues:")
        for error in safety_errors:
            print(f"  - {error}")

    # If we had blocks and connections, we could validate them
    if hasattr(dsim, "blocks_list") and hasattr(dsim, "line_list"):
        print("\\n=== Block Connection Validation ===")
        if dsim.blocks_list:  # Only if we have blocks
            perf_helper.start_timer("validation")

            is_valid, conn_errors = ValidationHelper.validate_block_connections(
                dsim.blocks_list, dsim.line_list
            )

            no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(
                dsim.blocks_list, dsim.line_list
            )

            validation_time = perf_helper.end_timer("validation")

            if is_valid and no_loops:
                print(f"✓ All validations passed ({validation_time:.4f}s)")
            else:
                print("✗ Validation issues found:")
                for error in conn_errors + loop_errors:
                    print(f"  - {error}")
        else:
            print("No blocks to validate (empty diagram)")

    # Demonstrate performance monitoring
    print("\\n=== Performance Monitoring Demo ===")
    perf_helper.start_timer("demo_operation")

    # Simulate some work
    import time

    time.sleep(0.1)

    duration = perf_helper.end_timer("demo_operation")
    print(f"Demo operation took {duration:.4f} seconds")

    # Log performance stats
    perf_helper.log_stats()


def demonstrate_validation_helpers():
    """Demonstrate validation helper functions."""

    print("\\n" + "=" * 50)
    print("VALIDATION HELPERS DEMONSTRATION")
    print("=" * 50)

    # Create mock blocks and connections for testing
    class MockBlock:
        def __init__(self, sid, name, in_ports, out_ports, b_type=2):
            self.sid = sid
            self.name = name
            self.in_ports = in_ports
            self.out_ports = out_ports
            self.b_type = b_type

    class MockLine:
        def __init__(self, block_src_id, srcport, block_dst_id, dstport):
            self.block_src_id = block_src_id
            self.srcport = srcport
            self.block_dst_id = block_dst_id
            self.dstport = dstport

    # Create test blocks
    blocks = [
        MockBlock(1, "Step", 0, 1, 0),  # Source block
        MockBlock(2, "Gain", 1, 1, 2),  # Function block
        MockBlock(3, "Integrator", 1, 1, 1),  # Memory block
        MockBlock(4, "Scope", 1, 0, 3),  # Sink block
    ]

    # Create connections: Step -> Gain -> Integrator -> Scope
    connections = [
        MockLine(1, 0, 2, 0),  # Step to Gain
        MockLine(2, 0, 3, 0),  # Gain to Integrator
        MockLine(3, 0, 4, 0),  # Integrator to Scope
    ]

    print("\\nTest setup:")
    print("- Step (source) -> Gain -> Integrator (memory) -> Scope (sink)")

    # Test connection validation
    print("\\n=== Connection Validation Test ===")
    is_valid, messages = ValidationHelper.validate_block_connections(blocks, connections)
    if is_valid:
        print("✓ Connections are valid")
    else:
        print("✗ Connection issues:")
        for msg in messages:
            print(f"  - {msg}")

    # Test algebraic loop detection
    print("\\n=== Algebraic Loop Detection Test ===")
    no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(blocks, connections)
    if no_loops:
        print("✓ No algebraic loops detected")
    else:
        print("✗ Algebraic loops found:")
        for error in loop_errors:
            print(f"  - {error}")

    # Test with algebraic loop
    print("\\n=== Testing with Algebraic Loop ===")
    loop_connections = connections + [
        MockLine(4, 0, 2, 1)
    ]  # Add feedback (invalid port but for demo)

    # Create a proper loop for testing
    loop_blocks = [
        MockBlock(1, "Gain1", 1, 1, 2),
        MockBlock(2, "Gain2", 1, 1, 2),
        MockBlock(3, "Gain3", 1, 1, 2),
    ]

    loop_conns = [
        MockLine(1, 0, 2, 0),  # Gain1 -> Gain2
        MockLine(2, 0, 3, 0),  # Gain2 -> Gain3
        MockLine(3, 0, 1, 0),  # Gain3 -> Gain1 (creates loop)
    ]

    no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(loop_blocks, loop_conns)
    if no_loops:
        print("✓ No algebraic loops detected")
    else:
        print("✗ Algebraic loops found (as expected):")
        for error in loop_errors:
            print(f"  - {error}")


if __name__ == "__main__":
    print("DiaBloS Incremental Improvements Demonstration")
    print("=" * 60)

    try:
        demonstrate_improved_simulation()
        demonstrate_validation_helpers()

        print("\\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\\nKey benefits of these improvements:")
        print("✓ Better error handling and validation")
        print("✓ Performance monitoring and optimization")
        print("✓ Enhanced logging for debugging")
        print("✓ Safety checks to prevent crashes")
        print("✓ All improvements work with existing DSim code")
        print("✓ Zero breaking changes to existing functionality")

    except Exception as e:
        print(f"\\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
