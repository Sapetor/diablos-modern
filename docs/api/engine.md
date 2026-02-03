# Simulation Engine API

The simulation engine handles execution logic, ODE integration, and diagram analysis.

## SimulationEngine

Main engine class for simulation execution.

::: lib.engine.simulation_engine.SimulationEngine
    options:
      show_root_heading: true
      members:
        - __init__
        - initialize_execution
        - execute_block
        - run_compiled_simulation
        - check_diagram_integrity
        - detect_algebraic_loops

## SystemCompiler

Compiles block diagrams into efficient ODE systems.

::: lib.engine.system_compiler.SystemCompiler
    options:
      show_root_heading: true

## Flattener

Flattens hierarchical subsystems for simulation.

::: lib.engine.flattener.Flattener
    options:
      show_root_heading: true
