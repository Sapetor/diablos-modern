# Core Library API

The core library provides the main simulation management functionality.

## DSim Class

The main simulation interface class.

::: lib.lib.DSim
    options:
      show_root_heading: true
      members:
        - __init__
        - save
        - load
        - run_simulation
        - enter_subsystem
        - exit_subsystem
        - get_current_path

## SimulationModel

Data model for the simulation diagram.

::: lib.models.simulation_model.SimulationModel
    options:
      show_root_heading: true

## FileService

File I/O operations for diagrams.

::: lib.services.file_service.FileService
    options:
      show_root_heading: true
      members:
        - save
        - load
        - serialize

## Type Definitions

Common type aliases used throughout the codebase.

::: lib.types
    options:
      show_root_heading: true
