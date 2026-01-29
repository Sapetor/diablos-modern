# Subsystems Architecture & Flattening

## Overview

Modern DiaBloS simulates hierarchical models (models containing Subsystems) by **flattening** the hierarchy at runtime. This means that while the user sees a nested structure (Parent -> Subsystem -> Child), the simulation engine sees a single flat list of blocks (Parent -> Subsystem/Child).

## Flattening Process

The flattening logic resides in `lib/engine/flattener.py`.

1.  **Recursion**: The `Flattener` recursively traverses the block list.
2.  **Naming**: It renames blocks using a path-like convention: `SubsystemName/BlockName`.
3.  **Filtering**: It separates "Primitves" (functional blocks like Gain, TransferFcn) from "Containers" (Subsystem blocks, Inports, Outports).
    *   **Crucial**: The `active_blocks_list` used by the engine contains ONLY the Primitives. The Subsystem block itself is excluded because it has no mathematical function.
4.  **Connection Resolution**: It traces connections through layers.
    *   `Source -> Subsystem/Inport -> InternalBlock` becomes `Source -> InternalBlock` in the flat graph.
    *   This logic handles multiple levels of nesting.

## Execution Model

The `SimulationEngine` uses two lists:
1.  `model.blocks_list`: The original hierarchical model (used for UI, saving, etc).
2.  `active_blocks_list`: The flattened execution graph (used for physics).

### Initialization Phase (`initialize_execution`)

This phase iterates over `active_blocks_list` to:
*   Execute Source blocks (b_type=0) like Step, Sine.
*   Initialize Memory blocks (Integrators).

**Critical Logic**: We must iterate `active_blocks_list`. Iterating `model.blocks_list` is incorrect because it would try to execute the Subsystem container (which does nothing) and skip the internal Source blocks, causing the simulation to stall or produce no data.

### Propagation Phase (`propagate_outputs`)

When a block computes its output, it must push data to its neighbors.
*   The engine looks up neighbors using the flat connection graph.
*   It must update the `input_queue` of the **destination block instance in `active_blocks_list`**.
*   Updating the original `model.blocks_list` instance has no effect on the running simulation.

### Plotting (`ScopePlotter`)

The Plotter must read data from the running simulation.
*   It accesses `engine.active_blocks_list`.
*   It finds the Scope block instance (e.g., named `subsystem1/scope0` or just `scope0`).
*   It reads the `vector` param from that active instance.

## Debugging Tips

If a subsystem simulation appears to run but produces no data (empty plots):
1.  **Check Initialization**: Are the internal source blocks executing? Enable `LOOP` debug logs in `simulation_engine.py`.
2.  **Check Propagation**: Is data crossing the boundary? Check `ENGINE PROPAGATE` logs.
3.  **Check Plotter Target**: Is the plotter reading the `active_blocks_list`?

## Future Improvements

*   **Compiled Flattening**: Currently, flattening is done for the Interpreter mode. The `SystemCompiler` (Fast Solver) needs to implement similar flattening logic to support Subsystems.
*   **Vectorized Ports**: Support for vector signals passing through subsystem ports.
