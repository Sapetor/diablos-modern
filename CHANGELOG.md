# Changelog

All notable changes to DiaBloS will be documented in this file.

## [Unreleased] - 2026-01-26

### New Features
- **MIMO Subsystem Support**: Subsystems now automatically synchronize their external ports based on internal `Inport` and `Outport` blocks. This allows for subsystems with arbitrarily many inputs and outputs.
- **Fast Solver Subsystem Support**: The Fast Solver (Compiled Mode) now recursively compiles and flattens Subsystems, allowing complex hierarchical models to run with compiled performance (10-100x speedup).
- **Subsystem Port Sync Fix**: Fixed a bug where adding input/output ports inside a subsystem would not correctly update the simulation parameters on the outside block, leading to simulation failures.
- **Subsystem Copy Fix**: Fixed a crash when simulating copied subsystems by correctly restoring the internal structure (`sub_blocks`, `sub_lines`) and maintaining the `Subsystem` class identity during paste.
- **Copy-Paste Connections Fix**: Fixed an issue where connections were lost after pasting by ensuring lines are registered before trajectory calculation.
- **Serialization Fix**: Resolved `QPainterPath` pickling errors during copy operations by implementing custom deepcopy logic for connections.




### Major Refactoring
This release includes significant architectural improvements to reduce code complexity.

#### DSim Reduction: 2,200 → 1,584 lines (28% reduction)

**State Unification (Option 1)**
- Added 8 properties for DSim/SimulationEngine state sharing:
  - `timeline`, `time_step`, `global_computed_list`
  - `execution_initialized`, `execution_stop`, `error_msg`
  - `execution_time_start`, `memory_blocks`
- DSim and engine now share the same state via live properties

**Legacy Code Removal (Option 2)**
- Removed all 4 `execution_function` fallback patterns from DSim
- All blocks now use `block.block_instance.execute()` directly
- Removed unused `functions` import and `execution_function` assignment from DSim
- Note: `functions.py` is still used by some block classes (integrator, statespace, etc.)

**SignalPlot Extraction (Option 3)**
- Extracted SignalPlot class (~320 lines) to `lib/plotting/signal_plot.py`
- Created `lib/plotting/__init__.py`

### Previous Changes
- **Polymorphic rendering**: All 31 blocks have `draw_icon()` methods
- **File I/O delegation**: `save()` and `open()` delegate to FileService
- **Safe delegations**: `check_diagram_integrity`, `get_neighbors`, `get_outputs`

### Bug Fixes
- Fixed algebraic loop detection for memory blocks
- Fixed integrator SOLVE_IVP `y0 must be 1-dimensional` error
- Fixed KeyError in scope/export with `.get('_init_start_', True)`

### Technical Improvements
- All 42 blocks have `execute()` methods in dedicated class files
- Test suite: 54 tests passing
