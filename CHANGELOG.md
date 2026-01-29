# Changelog

All notable changes to DiaBloS will be documented in this file.

## [Unreleased] - 2026-01-29

### New UI/UX Features

#### Alignment Tools
Align and distribute multiple selected blocks for cleaner diagrams:
- **Align Left/Right/Center (Horizontal)**: Align blocks horizontally
- **Align Top/Bottom/Center (Vertical)**: Align blocks vertically
- **Distribute Horizontally/Vertically**: Space blocks evenly (requires 3+ blocks)

Access via:
- Right-click context menu → Align submenu (when 2+ blocks selected)
- Keyboard shortcuts: `Ctrl+Shift+L` (Left), `Ctrl+Shift+R` (Right), `Ctrl+Shift+H` (Center H), `Ctrl+Shift+T` (Top), `Ctrl+Shift+B` (Bottom)

#### Single-Step Simulation
Debug simulations one timestep at a time:
- Press **F8** to step through simulation
- Works from stopped state (initializes at t=0) or paused state
- Each step advances exactly one `dt` and pauses automatically
- Useful for debugging and understanding block behavior

#### Minimap Widget
Overview navigation for large diagrams:
- Toggle via **View → Minimap** or `Ctrl+Shift+M`
- Shows scaled overview of entire diagram
- Current viewport highlighted as rectangle
- Click on minimap to pan main canvas to that location
- Dockable on left or right side

### Bug Fixes
- **Block Resize Port Alignment**: Fixed a visual glitch when resizing blocks with multiple input/output ports. The `rect` property was not being updated after minimum height enforcement in `update_Block()`, causing inconsistencies between block dimensions and port positions.
- **Subsystem Resize Port Scaling**: Fixed an issue where subsystem ports would not scale when resizing. Port positions were stored as absolute pixel values at creation time and never recalculated. Ports now scale proportionally with block dimensions.
- **Subsystem Naming Collision**: Fixed a bug where creating multiple subsystems would give them all the same name ("subsystem1"). The uniqueness check was comparing capitalized names ("Subsystem1") against lowercase block names ("subsystem1").
- **Subsystem Loop Detection**: Improved algebraic loop detection to look inside subsystems. A subsystem containing a memory block (Integrator, etc.) now correctly breaks algebraic loops, allowing valid feedback connections.

### Improvements
- **Resize Limit Feedback**: The cursor now changes to a "forbidden" indicator when trying to resize a block below its minimum size, providing visual feedback about resize constraints.
- **Smoother Port Positioning**: Disabled port grid snapping by default for smoother resize behavior. Blocks can opt-in via `block_instance.use_port_grid_snap = True`.
- **Code Cleanup**: Removed unused `port_spacing` calculation in `DBlock.update_Block()`.
- **Test Coverage**: Added 11 unit tests for block resize behavior (`tests/unit/test_block_resize.py`).

### New Features
- **Fast Solver Block Expansion**: Added 5 new blocks to the Fast Solver (Compiled Mode):
  - `WaveGenerator`: Multi-waveform source (Sine, Square, Triangle, Sawtooth)
  - `Noise`: Gaussian random noise generator
  - `MathFunction`: Standard math functions (sin, cos, exp, log, sqrt, etc.)
  - `Selector`: Vector element extraction
  - `Hysteresis`: Relay with upper/lower thresholds
- **MIMO Subsystem Support**: Subsystems now automatically synchronize their external ports based on internal `Inport` and `Outport` blocks. This allows for subsystems with arbitrarily many inputs and outputs.
- **Fast Solver Subsystem Support**: The Fast Solver (Compiled Mode) now recursively compiles and flattens Subsystems, allowing complex hierarchical models to run with compiled performance (10-100x speedup).
- **Subsystem Port Sync Fix**: Fixed a bug where adding input/output ports inside a subsystem would not correctly update the simulation parameters on the outside block, leading to simulation failures.
- **Subsystem Copy Fix**: Fixed a crash when simulating copied subsystems by correctly restoring the internal structure (`sub_blocks`, `sub_lines`) and maintaining the `Subsystem` class identity during paste.
- **Copy-Paste Connections Fix**: Fixed an issue where connections were lost after pasting by ensuring lines are registered before trajectory calculation.
- **Serialization Fix**: Resolved `QPainterPath` pickling errors during copy operations by implementing custom deepcopy logic for connections.
- **Variable Viewer Sync**: Fixed an issue where the Workspace Viewer table would not automatically refresh after running a script in the Variable Editor.
- **Property Editor Variables**: Updated the Property Editor to accept workspace variable names (e.g., "K", "A") in numeric fields without validation errors.
- **UI Shortcuts**: Unified shortcuts for Variable Editor (`Ctrl+Shift+V`) and Workspace Viewer (`Ctrl+Shift+W`).




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
- Fixed rectangle selection appearing after creating a block via double-click on canvas. The fix ensures proper event handling and state reset when focus returns from the command palette.
- Fixed Fast Solver replay loop missing handlers for `WaveGenerator`, `Noise`, `MathFunction`, `Selector`, `Hysteresis`, `Mux`, and `Demux` blocks, which caused empty Scope plots.

### Technical Improvements
- All 42 blocks have `execute()` methods in dedicated class files
- Test suite: 54 tests passing
