# Changelog

All notable changes to DiaBloS will be documented in this file.

## [Unreleased]

### Added
- **Drag-to-connect**: press an output port, drag, release on an input port. Click-click still works, a wire can be started from a free input port, and the preview snaps to the hovered port and turns green/red for accepted/rejected targets.
- **Wire editing**: right-click a wire for **Auto-route wire** and **Reset routing**; double-click a bend handle to remove it; dragging a straight or curved wire creates a proper three-segment bend; bends snap to the grid.
- **Block shapes**: blocks declare an outline via `BaseBlock.shape`. MatrixGain is now a triangle like Gain and both show their gain value inside; Sum and Product are circles (up to three inputs) with a `+`/`-` or `×`/`÷` glyph at each input port; Goto/From are pointed tags showing `[tag]`.

### Changed
- Block moves no longer convert bezier wires into Manhattan routes. Bezier wires stay curved; only orthogonal, auto-routed wires are re-routed, and hand-bent wires keep their bends (end segments stay axis-aligned). **Auto-route** switches the routed wires to orthogonal mode so the routing menu reflects what is drawn.
- Orthogonal wires have small rounded corners; the router keeps clear of block name labels.
- Arrowheads sit in front of the port disc instead of under it, wire labels sit at the true midpoint of the drawn path, and crossing wires show an over/under gap.

### Fixed
- Clicking or hovering a curved wire tested the straight chord between its ports, so clicks on the curve often missed (and clicks on empty space along the chord selected it).
- A freshly created wire was never routed with block knowledge, so feedback and obstacle-avoiding routes only appeared after a block was moved; changing a wire's routing mode from the menu had the same problem.
- Undo of a block move or resize needed two Ctrl+Z presses (the undo entry was pushed after the move). Wire bends and hand-made routes now survive undo/redo.

## [1.0.0] - 2026-09-03

First tagged release. Everything below landed after the 2026-01-29 development
log at the end of this file. Prebuilt apps (macOS arm64 DMG, Windows x64 zip)
are published on the GitHub Releases page.

### Added

**Analysis & experiments**
- **Linearize & Analyze** (Analysis menu): numeric linearization of the compiled ODE (A/B/C/D by finite differences), with pole-zero map, Bode plot, gain/phase margins and a summary of stability, time constants, oscillatory modes, controllability and observability.
- **Step / Impulse response** tabs in the linearization window whenever a SISO transfer function is available.
- **Find Operating Point (Trim)** (Analysis menu): solves `f(0, y) = 0` on the compiled right-hand side and shows the equilibrium state.
- **Linearized-model export**: Copy as Python (numpy + python-control), Copy as MATLAB, and Save Data as `.mat` / `.npz`.
- **Monte Carlo ensembles** (Analysis menu): threaded, cancellable, seeded runs with mean/percentile bands and per-run outcome histograms. Every block exposing a `seed` derives its sub-seed from one master seed, so an ensemble is reproducible from a single number.
- **Parameter sweeps, 1-D and 2-D** (Analysis menu): response-family and metric-vs-parameter views for 1-D, an outcome-metric heatmap for 2-D.
- **Stochastic blocks** for ensembles: PacketLoss (Gilbert-Elliott bursty loss), NetworkChannel, RandomSource.
- **Live parameter tuning**: pin a parameter from the property editor, drag a slider in the Parameter Tuning Panel (`Ctrl+Shift+T`), and watch scope plots redraw from a debounced headless re-simulation.

**Export & automation**
- **Export as Python Script** (File > Export, and the `export-python` CLI subcommand): writes the diagram as a self-contained numpy + scipy script that mirrors the compiled solver, with its own `--time` / `--dt` / `--out` / `--no-plot` CLI. Unsupported blocks, discrete sample times and algebraic loops are reported instead of producing broken code.
- **Headless CLI**: `python diablos_modern.py run diagram.diablos -o out.csv [--time --dt --solver interpreter]` simulates without the GUI and exports Scope traces to CSV or NPZ.
- **Export as TikZ** (File > Export): publication-ready TikZ with live preview, standalone/snippet modes and clipboard copy.
- **Export as Image** (File > Export): chrome-free PNG (3x) or SVG of the diagram content, plus **Copy Diagram as Image** in the Edit menu.
- **Publication figure export** from scope windows: serif matplotlib figures to PDF, PNG (300 dpi) or SVG.
- **Previous-run overlay** in scope windows: the previous run's traces drawn dimmed and dashed behind the live ones.

**Blocks**
- PDE family (1D/2D heat, wave, advection; diffusion-reaction) with FieldScope / FieldScope2D visualisation and GIF/MP4 animation export, plus a symbolic computation layer.
- Optimization primitives (ObjectiveFunction, NumericalGradient, VectorPerturb, StateVariable, VectorGain, VectorSum, LinearSystemSolver, RootFinder, ResidualNorm, Momentum, Adam) for building optimization algorithms as diagrams.
- **Function** block: a sandboxed Python expression of the inputs `u[i]` / `u1..` and time `t`, with a variable number of input ports.
- Logic blocks: RelationalOperator, CompareToConstant, LogicalOperator.
- **FromFile** data-import source (CSV / NPZ / MAT / TXT time-series with linear/zoh/nearest interpolation and hold/loop end behaviour) and **LookupTable1D / LookupTable2D**.
- LQR block and analyzer; MatrixGain; Impulse and Chirp sources; AgentScope multi-agent trajectory view.
- Multi-rate simulation support with RateTransition and FirstOrderHold.
- `draw_icon()` glyphs for 25 previously icon-less blocks; every registered block now defines its own icon, apart from a documented set whose glyph is painter-drawn text (`1/s`, `B(s)/A(s)`, `PID`, ...).

**Simulation**
- **Solver selection** in the Simulation Configuration dialog: RK45, RK23, DOP853, Radau, BDF, LSODA plus fixed-step RK4 and Euler, with rtol/atol fields. The choice is saved in the `.diablos` file.
- Fast-solver coverage extended to WaveGenerator, Noise, MathFunction, Selector, Hysteresis, Demux and the logic blocks, and to recursive **subsystem flattening**.
- Compiled-system cache and solver diagnostics reported in the status bar after a run.
- PDE Phase 1: **periodic** boundary conditions on the heat and wave families, **Robin BCs in 2D** with per-edge coefficients, **time-varying Robin `h`** through optional input ports, and new initial-condition presets (`linear`, `step`, `random`, `checkerboard`, `radial`), all single-sourced so the interpreter and compiled paths agree.

**UI**
- Keyboard-shortcuts reference dialog (**F1**), generated from the same table the command palette uses.
- Command palette keyboard navigation, plus palette Favorites and Recently-used sections.
- Smart alignment guides while dragging blocks; auto-route wires via a grid A* router; block rename.
- Block-palette colour schemes (Solarized, Tailwind, Catppuccin Frappé), a Solid Block Fills toggle, and persisted theme/palette preferences.
- Live overlay of output value chips on the canvas during a run.
- Discoverability pass: empty-canvas hint, Help menu, tooltips, parameter units, first-run guidance.

**Packaging & CI**
- PyInstaller packaging with a macOS DMG installer (arm64 and x86_64) and Windows/Ubuntu build instructions.
- Release workflow: pushing a `v*` tag builds and publishes a macOS arm64 DMG and a Windows x64 zip as a GitHub release. The app version is single-sourced from `pyproject.toml` and shown in the window title.
- GitHub Actions CI on a Python 3.9 + 3.12 matrix, headless, with coverage, a `ruff check` gate and a `ruff format --check` gate.
- New example diagrams: SIS epidemics, Kuramoto synchronisation, distributed subgradient.

### Changed
- The compiled fast solver is the default and the numerically accurate path; the interpreter is the fixed-step full-coverage fallback. The two-engine contract is documented in `docs/ARCHITECTURE.md`.
- All 13 `eval()` call sites replaced by `lib/safe_eval.py`, an allowlist AST interpreter; the External block's exec path is disabled and `.npz` loading no longer allows pickle.
- Design-token theming (spacing, radius, type scale, elevation, canonical fonts) applied across the UI, with the theme system moved into `lib/theming` so `lib/` no longer imports `modern_ui/`.
- `ruff` adopted as the linter and formatter (replacing black/pylint); the whole repository reformatted.
- numpy / scipy / PyQt5 upper bounds pinned for reproducible installs.
- `.diablos` is the default file dialog format and extension.
- Simulink/MATLAB references removed from the codebase and UI text.

### Fixed
- Compiled solver: closed-loop feedthrough (`D != 0`) bug; state blocks now run after algebraic blocks; `t_eval` floating-point overshoot; feedback loops previously ignored; `SgProd` name mismatch; vector-signal safety in the Product, MathFunction, Exponential, Deadband, Switch, PID and Hysteresis kernels; workspace variables resolved in algebraic executors.
- Interpreter: sampled-block timing, simulation horizon and RK45 stepping; RateLimiter slewing at twice the configured rate; state blocks integrating at a fixed 0.01 s regardless of `sim_dt`; Selector comma-list indices; 2D PDE blocks never actually integrating.
- Discrete / sampled-data: z-domain blocks default to an inherited sample rate; feedthrough memory blocks deliver their fresh output within the same step; discrete transfer functions no longer output zero when `sampling_time > 0`.
- WaveEquation1D energy divergence between solve and replay; advection numerical diffusion (second-order upwind); the PDE CFL warning now re-arms on each run.
- Subsystems: children lost on save/reload, port sync from internal Inport/Outport blocks, port scaling on resize, naming collisions, and crashes when simulating copied subsystems.
- Copy/paste: lost connections, `QPainterPath` pickling errors, and block category preserved through copy/paste, undo/redo and duplicate.
- Algebraic-loop detection: false positives after undo/redo, memory blocks inside subsystems, and stateful-block classification.
- Canvas: runaway-zoom crash, source block left selected after a Ctrl+Click chained connect, stray rectangle selection, duplicate block creation, and alignment guides suppressed by grid snapping.
- Platform: invisible text cursor on macOS arm64 (Fusion style), multi-monitor popup placement, Windows 11 title-bar flash, Windows dark-mode contrast.
- Scope and Export buffers went from O(n^2) to amortized O(1) per step; experiment worker threads are cancelled and joined on close.
- Goto/From virtual lines are persisted to disk; Demux honours user-added outputs and scalar inputs; PRBS no longer outputs a constant zero; the Noise block no longer hangs the solver; startup crash from auto-route wire ordering.
- Accidentally committed SSH keys removed from the repository.

---

## Pre-1.0 development log - 2026-01-29

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
