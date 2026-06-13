# DiaBloS Modern - Consolidated TODO

> Single source of truth for all pending work items.
> Last updated: February 2026

---

## High Priority

### Testing & Quality
- [x] **Regression test suite** - Created `tests/regression/test_regression_suite.py` with 26 tests
  - Numerical accuracy tests (integrators, TFs, state-space, transport delay, PID)
  - Bug fix regression tests (Step/Scope params, External block, StateVariable)
  - PDE block tests (Heat1D conservation, Heat2D initialization)
  - Optimization primitives tests (ObjectiveFunction, VectorGain, VectorSum)
- [x] **Performance profiling** - Profiled simulation engine (`tests/profiling/profile_simulation.py`)
  - Block-level simulation: 1000 steps in 0.03s (excluding imports)
  - PDE simulation: 100 steps × 100 nodes in 6ms
  - Main bottleneck: Import overhead (scipy), not execution
  - Recommendations documented in profiling script

### Block Implementation
- [x] **External block** - Returns proper error dict with message (was stub returning None)
  - File: `blocks/external.py`
  - Status: Properly handles missing file with informative error

### Known Issues
- [x] **Legacy test files** - Marked as skipped with proper documentation
  - `tests/test_blocks.py` - DSim requires GUI
  - `tests/test_sine_params.py` - Legacy DBlock API
  - `tests/test_transfer_function_exec.py` - DSim requires GUI

---

## Medium Priority

### Refactoring (from REFACTORING_TODO.md)
- [x] **Extract SubsystemManager from lib.py** - Done in Phase 3 of improvement plan
  - File: `lib/managers/subsystem_manager.py` (extracted)
  - Delegated from `lib/lib.py`
- [x] **Standardize block error returns** - Done in Phase 2 of improvement plan
  - Fixed blocks: demux.py, sigproduct.py, pid.py
  - All blocks now return `{0: value, 'E': False}` or `{'E': True, 'error': msg}`

### PDE Phase 1: Quick Wins (from PDE_ROADMAP.md)
- [ ] **Periodic BCs** - Add to HeatEquation1D/2D, WaveEquation1D/2D
  - Pattern exists in `AdvectionEquation1D`
- [ ] **Dynamic BC coefficients** - Time-varying Robin BC via input ports
  - Files: `heat_equation_1d.py`, `heat_equation_2d.py`
- [ ] **More initial conditions** - Add 'linear', 'step', 'random', 'checkerboard'
  - Location: `get_initial_state()` methods
- [ ] **Robin BC for 2D** - Extend from 1D to HeatEquation2D
  - Per-edge h coefficients

---

## Low Priority

### Refactoring (optional)
- [x] **Split modern_canvas.py** - Done in Phase 4 of improvement plan
  - Extracted: ClipboardManager, ZoomPanManager
  - File: `modern_ui/managers/`

### Documentation
- [x] **API documentation** - Done in Phase 7 of improvement plan
  - Files: `mkdocs.yml`, `docs/api/*.md`
  - Using mkdocs + mkdocstrings with Google-style docstrings
- [ ] **Video tutorials** - Demo videos for key features

---

## Feature Ideas

### Teaching & Interaction
- [x] **Live parameter tuning** — Manipulate-style interactive tuning: pin block parameters to a tuning panel, drag sliders and watch scope plots update in real-time via headless re-simulation. Supports float params and individual list elements (e.g., transfer function coefficients). Scope window stays on top during tuning. Right-click slider rows to set custom range.
- [x] **Custom Python Function block** — `blocks/function.py` ("Function"). Users type an expression of the inputs and time (e.g. `sin(u[0]**2) + u[1]`). Inputs exposed as 0-indexed `u[i]` and 1-indexed `u1`/`u2`; `t` is sim time. Variable input-port count via `io_editable='input'`. Evaluated through the hardened `safe_expr` AST walker (numpy math allowed; imports/attribute escapes rejected). List expressions yield vector outputs. Diagrams containing it use the interpreted engine (not in `COMPILABLE_BLOCKS`). Tests: `tests/unit/test_function_block.py` (19), integration chains in `test_simulation_execution.py`.
- [x] **Diagram-to-LaTeX/TikZ export** — Export block diagrams as TikZ figures for papers and lecture notes. Saves hours of redrawing diagrams for ACC/IFAC publications. File > Export > Export as TikZ... with live preview, clipboard copy, and configurable options.

### Research & Data
- [ ] **Data Import block** — Read time-series from CSV/MAT files as a source signal. Essential for comparing simulation against experimental data (e.g., QCar2) or model fitting.
- [ ] **Linearization tool** — Select input/output points in a diagram, compute the linearized transfer function, and generate Bode plot + pole-zero map. Ties together existing BodeMagnitude/BodePhase blocks.
- [ ] **Code generation** — "Export as standalone Python script" so a diagram becomes a self-contained `.py` file. Useful for sharing with collaborators who don't have DiaBloS installed.

### UI Polish
- [x] **Dark mode fixes** — Fixed invalid theme key lookups (`accent`, `block_fill`, `connection_line` → proper keys). Property editor "Documentation" title, command palette, minimap all now use correct theme colors. Error panel severity backgrounds are theme-aware. Canvas selection rect, connection preview, and error indicators use theme colors. Block renderer icons use `block_icon_color` theme key.
- [x] **Compact toolbar** — Switched to `ToolButtonIconOnly` with `setIconSize(20,20)`, 14px emoji font, 70px zoom slider. Removed redundant status label (main status bar already exists). Theme button always visible at default window size on macOS. Theme also accessible via View > Toggle Theme (Ctrl+T).
- [ ] **Minimap** — Small overview panel showing the full diagram with a viewport rectangle, helpful for navigating large diagrams.

---

## Future / Roadmap

### PDE Phase 2: Mesh Abstraction
- [ ] Create `MeshBase` abstract class
- [ ] Create `blocks/pde/mesh/` directory structure
- [ ] Refactor 2D PDE blocks to use mesh interface
- [ ] Add curvilinear mesh support

### PDE Phase 3: Unstructured Meshes
- [ ] Mesh loader block (read .msh, .vtk, .stl)
- [ ] Mesh generator block (circle, L-shape)
- [ ] Mesh exporter block (VTK for ParaView)
- [ ] FEM-based spatial operators (P1 triangles)
- [ ] Update FieldScope2D for triangulation rendering
- [ ] Add FieldExportVTK block

### PDE Phase 4: Advanced Features
- [ ] 3D PDE support (HeatEquation3D, WaveEquation3D)
- [ ] Absorbing BCs / PML for wave equations
- [ ] Adaptive mesh refinement
- [ ] Domain decomposition for parallel solving

---

## Completed

### February 2026
- [x] **7-Phase Improvement Plan** - Comprehensive code quality improvements
  - Phase 1: Bug fixes (FileService.save, SimulationEngine duplicates, sys.path)
  - Phase 2: Block error handling standardization
  - Phase 3: SubsystemManager extraction from lib.py
  - Phase 4: modern_canvas.py split (ClipboardManager, ZoomPanManager)
  - Phase 5: Config-driven logging (`lib/logging_config.py`, `config/logging.json`)
  - Phase 6: Type hints (`lib/types.py`, base_block.py)
  - Phase 7: API documentation (mkdocs + mkdocstrings)
- [x] **Advection equation fix** - Second-order upwind scheme reduces error from 30% to <1%
  - Files: `blocks/pde/advection_equation_1d.py`, `lib/engine/system_compiler.py`
- [x] **Animation export for FieldScope** - GIF/MP4 export with dialog
  - Files: `lib/plotting/animation_exporter.py`, `modern_ui/widgets/animation_export_dialog.py`
- [x] **Unit tests for 14 untested blocks** - 160 new tests added
  - TransportDelay, DiscreteTranFn, External, Assert, FFT, Subsystem, Inport, Outport, Abs, Terminator
- [x] **Test coverage improvement** - 44% → 57% of blocks tested
- [x] **Total tests** - 346 → 573 (422 unit + 152 integration)

### January 2026
- [x] **Optimization Primitives** - 11 blocks for visual algorithm building
- [x] **PDE 2D blocks** - HeatEquation2D, WaveEquation2D, AdvectionEquation2D
- [x] **FieldScope2D** - Interactive time slider visualization

### Previous
- [x] All Priority 1-7 refactoring tasks (see REFACTORING_TODO.md)
- [x] StateSpaceBaseBlock consolidation
- [x] Circular import fixes
- [x] Canvas and MainWindow modularization

---

## Change Log

| Date | Change |
|------|--------|
| 2026-06-13 | Added **1-D/2-D Lookup Table** + **FromFile** blocks: `blocks/lookup_table.py` (`LookupTable1D` via `interp1d`, `LookupTable2D` via `RegularGridInterpolator`; linear/nearest interp, clip/linear extrapolation; tables parsed with `safe_literal`); `blocks/from_file.py` (`FromFile` source replays CSV/NPZ/MAT/TXT time-series with linear/zoh/nearest interp and hold/loop end-behavior; data cached in `params`, reloaded on `_init_start_`/path change). New shared loader `lib/services/timeseries_loader.py` (`load_timeseries`, `allow_pickle=False`); `data_fit._load_data` refactored to delegate to it (DRY). Both blocks run on the interpreter path. Tests: `test_lookup_table.py` (16), `test_from_file.py` (13), `test_timeseries_loader.py` (10). |
| 2026-06-13 | Added **Find Operating Point (Trim)** (Analysis menu): `AnalysisController.find_trim()` solves `f(0,y)=0` on the compiled ODE RHS via `Linearizer.find_operating_point`; `modern_ui/widgets/operating_point_window.py` shows the equilibrium state table with copy-to-clipboard (handles no-states / uncompilable cleanly). Synchronous on the UI thread (mirrors Linearize & Analyze). Tests: `test_operating_point_window.py` (5), `test_analysis_controller.py::TestFindTrim` (3). |
| 2026-06-13 | Added **Step / Impulse response** to Linearize & Analyze: `AnalysisController._assemble` now computes `step_response`/`impulse_response` (scipy.signal) whenever a SISO transfer function is available; `LinearizationResultWindow` gained **Step** and **Impulse** tabs (show a hint when no I/O is designated). Contract extended in both docstrings + `_empty_result`. Tests: `test_analysis_controller.py::TestStepImpulseResponse` (2), `test_linearization_result_window.py` updated (3→5 tabs). |
| 2026-06-13 | Added **Parameter Sweep (1-D/2-D)** (Analysis > Parameter Sweep...): sweep one or two block parameters across a grid on the headless re-sim path. New `lib/analysis/resim.py` (shared `OUTCOME_METRICS` + `harvest_scope_signals`, extracted from MonteCarlo); `lib/analysis/parameter_sweep.py` (`ParameterSweepRunner`, restores params, cancellable, partial-on-cancel); `modern_ui/widgets/parameter_sweep_worker.py` (QThread), `parameter_sweep_dialog.py` (axis/range pickers), `sweep_result_window.py` (1-D response-family overlay + metric-vs-parameter; 2-D outcome-metric heatmap). 1-D yields per-value traces+metrics; 2-D yields a per-run metric grid. Tests: `test_parameter_sweep.py`, `_worker`, `_dialog`, `test_sweep_result_window.py` (21). MonteCarlo refactored onto `resim` (re-exports `OUTCOME_METRICS`; tests still green). |
| 2026-06-02 | Added solver selection: `SimulationDialog` (lib/dialogs.py) now offers a solver dropdown (RK45/RK23/DOP853/Radau/BDF/LSODA adaptive + fixed-step RK4/Euler) and rtol/atol fields. Compiled solver (`simulation_engine.py run_compiled_simulation`) dispatches on `solver_method`; new module fn `integrate_fixed_step` does in-house Euler/RK4; stochastic systems still force Euler; unknown method → RK45. Settings persist in `.diablos` (`solver_method`/`rtol`/`atol` via file_service + lib.py save/serialize/deserialize) and surface read-only in the property editor. Tests: `tests/unit/test_solver_selection.py` (19, incl. end-to-end runs across all solvers). |
| 2026-06-02 | Added Logic blocks: `RelationalOperator` (in1 OP in2), `CompareToConstant` (in OP constant), `LogicalOperator` (AND/OR/NAND/NOR/XOR/NOT, variable inputs). Category "Logic"; output 1.0/0.0 element-wise. Tests: `tests/unit/test_logic_blocks.py` (27). |
| 2026-06-02 | Added Custom Python Function block (`blocks/function.py`, "Function"): expression of inputs `u[i]`/`u1..` and time `t`, variable input ports (`io_editable='input'`), `safe_expr` sandbox, vector output via list expressions. Tests: `tests/unit/test_function_block.py`, integration chains in `test_simulation_execution.py`. |
| 2026-02-12 | Added TikZ export feature: File > Export > Export as TikZ... with live preview, standalone/snippet modes, configurable options. New files: `lib/export/tikz_exporter.py`, `modern_ui/widgets/tikz_export_dialog.py`. |
| 2026-02-11 | Fixed compiled solver execution order bug: state blocks (TranFn, Integrator) now run after algebraic blocks. Fixed cursor visibility in property editor. |
| 2026-02-06 | Dark mode fixes: invalid theme keys, block icon colors, error panel, canvas renderer. Compact toolbar (icon-only, 20px icons) |
| 2026-02-05 | Added Feature Ideas section (live tuning, Python function block, TikZ export, data import, linearization, code gen, dark mode, minimap) |
| 2026-02-05 | Marked completed items in REFACTORING_TODO.md (SubsystemManager, block error returns) |
| 2026-02-03 | Completed 7-phase improvement plan (bugs, refactoring, code quality) |
| 2026-02-03 | Fixed advection equation numerical diffusion (second-order upwind) |
| 2026-02-02 | Created consolidated TODO from REFACTORING_TODO.md, PDE_ROADMAP.md, CLAUDE.md |
| 2026-02-02 | Added animation export feature to completed |
| 2026-02-02 | Added 160 new unit tests to completed |

---

## Code Quality Review (2026-06-13) — follow-ups

A whole-app review confirmed 334 findings. All concrete defects, correctness, and
performance items with a safe fix are now **fixed** (see
`tasks/code-quality-review-2026-06-13.md` and its `-deferred.md` companion).

### Done
- [x] Impulse/Step-`impulse` vs adaptive solver — routed to the interpreter path.
- [x] Vectorize per-node Python loops in the compiled PDE RHS (1D & 2D).
- [x] `Scope`/`Export` O(n²) per-step concat → amortized-O(1) geometric buffers.
- [x] Compiled heat **Robin BC** reconciled with the interpreted block.
- [x] `base_analyzer` discrete PID TF (c2d); `integrator` configurable method.
- [x] `connection` routing from port orientation; `draw_grid`/minimap/block_renderer caching.
- [x] Unified 1D/2D PDE `compute_derivatives` signatures.
- [x] Extracted `MainWindow._init_core_managers` (constructor altitude).

### Remaining — architectural backlog (no behavior change; do with dedicated tests)
- [ ] **Single-source the PDE finite-difference/BC kernels** shared by the blocks
  and `SystemCompiler` (the Robin *correctness* divergence is already fixed; this
  is the larger merge-into-one-kernel refactor).
- [ ] **`lib.py` interpreter hot path**: O(blocks²) per-step re-iteration,
  duplicated multi-rate loops, `DSim` facade, engine-state re-copy.
- [ ] **`modern_canvas` god object**; consolidate the ~18-manager layer.
- [ ] Break the `lib/` ↔ `modern_ui/` import layering via dependency inversion
  (move shared theming into `lib`), instead of function-local imports.
- [ ] Add compiled-vs-interpreted equivalence tests for each compiled stateful
  block (RateLimiter, PID, TransportDelay, Selector) and an all-Neumann 2D PDE
  corner integration test.

---

## References

- `tasks/code-quality-review-2026-06-13.md` - Full whole-app review (334 findings)
- `docs/REFACTORING_TODO.md` - Detailed refactoring history
- `docs/PDE_ROADMAP.md` - Full PDE enhancement roadmap with architecture diagrams
- `CLAUDE.md` - Project overview and recent work
