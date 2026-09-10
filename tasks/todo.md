# DiaBloS Modern - Consolidated TODO

> Single source of truth for all pending work items.
> Last updated: 2026-09-08

---

## Release 1.1.0 (pending — needs the user's go-ahead to merge/push)

Everything since the local `v1.0.0` tag (2026-09-03) lives on `feat/maturity`
(~72 commits ahead of `main`, nothing pushed) and in the CHANGELOG `[Unreleased]`
section: zero-crossing detection, Spanish i18n, library blocks with masks,
drag-to-connect + wire editing + block shapes, validation suite, solver semantics
and stiffness guidance, examples gallery test, block API / user blocks, docs site.
`release.yml` only fires on a pushed `v*` tag, so no packaged build has run yet.

- [ ] Run the full suite + `ruff check .` + `ruff format --check .` on `feat/maturity`
- [ ] Bump `pyproject.toml` version to 1.1.0; move CHANGELOG `[Unreleased]` to `[1.1.0]`
- [ ] Merge `feat/maturity` into `main` (fast-forward; `main` has not moved)
- [ ] Push `main` and tags (`v1.0.0`, `v1.1.0`); confirm `release.yml` and `docs.yml` go green
- [ ] Delete the merged `feat/*` agent branches (`i18n-spanish`, `zero-crossing`,
  `library-blocks`, `i18n-masks`, `i18n-events-libraries`, `block-api`, `docs-site`,
  `examples`); keep `feat/paper` (JOSS draft) and `backup/pre-merge-main`
  (200 commits that exist nowhere else)

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

### PDE Phase 1: Quick Wins (from PDE_ROADMAP.md) — DONE (September 2026)
- [x] **Periodic BCs** - Added to HeatEquation1D/2D and WaveEquation1D/2D
  - Implemented in `lib/engine/pde_ops.py` (`is_periodic`), so the interpreter
    blocks and `lib/engine/compiler_kernels/pde.py` are equivalent by construction
  - Periodic on either end wraps that whole axis; the N nodes wrap as a ring
- [x] **Dynamic BC coefficients** - Time-varying Robin `h` via input ports
  - `heat_equation_1d.py`: `h_left` (3), `h_right` (4);
    `heat_equation_2d.py`: `h_left` (5), `h_right` (6), `h_bottom` (7), `h_top` (8)
  - Unconnected ports fall back to the matching param, so old diagrams load unchanged
- [x] **More initial conditions** - 1D `linear`/`step`/`random`;
  2D `linear`/`step`/`random`/`checkerboard`/`radial`
  - Single-sourced in `lib/engine/pde_helpers.py` (`parse_pde_*_initial_condition`),
    which the blocks and the compiler both call; `random` takes a `seed`
- [x] **Robin BC for 2D** - HeatEquation2D supports Robin on any edge with
  per-edge `h_left`/`h_right`/`h_bottom`/`h_top` and a shared `k_thermal`
- Tests: `tests/unit/test_pde_phase1.py`, `tests/regression/test_equiv_pde_phase1.py`
- Details: `docs/PDE_ROADMAP.md` (Phase 1), `docs/wiki/PDE.md`

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
- [x] **Data Import block** — `blocks/from_file.py` (`FromFile`, category Sources) replays a recorded CSV/NPZ/MAT/TXT time-series as a driving signal, interpolated (`linear`/`zoh`/`nearest`) onto the sim grid with `hold`/`loop` end behavior; parsed data cached in `params`. Shared reader: `lib/services/timeseries_loader.py` (`load_timeseries`, `allow_pickle=False`), also used by `blocks/optimization/data_fit.py`. Companions: `blocks/lookup_table.py` (1-D/2-D static maps). Tests: `tests/unit/test_from_file.py` (13), `test_timeseries_loader.py` (10), `test_lookup_table.py` (16).
- [x] **Linearization tool** — `lib/analysis/linearizer.py` computes A/B/C/D by finite differences over the compiled ODE RHS; `modern_ui/controllers/analysis_controller.py` assembles the result dict (poles/zeros, Bode, margins, step/impulse, controllability/observability). UI: Analysis > **Linearize & Analyze...** (`modern_ui/widgets/linearize_dialog.py` → `linearization_result_window.py`, 5 tabs + Python/MATLAB/.mat/.npz export via `lib/analysis/linearization_export.py`) and Analysis > **Find Operating Point (Trim)...** (`operating_point_window.py`). The BodeMag/BodePhase/Nyquist/RootLocus/LQR blocks remain the per-block path (`lib/analysis/analyzers/`).
- [x] **Code generation** — File > Export > **Export as Python Script...** (`MainWindow.export_python_script`) and the `export-python` CLI subcommand emit a self-contained numpy+scipy `.py` via `lib/export/python_codegen.py`. The script mirrors the compiled solver (one `rhs(t, x)` under `solve_ivp`, same three-group order) and its `--out` CSV/NPZ matches `python -m lib.cli run` column-for-column. Supported set: `SUPPORTED_BLOCK_NAMES` (19 blocks); unsupported blocks, discrete sample times and algebraic loops raise `CodegenUnsupportedError`/`CodegenError` instead of emitting broken code. Tests: `tests/unit/test_python_codegen.py`, `tests/modern_ui/test_export_python_menu.py`.

### UI Polish
- [x] **Dark mode fixes** — Fixed invalid theme key lookups (`accent`, `block_fill`, `connection_line` → proper keys). Property editor "Documentation" title, command palette, minimap all now use correct theme colors. Error panel severity backgrounds are theme-aware. Canvas selection rect, connection preview, and error indicators use theme colors. Block renderer icons use `block_icon_color` theme key.
- [x] **Compact toolbar** — Switched to `ToolButtonIconOnly` with `setIconSize(20,20)`, 14px emoji font, 70px zoom slider. Removed redundant status label (main status bar already exists). Theme button always visible at default window size on macOS. Theme also accessible via View > Toggle Theme (Ctrl+T).
- [x] **Minimap** — `modern_ui/widgets/minimap_widget.py`: dockable overview panel (left/right) with a viewport rectangle; click to pan, drag for continuous panning. View > Minimap or `Ctrl+Shift+M` (`MainWindow.toggle_minimap`). Coverage: `tests/modern_ui/test_main_window_view_actions.py`.

---

### Potential Future Additions (idea review, September 2026)
Picked for implementation in the Sept 2026 feature campaign: **Spanish localization
with a pluggable i18n system**, **zero-crossing / event detection** in the compiled
solver, and **user block libraries with masks**. The remaining ideas from that review
are parked here:

- [x] **Stiff solver choice in the UI** — already shipped: the Simulation Settings
  dialog (`lib/dialogs.py`, `SOLVER_METHODS`) offers RK45/RK23/DOP853/Radau/BDF/
  LSODA/RK4/Euler/auto with rtol/atol; `auto` resolves to LSODA. The headless `run`
  command reads the file's solver settings and accepts `--method/--rtol/--atol`.
- [ ] **Sample-time coloring** — tint blocks/wires by sample rate (continuous vs each
  discrete period) so multirate mistakes around ZOH/FOH/RateTransition are visible.
- [x] **User blocks directory for frozen builds** — done in the Sept 2026 maturity
  campaign: `lib/user_blocks.py` (`user_blocks_dir`) is scanned at startup, user
  blocks appear in the palette with a reload action and a missing-block warning.
- [ ] **One-click auto-layout** — layered (Sugiyama-style) layout action in the Edit
  menu, generalizing `scripts/fix_diagram_overlaps.py`.
- [ ] **Block-diagram algebra** — select a source and a sink, show the closed-loop
  transfer function symbolically with reduction steps (builds on `symbolic_execute()`).
- [ ] **Run comparison overlay** — keep the previous run as a ghost trace in the
  Scope / waveform inspector when parameters change (reuse the Monte Carlo harvesting).
- [ ] **System identification block** — fit an n-th order transfer function to
  FromFile data (FromFile + DataFit already provide the pieces).
- [ ] **Interactive source blocks** — Slider/Knob source and Manual Switch adjustable
  on the canvas during a run.
- [ ] **Real-time pacing and hardware I/O** — wall-clock-paced run mode plus UDP and
  serial source/sink blocks for lab rigs (Arduino etc.).
- [ ] **Autosave and crash recovery** — periodic recovery file per open diagram.
- [ ] **Semantic diff for `.diablos` files** — CLI subcommand diffing by block and
  connection rather than by JSON line.

### Open gaps left by the September 2026 campaigns
- [ ] **Library / user blocks are not compilable** — diagrams containing them fall
  back to the interpreted engine (not in `COMPILABLE_BLOCKS`). Compile a library
  block by flattening its subsystem the way `Flattener` already does for inline ones.
- [ ] **Outcome-metric labels and validator messages stay English** —
  `OUTCOME_METRICS` combo labels (`lib/analysis/resim.py`) and the validation-suite
  messages are not routed through `tr()`; the Spanish catalog is otherwise complete.
- [ ] **Stiffness diagnostic caps at 64 states** — `STIFFNESS_MAX_STATES` in
  `lib/engine/solver_diagnostics.py` skips the Jacobian eigen-analysis on larger
  systems (PDE diagrams), so they get no solver suggestion. Use a sparse/power-
  iteration estimate or sample the spectrum instead of skipping.
- [ ] **Event-triggered state reset** — zero-crossing detection locates switching
  instants but a block cannot yet reset an integrator state at the event (needed for
  bouncing-ball / impact examples).
- [ ] **pyqtgraph teardown segfault** in `tests/modern_ui/test_linearization_result_window.py`
  is only dodged (module-scoped fixture + `gc.collect()`); the result window's own
  teardown is still unfixed. See `tasks/lessons.md`, "pyqtgraph teardown segfaults".
- [ ] **PyQt6 migration** — deliberately deferred; PyQt5 is EOL-adjacent and
  `lib/theming/theme_manager.py` already guards the 5.9-vs-5.15 `setFamilies` split.
- [ ] **PathSim benchmark** — the competitive analysis asked for a head-to-head
  timing vs PathSim on the examples gallery; skipped because `pathsim` was not
  installed in the `diablos` env.

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

## Follow-ups (flagged September 2026)

- [x] **PDE BC params should be dropdowns.** Done: the specs in
  `lib/engine/pde_helpers.py` carry `options` (Dirichlet/Neumann/Robin/Periodic),
  the property editor renders any param with `choices`/`options` as a QComboBox,
  and `tests/regression/test_choice_param_dispatch.py` covers HeatEquation1D/2D.
- [ ] **Re-verify the reported compiled-path PID ordering bug.** The codegen agent
  (2026-09-03) claimed that on the compiled path a 1-port PID was ordered *before*
  the Sum feeding it, freezing the loop at zero. The only reproduction was the
  legacy `examples/pid_control_loop.json`, which was retired in 5559515, and both
  engines gave zeros on that stale file, so the claim is unconfirmed. Check with
  `examples/pid_second_order.diablos` (compiled vs interpreted must agree; the
  equivalence tests in `tests/regression/test_equiv_pid.py` are the template). If
  it reproduces, a feedthrough PID belongs in the algebraic middle group *after*
  its upstream Sum (`lib/engine/system_compiler.py`, `_is_d0_state_block`).
- [x] **`lib/ui/button.py` is vestigial.** Deleted 2026-09-10 with the DSim
  facade cleanup: the single `.active` write in `execution_init` had no reader.

---

## Completed

### September 2026 — 1.0.0 release campaign
- [x] **Release infrastructure** - `pyproject.toml` gained a `[project]` table
  that is the single source of truth for the version; `modern_ui/__init__.py`
  reads it back via `importlib.metadata` and the main-window title shows it
  (`modern_ui/managers/window_setup_manager.py:WINDOW_TITLE`); `diablos.spec`
  and `tools/build.sh` parse the same value so the macOS bundle version and the
  DMG filename match the tag (`dist/DiaBloS-<version>-arm64.dmg`).
  `.github/workflows/release.yml` builds a macOS arm64 DMG and a Windows x64 zip
  on every `v*` tag and publishes them as a GitHub release.
- [x] **`draw_icon()` for 25 previously icon-less blocks** - Abs, Assert,
  BodeMag, CompareToConstant, Delay, DiscreteStateSpace, Exponential, External,
  LogicalOperator, LQR, RelationalOperator, Terminator, TransportDelay,
  VariableTransportDelay and all 11 optimization primitives. Tests:
  `tests/unit/test_block_icons.py`.
- [x] **PDE Phase 1** - periodic BCs, dynamic Robin `h` ports, 2D Robin, and the
  new initial-condition presets (see the PDE Phase 1 section above).
- [x] **Python code generation** - `lib/export/python_codegen.py`, the
  File > Export > Export as Python Script... menu entry, and the
  `export-python` CLI subcommand (see Feature Ideas > Code generation).
- [x] **Docs reconciliation** - `docs/USER_MANUAL.md` extended to cover solver
  selection, scopes/FieldScope, the Analysis menu, tuning, experiments, exports,
  the headless CLI, PDE/optimization blocks, autosave and appearance;
  `mkdocs.yml` nav repaired (`wiki/Optimization-Primitives.md`) and extended
  with the architecture/developer/manual/fast-solver/building/roadmap pages;
  README test count and release/CLI/codegen mentions refreshed; CHANGELOG
  cut over to `[1.0.0]`.
- [x] **Hygiene** - `QFont.setFamilies` guarded behind `hasattr` in
  `lib/theming/theme_manager.py` (it needs Qt >= 5.13; the `QFont(family)`
  constructor is the fallback); the dead one-shot codemod
  `tools/integrate_variable_editor.py` deleted.

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
- [x] All Priority 1-7 refactoring tasks (see docs/archive/REFACTORING_TODO.md)
- [x] StateSpaceBaseBlock consolidation
- [x] Circular import fixes
- [x] Canvas and MainWindow modularization

---

## Change Log

| Date | Change |
|------|--------|
| 2026-09-08 | **Scope signal names consistent across solver paths**: `harvest_scope_signals` (`lib/analysis/resim.py`) keyed a single-channel Scope by its *label* only for the compiled replay's 2-D buffer and by the *block name* for the interpreter's flat buffer, so ensemble/sweep results renamed signals (and dropped the user's `labels` entry) depending on which solver ran. Both layouts are now normalised to `(n, vec_dim)` and every channel is keyed by `vec_labels[j]`, block name only as a fallback. Tests: `tests/unit/test_resim_harvest.py` (layout stubs), `tests/regression/test_harvest_scope_signals.py` (one diagram, both paths, identical keys). |
| 2026-09-03 | **1.0.0 release campaign**: release infra (`[project]` table in `pyproject.toml` as the single version source, read back by `modern_ui/__init__.py` and parsed by `diablos.spec`/`tools/build.sh`; `.github/workflows/release.yml` builds a versioned macOS arm64 DMG + Windows x64 zip on `v*` tags); `draw_icon()` for 25 icon-less blocks (`tests/unit/test_block_icons.py`); PDE Phase 1 (periodic BCs, dynamic Robin `h` ports, 2D Robin, new IC presets); standalone Python script export (`lib/export/python_codegen.py`, File > Export > Export as Python Script..., `export-python` CLI subcommand); docs reconciliation (USER_MANUAL, mkdocs nav, README, CHANGELOG 1.0.0); hygiene (`QFont.setFamilies` hasattr guard for Qt < 5.13, dead `tools/integrate_variable_editor.py` removed). |
| 2026-07-12 | Added **Scope "Previous run" overlay + publication figure export** (implemented via multi-agent workflow, then hand-verified). Overlay: `ScopePlotter` stashes each run's timeline+vectors (`_stash_run`, rotation keyed on timeline object identity; held runs dropped by `reset_held_runs()` from both `DSim.clear_all()` and `deserialize()`); `SignalPlot` gains a "Previous run" checkbox (disabled until a second run exists) drawing dimmed/dashed alpha-66 curves behind the live ones. Figure export: "Export Figure..." button renders a matplotlib (Agg, no pyplot) publication figure via new `lib/plotting/publication_figure.py` (serif fonts, Time (s) axis, legend, grid, `step(where='post')` for step traces) to PDF/PNG(300dpi)/SVG; CSV+figure export share `_collect_figure_traces`. Hand-verification caught a workflow bug: success feedback used PyQt5-unavailable `QTimer.singleShot(msec, obj, callback)` overload → every successful export raised and popped a false "Export Failed" dialog (masked in tests by conftest's QMessageBox neutralization); fixed with a button-parented QTimer. Tests: `test_signal_plot.py` (23), `test_scope_plotter_prev_run.py`, `test_publication_figure.py` (10), `tests/integration/test_prev_run_overlay.py` (interpreter+compiled). |
| 2026-07-11 | Added **Linearized-model export** to LinearizationResultWindow: "Copy as Python" (numpy + python-control snippet), "Copy as MATLAB", and "Save Data..." (.mat via scipy.io / .npz) buttons below the tabs (ok-results only). Formatting/serialization lives Qt-free in `lib/analysis/linearization_export.py` (full repr-precision round-trip, `ss()` only when A/B/C non-empty with zero-D synthesis, `tf()` only when num/den non-empty, `import control` omitted when unused). Tests: `tests/unit/test_linearization_export.py` (17) + export-bar tests in `test_linearization_result_window.py` (5). |
| 2026-07-11 | Added **Export diagram as image**: File > Export > Export as Image... (PNG at 3x, SVG via QSvgGenerator) and Edit > Copy Diagram as Image (clipboard), plus command-palette entries. New `modern_ui/tools/diagram_image_exporter.py` renders the chrome-free content trio (blocks/lines/ports) against the theme background, framed by the true content bounding rect (Bezier wire bows covered via `line.path.boundingRect()`), independent of zoom/pan; selection/hover state suppressed during render and restored in a `finally`. Tests: `tests/modern_ui/test_diagram_image_export.py` (15). |
| 2026-07-06 | Canvas manager-layer consolidation (first pass): DragResizeManager merged into InteractionManager; duplicate canvas `_paste_blocks` removed (context-menu paste now preserves connections via `ClipboardManager.paste_blocks(pos)`); connection validation moved into ConnectionManager; rect-selection block pass single-sourced in SelectionManager. |
| 2026-07-05 | Interpreter hot-path cleanup: deduplicated `execution_loop`/`execution_loop_headless` into `DSim._interpreter_step(interactive)`; `SimulationEngine.update_global_list` now O(1) via identity-tracked name index; `max_hier`/`rk45_len`/`rk_counter` became engine property bridges (re-copy blocks in `execution_init`/`run_tuning_simulation` deleted). Repo-wide `ruff format` sweep (414 files) + `ruff format --check` in CI (ruff pinned 0.15.18); E701/E702 ignores retired. |
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

#### Refactoring round (started 2026-09-10) — ranked by payoff
Measured with `ruff check --select C901` (max-complexity 25) and an AST pass over
function lengths; only eleven functions in `lib/`, `modern_ui/`, `blocks/` exceed
complexity 25, all listed here.
- [x] **Legacy icon switch in `block_renderer.py`** — done 2026-09-10.
  `_draw_legacy_icon` (356 lines, 51 branches, C901 = 52) ran *after* each
  block's `draw_icon` and appended a second copy of the shape: 14 blocks were
  double-stroked, FFT/RootLocus/XYGraph drew two different sketches on top of
  each other, RateLimiter's path was stroked with a width-1 pen. Replaced by
  table-driven `_draw_icon_text` (`_FRACTION_TEXT_ICONS`, `_CENTERED_TEXT_ICONS`,
  `_DYNAMIC_TEXT_ICONS`); RateLimiter/Hysteresis/PRBS absorbed their legacy
  fragments into `draw_icon`; Subsystem's nested squares live in
  `_icon_source_path`. Renderer 1514 → 1255 lines. Verified by pixel-diffing old
  vs new `draw_block` for all 82 block types (79 identical; the 20 changed are
  exactly the double-stroke/overlay set). Tests:
  `tests/modern_ui/test_block_renderer_icon_text.py`.
- [x] **`replay_compiled_signals`** — done 2026-09-10. Was 616 lines, C901 = 100
  (the worst in the repo). Now a ~70-line orchestrator (C901 < 10) over three
  helpers in `compiled_runner.py` (`_replay_order`, `_connections_by_destination`,
  `_collect_inputs`) and a new `lib/engine/replay_handlers.py`: `REPLAY_HANDLERS`
  (canonical_fn -> handler for the 16 inline block branches), `MATHFUNCTION_OPS`
  (the 17-way if/elif as a table of domain-guarded numpy callables), `RECORDERS` +
  `finalize_recorders` (Scope / FieldScope / FieldScope2D history), and
  `replay_fallback`. Verified by running all 56 examples on the compiled path before
  and after and comparing every Scope / FieldScope history: 115 arrays across 33
  compiled diagrams, all bit-identical. Tests: `tests/unit/test_replay_handlers.py`
  (40) plus the existing golden / equivalence suites.
- [x] **`SystemCompiler.compile_system`** — done 2026-09-10. Was 434 lines, C901 = 41.
  Now a ~40-line phase sequence over module-level helpers in `system_compiler.py`:
  `_build_input_map`, `STATE_ALLOCATORS` + `_allocate_states` (the 12-way state
  allocation if/elif as a table of `(block, params) -> (n, y0, matrices)`
  functions), `SOURCE_FNS` / `STATE_FNS` / `_is_d0_state_block` /
  `_execution_groups`, `_state_output_preloads`, `_make_model_func`, and a
  `_build_executors` method. Same trace-diff verification as the replay split
  (115/115 arrays bit-identical). Tests: `tests/unit/test_system_compiler_phases.py`.
- [x] **`DSim._interpreter_step`** (`lib/lib.py`, was 296 lines, C901 = 41) —
  done 2026-09-10. Now a ~50-line sequence over named phases: `_advance_clock`,
  `_publish_memory_outputs`, `_run_hierarchy_passes` / `_execute_ready_block`,
  `_is_end_of_run`, `_finish_run`, plus `_block_failed`, `_propagate_held_outputs`
  and `_has_enough_inputs` (worst remaining C901 = 10). Verified with the
  interpreted-path trace-diff (`use_fast_solver=False`, all 55 examples: 114/114
  arrays bit-identical). Tests: `tests/unit/test_interpreter_step_phases.py` (39).
  `docs/SOLVER_SEMANTICS.md` 4.2 names the phase methods.
  - [ ] Found while verifying (pre-existing, not fixed to keep the refactor
    behaviour-preserving): `examples/van_der_pol_stiff.diablos` on the
    **interpreted** path dies with `argument of type 'bool' is not iterable`.
    A block's `execute()` raises inside `solve_ivp` (`array must not contain
    infs or NaNs`), `SimulationEngine.execute_block` returns a bool instead of
    an error dict, and `DSim._block_failed` does `"E" in out_value`. Make
    `execute_block` return `{'E': True, 'error': ...}` (or make `_block_failed`
    accept non-dicts) so the user sees the block error, not a TypeError.
- [x] **`DSim` facade** (`lib/lib.py`, was 1825 lines / 87 methods) — done
  2026-09-10 (1736 lines). Measured every one-line delegation against its
  callers: 13 had no caller anywhere (`update_global_list`, `check_global_list`,
  `count_computed_global_list`, `get_max_hierarchy`, `detect_algebraic_loops`,
  `get_outputs`, `children_recognition`, `_plot_xygraph`, `_plot_fft`,
  `_is_discrete_upstream`, `_scope_step_modes`, `get_scope_traces`,
  `count_rk45_ints`) and were deleted; `reset_memblocks` / `reset_execution_data`
  were internal-only and are now direct `self.engine.*` calls (the one script
  caller moved too); `get_neighbors` went to the owner (`ScopePlotter.
  _input_connections` queries `dsim.engine`, no-engine stub → no inputs). Kept
  on purpose: the property bridges (`time_step`, `timeline`, `error_msg`, …),
  the subsystem navigation, run-history, plotting and `check_diagram_integrity`
  entry points — each has GUI/test/doc callers and is the intended MVC seam
  (`docs/DEVELOPER_GUIDE.md` "Controller Layer"). Also removed the pygame-era
  leftovers: `main_buttons_init` / `buttons_list` / `Button` (`lib/ui/` deleted,
  the only reader was a write nobody read; 9 test files dropped their
  `buttons_list` stubs) and the dead `canvas_*_limit`, `l_width`, `ls_width`,
  `line_creation`, `only_one`, `enable_line_selection`, `holding_CTRL` attributes.
  Verified: full suite + trace-diff on both solver paths bit-identical.
  Not done (deliberate): `lib/dialogs.py` stays — `SimulationDialog` is the live
  Run dialog (`execution_init` → `execution_init_time`) and `PortDialog` is used
  by `DBlock.change_port_numbers`; moving them under `modern_ui/` would make
  `lib/` import the GUI package. `lib/simulation/menu_block.py` has ~40 users
  across model/services/GUI; a rename/move is its own round.
- [x] **`lib/improvements.py`** (448 lines) — deleted 2026-09-10. The earlier
  call-site count was off: `PerformanceHelper` *was* live (main-window tick /
  step timers, canvas paint timer) and now lives in `modern_ui/perf_helper.py`;
  `ValidationHelper.validate_block_connections` / `detect_algebraic_loops` and
  `SafetyChecks.check_simulation_state` / `check_block_integrity` became plain
  functions at the bottom of `lib/diagram_validator.py` ("Pre-flight checks"),
  called by `ConnectionManager.validate_connection` and
  `SimulationController.start` / `ModernMainWindow.safe_update`. Dead and gone:
  `SimulationConfig` (instantiated, never read), `LoggingHelper`,
  `create_default_colors`, `validate_simulation_parameters`,
  `safe_execute_block_function`, the canvas's unused `validator`/`safety`
  instances, and `examples/example_usage.py` (a demo of the module; README/wiki
  rows removed, CHANGELOG "Removed" entry). Behaviour change, deliberate: the
  connection manager no longer has an `except AttributeError: pass` branch for a
  "validator not available" build — any validator crash rejects the wire (the
  tolerated-absence test was dropped). Tests: `tests/unit/test_diagram_preflight.py`
  (19). Follow-up worth its own round: `validate_block_connections`' duplicate-
  input check overlaps `DiagramValidator._check_duplicate_connections`.
- [x] **`SimulationController._print_terminal_verification`** (was 225 lines,
  C901 = 40) — done 2026-09-10. The report lives in
  `lib/engine/verification_report.py`: collectors (`collect_display_values`,
  `collect_state_variables`, `collect_scope_convergence`), judgement
  (`classify_scope`, `state_variable_lines`, `scope_lines`) and
  `build_verification_report(blocks) -> VerificationReport(text, passed, has_data)`,
  plus `report_blocks(dsim)` for the active-list-else-blocks_list choice. The
  controller method is ~15 lines that log the text. New CLI flag `run --verify`
  prints the same report and exits 3 on a failed check (README, USER_MANUAL,
  CHANGELOG). Text format unchanged. Tests: `tests/unit/test_verification_report.py`
  (44), `tests/integration/test_cli.py::TestVerifyFlag` (3). One fix folded in:
  Display blocks always reported `---` because the collector read
  `block.params["_display_value_"]` while the block writes it into the dict
  `execute()` receives (`exec_params`); it now reads through `runtime_params`
  like the renderer.
- [ ] **`SubsystemManager.create_subsystem_from_selection`** (406 lines,
  C901 = 43); `ClipboardManager.paste_blocks` (228); `solve_with_events` (232).
  Large but routine; extract when next touched.
- [ ] **Palette glyphs** — `modern_palette._draw_glyph` (156 lines, C901 = 36)
  is a second icon system with its own switch; reuse the blocks' `draw_icon`
  paths (mapped into the palette tile) so a block's icon is defined once.
- [x] **Single-source the PDE finite-difference/BC kernels** shared by the blocks
  and `SystemCompiler` — done 2026-07-05: shared pure ops in `lib/engine/pde_ops.py`
  consumed by both `blocks/pde/*` and `lib/engine/compiler_kernels/pde.py`
  (compiled goldens bit-identical; `boundary_mode` kwarg where the paths
  legitimately differ on Dirichlet handling).
- [x] **`lib.py` interpreter hot path** — partial, 2026-07-05: the two
  near-verbatim per-step loops (`execution_loop` / `execution_loop_headless`)
  are now one shared `DSim._interpreter_step(interactive)` (UI side effects
  gated, numerics identical); `update_global_list`'s per-execution O(blocks)
  name scan replaced with an identity-tracked dict index; the
  `max_hier`/`rk45_len`/`rk_counter` engine→DSim re-copies replaced with
  property bridges (engine is the single owner). NOT done (deliberate): the
  hierarchy fixpoint re-scan itself (worst-case O(blocks²) readiness checks
  per step, each block still executes once) — replacing it with a precomputed
  topological order would change discrete/memory-block ordering semantics and
  needs its own design + tests; the `DSim` facade stays (its delegation is
  the intended MVC bridge).
- [x] **`modern_canvas` god object / manager-layer consolidation** — first
  pass, 2026-07-06: merged `DragResizeManager` into `InteractionManager`
  (one gesture pipeline; it already imported `State` and drove
  `canvas.state`), deleted the canvas's duplicate ~70-line `_paste_blocks`
  (context-menu paste now goes through `ClipboardManager.paste_blocks(pos)`
  and preserves connections, matching Ctrl+V), moved `_validate_connection`
  into `ConnectionManager` (killing the manager→canvas validation callback),
  and single-sourced the rect-selection block pass in `SelectionManager`.
  Canvas managers 9→8; all moves behavior-preserving behind the GUI suite.
  Done 2026-07-19: the 63 `canvas_state` property proxies removed — each state
  slice now lives with its owning manager (zoom/pan → ZoomPanManager, gesture
  → InteractionManager, connection → ConnectionManager, validation →
  RenderingManager), `CanvasState` dissolved with grid the one canvas-owned
  slice (`ModernCanvas.grid`); the canvas keeps only read-only `zoom_factor`/
  `pan_offset` getters plus the grid getters/setters. See
  `tasks/canvas_state_ownership_scope.md` for the measured data and plan.
  Remaining (separate, larger design rounds): the trivially small main-window
  managers (`window_setup_manager`, `view_actions_manager`,
  `property_controller`, ...) which the 2026-06-13 review rated severity-low.
  Post-merge cleanup 2026-07-19 (4-agent /simplify pass): managers now call the
  slice-dataclass transition methods instead of inlining them, `end_connection()`
  carries the idle-glow gate re-eval, dead `reset_gesture_state()` deleted,
  stale proxy-era comments fixed. Deferred polish (low severity, needs its own
  round): semantic pan lifecycle on ZoomPanManager (`begin_pan/pan_by/end_pan` —
  middle-button pan writes `state.is_panning` from InteractionManager) and a
  `center_on(world_point)` to dedup the pan-centering math (main_window
  center-on-error / fit-to-window / minimap click); `drag.offset` doubles as an
  absolute click pos for line point/segment drags (dedicated field or
  `start_line_item_drag()` API); `dragging_block`/`dragging_item` still plain
  canvas attrs outside `DragState`; `interaction_manager.state` (FSM proxy) vs
  `zoom_pan_manager.state` (owned slice) naming collision;
  `_evaluate_animation_state` is cross-manager API but underscore-private.
- [x] Break the `lib/` ↔ `modern_ui/` import layering via dependency inversion
  (move shared theming into `lib`), instead of function-local imports — done
  2026-07-18: the design-token/theme system moved to `lib/theming/theme_manager.py`
  (only stdlib+PyQt5 deps); `modern_ui/themes/theme_manager.py` is now a
  backward-compat shim re-exporting the same singleton so all ~71 consumer sites
  are untouched. The 5 `lib/` modules (`plotting/signal_plot.py` — the sole
  module-level offender — `models/simulation_model.py`, `services/diagram_service.py`,
  `simulation/menu_block.py`) now import from `lib.theming`. `grep modern_ui lib/`
  is clean except the genuine `AnimationExportDialog` UI-widget import in
  `field_scope_mixin.py` (a real UI dependency, left function-local; inverting the
  plotting layer itself is a separate, larger move).
- [x] Add compiled-vs-interpreted equivalence tests for each compiled stateful
  block (RateLimiter, PID, TransportDelay, Selector) and an all-Neumann 2D PDE
  corner integration test — done 2026-07-05: `tests/regression/test_equiv_*.py`.
  Note: TransportDelay turned out not to be compilable (not in
  `COMPILABLE_BLOCKS`; history-dependent), so its test pins the interpreter
  fallback + analytic delayed-sine instead.

### Interpreter-path bugs found by the equivalence tests — all fixed 2026-07-05

Found via the strict-xfail tripwires; compiled path was verified correct in
every case, so each fix corrected the interpreter to match.

- [x] **RateLimiter slewed at 2× the configured rate in the interpreter.**
  As a memory block it runs `execute()` twice per step (output_only + state
  pass) and advanced `params['_prev']` on both. Fixed by returning the held
  output without advancing on the output_only pass
  (`blocks/rate_limiter.py`). `test_equiv_ratelimiter.py` now passes.
- [x] **Interpreter state blocks integrated at dt=0.01 regardless of sim_dt.**
  `run_tuning_simulation` never synced `engine.sim_dt`, so
  `initialize_execution` re-stamped every block's `exec_params['dtime']` with
  the engine's default 0.01. Fixed by calling `engine.update_sim_params`
  before `initialize_execution` in `run_tuning_simulation` (`lib/lib.py`);
  the interactive path already did this. TF/PID/Integrator now discretize at
  the real sim_dt. (The PID trajectory `test_equiv_pid.py` xfail remains —
  its residual divergence is the memory-block feedback delay + derivative
  kick, not dtime, and it runs at dt=0.01 where the clobber was masked.)
- [x] **Selector comma-list indices were broken through the run pipeline.**
  `WorkspaceManager.resolve_params` safe_expr-evaluated `"1,2"` to a tuple:
  the interpreter crashed in `_parse_indices` and the compiled kernel
  silently fell back to index 0. Fixed with `normalize_indices_str`
  (`blocks/selector.py`), used by both the block and the compiled kernel
  (`lib/engine/compiler_kernels/nonlinear.py`). Locked in by
  `tests/unit/test_selector_comma_indices.py`.
- [x] **2D PDE blocks never integrated in the interpreter.** Their
  `execute()` only reshaped the compiled-replay `state` kwarg and returned
  the IC forever. Fixed by adding params-persisted Forward-Euler stepping
  (`_interp_step`, output-then-step so samples align with the compiled path)
  to HeatEquation2D / WaveEquation2D / AdvectionEquation2D, reusing each
  block's `compute_derivatives` (single-sourced through `pde_ops`).
  `test_equiv_pde_neumann2d.py` now passes (both paths track within 1e-2).

### CI follow-up
- [x] Dedicated `ruff format` commit (414 files), then add
  `ruff format --check .` to the CI lint job — done 2026-07-05. Ruff pinned
  to 0.15.18 in the lint job (bump the pin together with a reformat);
  E701/E702 ignores retired from `pyproject.toml` (formatter guarantees them).

### Foundation hardening from the external review (2026-07-05)

- [x] **Headless CLI** (`lib/cli.py`): `python diablos_modern.py run
  diagram.diablos -o out.csv [--time --dt --solver interpreter]` runs a diagram
  without the GUI and exports Scope traces to CSV/NPZ. Defaults to the compiled
  path. Tests: `tests/integration/test_cli.py`.
- [x] **Analytic-solution regressions** for the compiled path
  (`tests/regression/test_analytic_solutions.py`): Integrator ramp, first-order
  lag step response, 1D heat eigenmode decay — each asserted against the closed
  form, so a kernel regression fails with a physical meaning.
- [x] **Documented the two-engine semantics contract** in
  `docs/ARCHITECTURE.md` (§3c): compiled = accurate source of truth, interpreter
  = fixed-step/full-coverage; the by-design differences (transient vs steady
  state, feedback delay, memory-block two-pass rule, PDE self-integration).
- [x] **safe_eval audit**: `lib/safe_eval.py` is an allowlist AST interpreter
  (no eval/exec/`__import__`, allocation guard, bounded array ctors); verified it
  blocks dunder traversal, lambdas, comprehensions, and arbitrary builtins.
  Combined with the `external.py` stub, the review's "remove eval()" concern is
  already addressed — no fix needed. Residual: `resolve_params` eval-ing every
  string param is a correctness footgun (it caused the Selector tuple bug),
  mitigated per-block via `normalize_indices_str`; a general opt-out is deferred.

---

## References

- `tasks/code-quality-review-2026-06-13.md` - Full whole-app review (334 findings)
- `docs/archive/REFACTORING_TODO.md` - Detailed refactoring history (archived)
- `docs/PDE_ROADMAP.md` - Full PDE enhancement roadmap with architecture diagrams
- `CLAUDE.md` - Project overview and recent work
