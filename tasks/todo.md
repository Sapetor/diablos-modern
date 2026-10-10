# DiaBloS Modern - Consolidated TODO

> Single source of truth for all pending work items.
> Last updated: 2026-10-10

---

## Open Items

Everything below is still open, verified against the code where cheaply
checkable (2026-10-02 pass noted per item).

### Parked feature ideas (idea review, September 2026)
- [ ] **Block-diagram algebra** — select a source and a sink, show the
  closed-loop transfer function symbolically with reduction steps (builds on
  `symbolic_execute()`). (confirmed absent 2026-10-02)
- [ ] **Run comparison overlay** — keep a run as a ghost trace in the Scope /
  waveform inspector *across a parameter sweep* (reuse the Monte Carlo
  harvesting in `lib/analysis/resim.py`). Note: a related but distinct feature
  already ships — the Scope's "Previous run" checkbox (2026-07-12,
  `ScopePlotter._stash_run`/`reset_held_runs`) overlays the immediately prior
  run of the *same* diagram. This item is specifically about comparing across
  changed parameters, not just reruns.
- [ ] **System identification block** — fit an n-th order transfer function to
  `FromFile` data (`FromFile` + `DataFit` already provide the pieces).
  (confirmed absent 2026-10-02)
- [ ] **Interactive source blocks** — Slider/Knob source and Manual Switch
  adjustable on the canvas during a run. (confirmed absent 2026-10-02; only a
  static `blocks/switch.py` exists)
- [ ] **Hardware I/O** — UDP and serial source/sink blocks for lab rigs
  (Arduino etc.). (confirmed absent 2026-10-02). Note: wall-clock **real-time
  pacing** itself already shipped separately — `DSim.real_time` /
  `SimulationEngine.real_time`, a "Run in real-time" checkbox in Simulation
  Settings (`lib/dialogs.py`) — so only the hardware-I/O blocks remain here.
- [ ] **Semantic diff for `.diablos` files** — CLI subcommand diffing by block
  and connection rather than by JSON line. (confirmed absent 2026-10-02)
- [x] **PathSim benchmark** (2026-10-10) — `docs/BENCHMARK_PATHSIM.md`,
  harness in `scripts/benchmarks/pathsim/` (separate venv, pathsim 0.27.0).
- [x] **Sampled blocks behind a ZOH acted one period late** (2026-10-10,
  `68e8289`), found by the PathSim benchmark (S1). Consumers of feedthrough
  memory blocks now wait for the producer's fresh output each step. Tests:
  `tests/regression/test_sampled_loop_delay.py`. The rate-inheritance
  staircase in `test_discrete_block_sample_time.py` had pinned the late
  response.

### Documentation
- [ ] **Video tutorials** — demo videos for key features.

### Open items from the October 2026 audit (2026-10-08)
Fixed the same day: undo/redo, between-run state, MC seeds, nested MC/sweep
(pickers + top-level root), selectable PID anti-windup (clamping default),
label clipping / port-label overlap / stale status + Properties counts, block
None/empty-input hardening and param docs. Still open:
- [ ] **UI scale only resizes the app font (found 2026-10-10).** Preferences >
  UI scale sets the app font to 10pt * factor and nothing else. Widgets whose
  QSS sets a point size from the TYPE tokens ignore it: menu bar, palette rows
  and headers, Properties panel, transport, toolbar icons. Only default-font
  widgets grow (status pill, zoom readout, search pill, filter box, "Main", the
  status-bar file name), so 150% looks patchy rather than scaled. Needs a decision:
  a true global scale (e.g. `QT_SCALE_FACTOR` set before `QApplication`, which
  also scales the canvas and every pixel literal and stacks with Windows DPI) or
  font-ratio scaling in every TYPE consumer (as the welcome overlay now does).

### Open items from the Windows session (2026-10-09)
All closed the same day (see the 2026-10-09 Change Log rows): resize,
Windows style, cp1252 logging, toolbar/status/glyph test portability, saved
undo state, Properties panel polish, palette drop lag, Enter ranking,
palette sideways shift, test isolation from the real QSettings store,
`remove_block` in subsystems, and a Windows CI leg. Nothing left open.

### Open gaps left by the September 2026 campaigns
- [ ] **Stateful user-block kernels** — a user block with a registered
  `@kernel` now compiles, but only algebraically: the compiler allocates ODE
  states per built-in fn (`_allocate_states` in `lib/engine/system_compiler.py`),
  so a user block cannot declare states. A `state_size`/initial-state hook on
  `BaseBlock` would let integrating user blocks compile too.
- [ ] **Stiffness diagnostic caps at 64 states** — `STIFFNESS_MAX_STATES` in
  `lib/engine/solver_diagnostics.py` (confirmed still `= 64`, 2026-10-02) skips
  the Jacobian eigen-analysis on larger systems (PDE diagrams), so they get no
  solver suggestion. Use a sparse/power-iteration estimate or sample the
  spectrum instead of skipping.
- [ ] **Event-triggered state reset** — zero-crossing detection locates
  switching instants but a block cannot yet reset an integrator state at the
  event (needed for bouncing-ball / impact examples).
- [ ] **pyqtgraph teardown segfault** in
  `tests/unit/test_linearization_result_window.py` (path corrected 2026-10-02;
  the file lives under `tests/unit/`, not `tests/modern_ui/` as previously
  noted here) is only dodged (module-scoped fixture + `gc.collect()`); the
  result window's own teardown is still unfixed. See `tasks/lessons.md`,
  "pyqtgraph teardown segfaults appear only in the full suite".
- [ ] **Follow-up: recreate the x86_64 build env.** `~/opt/anaconda3/envs/diablos_x86`
  (`docs/building.md`) is still Python 3.9 (Anaconda, confirmed via
  `docs/building.md` 2026-10-02), which `requires-python = ">=3.10"` now
  refuses, so the next **x86_64** macOS release will not build until that env
  is recreated on 3.10+. The arm64 release env (`~/.venvs/diablos-arm64/`,
  3.12) and the CI Windows/Linux builds are unaffected.

### Future / Roadmap — PDE Phase 2: Mesh Abstraction
- [ ] Create `MeshBase` abstract class
- [ ] Create `blocks/pde/mesh/` directory structure
- [ ] Refactor 2D PDE blocks to use mesh interface
- [ ] Add curvilinear mesh support

### Future / Roadmap — PDE Phase 3: Unstructured Meshes
- [ ] Mesh loader block (read .msh, .vtk, .stl)
- [ ] Mesh generator block (circle, L-shape)
- [ ] Mesh exporter block (VTK for ParaView)
- [ ] FEM-based spatial operators (P1 triangles)
- [ ] Update FieldScope2D for triangulation rendering
- [ ] Add FieldExportVTK block

### Future / Roadmap — PDE Phase 4: Advanced Features
- [ ] 3D PDE support (HeatEquation3D, WaveEquation3D)
- [ ] Absorbing BCs / PML for wave equations
- [ ] Adaptive mesh refinement
- [ ] Domain decomposition for parallel solving

---

## Completed

### October 2026 - Windows follow-up (2026-10-09)
- [x] **Review of the five commits `cf44b1f..cff6e81`:** no introduced
  correctness regression found. Checked minimum-size callers, multi-port
  layout, file-load height restoration, Fusion theme handling, cp1252 stream
  escaping and missing stdout/stderr in windowed builds. Resize tests and
  frozen-write checks pass except the known POSIX-permissions test.
- [x] **Toolbar overflow at the 1200px minimum width:** reduced the toolbar
  status cap to 300px. Native Windows size hint with a long message dropped
  from 1212px to 1150px; all four existing width regressions now pass.
- [x] **Platform-fragile status/glyph regressions:** the Windows offscreen
  plugin has no system font database. The red-pill test now accepts an
  elided diagnostic with the full text in its tooltip; the glyph test checks
  actual painter text and rendered ink instead of requiring different
  missing-glyph images. No tests skipped or marked xfail.
- [x] **Undo/redo to the saved state clears dirty:** history tracks the
  persisted content of the whole diagram, including nested subsystems and
  simulation settings. GUI/core saves update the marker; load/new clear old
  history and establish a clean origin. Autosaves and failed/cancelled saves
  leave it alone. Selection, theme colors and execution scratch data do not
  count. Regression coverage in `tests/regression/test_saved_history.py`;
  49 saved-state/history/dirty/new checks pass.
- [x] **Properties panel polish:** reset buttons use a Qt reload icon and
  zero padding, avoiding the inherited 24px padding that hid the symbol;
  Name, parameter and editable-port labels use 12px bold text. Native Windows
  before/after renders checked; all 40 related editor tests pass.

### September 2026 — 1.1.0 release
Everything since the local `v1.0.0` tag (2026-09-03) lived on `feat/maturity`
(~72 commits) plus the CHANGELOG `[Unreleased]` section: zero-crossing
detection, Spanish i18n, library blocks with masks, drag-to-connect + wire
editing + block shapes, validation suite, solver semantics and stiffness
guidance, examples gallery test, block API / user blocks, docs site.
- [x] Full suite + `ruff check .` + `ruff format --check .` green on
  `feat/maturity` (2026-09-10: 4064 passed / 28 skipped / 1 xfailed, ruff clean).
- [x] `pyproject.toml` bumped to 1.1.0; CHANGELOG `[Unreleased]` → `[1.1.0]`.
- [x] `feat/maturity` merged into `main` as a fast-forward (`main` had not
  moved); `main` + tags `v1.0.0`/`v1.1.0` pushed. `release.yml`/`docs.yml`/CI
  (3.9, 3.12, ruff) all green — release v1.1.0 published with an arm64 DMG,
  linux tar.gz, and windows zip.
- [x] Deleted the merged `feat/*` agent branches (`i18n-spanish`,
  `zero-crossing`, `library-blocks`, `i18n-masks`, `i18n-events-libraries`,
  `block-api`, `docs-site`, `examples`, `solver-semantics`, `validation`,
  `validation-bugfixes`, `wire-routing-and-block-shapes`); kept
  `backup/pre-merge-main` (200 commits that exist nowhere else).
  `feat/paper` (JOSS draft) and `feat/maturity` are now **also** fully merged
  into `main` and slated for deletion (status as of 2026-10-02).

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
- [x] **PDE Phase 1: Quick Wins** (from `PDE_ROADMAP.md`):
  - **Periodic BCs** added to HeatEquation1D/2D and WaveEquation1D/2D, shared
    via `lib/engine/pde_ops.py` (`is_periodic`) so the interpreter blocks and
    `lib/engine/compiler_kernels/pde.py` are equivalent by construction.
  - **Dynamic BC coefficients** - time-varying Robin `h` via input ports
    (`heat_equation_1d.py` ports 3/4, `heat_equation_2d.py` ports 5-8);
    unconnected ports fall back to the matching param.
  - **More initial conditions** - 1D `linear`/`step`/`random`; 2D adds
    `checkerboard`/`radial`; single-sourced in `lib/engine/pde_helpers.py`.
  - **Robin BC for 2D** - HeatEquation2D supports Robin on any edge.
  - Tests: `tests/unit/test_pde_phase1.py`, `tests/regression/test_equiv_pde_phase1.py`.
    Details: `docs/PDE_ROADMAP.md` (Phase 1), `docs/wiki/PDE.md`.
- [x] **Python code generation** - `lib/export/python_codegen.py`, the
  File > Export > Export as Python Script... menu entry, and the
  `export-python` CLI subcommand (see Feature campaign > Code generation).
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

### September 2026 — Feature campaign (i18n, zero-crossing, library blocks, misc)

**Teaching & Interaction**
- [x] **Live parameter tuning** — Manipulate-style interactive tuning: pin block parameters to a tuning panel, drag sliders and watch scope plots update in real-time via headless re-simulation. Supports float params and individual list elements (e.g., transfer function coefficients). Scope window stays on top during tuning. Right-click slider rows to set custom range.
- [x] **Custom Python Function block** — `blocks/function.py` ("Function"). Users type an expression of the inputs and time (e.g. `sin(u[0]**2) + u[1]`). Inputs exposed as 0-indexed `u[i]` and 1-indexed `u1`/`u2`; `t` is sim time. Variable input-port count via `io_editable='input'`. Evaluated through the hardened `safe_expr` AST walker (numpy math allowed; imports/attribute escapes rejected). List expressions yield vector outputs. Diagrams containing it use the interpreted engine (not in `COMPILABLE_BLOCKS`). Tests: `tests/unit/test_function_block.py` (19), integration chains in `test_simulation_execution.py`.
- [x] **Diagram-to-LaTeX/TikZ export** — Export block diagrams as TikZ figures for papers and lecture notes. Saves hours of redrawing diagrams for ACC/IFAC publications. File > Export > Export as TikZ... with live preview, clipboard copy, and configurable options.

**Research & Data**
- [x] **Data Import block** — `blocks/from_file.py` (`FromFile`, category Sources) replays a recorded CSV/NPZ/MAT/TXT time-series as a driving signal, interpolated (`linear`/`zoh`/`nearest`) onto the sim grid with `hold`/`loop` end behavior; parsed data cached in `params`. Shared reader: `lib/services/timeseries_loader.py` (`load_timeseries`, `allow_pickle=False`), also used by `blocks/optimization/data_fit.py`. Companions: `blocks/lookup_table.py` (1-D/2-D static maps). Tests: `tests/unit/test_from_file.py` (13), `test_timeseries_loader.py` (10), `test_lookup_table.py` (16).
- [x] **Linearization tool** — `lib/analysis/linearizer.py` computes A/B/C/D by finite differences over the compiled ODE RHS; `modern_ui/controllers/analysis_controller.py` assembles the result dict (poles/zeros, Bode, margins, step/impulse, controllability/observability). UI: Analysis > **Linearize & Analyze...** (`modern_ui/widgets/linearize_dialog.py` → `linearization_result_window.py`, 5 tabs + Python/MATLAB/.mat/.npz export via `lib/analysis/linearization_export.py`) and Analysis > **Find Operating Point (Trim)...** (`operating_point_window.py`). The BodeMag/BodePhase/Nyquist/RootLocus/LQR blocks remain the per-block path (`lib/analysis/analyzers/`).
- [x] **Code generation** — File > Export > **Export as Python Script...** (`MainWindow.export_python_script`) and the `export-python` CLI subcommand emit a self-contained numpy+scipy `.py` via `lib/export/python_codegen.py`. The script mirrors the compiled solver (one `rhs(t, x)` under `solve_ivp`, same three-group order) and its `--out` CSV/NPZ matches `python -m lib.cli run` column-for-column. Supported set: `SUPPORTED_BLOCK_NAMES` (19 blocks); unsupported blocks, discrete sample times and algebraic loops raise `CodegenUnsupportedError`/`CodegenError` instead of emitting broken code. Tests: `tests/unit/test_python_codegen.py`, `tests/modern_ui/test_export_python_menu.py`.

**UI Polish**
- [x] **Dark mode fixes** — Fixed invalid theme key lookups (`accent`, `block_fill`, `connection_line` → proper keys). Property editor "Documentation" title, command palette, minimap all now use correct theme colors. Error panel severity backgrounds are theme-aware. Canvas selection rect, connection preview, and error indicators use theme colors. Block renderer icons use `block_icon_color` theme key.
- [x] **Compact toolbar** — Switched to `ToolButtonIconOnly` with `setIconSize(20,20)`, 14px emoji font, 70px zoom slider. Removed redundant status label (main status bar already exists). Theme button always visible at default window size on macOS. Theme also accessible via View > Toggle Theme (Ctrl+T).
- [x] **Minimap** — `modern_ui/widgets/minimap_widget.py`: dockable overview panel (left/right) with a viewport rectangle; click to pan, drag for continuous panning. View > Minimap or `Ctrl+Shift+M` (`MainWindow.toggle_minimap`). Coverage: `tests/modern_ui/test_main_window_view_actions.py`.

**From the idea review (parked list), picked for this campaign**
- [x] **Stiff solver choice in the UI** — already shipped: the Simulation
  Settings dialog (`lib/dialogs.py`, `SOLVER_METHODS`) offers
  RK45/RK23/DOP853/Radau/BDF/LSODA/RK4/Euler/auto with rtol/atol; `auto`
  resolves to LSODA. The headless `run` command reads the file's solver
  settings and accepts `--method/--rtol/--atol`.
- [x] **User blocks directory for frozen builds** — `lib/user_blocks.py`
  (`user_blocks_dir`) is scanned at startup, user blocks appear in the palette
  with a reload action and a missing-block warning.
- [x] **PyQt6 migration** — done 2026-09-10: the GUI, the test suite, the
  PyInstaller spec and every dependency list moved to PyQt6 6.7+ (see the
  `[Unreleased] → Changed` entry in CHANGELOG.md). Released as v1.1.0.
- [x] **Bump the Python baseline to 3.10 and drop the `PyQt6<6.11` marker** —
  done 2026-09-19. Python 3.9 reached EOL in October 2025, and the stated
  reason for the pin was already stale: `readthedocs.yaml` had been building on
  3.12 all along, so the CI matrix leg was the only real consumer. Collapsed
  `PyQt6` to a single `>=6.7` line (`requirements.txt`, `pyproject.toml`),
  raised `requires-python` to `>=3.10`, ruff `target-version` to `py310`, CI
  matrix to `["3.10", "3.12"]`, and the 3.9 mentions in `README.md`,
  `readthedocs.yaml`, `docs/building.md`. (x86_64 build-env follow-up is still
  open — see Open Items.)

### October 2026 — library and user blocks on the fast solver (2026-10-02)
- [x] **Library / user blocks are not compilable** — half stale. *Library*
  blocks already compiled: an instance is a copied masked Subsystem, which
  `check_compilability` walks recursively and `Flattener` resolves with the mask
  scope (`tests/integration/test_masked_library_example.py` runs both engines).
  *User* blocks never compiled, even with a registered `@kernel`, because
  `COMPILABLE_BLOCKS` was the only gate. `SystemCompiler._has_user_kernel` now
  admits a user block whose kernel is registered, and `compiler_kernels._register`
  refuses to let a user module shadow a built-in kernel or event builder.
  Covered by `tests/regression/test_user_block_kernel.py`; docs in
  `docs/BLOCK_API.md` section 7.

### October 2026 — remaining UI strings translated (2026-10-02)
- [x] **Outcome-metric labels and validator messages translated** — new
  `OUTCOME_METRIC_LABELS` (`lib/analysis/resim.py`, marked with `tr_noop`) is
  translated at display time in the sweep/ensemble windows, whose metric combos
  now carry the English key as item data; `lib/diagram_validator.py` messages
  and suggestions go through `tr()`. 43 new `es.json` keys.

### September 2026 — newly verified completed items (2026-10-02 pass)
Found already fully implemented while tidying this file; no code changes made,
just evidence that these can come off the "parked ideas" backlog:
- [x] **Autosave and crash recovery** — `modern_ui/managers/project_manager.py`
  (`check_autosave_recovery`, `recover_autosave`, `cleanup_autosave`) plus a
  2-minute `QTimer` in `modern_ui/main_window.py` (`_auto_save`,
  `autosave_timer`); recovery is offered on startup
  (`QTimer.singleShot(500, self._check_autosave_recovery)`), the autosave file
  is removed on clean exit, and an in-flight autosave never clears the
  unsaved-changes flag.
- [x] **Sample-time coloring (blocks)** — see "Parked feature ideas" above;
  the block-side indicator already shipped well before the Sept 2026
  campaigns (predates `ab8b003`, multi-rate support with RateTransition/
  FirstOrderHold). Wire tinting followed on 2026-10-10.
- [x] **Wall-clock real-time pacing** (half of the "Real-time pacing and
  hardware I/O" idea) — `DSim.real_time` / `SimulationEngine.real_time`, a
  "Run in real-time" checkbox in `lib/dialogs.py`, consumed by
  `modern_ui/controllers/simulation_controller.py`. Hardware I/O (UDP/serial
  blocks) is the only part of that idea still open.

### September 2026 — Compiled-path PID dataflow ordering fix (2026-09-19)
Re-verified and fixed a reported compiled-path bug: an error-input PID loop
(`Step → Sum(+,-) → PID(port 0 only) → TranFn 1/(s+1) → Sum`) froze at exactly
zero on both engines, for two independent reasons.
- **Compiled path (the real bug)**: the middle execution group inherited the
  engine's memory-block-aware hierarchy order verbatim, so `exec_pid` read the
  Sum's output *before* it was written on every RHS evaluation — permanently
  zero error. Fixed with a new `_dataflow_order` (stable Kahn sort over
  middle-group edges) in `lib/engine/system_compiler.py`.
- **Interpreted path (a second bug found on the way)**: `blocks/pid.py`
  returned the stale `_last_output_` whenever the measurement port was
  unconnected, so the error-input wiring never computed at all; an
  unconnected port now reads 0.0, matching the compiled kernel.

Both paths now settle to the setpoint (0.999991 interpreted vs 0.999992
compiled). Tests: `TestErrorInputPIDOrdering` in
`tests/regression/test_equiv_pid.py`. Merged to main (fee40f4), CI green.

### Follow-ups (flagged September 2026, since resolved)
- [x] **PDE BC params should be dropdowns.** Done: the specs in
  `lib/engine/pde_helpers.py` carry `options` (Dirichlet/Neumann/Robin/Periodic),
  the property editor renders any param with `choices`/`options` as a QComboBox,
  and `tests/regression/test_choice_param_dispatch.py` covers HeatEquation1D/2D.

### Code Quality Review (2026-06-13) — follow-ups

A whole-app review confirmed 334 findings. All concrete defects, correctness,
and performance items with a safe fix are now **fixed** (see
`tasks/code-quality-review-2026-06-13.md` and its `-deferred.md` companion).

**Quick fixes**
- [x] Impulse/Step-`impulse` vs adaptive solver — routed to the interpreter path.
- [x] Vectorize per-node Python loops in the compiled PDE RHS (1D & 2D).
- [x] `Scope`/`Export` O(n²) per-step concat → amortized-O(1) geometric buffers.
- [x] Compiled heat **Robin BC** reconciled with the interpreted block.
- [x] `base_analyzer` discrete PID TF (c2d); `integrator` configurable method.
- [x] `connection` routing from port orientation; `draw_grid`/minimap/block_renderer caching.
- [x] Unified 1D/2D PDE `compute_derivatives` signatures.
- [x] Extracted `MainWindow._init_core_managers` (constructor altitude).

**Refactoring round (2026-09-10, ranked by `ruff check --select C901` complexity)**
Eleven functions in `lib/`, `modern_ui/`, `blocks/` exceeded complexity 25; all
eleven are now fixed, each verified by trace-diffing old vs new output across
the full example set (bit-identical) unless noted otherwise. Full narrative
(exact helper names, before/after line counts, every verification scenario) is
in git history / `tasks/lessons.md` — this is the condensed index:
- [x] **`block_renderer._draw_legacy_icon`** (356 lines, C901=52 → table-driven
  `_draw_icon_text`) — was double-drawing 14 block icons on top of their own
  `draw_icon()`. Renderer 1514→1255 lines. Tests:
  `tests/modern_ui/test_block_renderer_icon_text.py`.
- [x] **`replay_compiled_signals`** (616 lines, C901=100, worst in the repo →
  ~70-line orchestrator + `lib/engine/replay_handlers.py`). Tests:
  `tests/unit/test_replay_handlers.py` (40).
- [x] **`SystemCompiler.compile_system`** (434 lines, C901=41 → ~40-line phase
  sequence: `_build_input_map`, `STATE_ALLOCATORS`, `SOURCE_FNS`/`STATE_FNS`,
  `_execution_groups`, `_make_model_func`). Tests:
  `tests/unit/test_system_compiler_phases.py`.
- [x] **`DSim._interpreter_step`** (296 lines, C901=41 → phase methods:
  `_advance_clock`, `_publish_memory_outputs`, `_run_hierarchy_passes`,
  `_finish_run`). Folded-in fix: `SimulationEngine.execute_block` now returns
  `{"E": True, "error": ...}` when a block's `execute()` raises, instead of a
  bare bool that crashed `_block_failed` with a `TypeError` (hit by
  `van_der_pol_stiff` on the interpreted path). Tests:
  `tests/unit/test_interpreter_step_phases.py` (39),
  `tests/unit/test_execute_block_errors.py` (6).
- [x] **`DSim` facade** (`lib/lib.py`, 1825→1736 lines) — deleted 13 uncalled
  one-line delegations; removed the pygame-era `lib/ui/` package entirely
  (`Button`/`buttons_list`/the vestigial `.active` flag in `execution_init`
  had no reader). Kept the property bridges and MVC-seam entry points
  deliberately (GUI/test/doc callers).
- [x] **`lib/improvements.py`** (448 lines) — deleted. Live pieces relocated
  (`PerformanceHelper` → `modern_ui/perf_helper.py`; validation/safety checks →
  plain functions in `lib/diagram_validator.py`); dead config/demo code
  removed. Deliberate behavior change: the connection validator no longer
  silently tolerates a missing validator. Tests:
  `tests/unit/test_diagram_preflight.py` (19).
- [x] **`SimulationController._print_terminal_verification`** (225 lines,
  C901=40 → report logic moved to `lib/engine/verification_report.py`
  collectors + `build_verification_report`; new `run --verify` CLI flag).
  Folded-in fix: Display blocks' collector now reads through `runtime_params`
  instead of the wrong dict (fixed a `---` placeholder bug). Tests:
  `tests/unit/test_verification_report.py` (44).
- [x] **`SubsystemManager.create_subsystem_from_selection`** (406 lines,
  C901=43 → ~90 lines, C901≤6; four near-identical port-creation blocks
  collapsed into `_add_inport`/`_add_outport`). Tests:
  `tests/unit/test_subsystem_creation_helpers.py`.
- [x] **`ClipboardManager.paste_blocks`** (228 lines, C901=23 → 28-line
  orchestrator). Four real bugs fixed along the way: (a) pasting a Subsystem
  dropped its connections (ports never restored); (b) masking/renaming a
  subsystem created from a selection didn't stick
  (`lib/masks.py::_has_default_username`); (c) a failure mid-paste left
  orphaned blocks + a dangling undo entry (added rollback); (d) **wider bug**:
  `FileService._construct_block` had the same flipped-port defect, so *every
  saved diagram with a flipped block reopened with swapped ports* — fixed by
  making `DBlock.flipped` a property whose setter re-lays ports. Tests:
  `test_clipboard_paste_phases.py` (26), `test_clipboard_paste_subsystem.py`
  (6), `test_clipboard_paste_rollback.py` (6), `test_mask_default_label.py`
  (13), `test_flipped_block_ports.py` (5).
- [x] **`solve_with_events`** (`lib/engine/zero_crossing.py`, 232 lines,
  C901=25 → ~80-line loop, C901=9, over a `_SegmentLoop` dataclass). Fixed
  2026-09-22: `_restart_point`'s `nextafter` now steps toward `np.inf` instead
  of `tf+1.0` (was a no-op at extreme |t|). Tests:
  `tests/unit/test_zero_crossing_phases.py` (28).
- [x] **Palette glyphs** (`modern_palette._draw_glyph`, 158 lines, C901=36 →
  9-line lookup, C901=3, into `_GLYPHS`). Two bugs fixed along the way: a
  shadowed-prefix table bug (MatrixGain/Export/Demux showed the wrong glyph);
  and (2026-09-22) canvas fill / palette chip / category dot unified into one
  table, `lib/theming/categories.py::category_theme_key`. Declined: reusing
  blocks' `draw_icon()` paths in the palette (spiked, rated not worth it at
  the current 22px tile size). Tests:
  `tests/modern_ui/test_palette_glyph_registry.py` (55).
- [x] **Test-suite ordering fragility** — found and fixed the same day: a
  module-scoped `qapp` fixture in one test file was garbage-collecting the Qt
  `ThemeManager` singleton for later tests. Fixed by making `qapp`
  session-scoped + autouse; rule: never build a `QApplication` in a test file.

**Earlier architectural work (2026-07)**
- [x] **`lib.py` interpreter hot path** — partial, 2026-07-05: merged
  `execution_loop`/`execution_loop_headless` into `DSim._interpreter_step`;
  O(blocks) name scan → identity-tracked dict index; engine↔DSim re-copies →
  property bridges. Deliberately not done: the O(blocks²) hierarchy fixpoint
  re-scan (would need its own design + tests).
- [x] **`modern_canvas` god object / manager-layer consolidation** — first
  pass 2026-07-06 (`DragResizeManager` merged into `InteractionManager`,
  duplicate `_paste_blocks` removed, validation moved into
  `ConnectionManager`); 63 `canvas_state` proxies removed 2026-07-19 (state
  moved to owning managers, `CanvasState` dissolved); `/simplify` cleanup
  2026-07-19. See `tasks/canvas_state_ownership_scope.md` for the measured
  data, plan, and the remaining low-severity deferred polish items (pan
  lifecycle on `ZoomPanManager`, a `center_on()` dedup, a couple of naming
  collisions).
- [x] Break the `lib/` ↔ `modern_ui/` import layering via dependency
  inversion — done 2026-07-18: theming moved to
  `lib/theming/theme_manager.py` (stdlib + PyQt only);
  `modern_ui/themes/theme_manager.py` kept as a backward-compat shim.
  `grep modern_ui lib/` is clean except one genuine UI import left
  function-local (`field_scope_mixin.py`).
- [x] Add compiled-vs-interpreted equivalence tests for each compiled stateful
  block (RateLimiter, PID, TransportDelay, Selector) and an all-Neumann 2D PDE
  corner integration test — done 2026-07-05: `tests/regression/test_equiv_*.py`.
  TransportDelay turned out not to be compilable (not in `COMPILABLE_BLOCKS`;
  history-dependent), so its test pins the interpreter fallback + analytic
  delayed-sine instead.

**Interpreter-path bugs found by the equivalence tests — all fixed 2026-07-05**

Found via the strict-xfail tripwires; the compiled path was verified correct
in every case, so each fix corrected the interpreter to match.
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

**CI follow-up**
- [x] Dedicated `ruff format` commit (414 files), then add
  `ruff format --check .` to the CI lint job — done 2026-07-05. Ruff pinned
  to 0.15.18 in the lint job (bump the pin together with a reformat);
  E701/E702 ignores retired from `pyproject.toml` (formatter guarantees them).

**Foundation hardening from the external review (2026-07-05)**
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

### Early 2026 — testing & quality hardening (date not recorded in prior versions of this file)
- [x] **Regression test suite** — `tests/regression/test_regression_suite.py`
  (26 tests): numerical accuracy (integrators, TFs, state-space, transport
  delay, PID), bug-fix regressions (Step/Scope params, External block,
  StateVariable), PDE block tests (Heat1D conservation, Heat2D init),
  optimization primitives (ObjectiveFunction, VectorGain, VectorSum).
- [x] **Performance profiling** — `tests/profiling/profile_simulation.py`:
  block-level sim 1000 steps/0.03s, PDE sim 100 steps x 100 nodes/6ms; main
  bottleneck was scipy import overhead, not execution.
- [x] **External block stub** — `blocks/external.py` returns a proper error
  dict with a message instead of `None` when the target file is missing
  (still a stub for actual execution — see CLAUDE.md Known Issues).
- [x] **Legacy test files** marked skipped with documentation:
  `tests/test_blocks.py`, `tests/test_sine_params.py`,
  `tests/test_transfer_function_exec.py` (pre-GUI-refactor / legacy DBlock
  API).

### February 2026
- [x] **7-Phase Improvement Plan** - Comprehensive code quality improvements
  - Phase 1: Bug fixes (FileService.save, SimulationEngine duplicates, sys.path)
  - Phase 2: Block error handling standardization (fixed `demux.py`,
    `sigproduct.py`, `pid.py`; all blocks now return `{0: value, 'E': False}`
    or `{'E': True, 'error': msg}`)
  - Phase 3: SubsystemManager extraction from lib.py
    (`lib/managers/subsystem_manager.py`, delegated from `lib/lib.py`)
  - Phase 4: modern_canvas.py split (ClipboardManager, ZoomPanManager, under
    `modern_ui/managers/`)
  - Phase 5: Config-driven logging (`lib/logging_config.py`, `config/logging.json`)
  - Phase 6: Type hints (`lib/types.py`, base_block.py)
  - Phase 7: API documentation (`mkdocs.yml`, `docs/api/*.md`, mkdocs +
    mkdocstrings with Google-style docstrings)
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
| 2026-10-10 | **Port labels no longer forced on by selection**: `BlockRenderer.draw_port_labels` drew every port label on a selected block even when they did not fit, stacking "setpoint"/"measurement" over the PID's own text and sp/pv marks. Labels now show only when they fit, selected or not; hovering a port names it (tooltips from 89b00c4). Test: `test_canvas_labels_and_status_bar.py::test_pid_port_labels_never_cover_the_face`. |
| 2026-10-10 | **Zoom -/+ glyphs no longer dots**: `QToolBar#ModernToolBar QToolButton { padding: 4px 6px }` outranked the `QToolButton#ZoomRockerBtn` / `#Transport*` rules, so the 22px zoom buttons left ~8px for a 14px icon (drawn as 2-3px dots) and Play was clipped. Added equally specific overrides (padding 0, sizes restated since QSS min sizes replace setFixedSize); zoom buttons 24px with 16px icons and full-width glyphs. Test: `tests/modern_ui/test_toolbar_icon_size.py`. |
| 2026-10-10 | **Align Top shortcut**: Edit > Align > Align Top is now Ctrl+Shift+U ("up"; Ctrl+Shift+T stays with the tuning panel). USER_MANUAL.md listed Ctrl+Shift+T for it; corrected. Tests: `test_menu_shortcuts_and_preferences.py::test_ctrl_shift_u_aligns_tops` and a Ctrl+Shift+U case in `test_each_key_fires_its_action_exactly_once`. |
| 2026-10-10 | **Shared block search**: new `block_match_rank` (modern_palette.py; 0 exact / 1 prefix / 2 other / None) drives both the palette's Enter and the command palette's block scoring (canvas double-click quick-insert, Ctrl+K). Fixes: a recent Discrete Transfer Function outranked an exact "transfer function"; "statespace" tied the two State Space blocks. Tiers are 25 apart so the recents bonus never crosses them. Side effect: exact "step"/"export"/"parameter" now list the block above the action. Tests: `test_main_window_command_palette.py::TestBlockSearchMatchesPalette`, `test_palette_display_names.py::test_block_match_rank_tiers`. |
| 2026-10-10 | **Welcome overlay at UI scale 125%/150%**: checked with native renders. The overlay used fixed TYPE points, so it never scaled (its comments said it did). `_apply_styling` now scales its fonts, card height and max width by app font / 10pt (clamped at 1, so 100% is unchanged); the card min width is capped by what fits the canvas and re-applied on resize, since a scaled min overlapped cards and buttons on a 1000px window at 150%. Found that UI scale is font-only app-wide (open item). Tests: `test_welcome_overlay_and_tooltips.py::test_overlay_follows_the_ui_scale`, `::test_scaled_cards_fit_a_narrow_canvas`. |
| 2026-10-10 | **Sample-time coloring for wires**: discrete wires are tinted with the block dot's red (1 ms) to blue (1 s) log scale and dashed, before any run too. New `modern_ui/renderers/sample_time_colors.py` (`rate_color`, shared with the block dot; `block_sample_times` / `wire_sample_times` repeat the engine's rule: declared `resolve_sample_time()`, inherited = fastest discrete input, wire = source rate; a RateTransition's output wire uses `output_sample_time`). Selected wires keep the accent color. Test: `tests/modern_ui/test_wire_sample_time_colors.py`. |
| 2026-10-10 | **One-click auto-layout**: Edit > Auto Layout (Ctrl+Shift+A, also in the command palette). New pure `modern_ui/tools/auto_layout.py` (`layered_layout`: DFS back edges dropped for layering, longest-path layers via Kahn, barycenter sweeps with dst-port tie-break, columns centered, components stacked, grid snap; deterministic, input-order ties). `ModernCanvas.auto_layout` keeps the top-left corner, relocates blocks, re-routes every wire via `route_line_for_mode`, one undo step. es catalog + USER_MANUAL updated. Tests: `tests/modern_ui/test_auto_layout.py` (pure + examples with undo), Ctrl+Shift+A case in `test_each_key_fires_its_action_exactly_once`. |
| 2026-10-09 | **Windows follow-ups (2)**: palette drop lag -- `record_recent` rebuilt the whole palette (~400 widgets) on every drop, now swaps only the Recent section (~90 -> ~20 ms per drop, and the active filter survives); Enter in the palette filter ranks exact > prefix > substring instead of taking filesystem order; long (Spanish) names elide instead of widening the palette, whose hidden horizontal scroll shifted the list sideways on focus; `ui_settings()` honours `DIABLOS_SETTINGS_INI` and `tests/conftest.py` uses a throwaway INI, so the suite no longer reads or overwrites the developer's registry prefs; `SimulationModel.remove_block` filters `line_list` in place, so a block deleted inside a subsystem no longer leaves its wires in `sub_lines`; CI gained a `windows-latest` job (TikZ tests pass a bare file name to pdflatex; the POSIX read-only guard is skipped on Windows). |
| 2026-10-09 | **Windows follow-up**: reviewed the prior five fixes; repaired toolbar width, made status/glyph tests independent of installed fonts, implemented saved-document identity for undo/redo with explicit-save and load/new lifecycle handling, and polished Properties reset controls and row-label sizing. Moved resolved items to Completed; retained the four missing-pdflatex failures and the POSIX-permissions failure as environment gaps. |
| 2026-10-09 | **Windows fixes**: resized blocks can shrink back vertically (`DBlock.calculate_min_size` is port-only; regression test in `tests/modern_ui/test_drag_resize.py`); Fusion style forced on Windows too so the Properties spinboxes are readable (`qss_styles._maybe_use_fusion_style`); stdout/stderr use `errors="backslashreplace"` and the log file is UTF-8, so emoji log records no longer raise on a cp1252 console. Remaining Windows-only test failures logged under Open Items. |
| 2026-09-12 | **Play no longer pops the Simulation-settings modal**: `DSim.execution_init` called `execution_init_time()` -- which constructs and `exec()`s a `SimulationDialog` -- on *every* run, and every test that reached it stubbed that method out, so nothing caught it. `execution_init(ask=None)` now resolves the new `ask_before_run` preference (QSettings `simulation/ask_before_run`, default off, `lib/sim_prefs.py`) and otherwise runs straight away with the stored `sim_time`. The dialog moved to its own entry point: **Simulation > Simulation Settings...** (Ctrl+E) and a gear in the toolbar transport, wired through `SimulationActionsManager.open_settings`, pre-filled by `DSim.open_simulation_dialog` and applied by `DSim.apply_sim_settings` (which dirties the diagram only when a setting the `.diablos` file stores actually changed). The dialog grew an **Ask before every run** checkbox for anyone who wants the old flow. Tests: `tests/regression/test_play_does_not_ask.py` (27). |
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

## References

- `tasks/code-quality-review-2026-06-13.md` - Full whole-app review (334 findings)
- `docs/archive/REFACTORING_TODO.md` - Detailed refactoring history (archived)
- `docs/PDE_ROADMAP.md` - Full PDE enhancement roadmap with architecture diagrams
- `CLAUDE.md` - Project overview and recent work
