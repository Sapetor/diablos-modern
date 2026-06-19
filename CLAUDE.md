# DiaBloS Modern

Block diagram simulation tool (Simulink-style) built with Python/PyQt5. Entry point: `python diablos_modern.py`.

## Architecture

```
modern_ui/          PyQt5 GUI: main_window.py (top-level); subdirs widgets/ [modern_canvas.py],
                    controllers/, managers/ [undo/redo, clipboard, connections, selection],
                    renderers/, builders/, interactions/, tools/, themes/, styles/
lib/                Core engine (engine/, simulation/, analysis/, plotting/, services/,
                    export/, managers/, models/, + config_manager.py)
blocks/             Block implementations: mostly flat .py files (one block class each);
                    subdirs pde/, optimization/, optimization_primitives/ group related families
config/             Defaults: default_config.json, logging.json, block_sizes.py
examples/           Example diagrams (mostly .diablos; some legacy .json/.py pairs + sample data)
docs/               ARCHITECTURE.md, DEVELOPER_GUIDE.md, USER_MANUAL.md, FAST_SOLVER.md, building.md
scripts/            Maintenance scripts (resave_examples.py, fix_diagram_overlaps.py, audit_wiki_docs.py)
tests/              Unit, integration, regression, and GUI (modern_ui/) tests
tools/              Build scripts (build.sh, sync_block_registry.py)
tasks/              TODO list (todo.md) and lessons learned (lessons.md)
```

### Simulation Data Flow

1. **Canvas** -- user builds diagram (blocks + connections)
2. **Save/Load** -- `.diablos` JSON format via `FileService` (`lib/services/file_service.py`)
3. **Flattening** -- `Flattener` (`lib/engine/flattener.py`) expands nested subsystems into flat block list
4. **Init** -- `SimulationEngine.initialize_execution()` resolves hierarchy, detects algebraic loops
5. **Compilation** (fast path) -- `SystemCompiler.compile_system()` converts diagram to ODE for `scipy.integrate.solve_ivp`
6. **Execution** -- time-step loop calling `block.execute()` in hierarchy order, propagating outputs
7. **Plotting** -- `ScopePlotter` renders results in Scope/FieldScope windows

### Compiled Solver Execution Order

Blocks execute in three groups: **sources -> middle (algebraic) -> D=0 state blocks**. State blocks (TranFn/StateSpace, Integrator, PID, RateLimiter, PDE) are classified by their feedthrough term D. Strictly-proper state blocks (D=0: strictly-proper TFs, Integrator, RateLimiter, PDE) run last so their derivative computations use correct inputs; feedthrough state blocks (D!=0, which always includes PID) execute with the algebraic middle group instead. See `system_compiler.py` (`_is_d0_state_block`, `state_fns`) and `tasks/lessons.md` for the full rationale.

### Analysis & Experiment Subsystem (`lib/analysis/`)

Headless re-simulation and control-analysis tools that re-run a diagram without the GUI and aggregate results. The two runners share `lib/analysis/resim.py` (`harvest_scope_signals`, `OUTCOME_METRICS`) and snapshot/restore the original block params, so they never mutate the user's diagram:

- `monte_carlo.py` (`MonteCarloRunner`) -- N-run seeded ensembles; each block exposing a `seed` param (e.g. `packet_loss.py`, `random_source.py`, `network_channel.py`, `noise.py`) gets a sub-seed from `derive_seed(master_seed, run_index, block_name)`, so the whole experiment is reproducible from one master seed. Driven off the GUI thread by `modern_ui/widgets/monte_carlo_worker.py`.
- `parameter_sweep.py` (`ParameterSweepRunner`) -- deterministic 1-D/2-D sweeps over block params (response family / heatmap).
- `linearizer.py` (`Linearizer`) -- numeric Jacobian linearization (finite differences over the compiled ODE) yielding A/B/C/D; `control_system_analyzer.py` + `analyzers/` (`BodeAnalyzer`, `NyquistAnalyzer`, `RootLocusAnalyzer`, `LQRAnalyzer`) drive Bode/Nyquist/root-locus/LQR. Pole-zero data and step/impulse responses are produced separately by `analysis_controller.py` (from the linearized transfer function).

GUI side: `modern_ui/controllers/analysis_controller.py` (linearize / trim via `find_trim` / step-impulse) and `tuning_controller.py`, with result windows in `modern_ui/widgets/` (`linearization_result_window.py`, `operating_point_window.py`, `ensemble_result_window.py`, `sweep_result_window.py`).

## Block Contract

All blocks inherit from `BaseBlock` (`blocks/base_block.py`). The four required members are abstract `@property` methods (not plain class attributes):

```python
class ScaleBlock(BaseBlock):               # class convention: <BlockName>Block (blocks/abs_block.py -> AbsBlock)
    @property
    def block_name(self):                  # User-facing name
        return "Scale"

    @property
    def category(self):                    # OPTIONAL (defaults to "Other"); drives port-requirement defaults
        return "Math"

    @property
    def params(self):                      # spec dict: name -> {type, default, doc}
        return {"gain": {"type": "float", "default": 1.0, "doc": "Scale factor"}}

    @property
    def inputs(self):                      # list of port dicts (PortDefinition = Dict[str, str], a type alias)
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs) -> dict:
        # inputs: {port_idx: value}, params: flattened {name: value} (params['gain'] is 1.0 here)
        # Return {port_idx: output_value} or {'E': True, 'error': 'msg'}
        return {0: np.atleast_1d(inputs.get(0, 0)) * params['gain']}
```

Note: `PortDefinition` (in `base_block.py`) is a type alias for `Dict[str, str]` used only in annotations -- ports are plain `{"name": ..., "type": ...}` dicts, never `PortDefinition(...)`. The `params` **property** returns a nested spec dict, while the `params` passed to `execute()` is the flattened `{name: value}` map.

Optional overrides: `category` (port-requirement defaults: Sources make inputs optional, Sinks/Other make outputs optional -- via `requires_inputs`/`requires_outputs` in `base_block.py`), `requires_inputs`/`requires_outputs` (override directly when `category` isn't enough), `optional_inputs`, `optional_outputs`, `draw_icon()`, `symbolic_execute()`.

**Critical rule**: All block state that persists across time steps must be stored in `params` (e.g., `params['_t_old_']`), never on `self`. The engine's `reset_memblocks()` (`lib/engine/simulation_engine.py:713`) sets `_init_start_ = True` in each block's `params` and `exec_params` and clears stale `exec_params` accumulators (`_prev`, `mem`, `output`); it never touches instance attributes, so state stored on `self` survives resets invisibly and leaks between runs.

## Adding New Blocks

1. Create `.py` file in `blocks/` (or a subdirectory with `__init__.py`)
2. Inherit from `BaseBlock`, implement the required `@property` members and `execute()`. Class name convention is `<BlockName>Block` (block "Abs" in `blocks/abs_block.py` -> class `AbsBlock`).
3. Run `python tools/sync_block_registry.py` to update `_BLOCK_MODULES` in `lib/block_loader.py`

The build script (`tools/build.sh`) runs the sync automatically. In dev mode, blocks are also discovered dynamically.

## Testing

```bash
pytest tests/ -v                    # Entire suite (testpaths=tests in pytest.ini)
pytest tests/unit/ -v               # Unit tests only
pytest tests/integration/ -v        # Integration tests only
pytest tests/regression/ -v         # Regression tests only
pytest tests/modern_ui/ -v          # PyQt GUI tests
```

In addition to those subdirs (and `tests/profiling/`), ~15 `test_*.py` files live directly under `tests/`; `pytest tests/` (or bare `pytest`, via `testpaths = tests`) runs the whole suite. Markers are a fixed set declared in `pytest.ini` (`unit`, `integration`, `regression`, `slow`, `qt`, `file_io`) enforced via `--strict-markers` -- add new ones there before use.

**Headless / WSL / CI**: tests construct real PyQt5 widgets, so on a display-less machine run with `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg`. A session-scoped autouse fixture in `tests/conftest.py` (`_no_modal_dialogs`) neutralizes `QDialog`/`QMessageBox` so modal dialogs never block. On Windows/WSL the repo ships a prebuilt interpreter at `.venv-win/Scripts/python.exe` -- use it to run the suite.

**CI**: `.github/workflows/ci.yml` runs the suite on a Python **3.9 + 3.12** matrix (ubuntu-latest) per push/PR. 3.9 is the baseline (matches the local/readthedocs env), so avoid 3.10+-only syntax -- code that passes locally on a newer Python can still break the 3.9 leg. A separate `lint` job runs `ruff check .`.

**Linting**: `ruff check .` (config in `pyproject.toml`, targets py39) is the lint gate -- run it before pushing. Ruff is the project's linter/formatter (replaces black/pylint, in `requirements-dev.txt`); the rule set keeps pyflakes + error checks and silences high-volume stylistic E7xx (see `pyproject.toml`). `ruff format` is available but not yet enforced in CI.

### Test Pattern

```python
@pytest.mark.unit
class TestScaleBlock:
    def test_basic_output(self):
        from blocks.scale import ScaleBlock
        block = ScaleBlock()
        params = {'gain': 2.0, '_init_start_': True}
        result = block.execute(time=0.5, inputs={0: np.array([3.0])}, params=params, dtime=0.01)
        assert np.isclose(result[0][0], 6.0)
```

Instantiate block, call `execute()` with `time`, `inputs` dict (keyed by port index), `params`, and optional `dtime`. Use `np.isclose()` for floating-point assertions. The `_init_start_: True` flag triggers first-call initialization.

## Build & Packaging

See [`docs/building.md`](docs/building.md) for PyInstaller packaging (macOS, Windows, Linux).

Quick reference (macOS arm64, the recommended release env): `source ~/.venvs/diablos-arm64/bin/activate && ./tools/build.sh`. For Windows/Linux create a generic `.venv` (`python -m venv .venv`); see `docs/building.md` for full per-platform steps.

Key files: `diablos.spec` (PyInstaller config), `lib/app_paths.py` (frozen vs dev path resolution), `lib/block_loader.py` (`_BLOCK_MODULES` static registry for frozen mode).

## Known Issues

- `blocks/external.py` is a stub -- returns error dict (not implemented)

## Key Dependencies

- **Runtime**: PyQt5, numpy, scipy, matplotlib, pyqtgraph, Pillow, tqdm
- **GIF export**: Pillow >= 8.0.0
- **MP4 export**: ffmpeg (external, `brew install ffmpeg`)
