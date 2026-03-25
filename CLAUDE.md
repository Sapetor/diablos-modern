# DiaBloS Modern

Block diagram simulation tool (Simulink-style) built with Python/PyQt5. Entry point: `python diablos_modern.py`.

## Architecture

```
modern_ui/          PyQt5 GUI (main_window, modern_canvas, widgets/, controllers/)
lib/                Core engine (engine/, plotting/, services/, export/)
blocks/             Block implementations by category (sources/, sinks/, math/, pde/, etc.)
examples/           Example diagrams (.diablos files)
tests/              Unit, integration, regression tests
tools/              Build scripts (build.sh, sync_block_registry.py)
tasks/              TODO list (todo.md) and lessons learned (lessons.md)
```

### Simulation Data Flow

1. **Canvas** -- user builds diagram (blocks + connections)
2. **Save/Load** -- `.diablos` JSON format via `FileService` (`lib/services/file_service.py`)
3. **Flattening** -- `Flattener` expands nested subsystems into flat block list
4. **Init** -- `SimulationEngine.initialize_execution()` resolves hierarchy, detects algebraic loops
5. **Compilation** (fast path) -- `SystemCompiler.compile_system()` converts diagram to ODE for `scipy.integrate.solve_ivp`
6. **Execution** -- time-step loop calling `block.execute()` in hierarchy order, propagating outputs
7. **Plotting** -- `ScopePlotter` renders results in Scope/FieldScope windows

### Compiled Solver Execution Order

Blocks execute in three groups: **sources -> algebraic -> state blocks**. State blocks (TranFn, Integrator, StateSpace, PDE) run last so their derivative computations use correct inputs. D!=0 state blocks (feedthrough) execute with the algebraic group instead. See `system_compiler.py` and `tasks/lessons.md` for the full rationale.

## Block Contract

All blocks inherit from `BaseBlock` (`blocks/base_block.py`). Required:

```python
class MyBlock(BaseBlock):
    block_name = "MyBlock"                    # User-facing name
    params = {"gain": 1.0}                    # Default parameters
    inputs = [PortDefinition(...)]            # Input port specs
    outputs = [PortDefinition(...)]           # Output port specs

    def execute(self, time, inputs, params, **kwargs) -> dict:
        # inputs: {port_idx: value}, params: {name: value}
        # Return {port_idx: output_value} or {'E': True, 'error': 'msg'}
        return {0: inputs[0] * params['gain']}
```

Optional overrides: `optional_inputs`, `optional_outputs`, `requires_inputs` (False for sources), `requires_outputs` (False for sinks), `draw_icon()`, `symbolic_execute()`.

**Critical rule**: All block state that persists across time steps must be stored in `params` (e.g., `params['_t_old_']`), never on `self`. The engine's `reset_memblocks()` only resets params -- instance attributes survive resets invisibly.

## Adding New Blocks

1. Create `.py` file in `blocks/` (or a subdirectory with `__init__.py`)
2. Inherit from `BaseBlock`, implement required properties and `execute()`
3. Run `python tools/sync_block_registry.py` to update `_BLOCK_MODULES` in `block_loader.py`

The build script (`tools/build.sh`) runs the sync automatically. In dev mode, blocks are also discovered dynamically.

## Testing

```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/integration/ -v        # Integration tests only
pytest tests/regression/ -v         # Regression tests only
```

### Test Pattern

```python
@pytest.mark.unit
class TestMyBlock:
    def test_basic_output(self):
        from blocks.myblock import MyBlockBlock
        block = MyBlockBlock()
        params = {'gain': 2.0, '_init_start_': True}
        result = block.execute(time=0.5, inputs={0: np.array([3.0])}, params=params, dtime=0.01)
        assert np.isclose(result[0][0], 6.0)
```

Instantiate block, call `execute()` with `time`, `inputs` dict (keyed by port index), `params`, and optional `dtime`. Use `np.isclose()` for floating-point assertions. The `_init_start_: True` flag triggers first-call initialization.

## Build & Packaging

See [`docs/building.md`](docs/building.md) for PyInstaller packaging (macOS, Windows, Linux).

Quick reference: `source ~/.venvs/diablos-arm64/bin/activate && ./tools/build.sh`

Key files: `diablos.spec` (PyInstaller config), `lib/app_paths.py` (frozen vs dev path resolution), `lib/block_loader.py` (`_BLOCK_MODULES` static registry for frozen mode).

## Known Issues

- `blocks/external.py` is a stub -- returns error dict (not implemented)

## Key Dependencies

- **Runtime**: PyQt5, numpy, scipy, matplotlib, pyqtgraph, Pillow, tqdm
- **GIF export**: Pillow >= 8.0.0
- **MP4 export**: ffmpeg (external, `brew install ffmpeg`)
