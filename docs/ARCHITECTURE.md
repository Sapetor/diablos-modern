# DiaBloS Modern - Architecture Documentation

## Overview

DiaBloS Modern follows a **Model-View-Controller (MVC)** architecture, separating concerns between data management, business logic, and user interface.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│  modern_ui/                                                  │
│  ├── main_window.py          (Main window, menus, toolbar)  │
│  ├── widgets/                                                │
│  │   ├── modern_canvas.py    (Diagram canvas, interactions) │
│  │   ├── modern_palette.py   (Block palette)               │
│  │   ├── property_editor.py  (Property editing)            │
│  │   └── modern_toolbar.py   (Toolbar controls)            │
│  └── themes/                  (Theme management)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Controller Layer                         │
│  lib/lib.py (DSim)                                          │
│  - Coordinates between UI and backend                        │
│  - Delegates to MVC components                               │
│  - Maintains backward compatibility                          │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌─────────────────────────┐
│      Model Layer         │  │   Engine Layer          │
│  lib/models/             │  │  lib/engine/            │
│  SimulationModel         │  │  SimulationEngine       │
│  - blocks_list           │  │  - Diagram validation   │
│  - line_list             │  │  - Execution logic      │
│  - menu_blocks           │  │  - Algebraic loops      │
│  - colors                │  │  - Hierarchy analysis   │
└──────────────────────────┘  └─────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  lib/services/                                               │
│  ├── FileService           (Save/load diagrams)             │
│  ├── RunHistoryService     (Simulation history/persistence) │
│  └── (future services)                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layer                             │
│  lib/simulation/                                             │
│  ├── block.py              (DBlock - Block entities)        │
│  ├── connection.py         (DLine - Connection entities)    │
│  └── menu_block.py         (MenuBlocks - Block templates)   │
│                                                              │
│  blocks/                   (Block implementations)           │
│  ├── base_block.py         (Base class for all blocks)      │
│  ├── integrator.py         (Integrator block)               │
│  ├── transfer_function.py  (Transfer function block)        │
│  └── ... (69 block types)                                   │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model Layer (`lib/models/`)

**SimulationModel** (`simulation_model.py`)
- **Responsibility**: Manages diagram data (blocks, connections, state)
- **Key attributes**:
  - `blocks_list: List[DBlock]` - Instantiated blocks in diagram
  - `line_list: List[DLine]` - Connections between blocks
  - `menu_blocks: List[MenuBlocks]` - Available block types
  - `colors: Dict[str, QColor]` - Color palette
  - `dirty: bool` - Has diagram been modified?

- **Key methods**:
  - `add_block(block, position)` - Add block to diagram
  - `remove_block(block)` - Remove block and its connections
  - `add_line(src_data, dst_data)` - Create connection
  - `get_block_by_name(name)` - Lookup block by name
  - `is_port_available(dst_line)` - Check if port is connected
  - `load_all_blocks()` - Load block types from blocks/ directory

### 2. Engine Layer (`lib/engine/`)

**SimulationEngine** (`simulation_engine.py`)
- **Responsibility**: Business logic for simulation execution
- **Key attributes**:
  - `model: SimulationModel` - Reference to data model
  - `execution_initialized: bool` - Execution state
  - `sim_time: float` - Total simulation time
  - `sim_dt: float` - Time step
  - `global_computed_list: List[Dict]` - Execution tracking

- **Key methods**:
  - `check_diagram_integrity()` - Validate all ports connected
  - `get_neighbors(block_name)` - Get block's connections
  - `detect_algebraic_loops(blocks)` - Find feedback loops
  - `children_recognition(block, list)` - Find downstream blocks
  - `reset_execution_data()` - Clear execution state
  - `get_max_hierarchy()` - Find max hierarchy level

### 3. Services Layer (`lib/services/`)

**FileService** (`file_service.py`)
- **Responsibility**: Persistence (save/load diagrams)
- **Key methods**:
  - `save(autosave, ui_data, sim_params)` - Save to JSON file
  - `load(filepath)` - Load from JSON file
  - `apply_loaded_data(data)` - Reconstruct diagram from data

- **File format**: JSON with structure:
  ```json
  {
    "version": "2.0",
    "sim_data": {...},
    "blocks_data": [{...}],
    "lines_data": [{...}],
    "modern_ui_data": {...}
  }
  ```

### 4. Controller Layer (`lib/lib.py`)

**DSim Class**
- **Responsibility**: Coordinates between UI and backend
- **Pattern**: Delegates to MVC components while maintaining backward compatibility
- **Key delegations**:
  - `self.model = SimulationModel()` - Data management
  - `self.engine = SimulationEngine(model)` - Business logic
  - `self.file_service = FileService(model)` - Persistence
  - Exposes common properties: `blocks_list`, `line_list`, `colors`

### 5. Domain Entities (`lib/simulation/`)

**DBlock** (`block.py`)
- Represents a functional block in the diagram
- **Key attributes**:
  - `name: str` - Unique identifier
  - `block_fn: str` - Block type (e.g., "Integrator")
  - `fn_name: str` - Execution function name
  - `params: Dict` - Block parameters
  - `in_ports: int`, `out_ports: int` - Port counts
  - `in_coords: List[QPoint]`, `out_coords: List[QPoint]` - Port positions
  - `hierarchy: int` - Execution order level
  - `computed_data: bool` - Has been computed this step

**DLine** (`connection.py`)
- Represents a connection between blocks
- **Key attributes**:
  - `srcblock: str`, `dstblock: str` - Connected blocks
  - `srcport: int`, `dstport: int` - Port numbers
  - `points: List[QPoint]` - Line path
  - `path: QPainterPath` - Rendered path

**MenuBlocks** (`menu_block.py`)
- Template for creating blocks from palette
- Contains default parameters and metadata

### 6. Block Implementations (`blocks/`)

All blocks inherit from **BaseBlock** (`base_block.py`):

```python
class BaseBlock(ABC):
    # --- Required (abstract) members ---
    @property
    @abstractmethod
    def block_name(self): ...       # e.g., "Integrator"

    @property
    @abstractmethod
    def params(self): ...           # Default parameters (nested spec dict)

    @property
    @abstractmethod
    def inputs(self): ...           # Input port definitions

    @property
    @abstractmethod
    def outputs(self): ...          # Output port definitions

    @abstractmethod
    def execute(self, time, inputs, params, **kwargs): ...

    # --- Optional overrides ---
    @property
    def category(self): ...         # e.g., "Control" (defaults to "Other")

    @property
    def fn_name(self): ...          # Only some blocks (e.g. statespace) override this
    
    def draw_icon(self, block_rect):  # Optional: custom icon
        """Return QPainterPath in 0-1 normalized coordinates, or None for fallback."""
        ...
```

**Block types include**:
- **Sources**: Step, Ramp, Sine, Noise, Exponential
- **Math**: Sum, Product (SgProd), Gain
- **Control**: Integrator, Derivative, Transfer Function
- **Utilities**: Mux, Demux, Scope, Export, Terminator
- **Routing**: Goto (tagged sink) / From (tagged source). At simulation init the model inserts a hidden virtual line from the Goto’s upstream source to matching From blocks (same tag). Hidden lines are skipped in hit-testing/drawing. Optional `signal_name` defaults to the tag and is used as the virtual line label.
- **Analysis**: BodeMag, RootLocus
- **Advanced**: External (custom Python code)

## Data Flow

### 1. Block Creation Flow
```
User clicks block in palette
  ↓
modern_canvas.on_palette_block_clicked()
  ↓
DSim.add_block(menu_block, position)
  ↓
SimulationModel.add_block(menu_block, position)
  ↓
Creates DBlock instance
  ↓
Adds to blocks_list, sets dirty=True
```

### 2. Connection Creation Flow
```
User clicks output port, then input port
  ↓
modern_canvas._finish_line_creation()
  ↓
DSim.add_line(src_data, dst_data)
  ↓
SimulationModel.add_line(src_data, dst_data)
  ↓
Creates DLine instance
  ↓
Adds to line_list, sets dirty=True
```

### 3. Simulation Execution Flow
```
User clicks Play button
  ↓
modern_canvas.start_simulation()
  ↓
DSim.execution_init_time() - Get parameters
  ↓
DSim.execution_init() - Initialize simulation
  ├→ SimulationEngine.check_diagram_integrity()
  ├→ SimulationEngine.reset_execution_data()
  ├→ Assign hierarchy levels
  └→ Initialize block functions
  ↓
DSim.step() - Execute timestep (called in loop)
  ├→ For each hierarchy level:
  │   └→ Execute blocks at that level
  ├→ Update block states
  └→ Collect outputs
  ↓
Results displayed in Scope blocks
```

### 3b. Compiled Fast-Solver Path

For diagrams that pass `check_compilability()`, DiaBloS bypasses the per-timestep
interpreter loop above and compiles the whole diagram into a single ODE solved by
SciPy:

```
Flattening    - Flattener (lib/engine/flattener.py) expands nested
                subsystems into a flat block list
  ↓
Compilation   - SystemCompiler.compile_system() (lib/engine/system_compiler.py)
                builds a per-block executor closure for every block by dispatching
                to the kernel registry in lib/engine/compiler_kernels/
                (@kernel-decorated builders, one block family per module).
                Blocks run in three groups:
                sources → algebraic middle → strictly-proper (D=0) state blocks
  ↓
Solve         - the assembled derivative function is handed to
                scipy.integrate.solve_ivp (RK45 by default)
  ↓
Replay        - run_compiled_simulation recomputes block outputs at each saved
                time step to populate Scope traces
```

If compilation fails for any reason, the engine falls back to the interpreter path
above. See [FAST_SOLVER.md](FAST_SOLVER.md) for the full rationale and the
compiled-solver execution order.

### 3c. Compiled vs interpreter semantics

DiaBloS has **two simulation engines** with deliberately different numerics, and
it matters which one a result came from. Treat the **compiled path as the source
of truth**: it integrates the continuous ODE with an adaptive scipy solver
(RK45 by default) and is validated against closed-form analytic solutions in
`tests/regression/test_analytic_solutions.py`. The interpreter is a fixed-step,
per-block loop that trades accuracy for the ability to run *any* block
(including blocks with no compiled kernel).

They are **not** expected to be bit-identical, and for state-heavy diagrams they
legitimately diverge. `tests/regression/test_equiv_*.py` pins where the two agree
and characterizes where they do not. The known, by-design differences:

| Aspect | Compiled path | Interpreter path |
| --- | --- | --- |
| Time integration | Adaptive RK45 over each step (`solve_ivp`) | Single fixed-step update per `sim_dt` (Forward Euler for PDEs; per-block discretization for LTI blocks) |
| Algebraic loops | Resolved by pre-populating D=0 state blocks and ordering sources → algebraic → strictly-proper | One-sample feedback delay through strictly-proper memory blocks (feedthrough ones refresh their consumers within the step) |
| Coverage | Only blocks with a `@kernel` builder and no block gated to a sample time (`check_compilability()` gates both); non-compilable diagrams fall back | Every block, via `block.execute()` |
| Determinism | Byte-deterministic across runs (golden-master friendly) | Depends on step size; convergent as `sim_dt → 0` |

Consequences worth knowing when reading or comparing traces:

- **Transients differ, steady states agree.** A closed loop (e.g. PID + plant)
  reaches the same steady state on both paths, but the transient can differ by
  O(0.1) on an O(1) signal because of the feedback-delay and derivative-kick
  differences above. Do not assert tight trajectory equality across paths.
- **Step size reaches the blocks.** Interpreter state blocks discretize at the
  actual `sim_dt`, or at their own sample period when gated to a discrete rate
  (`DBlock.execution_step()`, stamped into `exec_params['dtime']`). (This relies on `engine.sim_dt` being synced before
  `initialize_execution`; `run_tuning_simulation` and `execution_init` both do
  this via `update_sim_params` — a past bug pinned them to the default 0.01.)
- **Memory blocks run twice per step.** The interpreter executes memory blocks
  (Integrator, RateLimiter, PID, …) with `output_only=True` to feed downstream
  consumers, then again to advance state. A block's `execute()` **must not**
  advance state on the `output_only` pass. Memory blocks with direct feedthrough
  (`b_type == 2`: ZeroOrderHold, RateLimiter, PID) then re-propagate the value
  they just computed, since the `output_only` value is stale for them.
- **PDE blocks self-integrate in the interpreter.** 1D and 2D PDE blocks step
  their own field with Forward Euler and persist it in `params`
  (`_interp_step` / `params['T']`), reusing the same spatial operators as the
  compiled kernels (`lib/engine/pde_ops.py`). FTCS stability limits apply; use
  the compiled solver for stiff or fine grids.

For a headless run of either engine, `lib/cli.py`
(`python diablos_modern.py run diagram.diablos -o out.csv [--solver interpreter]`)
loads a diagram, simulates it, and exports the Scope traces to CSV/NPZ without
the GUI. It defaults to the compiled path.

### 4. File Save/Load Flow
```
Save:
User clicks Save
  ↓
DSim.save() → FileService.save()
  ├→ Serialize blocks_list to JSON
  ├→ Serialize line_list to JSON
  ├→ Include sim_data and UI state
  └→ Write to .dat file

Load:
User clicks Open
  ↓
DSim.open() → FileService.load()
  ↓
DSim.update_blocks_data() for each block
  ├→ Find matching menu_block
  ├→ Use menu_block.fn_name (not saved fn_name!)
  ├→ Create DBlock with correct parameters
  └→ Add to blocks_list
  ↓
DSim.update_lines_data() for each line
  └→ Create DLine instances
```

## Design Patterns

### 1. MVC (Model-View-Controller)
- **Model**: SimulationModel (data)
- **View**: modern_ui/ (UI components)
- **Controller**: DSim (coordination)

### 2. Delegation
DSim delegates responsibilities to specialized components:
```python
# In DSim.__init__()
self.model = SimulationModel()
self.engine = SimulationEngine(self.model)
self.file_service = FileService(self.model)

# Delegation example
def add_block(self, block, m_pos):
    new_block = self.model.add_block(block, m_pos)
    self.dirty = self.model.dirty
    return new_block
```

### 3. Plugin Architecture (Blocks)
Blocks are discovered dynamically from `blocks/` directory:
```python
# lib/block_loader.py
def load_blocks():
    for file in blocks_dir.glob("*.py"):
        if file.name != "base_block.py":
            # Import and load block class
            ...
```

### 4. Repository Pattern
SimulationModel acts as a repository for blocks and lines:
- `add_block()`, `remove_block()`
- `add_line()`, `remove_line()`
- `get_block_by_name()`
- `clear_all()`

## Key Design Decisions

### 1. Backward Compatibility
DSim maintains properties that delegate to the model:
```python
# Backward compatibility
self.blocks_list = self.model.blocks_list
self.line_list = self.model.line_list
self.colors = self.model.colors
```

### 2. fn_name Resolution
Critical fix: Always use `menu_block.fn_name` not `block_name.lower()`:
```python
# SimulationModel.load_all_blocks()
if hasattr(block, 'fn_name'):
    fn_name = block.fn_name  # Use custom fn_name
else:
    fn_name = block.block_name.lower()
```

### 3. Separation of Concerns
- **SimulationModel**: Pure data, no business logic
- **SimulationEngine**: Business logic, no data management
- **FileService**: I/O only, delegates to model for data
- **DSim**: Coordination, backward compatibility

### 4. Type Safety
All MVC components use type hints:
```python
def add_block(self, block: MenuBlocks, m_pos: QPoint) -> DBlock:
    ...
```

## Extension Points

### Adding a New Block Type

> **Note:** Blocks implement their logic directly in `execute()`. The legacy
> `lib/functions.py` has been removed — there is no separate execution-function file.

1. Create file in `blocks/` directory:
```python
# blocks/my_block.py
from blocks.base_block import BaseBlock
import numpy as np

class MyBlock(BaseBlock):
    @property
    def block_name(self):
        return "MyBlock"

    @property
    def params(self):
        return {
            "param1": {"default": 1.0, "type": "float"}
        }

    # ... implement other properties

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute block logic directly - no functions.py needed!
        """
        input_val = inputs.get(0, 0.0)
        param1 = params.get('param1', 1.0)
        result = input_val * param1
        return {0: result}  # outputs keyed by port index
```

2. Restart application - block automatically discovered!


### Adding a New Service

1. Create service file in `lib/services/`:
```python
# lib/services/my_service.py
class MyService:
    def __init__(self, model):
        self.model = model

    def my_method(self):
        # Use self.model for data access
        ...
```

2. Instantiate in DSim:
```python
# lib/lib.py
self.my_service = MyService(self.model)
```

## Testing

See [tests/README.md](https://github.com/Sapetor/diablos-modern/blob/main/tests/README.md) for testing documentation.

Tests are organized by component:
- `tests/unit/test_simulation_model.py` - Model layer tests
- `tests/unit/test_simulation_engine.py` - Engine layer tests
- `tests/integration/` - End-to-end workflow tests

## Performance Considerations

- **Block lookup**: Currently O(n) iteration. Consider dict-based lookup for large diagrams.
- **Connection queries**: `get_neighbors()` iterates all lines. Consider caching.
- **Rendering**: Canvas redraws entire diagram. Consider dirty regions.
- **Memory**: Deep copying of parameters. Consider copy-on-write.

## Security Considerations

- **External blocks**: Can execute arbitrary Python code. Validate before loading diagrams from untrusted sources.
- **File loading**: JSON parsing is safe, but parameters are passed to `eval()` in some cases.

## Future Architecture Improvements

1. **Event System**: Replace direct calls with event bus
2. **Dependency Injection**: Use DI container for services
3. **Command Pattern**: Implement undo/redo
4. **Observer Pattern**: Auto-update UI on model changes
5. **Async Simulation**: Run simulations in background threads

## References

- Original DiaBloS: [GitHub](https://github.com/Sapetor/diablos-modern)
- PyQt5 Documentation: [https://doc.qt.io/qtforpython-5/](https://doc.qt.io/qtforpython-5/)
- MVC Pattern: [Martin Fowler - GUI Architectures](https://martinfowler.com/eaaDev/uiArchs.html)
