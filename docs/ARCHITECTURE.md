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
│                    Services Layer                            │
│  lib/services/                                               │
│  ├── FileService           (Save/load diagrams)             │
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
│  └── ... (18+ block types)                                  │
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
class BaseBlock:
    @property
    def block_name(self): ...      # e.g., "Integrator"

    @property
    def fn_name(self): ...          # e.g., "integrator"

    @property
    def category(self): ...         # e.g., "Control"

    @property
    def inputs(self): ...           # Input port definitions

    @property
    def outputs(self): ...          # Output port definitions

    @property
    def params(self): ...           # Default parameters

    def execute(self, time, inputs, params, **kwargs): ...
```

**Block types include**:
- **Sources**: Step, Ramp, Sine, Noise, Exponential
- **Math**: Sum, Product (SgProd), Gain
- **Control**: Integrator, Derivative, Transfer Function
- **Utilities**: Mux, Demux, Scope, Export, Terminator
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

1. Create file in `blocks/` directory:
```python
# blocks/my_block.py
from blocks.base_block import BaseBlock
from lib import functions

class MyBlock(BaseBlock):
    @property
    def block_name(self):
        return "MyBlock"

    @property
    def fn_name(self):
        return "my_block"  # Must match function in lib/functions.py

    @property
    def params(self):
        return {
            "param1": {"default": 1.0, "type": "float"}
        }

    # ... implement other properties

    def execute(self, time, inputs, params, **kwargs):
        return functions.my_block(time, inputs, params, **kwargs)
```

2. Add function to `lib/functions.py`:
```python
def my_block(time, inputs, params, output_only=False):
    # Implementation
    return {'output': result}
```

3. Restart application - block automatically discovered!

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

See [tests/README.md](../tests/README.md) for testing documentation.

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
