# DiaBloS Modern - Developer Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Adding New Blocks](#adding-new-blocks)
5. [Working with the MVC Architecture](#working-with-the-mvc-architecture)
6. [Testing](#testing)
7. [Code Style](#code-style)
8. [Common Tasks](#common-tasks)
9. [Debugging](#debugging)
10. [Contributing](#contributing)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- A GUI environment (X11 on Linux, native on Windows/Mac)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/Sapetor/diablos-modern.git
cd diablos-modern

# Install runtime dependencies
pip install numpy matplotlib tk tqdm pyqtgraph pyqt5 scipy

# Install development dependencies
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
# Run the application
python diablos_modern.py

# Run tests
pytest

# Run tests with coverage
pytest --cov=lib --cov=modern_ui --cov-report=html
```

## Development Setup

### Recommended IDE Setup

**VS Code** with extensions:
- Python (Microsoft)
- Pylance (type checking)
- Python Test Explorer
- Python Docstring Generator

**PyCharm** (Community or Professional)
- Built-in Python support
- Excellent refactoring tools
- Integrated debugger

### Code Quality Tools

```bash
# Install development tools
pip install pylint black mypy

# Run linter
pylint lib/ modern_ui/

# Format code
black lib/ modern_ui/ tests/

# Type checking
mypy lib/ modern_ui/
```

## Project Structure

```
diablos-modern/
├── blocks/                  # Block implementations
│   ├── base_block.py       # Base class for all blocks
│   ├── integrator.py       # Integrator block
│   ├── step.py             # Step input block
│   └── ...                 # 18+ block types
│
├── lib/                     # Core library
│   ├── models/             # Model layer (data)
│   │   └── simulation_model.py
│   ├── engine/             # Engine layer (business logic)
│   │   └── simulation_engine.py
│   ├── services/           # Services (file I/O, etc.)
│   │   └── file_service.py
│   ├── simulation/         # Domain entities
│   │   ├── block.py        # DBlock class
│   │   ├── connection.py   # DLine class
│   │   └── menu_block.py   # MenuBlocks class
│   ├── lib.py              # Main DSim controller
│   ├── functions.py        # Block execution functions
│   ├── dialogs.py          # UI dialogs
│   ├── config_manager.py   # Configuration management
│   └── improvements.py     # Utilities and helpers
│
├── modern_ui/              # User interface
│   ├── main_window.py      # Main window
│   ├── widgets/            # UI widgets
│   │   ├── modern_canvas.py
│   │   ├── modern_palette.py
│   │   ├── property_editor.py
│   │   └── modern_toolbar.py
│   └── themes/             # Theme management
│
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   ├── conftest.py        # Pytest fixtures
│   └── README.md          # Testing guide
│
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md    # Architecture guide
│   └── DEVELOPER_GUIDE.md # This file
│
├── examples/              # Example diagrams
├── saves/                 # Saved diagrams
├── diablos_modern.py     # Application entry point
├── pytest.ini            # Pytest configuration
└── requirements-dev.txt  # Development dependencies
```

## Adding New Blocks

### Step 1: Create Block Class

Create a new file in `blocks/` directory:

```python
# blocks/my_custom_block.py
from blocks.base_block import BaseBlock
from lib import functions

class MyCustomBlock(BaseBlock):
    """
    My custom block that does something interesting.
    """

    @property
    def block_name(self):
        """Display name for the block."""
        return "MyCustom"

    @property
    def fn_name(self):
        """
        Function name in lib/functions.py
        IMPORTANT: This must match exactly!
        """
        return "my_custom"

    @property
    def category(self):
        """Category for block palette organization."""
        return "Custom"  # or "Math", "Control", "Sources", etc.

    @property
    def color(self):
        """Block color in diagram."""
        return "purple"  # or QColor(128, 0, 255)

    @property
    def inputs(self):
        """Define input ports."""
        return [
            {"name": "input1", "type": "float"},
            {"name": "input2", "type": "float"}
        ]

    @property
    def outputs(self):
        """Define output ports."""
        return [
            {"name": "output", "type": "float"}
        ]

    @property
    def params(self):
        """Define block parameters with defaults."""
        return {
            "gain": {"default": 1.0, "type": "float"},
            "offset": {"default": 0.0, "type": "float"},
            "mode": {"default": "normal", "type": "string"}
        }

    @property
    def doc(self):
        """Documentation shown in help/tooltips."""
        return "Multiplies inputs by gain and adds offset."

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute the block logic.
        This delegates to lib/functions.py
        """
        return functions.my_custom(time, inputs, params, **kwargs)
```

### Step 2: Implement Execution Function

Add the function to `lib/functions.py`:

```python
def my_custom(time, inputs, params, output_only=False):
    """
    Execute custom block logic.

    Args:
        time: Current simulation time
        inputs: Dict mapping port numbers to input values
        params: Block parameters dict
        output_only: If True, return output without state update

    Returns:
        Dict with 'output' key containing result value(s)
        or Dict with 'E': True and 'error': 'message' on error
    """
    try:
        # Get input values
        input1 = inputs.get(0, 0.0)  # Port 0
        input2 = inputs.get(1, 0.0)  # Port 1

        # Get parameters
        gain = params.get('gain', 1.0)
        offset = params.get('offset', 0.0)

        # Compute output
        result = (input1 + input2) * gain + offset

        # Return result (single output)
        return {'output': result}

        # For multiple outputs:
        # return {'output': [result1, result2, result3]}

    except Exception as e:
        logger.error(f"Error in my_custom: {e}")
        return {'E': True, 'error': str(e)}
```

### Step 3: Test Your Block

Blocks are automatically discovered and loaded. Just restart the application:

```bash
python diablos_modern.py
```

Your block should appear in the Block Palette under the category you specified!

### Routing (Goto / From)

- `Goto`: sink with `tag` (and optional `signal_name`). It consumes a wire and does not render an output port.
- `From`: source with the same `tag`/`signal_name`. It exposes one output port.
- At simulation init, the model creates a hidden virtual line from the Goto’s upstream source to each matching From (same tag). Hidden lines are not drawn or hit-tested.
- The virtual line label is set to `signal_name` (defaults to `tag`), which can later be used for workspace binding/export.

### Step 4: Write Tests

```python
# tests/unit/test_my_custom_block.py
import pytest
from blocks.my_custom_block import MyCustomBlock
from lib import functions

@pytest.mark.unit
class TestMyCustomBlock:
    def test_block_has_correct_name(self):
        block = MyCustomBlock()
        assert block.block_name == "MyCustom"

    def test_block_has_correct_fn_name(self):
        block = MyCustomBlock()
        assert block.fn_name == "my_custom"

    def test_execution_function_exists(self):
        assert hasattr(functions, 'my_custom')

    def test_execution_with_valid_inputs(self):
        inputs = {0: 5.0, 1: 3.0}
        params = {'gain': 2.0, 'offset': 1.0}

        result = functions.my_custom(0.0, inputs, params)

        assert result['output'] == 17.0  # (5+3)*2+1
```

## Working with the MVC Architecture

### Model Layer (Data Management)

Use `SimulationModel` for all diagram data operations:

```python
from lib.models.simulation_model import SimulationModel

model = SimulationModel()

# Add a block
from PyQt5.QtCore import QPoint
menu_block = model.menu_blocks[0]  # Get a block template
new_block = model.add_block(menu_block, QPoint(100, 100))

# Find a block
block = model.get_block_by_name('step0')

# Add a connection
src_data = (block1.name, 0, block1.out_coords[0])
dst_data = (block2.name, 0, block2.in_coords[0])
line = model.add_line(src_data, dst_data)

# Check if port is available
available = model.is_port_available(dst_data)

# Get statistics
stats = model.get_diagram_stats()
print(f"Blocks: {stats['blocks']}, Lines: {stats['lines']}")
```

### Engine Layer (Business Logic)

Use `SimulationEngine` for simulation operations:

```python
from lib.engine.simulation_engine import SimulationEngine

engine = SimulationEngine(model)

# Validate diagram
is_valid = engine.check_diagram_integrity()

# Get block connections
inputs, outputs = engine.get_neighbors('block_name')

# Update simulation parameters
engine.update_sim_params(sim_time=10.0, sim_dt=0.01)

# Get execution status
status = engine.get_execution_status()
```

### Services Layer (File I/O)

Use `FileService` for persistence:

```python
from lib.services.file_service import FileService

file_service = FileService(model)

# Save diagram
sim_params = {'sim_time': 10.0, 'sim_dt': 0.01, 'plot_trange': 100}
result = file_service.save(autosave=False, sim_params=sim_params)

# Load diagram
data = file_service.load(filepath='saves/my_diagram.dat')
if data:
    sim_params = file_service.apply_loaded_data(data)
```

### Controller Layer (Coordination)

`DSim` coordinates between layers:

```python
from lib.lib import DSim

dsim = DSim()

# DSim delegates to model
dsim.add_block(menu_block, pos)  # → model.add_block()

# DSim delegates to engine
dsim.check_diagram_integrity()   # → engine.check_diagram_integrity()

# DSim delegates to file_service
dsim.save()                       # → file_service.save()
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/unit/test_simulation_model.py

# Specific test
pytest tests/unit/test_simulation_model.py::TestAddBlock::test_add_block_creates_block_instance

# With coverage
pytest --cov=lib --cov=modern_ui --cov-report=html

# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration
```

### Writing Tests

Use fixtures from `conftest.py`:

```python
import pytest

@pytest.mark.unit
@pytest.mark.qt
def test_something(simulation_model, simulation_engine):
    """Test description."""
    # Arrange
    block = simulation_model.add_block(...)

    # Act
    result = simulation_engine.check_diagram_integrity()

    # Assert
    assert result == expected
```

See [tests/README.md](../tests/README.md) for comprehensive testing guide.

## Code Style

### Python Style Guide

Follow **PEP 8** with these conventions:

```python
# Class names: PascalCase
class SimulationModel:
    pass

# Function names: snake_case
def add_block(self, block, position):
    pass

# Constants: UPPER_CASE
MAX_BLOCKS = 1000

# Private methods: _leading_underscore
def _internal_helper(self):
    pass
```

### Type Hints

Always use type hints for new code:

```python
from typing import List, Dict, Optional, Tuple

def add_block(self, block: MenuBlocks, m_pos: QPoint) -> DBlock:
    """Add a block to the diagram."""
    pass

def get_neighbors(self, block_name: str) -> Tuple[List[Dict], List[Dict]]:
    """Get block's input and output connections."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of what the function does.

    Longer description if needed, explaining the purpose
    and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> my_function(42, "hello")
        True
    """
    pass
```

### Imports

Organize imports in this order:

```python
# Standard library
import os
import sys
import logging

# Third-party
import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QApplication

# Local application
from lib.models.simulation_model import SimulationModel
from lib.simulation.block import DBlock
```

## Common Tasks

### Adding a Menu Action

1. Edit `modern_ui/main_window.py`:

```python
def _create_menu_bar(self):
    # ... existing code ...

    # Add new menu item
    my_action = QAction("My Action", self)
    my_action.triggered.connect(self._on_my_action)
    my_action.setShortcut("Ctrl+M")
    file_menu.addAction(my_action)

def _on_my_action(self):
    """Handle my action."""
    # Implementation
    pass
```

### Adding a Toolbar Button

```python
def _create_toolbar(self):
    # ... existing code ...

    my_button = QAction(QIcon("icons/my_icon.svg"), "My Button", self)
    my_button.triggered.connect(self._on_my_button_clicked)
    toolbar.addAction(my_button)
```

### Adding a Keyboard Shortcut

```python
def keyPressEvent(self, event):
    """Handle keyboard events."""
    if event.key() == Qt.Key_Delete:
        self.remove_selected_items()
    elif event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_S:
        self.save_diagram()
    else:
        super().keyPressEvent(event)
```

### Adding a Theme Color

```python
# modern_ui/themes/theme_manager.py

THEMES = {
    'light': {
        # ... existing colors ...
        'my_new_color': '#FF5733',
    },
    'dark': {
        # ... existing colors ...
        'my_new_color': '#C70039',
    }
}

# Use in code:
from modern_ui.themes.theme_manager import theme_manager
color = theme_manager.get_color('my_new_color')
```

## Debugging

### Enable Debug Logging

```python
# In diablos_modern.py
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific modules
logger = logging.getLogger('lib.models.simulation_model')
logger.setLevel(logging.DEBUG)
```

### Use Python Debugger

```python
import pdb

def my_function():
    # Set breakpoint
    pdb.set_trace()

    # Code execution will pause here
    # Use commands: n (next), s (step), c (continue), p var (print)
```

### VS Code Debugging

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: DiaBloS",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/diablos_modern.py",
            "console": "integratedTerminal"
        }
    ]
}
```

### Common Issues

**Qt platform plugin errors on WSL:**
```bash
export QT_QPA_PLATFORM=xcb
python diablos_modern.py
```

**Import errors:**
```bash
# Make sure you're in project root
cd /path/to/diablos-modern
python diablos_modern.py
```

**Block not appearing:**
- Check `fn_name` matches function in lib/functions.py
- Restart application to reload blocks
- Check console for error messages

## Contributing

### Workflow

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run tests**
   ```bash
   pytest
   black lib/ modern_ui/ tests/
   pylint lib/ modern_ui/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add my new feature"
   ```

   Use conventional commits:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `test:` - Tests
   - `refactor:` - Code refactoring
   - `style:` - Formatting
   - `chore:` - Maintenance

6. **Push and create PR**
   ```bash
   git push origin feature/my-new-feature
   ```

### Code Review Checklist

- [ ] Code follows PEP 8 style guide
- [ ] All functions have type hints
- [ ] All functions have docstrings
- [ ] Tests are included
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] No commented-out code
- [ ] No debugging print statements

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/Sapetor/diablos-modern/issues)
- **Documentation**: See [docs/](../docs/)
- **Architecture**: See [ARCHITECTURE.md](ARCHITECTURE.md)

## Additional Resources

- **PyQt5 Tutorial**: [https://www.pythonguis.com/pyqt5-tutorial/](https://www.pythonguis.com/pyqt5-tutorial/)
- **NumPy Documentation**: [https://numpy.org/doc/](https://numpy.org/doc/)
- **pytest Documentation**: [https://docs.pytest.org/](https://docs.pytest.org/)
- **Type Hints**: [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
