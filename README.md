# DiaBloS - Modern

A modern, Python-based graphical tool for simulating dynamical systems. This project is an evolution of the original DiaBloS, featuring a completely revamped user interface built with PyQt5, a clean **MVC architecture**, and comprehensive testing infrastructure.

![Screenshot of DiaBloS Modern UI](screenshot.png)

## Key Features

### User Experience
- **Modern, Themable UI**: Switch between light and dark modes instantly
- **Interactive Canvas**: Zoomable, pannable canvas for building simulations
- **Drag-and-Drop**: Easily add blocks by dragging from the palette
- **Streamlined Property Editing**: Edit block parameters directly with auto-apply
- **Dynamic Layout**: Resizable panels with responsive design

### Technical Excellence
- **MVC Architecture**: Clean separation of Model, View, and Controller layers
- **Type-Safe Code**: Comprehensive type hints throughout the codebase
- **Test Infrastructure**: Unit test framework with pytest ready for contributors
- **Extensible Design**: Plugin-based block system with automatic discovery
- **Professional Quality**: Follows Python best practices and coding standards

## Getting Started

### Requirements

- Python 3.9+
- A GUI environment with X11 support (will not work headless)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Sapetor/diablos-modern.git
    cd diablos-modern
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To run the modern version of DiaBloS, execute the following command:

```bash
python diablos_modern.py
```

## Basic Usage

1.  **Add Blocks**: Drag blocks from the **Block Palette** on the left onto the main canvas
2.  **Connect Blocks**: Click output port → click input port to create connections
3.  **Edit Properties**: Click a block to edit its parameters in the **Properties** panel
4.  **Copy/Paste**: Select blocks and use `Ctrl+C` / `Ctrl+V` (or `Cmd+C` / `Cmd+V` on Mac) to duplicate
5.  **Run Simulation**: Click the **Play** button to start simulation
6.  **View Results**: Use `Scope` blocks to visualize outputs

### Keyboard Shortcuts

- **Ctrl+C / Cmd+C**: Copy selected block(s)
- **Ctrl+V / Cmd+V**: Paste block(s) with offset
- **Ctrl+F / Cmd+F**: Flip selected block(s) horizontally
- **Delete**: Remove selected block(s) or connection(s)
- **Escape**: Cancel current operation

## Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - MVC design, data flow, extension points
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - How to add blocks, contribute, best practices
- **[Testing Guide](tests/README.md)** - Running tests, writing tests, coverage

## For Developers

### Quick Start

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run with coverage
pytest --cov=lib --cov=modern_ui --cov-report=html

# Format code
black lib/ modern_ui/ tests/

# Type checking
mypy lib/ modern_ui/
```

### Architecture Overview

DiaBloS Modern follows a clean **MVC (Model-View-Controller)** architecture:

- **Model** (`lib/models/`) - Data management (blocks, connections, state)
- **Engine** (`lib/engine/`) - Business logic (validation, execution)
- **Services** (`lib/services/`) - File I/O and external operations
- **View** (`modern_ui/`) - User interface components
- **Controller** (`lib/lib.py`) - Coordination and delegation

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

### Adding a New Block

1. Create `blocks/my_block.py` with your block class
2. Add execution function to `lib/functions.py`
3. Restart application - block auto-discovered!

See [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#adding-new-blocks) for step-by-step guide.

### Code Quality

- **Type Hints**: All core files have comprehensive type annotations
- **Test Framework**: Pytest infrastructure with fixtures for unit testing
- **Documentation**: Google-style docstrings throughout
- **Standards**: Follows PEP 8 and Python best practices

## Block Types

DiaBloS includes 20+ built-in block types:

**Sources**: Step, Ramp, Sine, Noise, Exponential
**Math**: Sum, Product, Gain
**Control**: Integrator, Derivative, Transfer Function, State Space
**Analysis**: Bode Plot, Root Locus
**Utilities**: Scope, Export, Mux, Demux, Terminator
**Advanced**: External (custom Python code)

## Contributing

We welcome contributions! Please see [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#contributing) for:

- Development workflow
- Code style guidelines
- Testing requirements
- Pull request process

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with: PyQt5, NumPy, Matplotlib, SciPy, PyQtGraph
