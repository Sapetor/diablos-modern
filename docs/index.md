# DiaBloS Modern Documentation

Welcome to the documentation for **DiaBloS Modern** - a block diagram simulation tool for dynamic systems, built with Python and PyQt5.

## Overview

DiaBloS allows you to create, connect, and simulate dynamic systems using a visual block diagram interface. It supports:

- **Block Diagram Editing**: Drag-and-drop interface for creating block diagrams
- **Simulation Engine**: ODE integration with scipy for accurate simulation
- **PDE Support**: 1D and 2D partial differential equation blocks
- **Optimization Primitives**: Visual building blocks for optimization algorithms
- **Control System Analysis**: linearization with Bode, Nyquist, root locus, pole-zero and step/impulse views, plus trim solving
- **Experiments**: seeded Monte Carlo ensembles and 1-D/2-D parameter sweeps
- **Export**: TikZ, PNG/SVG images, and a standalone numpy + scipy Python script
- **Headless CLI**: simulate a diagram or export it as a script without the GUI

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [User Manual](USER_MANUAL.md)
- [Block Reference](user-guide/block-reference.md)
- [Fast Solver (compiled mode)](FAST_SOLVER.md)
- [Architecture](ARCHITECTURE.md) / [Developer Guide](DEVELOPER_GUIDE.md)
- [API Reference](api/lib.md)

## Architecture

DiaBloS Modern uses a Model-View-Controller (MVC) architecture:

- **Model** (`lib/models/`): Data structures for blocks, connections, and diagrams
- **Engine** (`lib/engine/`): Simulation execution and ODE integration
- **UI** (`modern_ui/`): PyQt5-based graphical interface
- **Blocks** (`blocks/`): Individual block implementations

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python diablos_modern.py
```

## Contributing

DiaBloS Modern is open source. Contributions are welcome!
