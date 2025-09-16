# GEMINI.md

This file provides guidance to GEMINI-CLI when working with code in this repository.

## Running the Application

The primary and recommended application is the **Modern UI**.

**Modern UI (Recommended):**
```bash
python diablos_modern.py
```

**Other Versions (for reference or specific testing):**
```bash
# Enhanced version with original UI
python diablos_improved.py

# Original legacy application
python diablosM1_main.py

# Experimental modern architecture (not fully functional)
python diablos_main_new.py
```

**Requirements:**
- Python 3.9.7+ (tested with 3.12+)
- GUI environment with X11 support (will not work headless)
- Install dependencies: `pip install pygame numpy matplotlib tk tqdm pyqtgraph pyqt5 scipy`

## Architecture Overview

The main application (`diablos_modern.py`) uses a modern PyQt5 user interface but leverages the original, battle-tested simulation engine from `lib/lib.py` and `lib/functions.py`. This engine has been significantly refactored and stabilized to handle complex simulation scenarios.

An experimental, clean architecture based on a `SimulationController` and distinct services exists in `diablos_main_new.py` but is not the current production version.

## Simulation Engine Enhancements

The core simulation engine (`lib/lib.py`) has undergone significant stabilization and feature enhancements:

- **Algebraic Loop Resolution:** The algebraic loop detection logic has been improved to use a topological sort algorithm, which accurately identifies algebraic loops.
- **Stateful Simulation:** Memory blocks (Integrators, Transfer Functions) are now correctly handled at initialization and during each step of the simulation loop, ensuring their internal states are properly updated.
- **Robust Error Handling:** The simulation engine now returns specific error messages to the UI, allowing for clear error messages to be displayed to the user instead of failing silently.
- **Batch Mode Simulation:** The simulation can now be run in "batch mode" for faster execution, decoupled from the UI's frame rate.
- **Robust Block Functions:** The `integrator` and `adder` blocks are now more robust to dimension mismatches between their inputs.

## Transfer Function Enhancements
- The transfer function block now correctly handles changes in denominator order. The `init_conds` parameter is dynamically resized to match the required number of states, with new states defaulting to zero.
- The `b_type` of the transfer function block is now dynamically updated when the parameters are changed, ensuring that it is correctly identified as a memory block when it is strictly proper.

## Property Editor Enhancements
- The modal parameter-setting dialog has been replaced with a non-intrusive property editor panel at the bottom of the window.
- Block parameters can be edited directly in the panel.
- Changes are applied automatically when editing is complete (e.g., on focus-out or when Enter is pressed), removing the need for 'Confirm'/'Cancel' buttons.
- The property editor panel is now more elegantly styled with better spacing and uniform control sizes.
- The panel dynamically resizes its height to fit the content of the selected block, reducing the need for scrolling.

## Modern UI (`diablos_modern.py`)

The modern UI provides a professional and feature-rich environment for building and running simulations.

**Key Features (Complete):**
- **Modern Theming**: Complete dark/light theme system with instant switching.
- **Enhanced Layout**: Resizable splitter panels for palette, canvas, and properties.
- **Modern Toolbar**: Zoom controls, theme toggle, and simulation controls with correct state management.
- **Professional Styling**: Consistent and modern appearance across all components.
- **Interactive Block Palette**: Draggable blocks organized by category.
- **Modern Canvas**: Full block rendering, mouse events, zoom, and pan.
- **Drag-and-Drop**: Create blocks by dragging from the palette to the canvas.
- **Full Mouse Interaction**: Select, drag, and connect blocks and lines.
- **Streamlined Property Editor**: Edit block parameters directly in a panel with changes applied automatically on completion. The panel also dynamically resizes to fit its content.
- **Stable Simulation**: Fully integrated with the enhanced simulation engine.

## Current Status Summary

**Modern UI (`diablos_modern.py`):**
- ✅ **Recommended Version**: Fully functional, stable, and feature-rich.
- ✅ **Complete Functionality**: Block creation, drag, connections, simulation, and plotting.
- ✅ **Fixed Issues**: All major simulation and algebraic loop issues have been resolved. UI state inconsistencies for simulation controls have been fixed.

**Other Versions:**
- `diablos_improved.py`: Stable, but uses the legacy UI. Good for reference.
- `diablosM1_main.py`: Legacy. Use for historical reference only.
- `diablos_main_new.py`: Experimental and incomplete. Do not use for production.

**Recommendation:**
- **Always use `diablos_modern.py` for all tasks.**
