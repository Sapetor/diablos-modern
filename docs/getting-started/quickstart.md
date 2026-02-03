# Quick Start Guide

This guide will walk you through creating your first simulation in DiaBloS Modern.

## Starting the Application

```bash
python diablos_modern.py
```

## Creating a Simple System

### Step 1: Add a Source Block

1. Open the block palette on the left
2. Find "Sources" category
3. Drag a "Step" block onto the canvas

### Step 2: Add a Dynamic Block

1. Find "Control" category
2. Drag an "Integrator" block onto the canvas

### Step 3: Add a Sink Block

1. Find "Sinks" category
2. Drag a "Scope" block onto the canvas

### Step 4: Connect the Blocks

1. Click on the output port of the Step block
2. Drag to the input port of the Integrator
3. Connect the Integrator output to the Scope input

### Step 5: Run the Simulation

1. Set simulation time (e.g., 10 seconds) in the toolbar
2. Click the "Run" button (or press F5)
3. View results in the Scope plot window

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F5 | Run simulation |
| Ctrl+S | Save diagram |
| Ctrl+O | Open diagram |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Delete | Delete selected |
| Ctrl+C | Copy |
| Ctrl+V | Paste |

## Next Steps

- Explore the [Block Reference](../user-guide/block-reference.md)
- Learn about [PDE Blocks](../wiki/PDE_Blocks.md)
- Try the [Optimization Primitives](../wiki/Optimization_Primitives.md)
