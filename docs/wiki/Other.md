# Other Blocks

List of available blocks in the **Other** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [BodeMag](#bodemag) | Right-click to generate a Bode magnitude plot from a connected Transfer Function block. |
| [External](#external) | External Function Block. |
| [RootLocus](#rootlocus) | Root Locus Plotter. |

---

### BodeMag

Right-click to generate a Bode magnitude plot from a connected Transfer Function block.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|

**Ports**: 1 In, 0 Out

---

### External

External Function Block.

> **⚠️ NOT IMPLEMENTED**: This block is a stub. It returns an error when executed.
> If you need custom Python code, consider using the Python block or submitting a feature request.

Intended to execute custom Python code loaded from an external file.

Parameters:
- Script Path: Path to the .py file.
- Function Name: Name of the function to call.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | string | `` | Path to external Python script |
| `function` | string | `execute` | Function name to call |

**Ports**: 1 In, 1 Out

**Status**: Returns error `{'E': True, 'error': 'External file not loaded: ...'}` when executed.

---

### RootLocus

Root Locus Plotter.

Analyzes the closed-loop poles of a system as a parameter varies.

Usage:
Connect to a Transfer Function or State Space block.
Right-click to generate plot.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|

**Ports**: 1 In, 0 Out

---
