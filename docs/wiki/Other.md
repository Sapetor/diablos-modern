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

Executes custom Python code loaded from an external file.

Parameters:
- Script Path: Path to the .py file.
- Function Name: Name of the function to call.

Usage:
Integrate custom logic, complex math, or hardware interfaces not provided by standard blocks.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | string | `<no filename>` |  |

**Ports**: 1 In, 1 Out

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
