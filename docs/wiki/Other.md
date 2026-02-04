# Other Blocks

List of available blocks in the **Other** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [BodeMag](#bodemag) | Right-click to generate a Bode magnitude plot from a connected Transfer Function block. |
| [BodePhase](#bodephase) | Right-click to generate a Bode phase plot from a connected Transfer Function block. |
| [External](#external) | External Function Block. |
| [Nyquist](#nyquist) | Right-click to generate a Nyquist plot from a connected dynamic block. |
| [RootLocus](#rootlocus) | Root Locus Plotter. |

---

### BodeMag

Right-click to generate a Bode magnitude plot from a connected Transfer Function block.

Displays the frequency response magnitude (gain in dB) vs frequency (rad/s) on a log scale.

Usage:
Connect to a Transfer Function or State Space block.
Right-click to generate plot.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|

**Ports**: 1 In, 0 Out

---

### BodePhase

Right-click to generate a Bode phase plot from a connected Transfer Function block.

Displays the frequency response phase (degrees) vs frequency (rad/s) on a log scale.

Usage:
Connect to a Transfer Function or State Space block.
Right-click to generate plot.

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

### Nyquist

Nyquist Plot.

Displays the frequency response as a polar plot (Real vs Imaginary parts) for stability analysis.

Usage:
Connect to a Transfer Function or State Space block.
Right-click to generate plot.
Use for stability analysis: check encirclements of the -1 point.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|

**Ports**: 1 In, 0 Out

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
