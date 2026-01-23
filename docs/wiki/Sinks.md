# Sinks Blocks

List of available blocks in the **Sinks** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Assert](#assert) | Stops simulation if input violates condition. Modes: >0, <0, >=0, <=0, ==0, !=0, finite. |
| [Display](#display) | Numerical Display. |
| [Export](#export) | Data Export. |
| [FFT](#fft) | Spectrum Analyzer (FFT). |
| [Scope](#scope) | Oscilloscope / Plotter. |
| [Term](#term) | Signal Terminator. |
| [XYGraph](#xygraph) | XY Plotter. |

---

### Assert

Stops simulation if input violates condition. Modes: >0, <0, >=0, <=0, ==0, !=0, finite.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `condition` | string | `>0` | Condition: >0, <0, >=0, <=0, ==0, !=0, finite |
| `message` | string | `Assertion failed` | Error message on failure. |
| `enabled` | bool | `True` | Enable/disable assertion check. |

**Ports**: 1 In, 0 Out

---

### Display

Numerical Display.

Shows the current value of the input signal.

Parameters:
- Format: standard Python f-string format (e.g. {:.2f}).
- Label: Text label prefix.

Usage:
Monitor scalar values during simulation.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `format` | string | `%.3f` | Printf-style format string. |
| `label` | string | `` | Optional label prefix. |

**Ports**: 1 In, 0 Out

---

### Export

Data Export.

Saves simulation data to a file (e.g., .npz, .mat, .csv).

Parameters:
- Filename: Destination file path.
- Variable Name: Name of variable in saved file.

Usage:
Save results for post-processing in Python or MATLAB.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `str_name` | string | `default` |  |

**Ports**: 1 In, 0 Out

---

### FFT

Spectrum Analyzer (FFT).

Computes and plots the Frequency Spectrum (Magnitude) of the input.

Parameters:
- Window: Tapering window (Hamming, Hanning, Rectangular).
- Log Scale: Use logarithmic X/Y axes.

Usage:
Analyze frequency content, resonances, or noise.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `title` | string | `FFT Spectrum` | Plot title. |
| `window` | string | `hann` | Window function: 'hann', 'hamming', 'blackman', 'none'. |
| `normalize` | bool | `True` | Normalize magnitude to 0-1. |
| `log_scale` | bool | `False` | Use dB scale for magnitude. |

**Ports**: 1 In, 0 Out

---

### Scope

Oscilloscope / Plotter.

Displays time-domain signals during or after simulation.

Parameters:
- Title: Plot window title.
- Labels: Comma-separated legend labels.

Usage:
The primary way to visualize simulation results.
Double-click after simulation to re-open the plot.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `labels` | string | `default` |  |

**Ports**: 1 In, 0 Out

---

### Term

Signal Terminator.

Safely terminates an unused output signal.

Usage:
Prevents 'Unconnected Output' warnings during validation.

**Ports**: 1 In, 0 Out

---

### XYGraph

XY Plotter.

Plots Y vs X (parametric plot).

Connnections:
- Port 1: X-axis signal.
- Port 2: Y-axis signal.

Usage:
Phase portraits, hysteresis loops, or orbital paths.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `x_label` | string | `X` | X-axis label. |
| `y_label` | string | `Y` | Y-axis label. |
| `title` | string | `XY Plot` | Plot title. |

**Ports**: 2 In, 0 Out

---
