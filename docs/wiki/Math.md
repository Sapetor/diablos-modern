# Math Blocks

List of available blocks in the **Math** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Abs](#abs) | Absolute Value. |
| [Deriv](#deriv) | Time Derivative (du/dt). |
| [Exp](#exp) | Exponential Signal. |
| [Function](#function) | Evaluates a sandboxed Python expression of the inputs and time. |
| [Gain](#gain) | Scales the input signal by a specified Gain. |
| [LookupTable1D](#lookuptable1d) | 1-D interpolated breakpoint table, y = f(x). |
| [LookupTable2D](#lookuptable2d) | 2-D interpolated table over a regular grid, z = f(x, y). |
| [MathFunction](#mathfunction) | Apply a mathematical function (sin, cos, sqrt, etc.). |
| [MatrixGain](#matrixgain) | Scalar, vector, or matrix gain. |
| [Product](#product) | Multiplies or divides multiple input signals. |
| [SgProd](#sgprod) | Computes the element-wise product of input signals. |
| [Sum](#sum) | Adds or subtracts multiple input signals. |

---

### Abs

Absolute Value.

Computes the absolute value of the input signal.
y = |u|

Usage:
Used in magnitude calculations, rectifiers, or error metrics.

**Ports**: 1 In, 1 Out

---

### Deriv

Time Derivative (du/dt).

Approximates the time derivative of the input.

Warning:
Derivative is sensitive to noise. Use with a low-pass filter if possible.

Parameters:
- Filter Coefficient: Bandwidth of internal filter (if implemented).

Usage:
Computing velocity from position, or rate of change.

**Ports**: 1 In, 1 Out

---

### Exp

Exponential Signal.

y(t) = Amplitude * exp(Rate * t)

Parameters:
- Amplitude: Initial value.
- Rate: Growth (+) or Decay (-) constant.

Usage:
Transient analysis or unstable system simulation.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `a` | float | `1.0` |  |
| `b` | float | `1.0` |  |

**Ports**: 1 In, 1 Out

---

### Gain

Scales the input signal by a specified Gain.

Supports:
- Scalar Gain: y = K * u (element-wise).
- Vector Gain: y = K * u (element-wise, if K is a vector).
- Matrix Gain: y = K @ u (Matrix Multiplication, if K is a matrix).

Usage:
Use Matrix Gain (nested lists like [[1], [2]]) to expand scalars to vectors.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gain` | matrix | `1.0` | Gain value: scalar, vector, or matrix. Matrix uses y = K @ u. |

**Ports**: 1 In, 1 Out

---

### Product

Multiplies or divides multiple input signals.

Similar to Sum block but for multiplication/division operations.

Parameters:
- Ops: A string of '*' and '/' characters defining the operation for each input port.
  Example: '*/' creates 2 ports: (in1 / in2).
  Example: '**' creates 2 ports: (in1 * in2).

Usage:
Signal modulation, ratio calculations, or Newton's method (f/f').

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `ops` | string | `**` | Operations string: '*' for multiply, '/' for divide |

**Ports**: 2 In (configurable), 1 Out

---

### SgProd

Computes the element-wise product of input signals.

Operation:
y = u1 * u2 * ... * un (Element-wise multiplication).

Parameters:
- Inputs: Number of input ports to multiply.

Usage:
Used for modulation, mixing, or non-linear scaling.

**Ports**: 2 In, 1 Out

---

### Sum

Adds or subtracts multiple input signals.

Parameters:
- Signs: A string of '+' and '-' characters defining the operation for each input port.
  Example: '+-+' creates 3 ports: (in1 - in2 + in3).

Usage:
Standard summing junction for feedback loops (set signs to '+-').

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sign` | string | `++` |  |

**Ports**: 2 In, 1 Out

---

### MathFunction

Apply a mathematical function to the input signal.

Supported Functions:
- Trigonometric: sin, cos, tan, asin, acos, atan
- Exponential: exp, log (ln), log10, sqrt, square
- Operational: sign, abs, ceil, floor, reciprocal

You can also enter any valid Python expression using `u` (input) and `t` (time), e.g., `u**2 + sin(t*10)`.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `function` | choice | `sin` | Function to apply (sin, cos, tan, exp, log, sqrt, etc.) |

**Ports**: 1 In, 1 Out

---
### Function

Evaluates a user-supplied Python expression of the block's inputs and time.

Inputs are exposed twice: as the 0-indexed list `u[0]`, `u[1]`, ... and as the
1-indexed aliases `u1`, `u2`, ... The current simulation time is bound to `t`.
Bare numpy math (`sin`, `cos`, `exp`, `sqrt`, `sign`, `abs`, ...), the prefixed
forms (`np.tanh(u[0])`, `math.atan2(u[1], u[0])`) and the constants `pi` and `e`
are available.

The number of input ports is editable from the property panel. Expressions go
through `lib/safe_eval.py`, an allowlist AST interpreter, so imports, attribute
escapes and statements are rejected rather than executed.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `expression` | string | `u[0]` | Python expression using `u[i]`/`u1`.. and `t`, e.g. `sin(u[0]) + t`. |

**Ports**: 1 In (editable count), 1 Out

---

### LookupTable1D

One-dimensional lookup table: `y = f(x)` by interpolating a breakpoint table.

Usage:
Model a measured nonlinearity — a valve characteristic, a motor torque curve, a
sensor calibration.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `x_values` | string | `[0, 1, 2, 3]` | Breakpoints (distinct). |
| `y_values` | string | `[0, 1, 4, 9]` | Table values, same length as x_values. |
| `interpolation` | choice | `linear` | `linear` or `nearest`. |
| `extrapolation` | choice | `clip` | Outside the table: `clip` (hold the edge) or `linear` (extend the end slope). |

**Ports**: 1 In, 1 Out

---

### LookupTable2D

Two-dimensional lookup table: `z = f(x, y)` over a regular grid.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `x_values` | string | `[0, 1, 2]` | Row breakpoints (distinct). |
| `y_values` | string | `[0, 1]` | Column breakpoints (distinct). |
| `z_table` | string | `[[0, 1], [2, 3], [4, 5]]` | Values, shape `[len(x_values)][len(y_values)]`. |
| `interpolation` | choice | `linear` | `linear` or `nearest`. |
| `extrapolation` | choice | `clip` | Outside the grid: `clip` or `linear`. |

**Ports**: 2 In, 1 Out

---

### MatrixGain

Gain block whose gain may be a scalar, a vector or a matrix.

- Scalar `2.5`: `y = K * u`
- Vector `[1, 2, 3]`: `y = K .* u` (element-wise, same length)
- Matrix `[[1, 0], [0, 2]]`: `y = K @ u` (matrix-vector product)

The gain field also accepts a workspace variable name defined in the
[Variable Editor](../VARIABLE_EDITOR_GUIDE.md).

Drawn as a triangle, like `Gain`, with the gain value inside.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gain` | string | `1.0` | Gain: scalar, vector, or matrix. |

**Ports**: 1 In, 1 Out

---
