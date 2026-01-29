# Math Blocks

List of available blocks in the **Math** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Abs](#abs) | Absolute Value. |
| [Deriv](#deriv) | Time Derivative (du/dt). |
| [Exp](#exp) | Exponential Signal. |
| [Gain](#gain) | Scales the input signal by a specified Gain. |
| [MathFunction](#mathfunction) | Apply a mathematical function (sin, cos, sqrt, etc.). |
| [SgProd](#sgprod) | Computes the product of input signals. |
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

### SgProd

Computes the product of input signals.

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
