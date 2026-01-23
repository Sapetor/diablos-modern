# Control Blocks

List of available blocks in the **Control** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Deadband](#deadband) | Dead Zone / Deadband. |
| [Delay](#delay) | Discrete Integer Delay (z^-N). |
| [DiscreteStateSpace](#discretestatespace) | Discrete State-Space Model. |
| [DiscreteTranFn](#discretetranfn) | Represents a discrete-time linear time-invariant system as a transfer function in z-domain. |
| [Hysteresis](#hysteresis) | Hysteresis Relay. |
| [Integrator](#integrator) | Continuous-time Integrator (1/s). |
| [PID](#pid) | PID Controller. |
| [RateLimiter](#ratelimiter) | Rate Limiter. |
| [Saturation](#saturation) | Limits the input signal to a specified range. |
| [StateSpace](#statespace) | Continuous State-Space Model. |
| [TranFn](#tranfn) | Represents a linear time-invariant system as a transfer function. |
| [TransportDelay](#transportdelay) | Transport Delay / Time Delay. |
| [ZeroOrderHold](#zeroorderhold) | Zero-Order Hold (ZOH). |

---

### Deadband

Dead Zone / Deadband.

Outputs zero when the input is within the specified range [Start, End].

Function:
- u < Start: y = u - Start
- Start <= u <= End: y = 0
- u > End: y = u - End

Parameters:
- Start/End: Lower and Upper bounds of the zero region.

Usage:
Models mechanical play (backlash) or noise thresholds.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `start` | float | `-0.5` | Start of dead zone (lower threshold). |
| `end` | float | `0.5` | End of dead zone (upper threshold). |

**Ports**: 1 In, 1 Out

---

### Delay

Discrete Integer Delay (z^-N).

Delays the input by a fixed number of execution steps.
y[k] = u[k - N]

Parameters:
- Delay Steps: Number of steps (N).
- Initial Value: Output for k < N.

Usage:
Models digital latency or buffer pipelines.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `delay_steps` | int | `1` | Number of time steps to delay. |
| `initial_value` | float | `0.0` | Output before delay buffer fills. |

**Ports**: 1 In, 1 Out

---

### DiscreteStateSpace

Discrete State-Space Model.

x[k+1] = Ax[k] + Bu[k]
y[k] = Cx[k] + Du[k]

Parameters:
- A, B, C, D: Discrete system matrices.
- Sampling Time: Execution rate.

Usage:
Digital Modern Control (MIMO).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `A` | list | `[[0.0]]` |  |
| `B` | list | `[[1.0]]` |  |
| `C` | list | `[[1.0]]` |  |
| `D` | list | `[[0.0]]` |  |
| `init_conds` | list | `[0.0]` |  |
| `sampling_time` | float | `-1.0` |  |

**Ports**: 1 In, 1 Out

---

### DiscreteTranFn

Represents a discrete-time linear time-invariant system as a transfer function in z-domain.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `numerator` | list | `[1.0, 0.0]` |  |
| `denominator` | list | `[1.0, -0.5]` |  |
| `sampling_time` | float | `-1.0` |  |

**Ports**: 1 In, 1 Out

---

### Hysteresis

Hysteresis Relay.

Switches output based on history (memory effect).

Logic:
- Output = ON (1) if Input > High Threshold
- Output = OFF (0) if Input < Low Threshold
- Retains previous state if Input is between thresholds.

Parameters:
- Low/High Thresholds: Switching points.

Usage:
Thermostats, Schmitt Triggers, On-Off Control.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `upper` | float | `0.5` | Threshold to switch high. |
| `lower` | float | `-0.5` | Threshold to switch low. |
| `high` | float | `1.0` | Output when high. |
| `low` | float | `0.0` | Output when low. |

**Ports**: 1 In, 1 Out

---

### Integrator

Continuous-time Integrator (1/s).

Computes the time integral of the input signal.
y(t) = y(0) + integral(u(t) dt)

Parameters:
- Initial Condition: Value of the output at start time.
- Limit Output: Enable saturation limits on the integral.
- Method: Integration method (e.g., RK45, Forward Euler).

Usage:
Fundamental block for building dynamic system models.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `init_conds` | float | `0.0` |  |
| `method` | string | `SOLVE_IVP` |  |

**Ports**: 1 In, 1 Out

---

### PID

PID Controller.

u(t) = P + I + D

Parameters:
- Proportional (P): Kp * error
- Integral (I): Ki * integral(error)
- Derivative (D): Kd * derivative(error)
- Filter Coeff (N): Derivative filter bandwidth (Low-pass).
  D term = Kd * N * s / (s + N)

Usage:
Feedback control. Tuning parameters Kp, Ki, Kd.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `Kp` | float | `1.0` | Proportional gain. |
| `Ki` | float | `0.0` | Integral gain. |
| `Kd` | float | `0.0` | Derivative gain. |
| `N` | float | `20.0` | Derivative filter coefficient (higher = less smoothing). |
| `u_min` | float | `-inf` | Output lower limit. |
| `u_max` | float | `inf` | Output upper limit. |

**Ports**: 2 In, 1 Out

---

### RateLimiter

Rate Limiter.

Limits the rate of change (slope) of the input signal.

Parameters:
- Rising Slew Rate: Max positive slope (dy/dt).
- Falling Slew Rate: Max negative slope (dy/dt) (usually negative).

Usage:
Prevents abrupt changes in control signals or models actuator speed limits.
Useful for smoothing setpoints.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `rising_slew` | float | `inf` | Max positive slope (units/sec). |
| `falling_slew` | float | `inf` | Max negative slope magnitude (units/sec). |

**Ports**: 1 In, 1 Out

---

### Saturation

Limits the input signal to a specified range.

Output:
- Upper Limit if u > Upper Limit
- Lower Limit if u < Lower Limit
- u otherwise

Usage:
Prevents windup or limits actuator signals.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `min` | float | `-inf` | Lower saturation limit. |
| `max` | float | `inf` | Upper saturation limit. |

**Ports**: 1 In, 1 Out

---

### StateSpace

Continuous State-Space Model.

dx/dt = Ax + Bu
y = Cx + Du

Parameters:
- A, B, C, D: System matrices.
- Initial State: x(0) vector.

Usage:
For Modern Control (MIMO systems). Can model any linear system.
Matrices can be entered as nested lists: [[1, 0], [0, 1]].

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `A` | list | `[[0.0]]` |  |
| `B` | list | `[[1.0]]` |  |
| `C` | list | `[[1.0]]` |  |
| `D` | list | `[[0.0]]` |  |
| `init_conds` | list | `[0.0]` |  |

**Ports**: 1 In, 1 Out

---

### TranFn

Represents a linear time-invariant system as a transfer function.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `numerator` | list | `[1.0]` |  |
| `denominator` | list | `[1.0, 1.0]` |  |

**Ports**: 1 In, 1 Out

---

### TransportDelay

Transport Delay / Time Delay.

Delays the input signal by a specified time amount.
y(t) = u(t - Delay)

Parameters:
- Time Delay: Amount of delay in seconds.
- Initial Output: Output value before t < Delay.
- Buffer Size: Max history length (increase if simulation is long/fast).

Usage:
Models pipe flow, conveyor belts, or communication latency.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `delay_time` | float | `0.1` | Delay time τ in seconds. |
| `initial_value` | float | `0.0` | Output before delay time elapses. |

**Ports**: 1 In, 1 Out

---

### ZeroOrderHold

Zero-Order Hold (ZOH).

Samples the input signal at a fixed rate and holds it constant between samples.

Parameters:
- Sampling Time: The period (in seconds) between samples.

Usage:
Converts continuous signals to discrete (digital) steps.
Models triggers or ADCs.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `sampling_time` | float | `0.1` |  |

**Ports**: 1 In, 1 Out

---
