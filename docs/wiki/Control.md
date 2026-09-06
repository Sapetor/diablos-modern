# Control Blocks

List of available blocks in the **Control** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Deadband](#deadband) | Dead Zone / Deadband. |
| [Delay](#delay) | Discrete Integer Delay (z^-N). |
| [DiscreteStateSpace](#discretestatespace) | Discrete State-Space Model. |
| [DiscreteTranFn](#discretetranfn) | Represents a discrete-time linear time-invariant system as a transfer function in z-domain. |
| [FirstOrderHold](#firstorderhold) | First-Order Hold (FOH) - linear extrapolation between samples. |
| [Hysteresis](#hysteresis) | Hysteresis Relay. |
| [Integrator](#integrator) | Continuous-time Integrator (1/s). |
| [LQR](#lqr) | Linear-quadratic regulator gain designer (right-click to compute). |
| [PID](#pid) | PID Controller. |
| [RateLimiter](#ratelimiter) | Rate Limiter. |
| [RateTransition](#ratetransition) | Rate Transition for multi-rate simulation. |
| [Saturation](#saturation) | Limits the input signal to a specified range. |
| [StateSpace](#statespace) | Continuous State-Space Model. |
| [TranFn](#tranfn) | Represents a linear time-invariant system as a transfer function. |
| [TransportDelay](#transportdelay) | Transport Delay / Time Delay. |
| [VariableTransportDelay](#variabletransportdelay) | Transport delay whose τ is supplied on an input port. |
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
| `sampling_time` | float | `0.0` | Sample period in seconds (0=inherit from the upstream rate, >0=fixed rate). A discrete state recursion has no continuous-time meaning, so -1 is not a useful setting here: with no rate to inherit the block advances one sample per solver step and its response then depends on the simulation step size. |

**Ports**: 1 In, 1 Out

---

### DiscreteTranFn

Represents a discrete-time linear time-invariant system as a transfer function in z-domain.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `numerator` | list | `[1.0, 0.0]` |  |
| `denominator` | list | `[1.0, -0.5]` |  |
| `sampling_time` | float | `0.0` | Sample period in seconds (0=inherit from the upstream rate, >0=fixed rate). A z-domain block has no continuous-time meaning, so -1 is not a useful setting here: with no rate to inherit the block advances one sample per solver step and its response then depends on the simulation step size. |

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
- Method: Integration method (e.g., RK45, Forward Euler), used by the
  interpreted solver only; the compiled solver integrates the whole
  diagram with the method from Simulation settings.

Usage:
Fundamental block for building dynamic system models.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `init_conds` | float | `0.0` | Initial condition value |
| `method` | string | `SOLVE_IVP` | Integration method, applied by the interpreted solver only. The compiled (fast) solver assembles the whole diagram into one ODE system and integrates it with the solver method set in Simulation settings, which offers Euler and RK4 as well. |
| `ivp_method` | string | `RK45` | scipy ODE solver used when Method is SOLVE_IVP |
| `sampling_time` | float | `-1.0` | Sample time (-1=continuous, 0=inherited, >0=discrete) |

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

### FirstOrderHold

First-Order Hold (FOH).

Samples the input signal at a fixed rate and linearly extrapolates between samples.
Unlike ZOH which holds constant, FOH computes the slope between the last two samples
and extrapolates forward, producing a sawtooth-like output.

Parameters:
- Input Sample Time: The period (in seconds) between input samples.
- Sampling Time: Block execution rate (-1 for continuous output).

Usage:
- Smoother output than Zero-Order Hold
- Introduces one sample delay for extrapolation
- Good for continuous-to-discrete conversion when smoothness matters
- Models DACs with linear interpolation

Note: Output = linear extrapolation from previous samples using computed slope.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_sample_time` | float | `0.1` | Sample period for input (seconds). |
| `sampling_time` | float | `-1.0` | Block runs continuously (-1) to interpolate. |

**Ports**: 1 In, 1 Out

---

### RateTransition

Rate Transition Block for multi-rate simulation.

Safely transfers signals between blocks running at different sample rates.
Handles both upsampling (slow to fast) and downsampling (fast to slow).

Parameters:
- Output Sample Time: Target output sample period (seconds). Set to -1 for continuous.
- Transition Mode: How to handle rate conversion:
  - **ZOH**: Zero-order hold (hold last sample, good for upsampling)
  - **Linear**: Linear interpolation between samples (smooth ramps)
  - **Filter**: Low-pass filter (anti-alias for downsampling)
  - **Sample**: Take latest sample (simple downsampling)
  - **Average**: Average samples in window (downsampling)
- Filter Cutoff: Normalized cutoff frequency for Filter mode (0-0.5).

Usage:
- Place between blocks with different sample rates
- For slow to fast (upsampling): Use ZOH or Linear
- For fast to slow (downsampling): Use Filter, Sample, or Average

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `output_sample_time` | float | `0.1` | Target output sample period (seconds). |
| `transition_mode` | string | `ZOH` | Mode: ZOH, Linear, Filter, Sample, Average. |
| `filter_cutoff` | float | `0.4` | Normalized cutoff for Filter mode (0-0.5). |
| `sampling_time` | float | `-1.0` | Block runs continuously (-1) for smooth output. |

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
### LQR

Linear-quadratic regulator gain designer. This is a design tool, not a
simulation block: `execute()` does nothing and the block produces no output
signal.

Right-click the block and choose **Compute LQR gain** to solve the continuous
algebraic Riccati equation for

```
min ∫ (xᵀQx + uᵀRu) dt   subject to   dx/dt = Ax + Bu
```

The result window (**LQR Result: _name_**) reports the optimal gain `K`
(`u = -Kx`), the closed-loop eigenvalues of `A - BK` and the cost matrix `P`.

Connect the `plant` input to a `StateSpace` block to read `A` and `B`
automatically; otherwise enter them by hand. Every matrix field also accepts a
workspace variable name defined in the
[Variable Editor](../VARIABLE_EDITOR_GUIDE.md).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `A` | string | `[[0, 1], [0, 0]]` | State matrix (n×n). Matrix or workspace variable. |
| `B` | string | `[[0], [1]]` | Input matrix (n×m). Matrix or workspace variable. |
| `Q` | string | `[[1, 0], [0, 1]]` | State cost matrix (n×n, positive semidefinite). |
| `R` | string | `[[1]]` | Input cost matrix (m×m, positive definite). |

**Ports**: 1 In, 0 Out

See also: [Analysis & Experiments](../user-guide/analysis.md#lqr-design).

---

### VariableTransportDelay

Input-driven transport delay: `y(t) = u(t - τ(t))`.

Like `TransportDelay`, but the delay τ arrives at runtime on the second input
port instead of being a fixed parameter. τ is clamped to `[0, max_delay]`, and a
`(time, value)` history buffer is interpolated linearly for sub-sample accuracy.

Usage:
Variable network latency, transport lag that depends on a flow rate, or a
delay driven by a `RandomSource`.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_delay` | float | `1.0` | Maximum delay τ (s); also the buffer retention window. |
| `initial_value` | float | `0.0` | Output before the requested sample exists. |

**Ports**: 2 In (`in`, `tau`), 1 Out

---
