# DiaBloS Block Reference

Blocks marked with ⚡ are supported by the **Fast Solver (Compiled Mode)**.

## Sources

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Step** ⚡ | Unit step signal | `delay`, `value` |
| **Ramp** ⚡ | Linear ramp signal | `slope`, `delay` |
| **Sine** ⚡ | Sinusoidal signal | `amplitude`, `frequency`, `phase`, `bias` |
| **Constant** ⚡ | Fixed value output | `value` (scalar or vector) |
| **Noise** | Random noise generator | `mean`, `std` |
| **PRBS** | Pseudo-random binary sequence | `amplitude`, `period` |
| **Exponential** ⚡ | Exponential signal | `amplitude`, `rate` |

## Math

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Sum** ⚡ | Add/subtract signals | `signs` (e.g., "++−") |
| **Gain** ⚡ | Multiply by scalar/vector/matrix | `gain` |
| **SigProduct** ⚡ | Element-wise multiplication | — |
| **Abs** ⚡ | Absolute value | — |

## Control

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Integrator** ⚡ | Continuous integration | `initial_condition`, `method` |
| **Derivative** | Continuous differentiation | `filter_coeff` |
| **PID** ⚡ | PID controller | `Kp`, `Ki`, `Kd` |
| **TranFn** ⚡ | Transfer function | `numerator`, `denominator` |
| **StateSpace** ⚡ | State-space model | `A`, `B`, `C`, `D` |
| **Delay** | Discrete N-step delay | `delay_steps` |
| **TransportDelay** | Continuous time delay | `delay_time` |

## Nonlinear

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Saturation** ⚡ | Clamp signal to limits | `lower_limit`, `upper_limit` |
| **Deadband** ⚡ | Zero output in dead zone | `start`, `end` |
| **Hysteresis** | Hysteresis loop | `rising_th`, `falling_th` |
| **RateLimiter** ⚡ | Limit rate of change | `rising_rate`, `falling_rate` |
| **Switch** ⚡ | Select input based on control | `threshold`, `n_inputs`, `mode` |

## Sinks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Scope** ⚡ | Plot time-domain signals | `title`, `labels` |
| **XYGraph** | Parametric X-Y plot | `title` |
| **Display** ⚡ | Show numerical value | `format`, `label` |
| **Export** | Save data to file | `filename` |
| **FFT** | Frequency spectrum | `window`, `log_scale` |
| **Terminator** ⚡ | Sink for unused signals | — |
| **BodeMag** | Bode magnitude plot | — |
| **RootLocus** | Root locus plot | — |

## Routing

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **Mux** ⚡ | Combine signals to vector | `n_inputs` |
| **Demux** ⚡ | Split vector to signals | `n_outputs` |
| **Selector** | Pick vector elements | `indices` |
| **Goto/From** ⚡ | Virtual connection | `tag` |

## Discrete

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| **ZeroOrderHold** | Sample and hold | — |
| **DiscreteTranFn** | Discrete transfer function | `numerator`, `denominator` |
| **DiscreteStateSpace** | Discrete state-space | `A`, `B`, `C`, `D` |
