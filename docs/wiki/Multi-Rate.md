# Multi-Rate Simulation

DiaBloS Modern supports multi-rate simulation where blocks can operate at different sample rates. This is essential for modeling systems that combine continuous-time dynamics with discrete-time controllers or mixed-rate digital systems.

## Overview

In multi-rate simulation:
- **Continuous blocks** execute every simulation timestep (`sampling_time = -1`)
- **Discrete blocks** execute only at their specified sample times (`sampling_time > 0`)
- **Rate transition blocks** handle signal conversion between different rates

## Sample Time Convention

The `sampling_time` parameter controls block execution:

| Value | Meaning |
|-------|---------|
| `-1.0` | Continuous (execute every timestep) |
| `0.0` | Inherited (match fastest connected input) |
| `> 0` | Fixed discrete rate (e.g., `0.1` = 10 Hz) |

## Key Blocks

### ZeroOrderHold (ZOH)

Samples a continuous signal at fixed intervals and holds the value constant between samples. Produces a staircase output.

```
Input:  ~~~~~ (continuous sine)
Output: _|‾|_|‾ (staircase)
```

### FirstOrderHold (FOH)

Samples at fixed intervals and linearly extrapolates using the slope between the last two samples. Produces a sawtooth-like output with linear segments.

```
Input:  ~~~~~ (continuous sine)
Output: /\/\/\ (linear segments)
```

### RateTransition

Converts signals between different sample rates with multiple modes:

- **ZOH Mode**: Hold last sample (simple, introduces steps)
- **Linear Mode**: Smooth interpolation between samples (recommended for upsampling)
- **Filter Mode**: Low-pass filter for anti-aliasing (recommended for downsampling)
- **Sample Mode**: Take latest sample (simple downsampling)
- **Average Mode**: Average samples in window (downsampling with smoothing)

## Example: Multi-Rate Demo

The `examples/multirate_demo.diablos` demonstrates:

1. A 2 Hz sine wave (continuous source)
2. ZOH sampling at 10 Hz (produces staircase)
3. RateTransition with Linear mode (smooths the staircase into ramps)
4. FOH for comparison (extrapolates with slope)

### Block Diagram

```
Sine 2Hz ──┬──> ZOH 10Hz ──┬──> RateTransition (Linear) ──> Scope
           │               │
           │               └──> Scope (ZOH comparison)
           │
           └──> FOH 10Hz ──> Scope (FOH comparison)
```

### Expected Output

| Signal | Appearance |
|--------|------------|
| Continuous | Smooth sine wave |
| ZOH 10Hz | Staircase (steps at 0.1s intervals) |
| RateTransition Linear | Smooth ramps between ZOH steps |
| FOH | Sawtooth extrapolation |

## Usage Guidelines

### When to Use Rate Transition

1. **Upsampling (slow to fast)**: Use `Linear` mode for smooth output
2. **Downsampling (fast to slow)**: Use `Filter` or `Average` mode to prevent aliasing
3. **Simple passthrough**: Use `ZOH` or `Sample` mode

### Best Practices

1. **Place RateTransition blocks** between blocks with different sample rates
2. **Use Linear mode** when smoothness matters (e.g., feeding continuous controllers)
3. **Use Filter mode** when downsampling to prevent aliasing artifacts
4. **Set `sampling_time: -1`** on RateTransition to ensure continuous output

### Validation Warnings

The diagram validator will warn about:
- Non-integer rate ratios between connected discrete blocks
- Discrete signals feeding continuous blocks without rate transition

## Technical Details

### Linear Mode Algorithm

The Linear mode implements smooth ramping between input samples:

1. Detects when input value changes (new sample from upstream)
2. Computes ramp duration from time between input changes
3. Interpolates: `output = start + alpha * (end - start)` where `alpha = (t - t_start) / duration`
4. Clamps alpha to [0, 1] to stay within the ramp

### Sample Time Propagation

During simulation initialization:
1. Explicit sample times are resolved from block parameters
2. Inherited rates (`sampling_time = 0`) propagate from fastest connected input
3. Discrete blocks schedule their execution times

### Held Outputs

When a discrete block skips execution (not at sample time):
- Its held outputs from the last execution are propagated to downstream blocks
- This ensures continuous blocks always have valid inputs

## See Also

- [Control Blocks](Control) - ZOH, FOH, RateTransition documentation
- [Examples](Examples) - Multi-rate demo and other examples
