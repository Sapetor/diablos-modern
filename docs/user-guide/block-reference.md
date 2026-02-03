# Block Reference

Complete reference for all blocks in DiaBloS Modern.

See the [Blocks API](../api/blocks.md) for programmatic documentation.

## Sources

| Block | Description |
|-------|-------------|
| Constant | Outputs a constant value |
| Step | Step function at specified time |
| Ramp | Linear ramp signal |
| Sine | Sinusoidal signal |
| Noise | Gaussian random noise |
| WaveGenerator | Multi-waveform generator |

## Math

| Block | Description |
|-------|-------------|
| Gain | Multiply by constant |
| Sum | Add/subtract signals |
| Product | Multiply signals |
| MathFunction | Apply math functions |
| Abs | Absolute value |
| Exponential | Exponential function |

## Control

| Block | Description |
|-------|-------------|
| Integrator | Numeric integration |
| Derivative | Numeric differentiation |
| PID | PID controller |
| TransferFcn | Transfer function |
| StateSpace | State-space system |
| Saturation | Output saturation |
| RateLimiter | Rate limiting |
| Deadband | Dead zone |
| Hysteresis | Hysteresis element |

## PDE

| Block | Description |
|-------|-------------|
| HeatEquation1D | 1D heat diffusion |
| WaveEquation1D | 1D wave equation |
| AdvectionEquation1D | 1D advection |
| DiffusionReaction1D | 1D diffusion-reaction |
| HeatEquation2D | 2D heat diffusion |
| WaveEquation2D | 2D wave equation |
| AdvectionEquation2D | 2D advection |

## Field Processing

| Block | Description |
|-------|-------------|
| FieldProbe | Sample 1D field at position |
| FieldProbe2D | Sample 2D field at position |
| FieldScope | Visualize 1D field |
| FieldScope2D | Visualize 2D field |
| FieldSlice | Extract 1D slice from 2D field |

## Sinks

| Block | Description |
|-------|-------------|
| Scope | Time-series plot |
| XYGraph | X-Y plot |
| Display | Numeric display |
| FFT | Frequency spectrum |
| Terminator | Signal terminator |

## Routing

| Block | Description |
|-------|-------------|
| Mux | Combine signals |
| Demux | Split vector |
| Switch | Signal switch |
| Selector | Select vector elements |
| Goto/From | Signal routing |

## Subsystem

| Block | Description |
|-------|-------------|
| Subsystem | Hierarchical grouping |
| Inport | Subsystem input |
| Outport | Subsystem output |
