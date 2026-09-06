# Block Reference

Every block DiaBloS ships, grouped by the category it appears under in the
palette. 96 block types are registered; one (`External`) is a disabled stub and
is hidden from the palette.

`scripts/audit_wiki_docs.py` checks these pages against `blocks/` and reports
any block that has no entry here.

## Block library

| Category | Contents |
|---|---|
| **[Sources](Sources.md)** | Step, Ramp, Sine, Constant, Chirp, PRBS, Impulse, Noise, RandomSource, WaveGenerator, FromFile |
| **[Sinks](Sinks.md)** | Scope, XYGraph, Display, FFT, Export, Assert, Term, AgentScope |
| **[Math](Math.md)** | Gain, MatrixGain, Sum, Product, SgProd, Abs, Exp, Deriv, MathFunction, Function, LookupTable1D/2D |
| **[Control](Control.md)** | Integrator, PID, TranFn, StateSpace, Saturation, RateLimiter, Deadband, Hysteresis, delays, discrete and multi-rate blocks, LQR |
| **[Logic](Logic.md)** | RelationalOperator, CompareToConstant, LogicalOperator |
| **[Routing](Routing.md)** | Mux, Demux, Switch, Selector, Goto/From tags, Inport/Outport, Subsystem, PacketLoss, NetworkChannel |
| **[Analysis](Analysis.md)** | BodeMag, BodePhase, Nyquist, RootLocus (right-click markers) |
| **[PDE](PDE.md)** | Heat, wave, advection and diffusion-reaction equations in 1D/2D, field processing, FieldScope |
| **[Optimization](Optimization.md)** | Parameter optimization with cost functions and constraints |
| **[Optimization Primitives](Optimization-Primitives.md)** | Build optimization algorithms as diagrams (gradients, Adam, Momentum, solvers) |
| **[Other](Other.md)** | External (stub) |

## Features

- **[Multi-Rate Simulation](Multi-Rate.md)** - discrete blocks at independent sample rates
- **[Subsystems](Subsystems_Architecture.md)** - hierarchy, masks and library blocks
- **[Modern UI](ModernUI_Architecture.md)** - how the canvas and its managers fit together

## Examples

- **[Examples](Examples.md)** - demo and verification diagrams

## Core documentation

- [User Manual](../USER_MANUAL.md)
- [Running Simulations](../user-guide/running-simulations.md)
- [Analysis & Experiments](../user-guide/analysis.md)
- [Developer Guide](../DEVELOPER_GUIDE.md)
