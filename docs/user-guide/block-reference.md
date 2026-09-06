# Block Reference

Every block registered in DiaBloS Modern, grouped by the category it appears
under in the palette. **In** and **Out** are the default port counts — several
blocks let you add ports from the property panel.

Each name links to the category page, which carries the full parameter table,
defaults and usage notes. `scripts/audit_wiki_docs.py` checks this listing
against `blocks/`.

For writing your own block, see the [Blocks API](../api/blocks.md) and the
[Developer Guide](../DEVELOPER_GUIDE.md).

## Sources

11 blocks. Full parameter tables: [Sources](../wiki/Sources.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Chirp](../wiki/Sources.md#chirp) | 0 | 1 | Generates a swept-frequency cosine (chirp) signal. |
| [Constant](../wiki/Sources.md#constant) | 0 | 1 | Outputs a constant value. |
| [FromFile](../wiki/Sources.md#fromfile) | 0 | 1 | Replays a recorded time-series from a file as a driving signal. |
| [Impulse](../wiki/Sources.md#impulse) | 0 | 1 | Generates a discrete impulse (Dirac delta approximation). |
| [Noise](../wiki/Sources.md#noise) | 0 | 1 | White Noise Generator. |
| [PRBS](../wiki/Sources.md#prbs) | 0 | 1 | Pseudo-Random Binary Sequence (PRBS). |
| [Ramp](../wiki/Sources.md#ramp) | 0 | 1 | Generates a Linear Ramp signal. |
| [RandomSource](../wiki/Sources.md#randomsource) | 0 | 1 | Random Source. |
| [Sine](../wiki/Sources.md#sine) | 0 | 1 | Generates a Sinusoidal signal. |
| [Step](../wiki/Sources.md#step) | 0 | 1 | Generates a Step function. |
| [WaveGenerator](../wiki/Sources.md#wavegenerator) | 0 | 1 | Generates various waveforms. |

## Math

12 blocks. Full parameter tables: [Math](../wiki/Math.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Abs](../wiki/Math.md#abs) | 1 | 1 | Absolute Value. |
| [Deriv](../wiki/Math.md#deriv) | 1 | 1 | Time Derivative (du/dt). |
| [Exp](../wiki/Math.md#exp) | 1 | 1 | Exponential Signal. |
| [Function](../wiki/Math.md#function) | 1 | 1 | Evaluate a Python expression of the inputs and time. |
| [Gain](../wiki/Math.md#gain) | 1 | 1 | Scales the input signal by a specified Gain. |
| [LookupTable1D](../wiki/Math.md#lookuptable1d) | 1 | 1 | 1-D lookup table: maps the input x through a breakpoint/value table by interpolation. |
| [LookupTable2D](../wiki/Math.md#lookuptable2d) | 2 | 1 | 2-D lookup table: maps inputs (x, y) through a regular grid of values by interpolation. |
| [MathFunction](../wiki/Math.md#mathfunction) | 1 | 1 | Apply a mathematical function to the input signal. |
| [MatrixGain](../wiki/Math.md#matrixgain) | 1 | 1 | Scales the input by a gain that can be a scalar, vector, or matrix. |
| [Product](../wiki/Math.md#product) | 2 | 1 | Multiplies or divides multiple input signals. |
| [SgProd](../wiki/Math.md#sgprod) | 2 | 1 | Computes the product of input signals. |
| [Sum](../wiki/Math.md#sum) | 2 | 1 | Adds or subtracts multiple input signals. |

## Logic

3 blocks. Full parameter tables: [Logic](../wiki/Logic.md).

| Block | In | Out | Description |
|---|---|---|---|
| [CompareToConstant](../wiki/Logic.md#comparetoconstant) | 1 | 1 | Compare the input against a constant: y = (in OP constant). |
| [LogicalOperator](../wiki/Logic.md#logicaloperator) | 2 | 1 | Boolean logic over the inputs (nonzero = True). |
| [RelationalOperator](../wiki/Logic.md#relationaloperator) | 2 | 1 | Compare two inputs: y = (in1 OP in2). |

## Control

17 blocks. Full parameter tables: [Control](../wiki/Control.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Deadband](../wiki/Control.md#deadband) | 1 | 1 | Dead Zone / Deadband. |
| [Delay](../wiki/Control.md#delay) | 1 | 1 | Discrete Integer Delay (z^-N). |
| [DiscreteStateSpace](../wiki/Control.md#discretestatespace) | 1 | 1 | Discrete State-Space Model. |
| [DiscreteTranFn](../wiki/Control.md#discretetranfn) | 1 | 1 | Represents a discrete-time linear time-invariant system as a transfer function in z-dom… |
| [FirstOrderHold](../wiki/Control.md#firstorderhold) | 1 | 1 | First-Order Hold (FOH) |
| [Hysteresis](../wiki/Control.md#hysteresis) | 1 | 1 | Hysteresis Relay. |
| [Integrator](../wiki/Control.md#integrator) | 1 | 1 | Continuous-time Integrator (1/s). |
| [LQR](../wiki/Control.md#lqr) | 1 | 0 | LQR optimal state-feedback gain designer. |
| [PID](../wiki/Control.md#pid) | 2 | 1 | PID Controller. |
| [RateLimiter](../wiki/Control.md#ratelimiter) | 1 | 1 | Rate Limiter. |
| [RateTransition](../wiki/Control.md#ratetransition) | 1 | 1 | Rate Transition Block |
| [Saturation](../wiki/Control.md#saturation) | 1 | 1 | Limits the input signal to a specified range. |
| [StateSpace](../wiki/Control.md#statespace) | 1 | 1 | Continuous State-Space Model. |
| [TranFn](../wiki/Control.md#tranfn) | 1 | 1 | Represents a linear time-invariant system as a transfer function. |
| [TransportDelay](../wiki/Control.md#transportdelay) | 1 | 1 | Transport Delay / Time Delay. |
| [VariableTransportDelay](../wiki/Control.md#variabletransportdelay) | 2 | 1 | Variable Transport Delay / Input-Driven Time Delay. |
| [ZeroOrderHold](../wiki/Control.md#zeroorderhold) | 1 | 1 | Zero-Order Hold (ZOH). |

## Routing

8 blocks. Full parameter tables: [Routing](../wiki/Routing.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Demux](../wiki/Routing.md#demux) | 1 | 2 | Demultiplexer (Demux). |
| [From](../wiki/Routing.md#from) | 0 | 1 | From Tag. |
| [Goto](../wiki/Routing.md#goto) | 1 | 0 | Goto Tag. |
| [Mux](../wiki/Routing.md#mux) | 2 | 1 | Multiplexer (Mux). |
| [NetworkChannel](../wiki/Routing.md#networkchannel) | 1 | 1 | Network Channel / Communication Link. |
| [PacketLoss](../wiki/Routing.md#packetloss) | 1 | 1 | Packet-Loss / Lossy Channel. |
| [Selector](../wiki/Routing.md#selector) | 1 | 1 | Selector / Indexer. |
| [Switch](../wiki/Routing.md#switch) | 3 | 1 | Signal Switch. |

## Sinks

8 blocks. Full parameter tables: [Sinks](../wiki/Sinks.md).

| Block | In | Out | Description |
|---|---|---|---|
| [AgentScope](../wiki/Sinks.md#agentscope) | 1 | 0 | Multi-agent scope: animated 2D scatter of N agents over time. |
| [Assert](../wiki/Sinks.md#assert) | 1 | 0 | Stops simulation if input violates condition. Modes: >0, <0, >=0, <=0, ==0, !=0, finite. |
| [Display](../wiki/Sinks.md#display) | 1 | 0 | Numerical Display. |
| [Export](../wiki/Sinks.md#export) | 1 | 0 | Data Export. |
| [FFT](../wiki/Sinks.md#fft) | 1 | 0 | Spectrum Analyzer (FFT). |
| [Scope](../wiki/Sinks.md#scope) | 1 | 0 | Oscilloscope / Plotter. |
| [Term](../wiki/Sinks.md#term) | 1 | 0 | Signal Terminator. |
| [XYGraph](../wiki/Sinks.md#xygraph) | 2 | 0 | XY Plotter. |

## Analysis

4 blocks. Full parameter tables: [Analysis](../wiki/Analysis.md).

| Block | In | Out | Description |
|---|---|---|---|
| [BodeMag](../wiki/Analysis.md#bodemag) | 1 | 0 | Right-click to generate a Bode magnitude plot from a connected Transfer Function block. |
| [BodePhase](../wiki/Analysis.md#bodephase) | 1 | 0 | Right-click to generate a Bode Phase plot from a connected dynamic block. |
| [Nyquist](../wiki/Analysis.md#nyquist) | 1 | 0 | Right-click to generate a Nyquist plot from a connected dynamic block. |
| [RootLocus](../wiki/Analysis.md#rootlocus) | 1 | 0 | Root Locus Plotter. |

## PDE

16 blocks. Full parameter tables: [PDE](../wiki/PDE.md).

| Block | In | Out | Description |
|---|---|---|---|
| [AdvectionEquation1D](../wiki/PDE.md#advectionequation1d) | 1 | 2 | 1D Advection Equation: ∂c/∂t + v * ∂c/∂x = 0 |
| [AdvectionEquation2D](../wiki/PDE.md#advectionequation2d) | 5 | 3 | 2D Advection-Diffusion: ∂c/∂t = -v·∇c + D∇²c + S |
| [DiffusionReaction1D](../wiki/PDE.md#diffusionreaction1d) | 3 | 3 | 1D Diffusion-Reaction: ∂c/∂t = D∇²c - kc^n + S |
| [FieldGradient](../wiki/PDE.md#fieldgradient) | 1 | 1 | Field Gradient: Compute spatial derivative. |
| [FieldIntegral](../wiki/PDE.md#fieldintegral) | 1 | 1 | Field Integral: Integrate field over domain. |
| [FieldLaplacian](../wiki/PDE.md#fieldlaplacian) | 1 | 1 | Field Laplacian: Compute second spatial derivative. |
| [FieldMax](../wiki/PDE.md#fieldmax) | 1 | 3 | Field Max: Find maximum value and location. |
| [FieldProbe](../wiki/PDE.md#fieldprobe) | 2 | 1 | Field Probe: Extract value at a specific location. |
| [FieldProbe2D](../wiki/PDE.md#fieldprobe2d) | 3 | 1 | 2D Field Probe: Extract value at (x,y) location. |
| [FieldScope](../wiki/PDE.md#fieldscope) | 1 | 0 | Field Scope: Visualize spatiotemporal field evolution. |
| [FieldScope2D](../wiki/PDE.md#fieldscope2d) | 1 | 0 | 2D Field Scope: Visualize 2D field evolution. |
| [FieldSlice](../wiki/PDE.md#fieldslice) | 2 | 1 | Field Slice: Extract 1D slice from 2D field. |
| [HeatEquation1D](../wiki/PDE.md#heatequation1d) | 5 | 2 | 1D Heat Equation: ∂T/∂t = α∇²T + q |
| [HeatEquation2D](../wiki/PDE.md#heatequation2d) | 9 | 3 | 2D Heat Equation: ∂T/∂t = α∇²T + q |
| [WaveEquation1D](../wiki/PDE.md#waveequation1d) | 3 | 3 | 1D Wave Equation: ∂²u/∂t² = c²∇²u |
| [WaveEquation2D](../wiki/PDE.md#waveequation2d) | 5 | 3 | 2D Wave Equation: ∂²u/∂t² = c²∇²u |

## Optimization

5 blocks. Full parameter tables: [Optimization](../wiki/Optimization.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Constraint](../wiki/Optimization.md#constraint-block) | 1 | 1 | Constraint for Constrained Optimization |
| [CostFunction](../wiki/Optimization.md#costfunction-block) | 2 | 1 | Cost Function for Optimization |
| [DataFit](../wiki/Optimization.md#datafit-block) | 1 | 2 | Data Fit - Model calibration against experimental data |
| [Optimizer](../wiki/Optimization.md#optimizer-block) | 0 | 0 | Optimizer - Triggers and configures optimization |
| [Parameter](../wiki/Optimization.md#parameter-block) | 0 | 1 | Tunable Parameter for Optimization |

## Optimization Primitives

11 blocks. Full parameter tables: [Optimization Primitives](../wiki/Optimization-Primitives.md).

| Block | In | Out | Description |
|---|---|---|---|
| [Adam](../wiki/Optimization-Primitives.md#adam) | 1 | 1 | Adam optimizer with adaptive learning rates. |
| [LinearSystemSolver](../wiki/Optimization-Primitives.md#linearsystemsolver) | 2 | 1 | Solves the linear system Ax = b for x. |
| [Momentum](../wiki/Optimization-Primitives.md#momentum) | 1 | 1 | Momentum-accelerated gradient descent. |
| [NumericalGradient](../wiki/Optimization-Primitives.md#numericalgradient) | 3 | 1 | Computes gradient from finite difference inputs. |
| [ObjectiveFunction](../wiki/Optimization-Primitives.md#objectivefunction) | 1 | 1 | Evaluates an objective function f(x) from a Python expression. |
| [ResidualNorm](../wiki/Optimization-Primitives.md#residualnorm) | 1 | 1 | Computes the norm of a vector for convergence checking. |
| [RootFinder](../wiki/Optimization-Primitives.md#rootfinder) | 1 | 1 | Computes one Newton step for solving F(x) = 0. |
| [StateVariable](../wiki/Optimization-Primitives.md#statevariable) | 1 | 1 | Holds state x(k) for iterative optimization. |
| [VectorGain](../wiki/Optimization-Primitives.md#vectorgain) | 1 | 1 | Scales a vector by a scalar gain: y = α * x |
| [VectorPerturb](../wiki/Optimization-Primitives.md#vectorperturb) | 1 | 1 | Perturbs x[index] by epsilon for finite difference gradient computation. |
| [VectorSum](../wiki/Optimization-Primitives.md#vectorsum) | 2 | 1 | Adds or subtracts multiple vector inputs. |

## Other

1 blocks. Full parameter tables: [Other](../wiki/Other.md).

| Block | In | Out | Description |
|---|---|---|---|
| [External](../wiki/Other.md#external) | 1 | 1 | External Function Block (NOT FULLY IMPLEMENTED). *(hidden stub)* |
---

## Not in this list

**Subsystem**, **Inport** and **Outport** are structural rather than registered
blocks; they are documented on the
[Routing](../wiki/Routing.md) and
[Subsystems](../wiki/Subsystems_Architecture.md) pages.

**Your own library blocks** — masked subsystems you saved — appear in the
palette under **USER LIBRARY**. See
[Creating Diagrams](creating-diagrams.md#masks-and-user-library-blocks).
