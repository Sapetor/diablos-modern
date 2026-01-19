# Fast Solver (Compiled Mode)

## Overview
The **Fast Solver** is a high-performance execution engine introduced in DiaBloS Modern. It compiles valid portions of a block diagram into a flat system of differential equations and solves them using extensive numerical libraries (`scipy.integrate`).

This approach typically yields **10x-100x speedups** compared to the standard Interpreted Mode, especially for stiff systems or long simulations.

## How it Works
1.  **Analysis**: The `SystemCompiler` analyzes the diagram connectivity and block types.
2.  **Compilation**:
    *   **State Mapping**: `Integrator`, `TransferFcn`, and `StateSpace` blocks are assigned slots in a global state vector $\mathbf{y}$.
    *   **Function Generation**: The system is flattened into a derivative function $\frac{d\mathbf{y}}{dt} = f(t, \mathbf{y})$.
    *   **Matrix Construction**: Linear Time-Invariant (LTI) subsystems are converted to State-Space ($A, B, C, D$) matrices for efficient computation.
3.  **Solving**: `scipy.integrate.solve_ivp` (default: RK45) integrates the system over the simulation time.
4.  **Replay**: The solver's output ($\mathbf{y}$ over time) is "replayed" through the topological graph to reconstruct intermediate signals for `Scope` blocks.

## Supported Blocks
The following blocks are fully supported in Fast Solver mode. If a diagram contains *only* these blocks, it will automatically run in Fast Mode.

### Linear Dynamics
*   **Integrator**: Core state element.
*   **TransferFcn**: Automatically converted to State-Space. Supports identifying inputs.
*   **StateSpace**: Executed using vector-matrix multiplication.

### Math & Logic
*   **Gain**: Scalar multiplication.
*   **Sum**: Multi-input addition/subtraction.
*   **Product** (`SgProd`): Element-wise multiplication.
*   **Abs**: Absolute value.
*   **Saturation**: Min/Max clipping.
*   **Deadband**: Zero output in dead zone.

### Control
*   **PID**: Proportional-Integral-Derivative controller (compiled to 2 states + feedthrough).
*   **RateLimiter**: Limits the rate of change of the signal (compiled as dynamic state).

### Sources
*   **Step**: Delayed step input.
*   **Ramp**: Linear growth with delay ($slope \times (t-delay)$).
*   **Sine**: Sine wave with amplitude, frequency, phase, bias.
*   **Constant**: Static value.
*   **Exponential**: Exponential source ($a \times e^{bx}$).
*   **From**: Tag receiver (compile-time resolution).

### Signal Routing
*   **Mux / Demux**
*   **Switch**: Routes signals based on control threshold or index.

### Sinks
*   **Scope**: Records data for the Waveform Inspector.
*   **Terminator**: Sink for unused signals.
*   **Display**: Sink.
*   **To**: Tag sender.

## Special Handling

### Zero-State Systems ("Pure Algebraic")
Systems without any states (e.g., `Ramp` -> `Gain` -> `Scope`) do not require an ODE solver. However, the Fast Solver stack handles them efficiently by:
1.  Detecting 0 states.
2.  Skipping `solve_ivp` overhead.
3.  Generating a time vector manually.
4.  Executing the **Replay Loop** to compute all algebraic signals at each time step.

### Mixed Mode / Fallback
If a diagram contains unsupported blocks (e.g., custom scripted blocks, `PythonFunction`, or legacy blocks), the Engine automatically falls back to **Interpreter Mode**. This is slower but guarantees compatibility.

## Troubleshooting

### "Solver returned no states" / Empty Plot
*   **Cause**: The system has no integrators or dynamic elements (0 states).
*   **Fix**: This occurred in older versions but is **Fixed** in the current release. The engine now correctly handles 0-state replay and syncs the timeline to the plotter.

### Algebraic Loops
*   **Issue**: A feedback loop without a delay/integrator (e.g., `Gain` -> `Sum` -> `Gain` -> ...).
*   **Behavior**: The compiler generally cannot resolve algebraic loops analytically. The simulation may fail to initialize or fallback to Interpreter Mode (which attempts iterative resolution).
*   **Fix**: Insert an `Integrator` or `TransferFcn` (with denominator order > numerator) in the feedback path.

## Performance
For a benchmark system of 50 coupled oscillators (100 states), Fast Solver runs in **~0.1s** vs **~3.5s** for the Interpreter.
