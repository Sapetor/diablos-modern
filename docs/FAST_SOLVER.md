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
The following blocks are fully supported in Fast Solver mode. If a diagram contains *only* these blocks, and none of them is gated to a discrete rate (`sampling_time > 0`), it will automatically run in Fast Mode.

### Linear Dynamics
*   **Integrator**: Core state element. Its per-block `method` is an interpreter-only setting: the compiled path integrates the assembled system with the solver method from Simulation settings, and logs a warning when it discards a non-default per-block method.
*   **TransferFcn**: Automatically converted to State-Space. Supports identifying inputs.
*   **StateSpace**: Executed using vector-matrix multiplication.

### Math & Logic
*   **Gain**: Scalar multiplication.
*   **Sum**: Multi-input addition/subtraction.
*   **Product** (`SgProd`): Element-wise multiplication.
*   **Abs**: Absolute value.
*   **Saturation**: Min/Max clipping.
*   **Deadband**: Zero output in dead zone.
*   **MathFunction**: Standard math functions (sin, cos, exp, log, sqrt, etc.).

### Control
*   **PID**: Proportional-Integral-Derivative controller (compiled to 2 states + feedthrough).
*   **RateLimiter**: Limits the rate of change of the signal (compiled as dynamic state).
*   **Hysteresis**: Relay with upper/lower thresholds. Compiled only with
    zero-crossing detection on (see below); otherwise the diagram falls back to
    the interpreter.

### Sources
*   **Step**: Delayed step input.
*   **Ramp**: Linear growth with delay ($slope \times (t-delay)$).
*   **Sine**: Sine wave with amplitude, frequency, phase, bias.
*   **WaveGenerator**: Multi-waveform source (Sine, Square, Triangle, Sawtooth).
*   **Constant**: Static value.
*   **Exponential**: Exponential source ($a \times e^{bx}$).
*   **Noise**: Gaussian random noise ($\mu + \sigma \cdot \text{randn}()$).
*   **From**: Tag receiver (compile-time resolution).

### Signal Routing
*   **Mux / Demux**
*   **Switch**: Routes signals based on control threshold or index.
*   **Selector**: Extracts specific elements from vector signals.
*   **Subsystems**: Recursive grouping supported. The compiler flattens subsystems into their primitive components.


### Sinks
*   **Scope**: Records data for the Waveform Inspector.
*   **Terminator**: Sink for unused signals.
*   **Display**: Sink.
*   **To**: Tag sender.

## Zero-Crossing Detection

A discontinuous block makes the assembled right-hand side *piecewise*, and an
adaptive Runge-Kutta step cannot represent a kink or a jump inside itself. Left
alone, `solve_ivp` either smears the switch across whatever step straddled it or
burns a pile of rejected steps shrinking onto it -- and either way the switching
instant is only located to step accuracy, so the answer moves when you change
the step size.

Zero-crossing detection gives the solver the switching surfaces instead. It is
**on by default** (Simulation settings -> *Detect zero crossings*), stored in the
`.diablos` `sim_data` alongside `solver_method`/`rtol`/`atol`, and it applies to
the compiled path only -- the interpreter is fixed-step and has no event
machinery.

### Which blocks contribute events

| Block | Event function(s) |
| --- | --- |
| `Saturation` | `u - min`, `u - max` (each finite limit only) |
| `Deadband` | `u - start`, `u - end` |
| `Switch` | `ctrl - threshold`, or each half-integer port boundary in index mode |
| `Hysteresis` | the *active* threshold: `u - upper` while low, `u - lower` while high |
| `Abs` | `u` (the corner at zero) |
| `MathFunction` | `u`, for `sign` and `abs` only |
| `Step` | `t - delay` (`up`/`down` shapes), monotonic |
| `Ramp` | `t - delay` (the start kink), monotonic |
| `WaveGenerator` | the periodic edge of `Square`/`Sawtooth`, the kink of `Triangle` |
| `PRBS` | bit boundaries (`t` a multiple of `bit_time`) |

`Hysteresis` is compilable *only* while detection is on: its latch is
path-dependent, so it can be part of an ODE right-hand side only if the mode is
frozen for the length of each segment and flipped exclusively at a located
switching instant. With detection off the diagram falls back to the interpreter.

Not covered: `RateLimiter` (its compiled form is a stiff first-order chase, and
a guard on that chase grazes its own clip boundary repeatedly rather than
crossing it cleanly), and `ceil`/`floor` in `MathFunction` (an unbounded event
train at every integer).

Each of these blocks carries a `zero_crossing` parameter (`auto` / `off`) to
drop just its own events while the rest of the diagram keeps theirs.

### Adding events to a new block

Declare them next to the kernel that implements the discontinuity, in
`lib/engine/compiler_kernels/`:

```python
@events("MyBlock")
def events_myblock(ctx):
    src = ctx.input_sources[0]

    def g(t, y, signals):
        return signal_scalar(signals, src) - threshold

    return [EventSpec(block=ctx.b_name, label="threshold", func=g)]
```

`SystemCompiler.compile_system` collects every registered builder's specs onto
the compiled RHS; the runner never learns about individual block types. An
`EventSpec` may also carry `on_event` (a discrete update applied at the root --
this is how the relay latch flips), `on_start` (reconcile a mode with the
initial condition), `direction`, and `monotonic`.

### The restart algorithm

Integration proceeds in segments. Every event is registered as *terminal*, so a
segment ends at the first located root:

1. Solve from the segment start to the end of the simulation, with `t_eval` set
   to the remaining output grid points and all events terminal.
2. If no event fired, the run is done -- the segment produced every remaining
   grid sample.
3. Otherwise take the earliest root, and apply the discrete update of *every*
   event at that instant (simultaneous crossings all get their update, not just
   the one scipy reported last).
4. Restart from the root plus a nudge of `1e-11` of the span, carrying the state
   across unchanged. The nudge is not optional: at the root `g == 0`, and
   scipy's `find_active_events` counts a zero as active, so a restart exactly at
   the root would re-detect the same crossing forever.
5. Write each segment's samples into the caller's grid. The returned `(t, y)`
   is exactly the array a single `solve_ivp` call would have returned, so
   replay, scope capture, `dense_output`, CSV/NPZ export and plotting see no
   difference.

`solve_ivp` compares event signs only at accepted step ends, so a *state*-
dependent guard can cross zero and come back inside one large step and be missed
entirely (a constant right-hand side happily takes one step across the whole
run). The step is therefore capped at the output `dt` whenever any registered
event is non-monotonic. Pure time events (`Step`, `Ramp`) are marked monotonic
and skip the cap, which keeps it away from the many diagrams whose only
discontinuity is a step.

### The chattering guard

A relay in sliding mode switches infinitely often in finite time; each restart
then lands on the previous one and the run never reaches `t_end`. Two bounds
stop that:

* **Minimum separation** -- events closer together than `1e-9` of the span, 20
  times in a row, are chattering.
* **Event cap** -- 10000 located events per run.

Tripping either logs a warning naming the block, and the remainder of the run is
integrated with a **fixed step** (RK4 on the output grid) rather than with
events switched off. That distinction matters: an adaptive solver turned loose
on a chattering system does not finish either, because its error control keeps
shrinking the step toward machine precision. The fixed-step tail always
terminates and matches what the interpreter would have produced.

Blocks holding a discrete mode (`Hysteresis`) are unfrozen at the same moment,
so they resume updating from the right-hand side instead of waiting for events
that will no longer arrive.

### Cost

Registering no events costs nothing: the `events=` argument is skipped entirely
and the solver is called exactly as before. Measured on this repo's examples
(median of 7 runs), diagrams with no discontinuity are unchanged within noise
(`c03_bode_frequency_response` 3.3 ms, `c11_opinion_dynamics` 8.0 ms both ways),
while a single located step edge costs one extra restart:
`c05_mass_spring_state_space` 10.0 -> 11.4 ms and `c01_tank_feedback` 7.7 ->
9.0 ms.

## Special Handling

### Zero-State Systems ("Pure Algebraic")
Systems without any states (e.g., `Ramp` -> `Gain` -> `Scope`) do not require an ODE solver. However, the Fast Solver stack handles them efficiently by:
1.  Detecting 0 states.
2.  Skipping `solve_ivp` overhead.
3.  Generating a time vector manually.
4.  Executing the **Replay Loop** to compute all algebraic signals at each time step.

### Mixed Mode / Fallback
If a diagram contains unsupported blocks (e.g., custom scripted blocks, `PythonFunction`, or legacy blocks), the Engine automatically falls back to **Interpreter Mode**. This is slower but guarantees compatibility. A block with a discrete sample time (`sampling_time > 0`) also forces the fallback, even if it is otherwise supported: the compiled ODE right-hand side has no notion of sample instants, so only the interpreter runs such a block at its own rate.

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
