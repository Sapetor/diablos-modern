# Running Simulations

## Starting a run

Press <kbd>F5</kbd>, click **Run** on the toolbar, or use **Simulation ▸ Run**.
<kbd>F6</kbd> pauses, <kbd>F7</kbd> stops, and <kbd>F8</kbd> advances exactly one
`dt` and pauses again — useful for watching a discrete block change state.

DiaBloS validates the diagram first (unconnected required ports, duplicate
inputs, tag problems, algebraic loops) and autosaves before it runs.

## Simulation Configuration

Every run opens the **Simulation Configuration** dialog, pre-filled from the
diagram's stored settings. The values you confirm are saved into the `.diablos`
file, so a diagram carries its own solver setup.

### Solver Configuration

| Field | Meaning |
|---|---|
| **Solver Method** | The integration method — see the table below. |
| **Base Step Size (dt) [s]** | The global solver step. Discrete blocks run at their own `sampling_time` or synchronize to this step. |
| **Simulation Duration [s]** | End time of the run. |
| **Rel. tol** | Relative tolerance for the adaptive solvers (default `1e-9`). |
| **Abs. tol** | Absolute tolerance for the adaptive solvers (default `1e-12`). |
| **Detect zero crossings (fast solver)** | See [Zero-crossing detection](#zero-crossing-detection) below. On by default. |
| **Run in real-time** | Throttle the run to wall-clock time. |

### Visualization

| Field | Meaning |
|---|---|
| **Plot Window Range [samples]** | How many samples the live plot window keeps. |
| **Enable Dynamic Plotting** | Draw scope traces while the run proceeds. |

### Solver methods

Eight methods are offered, in this order:

| Method | Kind | When to use it |
|---|---|---|
| `RK45` | Adaptive Runge-Kutta 4(5) | The default. Good for most non-stiff problems. |
| `RK23` | Adaptive Runge-Kutta 2(3) | Cheaper per step, lower order; loose tolerances. |
| `DOP853` | Adaptive Runge-Kutta, 8th order | Smooth problems where you want tight tolerances. |
| `Radau` | Implicit Runge-Kutta | Stiff systems. |
| `BDF` | Implicit multistep | Stiff systems, especially large ones. |
| `LSODA` | Automatic stiff/non-stiff switching | When you do not know whether the problem is stiff. |
| `RK4` | Fixed step | Reproducible stepping; ignores `rtol`/`atol`. |
| `Euler` | Fixed step | Teaching, and debugging a step-by-step trace. |

The first six are `scipy.integrate.solve_ivp` methods and honour `rtol`/`atol`.
`RK4` and `Euler` take fixed steps of the size set in **Base Step Size (dt)**.

## Two execution paths

DiaBloS has two engines and picks between them per run.

**The compiled (fast) path** flattens the whole diagram — subsystems included —
into a single right-hand side `rhs(t, x)` and hands it to `solve_ivp`. This is
the default and the numerically accurate path: the solver controls its own step
size, so it is both faster and more correct than fixed stepping. It is toggled
by **Simulation ▸ Enable Fast Solver (Experimental)**, which is checked by
default.

**The interpreted path** walks the diagram block by block at a fixed step,
calling each block's `execute()`. It covers every block, including those the
compiler does not know how to emit.

The engine falls back to the interpreter automatically when a diagram contains a
block outside the compiled set, a discrete sample time, or an algebraic loop.
Which one ran is reported in the diagnostics (below).

See [Fast Solver](../FAST_SOLVER.md) for the block coverage list and the
execution-order rules.

## Zero-crossing detection

Adaptive solvers take large steps. When a diagram contains a discontinuity — a
relay flipping, a saturation hitting its limit, a step firing — a large step
smears the switch across it, so the switching instant lands in the wrong place
and moves when you change the step size.

With **Detect zero crossings (fast solver)** on (the default), the compiled path
hands `solve_ivp` a scalar *event function* for each switching surface. The
solver stops exactly at each crossing, applies any discrete update, and restarts
a fresh integration from there. Switching instants land on their true time and
stop depending on the step size.

Blocks that contribute switching surfaces: `Switch`, `Saturation`, `Deadband`,
`Hysteresis`, `Abs`, `MathFunction` in `sign`/`abs` mode, the `Step` and `Ramp`
edges, `WaveGenerator` in square/sawtooth mode, and `PRBS`.

Two things are worth knowing:

- **`Hysteresis` only compiles when detection is on.** Its latch is frozen
  inside each integration segment and flipped at the located switching instant.
  With detection off, a diagram containing `Hysteresis` falls back to the
  interpreter.
- **A chattering guard protects sliding-mode diagrams.** A relay that switches
  infinitely often would otherwise stall the run; DiaBloS logs a warning naming
  the block and finishes with a fixed step.

Each of those blocks also has its own `zero_crossing` parameter (`auto` or
`off`) so you can drop one block's crossings without turning detection off for
the whole diagram.

From the command line: `--no-zero-crossing` on the `run` subcommand overrides
the diagram's setting for that run.

## Reading the solver settings in the property panel

Click empty canvas — with nothing selected, the **Properties** panel shows a
diagram inspector instead of block parameters. Its **Solver** section is the
quickest way to check what a diagram will actually do:

| Row | What it shows |
|---|---|
| `solver` | The integration method (`RK45`, `BDF`, …). |
| `step_size` | Base step `dt`, in seconds. |
| `duration` | Simulation end time, in seconds. |
| `rtol` | Relative tolerance — only used by the adaptive methods. |
| `atol` | Absolute tolerance — likewise. |
| `fast_solver` | `✓ on` if the compiled path is enabled, `off` otherwise. |
| `zero_crossing` | `✓ on` if event detection is enabled. Compiled path only. |

Below it, **Workspace** lists the variables defined in the
[Variable Editor](../VARIABLE_EDITOR_GUIDE.md), **Recent runs** lists the last
runs, and **Validation** reports diagram problems.

### Run diagnostics

The compiled solver's per-run statistics are *not* in the property panel — they
go to the status bar and the log file after each run, as one line:

```
method=RK45 backend=compiled states=3 points=1001 nfev=272 events=4 cache=hit compile=0.0121s solve=0.0338s replay=0.0042s total=0.0501s
```

| Field | Meaning |
|---|---|
| `method` | The solver actually used. |
| `backend` | `compiled` or the interpreted fallback. |
| `states` | Size of the state vector the compiler built. |
| `points` | Number of output time points. |
| `nfev` | Right-hand-side evaluations (`n/a` on the fixed-step paths). |
| `events` | Zero crossings located; annotated `(guard tripped)` if the chattering guard fired. |
| `cache` | `hit` if the compiled system was reused, `miss` if it was rebuilt. |
| `compile` / `solve` / `replay` / `total` | Wall-clock time of each phase. |

A `backend` that says the interpreter ran when you expected the compiled path,
or a rising `nfev`, is usually the first sign that a block fell out of the
compiled set or that a discontinuity is being hunted.

## Viewing results

### Scope plots

Scope blocks show time-series data. Multiple inputs appear as separate traces,
the mouse wheel zooms and dragging pans. The previous run's traces are drawn
dimmed and dashed behind the live ones for comparison.

**Show Plots** (Simulation menu) opens the Waveform Inspector: the last runs,
toggleable traces, a time slider, CSV export of the selected traces, and
optional on-disk persistence of the run history.

Scope windows can also export a publication figure — serif matplotlib output to
PDF, PNG at 300 dpi, or SVG.

### Field visualizations

For PDE blocks:

- **FieldScope** — a 1D field as a heatmap over time
- **FieldScope2D** — a 2D field with a time slider

Both can export the animation as GIF (needs Pillow) or MP4 (needs `ffmpeg` on
your PATH).

### Exporting results

Right-click a plot window to save it as an image, export an animation for field
plots, or copy it to the clipboard.

## Analysis and experiments

Everything under the **Analysis** menu re-runs the diagram headlessly and never
mutates it:

- **Linearize & Analyze…** — numeric linearization, poles/zeros, Bode, margins
- **Find Operating Point (Trim)…** — solve for an equilibrium
- **Parameter Sweep…** — 1-D and 2-D sweeps
- **Monte Carlo…** — seeded stochastic ensembles

These have their own page: [Analysis & Experiments](analysis.md).

## Running without the GUI

```bash
python diablos_modern.py run model.diablos -o out.csv
python diablos_modern.py run model.diablos -o out.npz --time 30 --dt 0.005
python diablos_modern.py run model.diablos --solver interpreter
python diablos_modern.py run model.diablos --no-zero-crossing
```

`run` simulates the diagram and writes every Scope trace to CSV or NPZ. With no
`-o` it writes next to the diagram with a `.csv` extension. `--time` and `--dt`
override the diagram's stored values; `--solver` chooses `compiled` (default) or
`interpreter`; `-q` silences the summary line.

## Troubleshooting

### "Algebraic Loop Detected"

The diagram has a feedback loop with no state in it. Insert an `Integrator`, a
`Delay` or a `TransportDelay` to break the loop. Note that a subsystem
containing a memory block counts as breaking the loop.

### "Unlinked Port"

A required port is unconnected. The error message names the block. Attach a
`Term` block to an output you deliberately want to discard.

### The run is slower than expected

Check the `backend` field in the run diagnostics. If it says the interpreter
ran, a block in the diagram is outside the compiled set — see
[Fast Solver](../FAST_SOLVER.md). A very large `events` count means the solver
is spending its time locating crossings; consider setting `zero_crossing` to
`off` on the offending block.

## Export as Python Script

**File → Export → Export as Python Script...** writes the current diagram out as a
self-contained `.py` file. The generated script depends only on **numpy** and
**scipy** (matplotlib is imported only when it plots), so it runs anywhere
without DiaBloS installed — handy for sharing a model, putting a simulation in
CI, or using the diagram as the starting point for hand-written code.

The script mirrors the compiled (fast) solver: the whole diagram becomes one
`rhs(t, x)` integrated with `scipy.integrate.solve_ivp`, so its results match a
run inside DiaBloS. It is written to be read and edited:

- a **parameters** section with one named constant per block parameter
  (`KP_GAIN = 3.0`), plus the `A`/`B`/`C`/`D` matrices of every transfer
  function and state-space block;
- the **state vector layout** as a comment, and the initial state `X0`;
- an `evaluate(t, x)` function in the solver's own evaluation order —
  strictly-proper state outputs, sources, algebraic blocks, then the remaining
  state derivatives;
- a plotting section that reproduces what each Scope block would show.

### Running the exported script

```bash
python model.py                    # simulate and plot
python model.py --no-plot          # headless (DIABLOS_NO_PLOT=1 works too)
python model.py --out run.csv      # write the Scope traces to CSV (or .npz)
python model.py --time 30 --dt 0.005
```

The CSV/NPZ columns carry the same signal names as a headless
`run` export, so the two can be compared directly.

### From the command line

```bash
python diablos_modern.py export-python diagram.diablos -o model.py
python diablos_modern.py export-python diagram.diablos --solver RK4 --time 30
```

### Supported blocks

Code generation covers the block families that can be written out as plain
Python without changing the numerics:

| Group | Blocks |
|-------|--------|
| Sources | Step, Sine, Ramp, Constant |
| Math | Gain, Sum, Product, SgProd, Abs |
| Nonlinear | Saturation |
| Routing | Mux, Demux |
| State | Integrator, TranFn, StateSpace, PID |
| Sinks | Scope, Display, Terminator |

Anything else — Noise, Hysteresis, MathFunction, RateLimiter, the PDE/Field
families, the optimization primitives — makes the export stop with a dialog
listing the offending blocks by name, rather than emitting a script that
quietly computes something else. The same applies to a `Step` set to *impulse*,
to blocks with a discrete sample time (both run on the interpreter, not the
compiled path), and to diagrams with an algebraic loop.

## Troubleshooting

### "Algebraic Loop Detected"

Your diagram has a feedback loop without a delay element. Add an Integrator or TransportDelay block to break the loop.

### "Unlinked Port"

All required ports must be connected. Check the error message for which block has unconnected ports.
