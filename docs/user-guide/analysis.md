# Analysis & Experiments

Beyond running a diagram once, DiaBloS can *interrogate* it: linearize the closed
loop, read its poles and margins, sweep a parameter, or run a seeded stochastic
ensemble. Everything on this page re-simulates the diagram headlessly and
restores the original block parameters afterwards, so none of it changes the
diagram you are looking at.

Four tools live on the **Analysis** menu. Four more are right-click actions on
[Analysis marker blocks](../wiki/Analysis.md). Live tuning is a dock panel.

| What you want | Where it is |
|---|---|
| Poles, zeros, Bode, margins of the whole loop | **Analysis ▸ Linearize & Analyze…** |
| An equilibrium to linearize about | **Analysis ▸ Find Operating Point (Trim)…** |
| A response family or metric heatmap over parameters | **Analysis ▸ Parameter Sweep…** |
| A stochastic ensemble with percentile bands | **Analysis ▸ Monte Carlo…** |
| Bode / Nyquist / root locus of one transfer function | Right-click a marker block |
| An LQR gain | Right-click an `LQR` block |
| Drag a gain and watch the plot move | **View ▸ Parameter Tuning Panel** (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>T</kbd>) |

---

## Linearization

**Analysis ▸ Linearize & Analyze…**

DiaBloS linearizes numerically, not symbolically: it compiles the diagram to its
ODE right-hand side and takes finite differences of that. Nonlinear blocks —
saturations, deadbands, lookup tables, `Function` expressions — are therefore
included, linearized about the operating point rather than ignored.

The setup dialog (**Linearize & Analyze**) asks for:

- **Inputs (sources)** — which source blocks count as inputs `u`
- **Outputs (signals)** — which signals count as outputs `y`
- **Find operating point (trim) first** — solve for the equilibrium and
  linearize there instead of at the current state

The result window is titled **Linearized System Analysis** and has five tabs:

| Tab | Contents |
|---|---|
| **Pole-Zero** | Pole-zero map in the complex plane |
| **Bode** | Magnitude (dB) and phase (deg) against frequency (rad/s) |
| **Step** | Step response, when a SISO transfer function is available |
| **Impulse** | Impulse response, same condition |
| **Summary** | The text report — see below |

The **Summary** tab reports the state/input/output counts and names, whether the
system is stable, its time constants and oscillatory modes, the gain and phase
margins with their crossover frequencies, the transfer function, whether the
system is controllable and observable, and the operating point it linearized
about.

### Getting the model out

Three buttons at the bottom of the window:

- **Copy as Python** — an `A`/`B`/`C`/`D` snippet using numpy and
  `python-control`
- **Copy as MATLAB** — the same as MATLAB source
- **Save Data…** — writes a MAT-file (`.mat`) or a NumPy archive (`.npz`)

## Operating point (trim)

**Analysis ▸ Find Operating Point (Trim)…**

Solves `f(0, y) = 0` on the compiled right-hand side to find an equilibrium
state. The **Operating Point (Trim)** window lists each state and its value, and
**Copy (Python dict)** puts them on the clipboard.

Use it before linearizing a plant that does not sit at the origin — an inverted
pendulum, a tank at a non-zero level, a vehicle at cruise.

## Bode, Nyquist and root locus of one block

The four [Analysis marker blocks](../wiki/Analysis.md) — `BodeMag`, `BodePhase`,
`Nyquist` and `RootLocus` — analyse a *single* `TranFn` or `StateSpace` block
rather than the whole diagram. Drop one on the canvas, wire its input to the
block you want, right-click it, and choose the generate action:

| Block | Right-click action | Window title |
|---|---|---|
| `BodeMag` | Generate Bode magnitude plot | `Bode Magnitude Plot: <name>` |
| `BodePhase` | Generate Bode phase plot | `Bode Phase Plot: <name>` |
| `Nyquist` | Generate Nyquist plot | `Nyquist Plot: <name>` |
| `RootLocus` | Generate root-locus plot | `Root Locus: <name>` |

No simulation run is needed. For the frequency response of a *closed loop with
nonlinearities in it*, use **Linearize & Analyze…** instead.

## LQR design

Drop an `LQR` block, wire its `plant` input to a `StateSpace` block (so `A` and
`B` are read automatically), set the cost matrices `Q` and `R`, then right-click
▸ **Compute LQR gain**.

The **LQR Result** window solves the continuous algebraic Riccati equation and
reports the optimal gain `K` for `u = -Kx`, the closed-loop eigenvalues of
`A - BK`, and the cost matrix `P`. Every matrix field also accepts a workspace
variable name from the [Variable Editor](../VARIABLE_EDITOR_GUIDE.md), so you
can build `Q` and `R` in a script and reference them by name.

`examples/c05_lqr_vs_open_loop.diablos` and `examples/c06_lqr_state_feedback.diablos`
are worked examples.

---

## Monte Carlo ensembles

**Analysis ▸ Monte Carlo…**

Runs the diagram N times with different random draws and aggregates the results.
The setup dialog (**Monte-Carlo Ensemble**) takes:

- **Number of runs**
- **Master seed**
- **Simulation time**
- **Step size (dt)**

### Reproducibility from one number

Every block that exposes a `seed` parameter — `Noise`, `RandomSource`,
`PacketLoss`, `NetworkChannel` — gets a sub-seed derived deterministically from
`(master seed, run index, block name)`. Two consequences:

- Re-running with the same master seed reproduces the ensemble exactly, run for
  run.
- Each block and each run still gets a statistically independent stream.

So an experiment is fully specified by one integer, which is what makes an
ensemble citable in a paper.

### Reading the results

The **Monte Carlo Ensemble** window reports `n_ok / n_runs successful runs`, and
offers two views per signal:

- **Time Series** — every run drawn faintly with the mean in bold and percentile
  bands around it.
- **Histogram** — the distribution of a single per-run *outcome metric*.

The outcome metrics are `final`, `mean`, `max`, `min`, `peak-to-peak` and `rms`,
each reducing one run's trace to one number. They are shared with the parameter
sweep, so a sweep heatmap and an ensemble histogram mean exactly the same thing
by "rms".

The run happens on a worker thread and is cancellable from the **Monte Carlo**
progress dialog.

## Parameter sweeps

**Analysis ▸ Parameter Sweep…**

Deterministic sweeps over one or two block parameters. The **Parameter Sweep**
dialog offers:

- **Sweep type** — `1-D (one parameter)` or `2-D (two parameters)`
- For each axis (**Parameter X**, **Parameter Y**): **Block**, **Parameter**,
  **Min**, **Max**, **Points**
- **Simulation time** and **Step size (dt)**

Only numeric scalar parameters can be swept; if no block exposes one, the dialog
says so.

Results, in a window also titled **Parameter Sweep**:

- **1-D** offers two views. **Response family** overlays every run's trace, so
  you see the shape change as the parameter moves. **Metric vs parameter** plots
  one outcome metric against the swept value — the usual way to find where
  overshoot or settling time turns over.
- **2-D** draws a heatmap of one outcome metric over the two parameters.

A 1-D sweep of a controller gain against `max` overshoot, or a 2-D sweep of
`Kp` × `Ki` against `rms` error, is the fast way to see a design trade-off
without deriving it.

## Live tuning

**View ▸ Parameter Tuning Panel** (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>T</kbd>)

Pin a parameter to the panel by right-clicking its block and choosing **Tune
live** (or a specific parameter from the **Add to Tuning** submenu). Each pinned
parameter becomes a slider; dragging it triggers a debounced headless
re-simulation and redraws the scope plots.

Right-click a slider for **Set Range…**, **Reset to _value_** and **Remove**.

You need one completed run first — the panel says
`Run simulation first (F5) before tuning` otherwise.

---

## How the experiment runners work

All of this is in `lib/analysis/`:

| Module | Role |
|---|---|
| `resim.py` | Shared headless re-simulation: `harvest_scope_signals`, `OUTCOME_METRICS` |
| `monte_carlo.py` | `MonteCarloRunner`, `derive_seed` |
| `parameter_sweep.py` | `ParameterSweepRunner` |
| `linearizer.py` | Numeric Jacobian linearization producing A/B/C/D |
| `control_system_analyzer.py`, `analyzers/` | Bode, Nyquist, root locus, LQR |

The two experiment runners snapshot the original block parameters and restore
them afterwards, and they run off the GUI thread, so a long ensemble never
freezes the window or leaves the diagram modified.

See the [Developer Guide](../DEVELOPER_GUIDE.md) and
[Architecture](../ARCHITECTURE.md) for the internals.
