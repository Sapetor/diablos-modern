# DiaBloS User Manual

## Getting Started

### Running the Application
```bash
python diablos_modern.py
```

### Interface Overview
- **Left Panel**: Block palette - drag blocks onto canvas
- **Center**: Canvas - build your diagram
- **Right Panel**: Property editor - configure selected block
- **Bottom**: Variable editor - define workspace variables

---

## Basic Workflow

### 1. Add Blocks
Drag blocks from the palette onto the canvas.

### 2. Connect Blocks
Press an output port, drag to an input port and release. You can also click
the output port, let go, and click the input port; or start from a free input
port and finish on an output port. While you drag, the preview turns green
over a port that will accept the wire and red over one that will not (already
connected, same block). Press **Esc**, or release on empty canvas, to cancel.
See [Wires](#wires) for routing, bending and auto-routing.

### 3. Configure Parameters
Click a block, then edit its parameters in the property editor.

### 4. Define Variables
Use the variable editor to define constants:
```python
K = 2.5
A = [1, 2, 3]
M = [[1, 0], [0, 1]]
```

### 5. Run Simulation
- **F5**, the toolbar's ▶, or **Simulation → Run** — the run starts immediately,
  using the solver, step size and duration stored in the diagram
- To change those first: **Ctrl+E**, the toolbar's gear, or
  **Simulation → Simulation Settings...**
- View results in Scope blocks

---

## Keyboard Shortcuts

### General
| Key | Action |
|-----|--------|
| Ctrl+N | New diagram |
| Ctrl+O | Open |
| Ctrl+S | Save |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+A | Select all |
| Delete | Delete selected |
| Ctrl+G | Create subsystem from selection |
| Ctrl+P | Command palette |
| F1 | Keyboard shortcuts reference |

### Simulation
| Key | Action |
|-----|--------|
| F5 | Run simulation |
| F6 | Pause simulation |
| F7 | Stop simulation |
| F8 | Single step (advance one timestep) |
| Ctrl+E | Simulation settings (solver, step size, duration) |

### Alignment (when 2+ blocks selected)
| Key | Action |
|-----|--------|
| Ctrl+Shift+L | Align left |
| Ctrl+Shift+R | Align right |
| Ctrl+Shift+H | Align center (horizontal) |
| Ctrl+Shift+T | Align top |
| Ctrl+Shift+B | Align bottom |

### View
| Key | Action |
|-----|--------|
| Ctrl++ / Ctrl+- | Zoom in / out |
| Ctrl+0 | Fit to window |
| Ctrl+Shift+G | Toggle grid |
| Ctrl+Shift+M | Toggle minimap |
| Ctrl+Shift+V | Toggle variable editor |
| Ctrl+Shift+W | Toggle workspace viewer |
| Ctrl+Shift+T | Toggle parameter tuning panel |
| Ctrl+T | Toggle theme (dark/light) |

> The full, always-current list is in **Help → Keyboard Shortcuts** (or **F1**).
> It is generated from the same table the command palette uses, so it never
> drifts from the real bindings.

### Command Palette
Press **Ctrl+P** (also **Edit → Command Palette** and **Help → Command Palette**)
to search and run without hunting through menus. The palette indexes:

- every block in the library ("Add *Gain* block") - placing it on the canvas
- simulation commands (run / pause / stop / step / toggle fast solver)
- view commands (zoom, fit, theme, grid, minimap, variable editor, workspace
  variables, tuning panel)
- file commands (new / open / save / load workspace, show plots, export as
  image / TikZ / Python script, copy diagram as image)
- the bundled examples and your recent files

---

## Simulation Settings and Solvers

Pressing **F5** (**Simulation → Run**) runs the diagram straight away with the
settings it already carries. To edit them, open the **Simulation Configuration**
dialog: **Ctrl+E**, the gear next to the transport buttons, or
**Simulation → Simulation Settings...**. It has two groups:

**Solver Configuration**
- **Solver Method**: `RK45` (default), `RK23`, `DOP853` (adaptive);
  `Radau`, `BDF`, `LSODA` (stiff); `RK4`, `Euler` (fixed-step, using the base
  step size below). This is the whole-diagram solver used by the compiled
  (fast) path. It is not the same setting as the **Integrator** block's own
  `Method`, which only applies on the interpreted path; the Integrator's
  fixed-step 4-stage scheme is listed there as `RK4` (diagrams saved before
  that label was corrected store `RK45` and still load).
- **Base Step Size (dt) [s]**: the global step. Discrete blocks run at their own
  `sampling_time` or synchronise to this step.
- **Simulation Duration [s]**.
- **Rel. tol / Abs. tol**: tolerances for the adaptive solvers.
- **Detect zero crossings (fast solver)** (on by default): stop the compiled
  solver exactly at each switching instant instead of letting an adaptive step
  smear it. See below.
- **Run in real-time**: pace the simulation against the wall clock.

**Visualization**
- **Plot Window Range [samples]** and **Enable Dynamic Plotting** for live
  plotting during a run.

Below the two groups, **Ask before every run** (off by default) restores the old
behaviour of re-opening this dialog every time you press Play. Unlike the
settings above it is an application preference, not diagram data: it is stored
in your Qt settings and follows you across diagrams.

Solver method, `rtol`, `atol` and the zero-crossing setting are saved inside the
`.diablos` file and shown read-only in the property editor. Accepting the dialog
after changing any of them marks the diagram as having unsaved changes.

### Zero-crossing detection

A discontinuous block -- `Switch`, `Saturation`, `Deadband`, `Hysteresis`,
`Abs`, `MathFunction` set to `sign`/`abs`, a `Step` or `Ramp` edge, a square or
sawtooth `WaveGenerator`, `PRBS` -- makes the compiled system's right-hand side
piecewise. Left alone, an adaptive step that straddles the switch either smears
it over the step or burns rejected steps shrinking onto it, and the switching
instant itself is only ever known to step accuracy.

With detection on (the default) the solver is given each block's switching
surface, stops exactly at the crossing, applies any discrete update (a relay
latch, say), and restarts from there. Switching instants then land on their true
time and stop moving when you change the step size. `Hysteresis` runs on the
compiled path *only* with this on -- its latch needs located instants to be
well-defined -- and falls back to the interpreter otherwise.

The cost is one extra solver restart per crossing, so a smooth diagram pays
nothing (no events are registered and the solver is called exactly as before).
Turn the setting off to reproduce the older behaviour for a whole diagram, or
set a single block's **zero_crossing** parameter to `off` to drop just that
block's crossings -- worth doing for a block that switches far faster than your
output step, where locating every crossing costs more than it buys.

If a relay chatters (switches infinitely often in finite time, as in sliding
mode), detection cannot converge. Rather than hang, DiaBloS logs a warning
naming the offending block and finishes the run with a fixed step. Adding
hysteresis to the switching element is the usual fix.

### Compiled vs interpreted execution

**Simulation → Enable Fast Solver (Experimental)** (checked by default) selects
the *compiled* path: the whole diagram is flattened into a single ODE
`dy/dt = f(t, y)` and integrated with `scipy.integrate.solve_ivp`. This is the
numerically accurate path and is typically 10x-100x faster than the
interpreter.

The engine falls back to the *interpreter* (fixed-step, block-by-block)
automatically when the diagram contains a block the compiler does not support,
or when any block declares a discrete `sampling_time > 0` - the compiled
right-hand side has no notion of sample instants. Unticking the menu item forces
the interpreter for every run.

See [FAST_SOLVER.md](FAST_SOLVER.md) for the supported-block list and
troubleshooting, and `docs/ARCHITECTURE.md` for the by-design differences
between the two engines.

---

## Viewing Results

### Scope windows
Scope blocks open a plot window after a run. Each window offers:

- a **columns** spinner plus **Auto** for the plot grid layout, and a minimum
  plot-height control
- **Previous run** - overlays the previous run's traces as dimmed dashed
  curves behind the live ones. The checkbox is disabled until a second run
  exists.
- **Export to CSV...** - pick which scopes to export in the
  *Select Scopes to Export* dialog
- **Export Figure...** - a publication-quality matplotlib figure (serif fonts,
  labelled time axis, legend, grid; step traces drawn as steps) written to
  **PDF**, **PNG** (300 dpi) or **SVG**

### Waveform Inspector
**Simulation → Show Plots** opens the *Waveforms* dock: a run-history overlay
across simulations.

- toggle individual runs and individual traces
- **Pin/Unpin** keeps a run when the history rotates
- scrub with the time slider to read values at the cursor
- **Export CSV** writes the selected traces
- **Persist history** stores the run history on disk so it survives a restart

### FieldScope (PDE and field visualisation)
`FieldScope` (1D) and `FieldScope2D` render a PDE block's field history. The 1D
block's `display_mode` chooses between `heatmap` (a space-time image) and
`slider` (an animated line plot with a time slider); the 2D block is an animated
heatmap with a frame slider. Both take a `colormap` (any matplotlib colormap
name) and a `title`. `AgentScope` animates multi-agent 2D trajectories the same
way.

Each of these windows has an **Export** button that opens the **Export
Animation** dialog: choose **GIF** or **MP4**, an FPS (1-60, with the resulting
playback duration shown) and a quality preset (72 / 100 / 150 dpi). Export runs
on a worker thread with a progress bar. MP4 requires `ffmpeg` on the PATH - the
radio button is disabled with a tooltip when it is missing.

---

## Analysis

The **Analysis** menu drives the linearization-based tools. All of them work on
the compiled ODE, so the diagram must be compilable.

### Linearize & Analyze...
Opens a dialog to pick which **source blocks act as system inputs** and which
blocks are **measured outputs**; leave both empty for an A-only
(eigenvalue/stability) analysis. Tick **Find operating point (trim) first** to
solve for equilibrium before linearizing.

The **Linearized System Analysis** window has five tabs:

| Tab | Contents |
|-----|----------|
| Pole-Zero | Pole/zero map of the linearized system |
| Bode | Magnitude and phase, with gain/phase margins and crossover frequencies |
| Step | Step response (when a SISO transfer function is available) |
| Impulse | Impulse response (same condition) |
| Summary | A/B/C/D matrices, stability, time constants, oscillatory modes, controllability and observability |

An export bar below the tabs offers **Copy as Python** (a numpy +
`python-control` snippet), **Copy as MATLAB**, and **Save Data...** (`.mat` or
`.npz`).

### Find Operating Point (Trim)...
Solves `f(0, y) = 0` on the compiled right-hand side and shows the equilibrium
state in the **Operating Point (Trim)** window, with copy-to-clipboard.

### Frequency-domain analyzer blocks
Nyquist, root-locus, Bode magnitude/phase and LQR are also available as blocks
you drop on the canvas and feed from a `TranFn` or `StateSpace` block:

- **BodeMag** / **BodePhase** - Bode magnitude / phase plot
- **Nyquist** - Nyquist plot
- **RootLocus** - root-locus plot
- **LQR** - computes the LQR gain

**Double-click** one of the four plot blocks - or right-click it and choose the
matching *Generate ... plot* entry - to produce the plot; no simulation run is
needed. The LQR block is driven from its right-click menu
(*Compute LQR gain*).

---

## Live Parameter Tuning

Manipulate-style interactive tuning: change a parameter with a slider and watch
the scope plots re-draw.

1. Run the simulation once so there is something to re-simulate.
2. Select a block, and in the property editor press **◉ Pin to tuning** next to
   the parameter you want to explore.
3. Open the panel with **View → Parameter Tuning Panel** (**Ctrl+Shift+T**).
4. Drag the slider. Changes are debounced, applied to the block, and a headless
   re-simulation updates the open scope window in place. The scope window is
   kept on top while tuning is active.
5. Right-click a slider row to set a custom min/max range.

Float parameters and individual list elements (for example transfer-function
coefficients) can both be pinned. Editing the diagram deactivates tuning.

---

## Experiments

Both experiment runners re-simulate the diagram headlessly and snapshot/restore
the original block parameters, so your diagram is never mutated.

### Monte Carlo ensembles
**Analysis → Monte Carlo...** opens the **Monte-Carlo Ensemble** dialog:
number of runs, a single **master seed**, simulation time and step size. Every
block that exposes a `seed` parameter (`PacketLoss`, `RandomSource`,
`NetworkChannel`, `Noise`, ...) derives its per-run sub-seed from
`(master_seed, run_index, block_name)`, so each run differs and the whole
ensemble is reproducible from that one number.

The **Monte Carlo Ensemble** result window shows a time-series view (mean,
percentile bands, sample traces) and a histogram view of a per-run outcome
metric.

### Parameter sweeps
**Analysis → Parameter Sweep...** sweeps one or two block parameters over a
grid.

- **1-D** produces a *response family* - one trace per swept value, coloured
  along the parameter - with a **View** toggle to a metric-vs-parameter plot.
- **2-D** produces a heatmap of an outcome metric over the (x, y) grid, with
  signal and metric pickers plus a colorbar.

Both experiments run on a worker thread with a progress dialog and can be
cancelled; a cancelled sweep keeps the runs it completed.

### Outcome metrics
Ensembles and sweeps share one metric vocabulary: `final`, `mean`, `max`,
`min`, `peak-to-peak`, `rms`.

---

## Export

### Export as Image...
**File → Export → Export as Image...** renders the diagram content only (no
grid, selection, or UI chrome) against the theme background, framed by the true
content bounding box and independent of the current zoom/pan. PNG is written at
3x; SVG is vector. **Edit → Copy Diagram as Image** puts the same render on the
clipboard.

### Export as TikZ...
**File → Export → Export as TikZ...** opens the **Export as TikZ** dialog with a
live preview, a standalone-document / snippet toggle, configurable options and
copy-to-clipboard - for dropping diagrams straight into a paper or slide deck.

**The figure is redrawn, not traced.** The exporter does not copy your canvas
coordinates: it follows the signal from its sources, puts the blocks on one
left-to-right row in that order, and routes everything that cannot run along
that row through a lane - feedback below, a forward wire that skips a block
above. Blocks you never wired up are parked at the end so they cannot split the
signal path. Consequences worth knowing:

- Blocks are chained with TikZ's `positioning` library, so the gap is measured
  *border to border*. A block with a long caption or a tall fraction pushes its
  neighbour along instead of overlapping it.
- Each block gets the outline the canvas paints it with: Gain/MatrixGain a
  triangle, Sum/Product a circle, Goto/From a tag, everything else a box.
- A **flipped** block only stays flipped where that means something - in the
  return path of a loop, which is where you flip a gain. The layout drops such a
  block into the loop lane pointing back at the summing junction. On the main
  row, where everything reads left to right, the flag is ignored.
- Where one output feeds several wires you get a junction dot, one per port.
- A scope hanging off a signal that already continues to another block is left
  out: the wire is drawn once, and a second arrowhead on it says nothing.

**Hand-tuning.** The generated file lists its knobs at the top. The useful ones
are `node distance` (the block gap, set once in the picture options) and the
picture's `x=`/`y=` units. Prefer those, or a shorter caption, over wrapping the
figure in `\resizebox`: scaling shrinks the type with the picture, and journals
set a minimum figure font size. `\resizebox` is available as an option but is
off by default.

**Pasting a snippet.** A snippet carries the `\usetikzlibrary` line it needs.
Keep that line outside any TeX group - if you paste a snippet inside
`\resizebox{..}{..}{..}` or a `minipage`, TeX records the libraries as loaded
globally but defines them only inside that group, and the *next* snippet then
fails with `You need to say \usetikzlibrary{calc}`. Moving the line to your
preamble once is the safe way to paste several figures.

### Export as Python Script...
**File → Export → Export as Python Script...** writes the diagram as a
self-contained `.py` file that depends only on **numpy** and **scipy**
(matplotlib is imported lazily, only to plot).

The generated script mirrors the **compiled** solver: the whole diagram becomes
one `rhs(t, x)` integrated by `scipy.integrate.solve_ivp`, reproducing the same
three-group evaluation order, so the script and a compiled DiaBloS run agree to
solver tolerance. Scope traces are labelled exactly as the headless CLI names
them, so the script's CSV lines up column-for-column with
`python diablos_modern.py run`.

The script has its own small CLI:

```bash
python model.py                # simulate and plot
python model.py --no-plot      # headless (also honours DIABLOS_NO_PLOT=1)
python model.py --out run.csv  # write the Scope traces to CSV or NPZ
python model.py --time 20 --dt 0.005
```

**Supported blocks.** Export covers a deliberate subset - the blocks whose
compiled kernel is a short, state-free expression that can be transcribed
literally:

> Abs, Constant, Demux, Display, Gain, Integrator, Mux, PID, Product, Ramp,
> Saturation, Scope, SgProd, Sine, StateSpace, Step, Sum, Terminator, TranFn

Subsystems are flattened first, so a diagram built from these blocks inside
subsystems still exports.

**Limitations.** The exporter refuses rather than emitting broken code, listing
the offending blocks:

- any block outside the list above - notably `Noise` / `PRBS` / `WaveGenerator`
  and `Hysteresis` / `StateVariable` (path- or RNG-dependent), the PDE and
  Field families, `MathFunction` (arbitrary user expressions), and
  `RateLimiter`
- any block with a discrete `sampling_time > 0` (sampled-data blocks need the
  interpreter)
- algebraic loops - break one with an Integrator or a strictly-proper transfer
  function before exporting

### Export the linearized model
See [Analysis](#analysis): the linearization result window exports A/B/C/D as
Python or MATLAB code, or as a `.mat` / `.npz` data file.

---

## Headless Command Line

Both subcommands run without opening a window, which makes them usable from
scripts and CI.

### Simulate a diagram
```bash
python diablos_modern.py run diagram.diablos -o out.csv
python diablos_modern.py run diagram.diablos --time 20 --dt 0.005 -o out.csv
python diablos_modern.py run diagram.diablos --solver interpreter -o out.npz
```

| Flag | Meaning |
|------|---------|
| `-o`, `--out` | Output file (`.csv` or `.npz`). Default: the diagram name with `.csv` |
| `-t`, `--time` | Simulation duration in seconds. Default: the diagram's `sim_time` |
| `--dt` | Time step in seconds. Default: the diagram's `sim_dt` |
| `--solver` | `compiled` (default) or `interpreter` |
| `--no-zero-crossing` | Disable compiled-path zero-crossing detection. Default: the diagram's own setting |
| `-q`, `--quiet` | Suppress the per-run summary on stdout |
| `--verify` | Print the post-run verification report (Display values, StateVariable convergence, Scope first→last samples judged by each Scope's `verify_mode`); exit code 3 when a check fails |

The Scope traces are written as columns (`t` first).

### Export a diagram as a Python script
```bash
python diablos_modern.py export-python diagram.diablos -o model.py
```

| Flag | Meaning |
|------|---------|
| `-o`, `--out` | Output `.py` file. Default: the diagram name with `.py` |
| `-t`, `--time` | Simulation duration baked into the script. Default: the diagram's `sim_time` |
| `--dt` | Time step baked into the script. Default: the diagram's `sim_dt` |
| `--solver` | Solver baked into the script (`RK45`, `LSODA`, `Euler`, `RK4`, ...). Default: the diagram's `solver_method` |
| `-q`, `--quiet` | Suppress the summary on stdout |

Both are also reachable as a module: `python -m lib.cli run diagram.diablos -o out.csv`.

---

## Importing Data: the FromFile Block

`FromFile` (category *Sources*) replays a recorded time series as a driving
signal - the way to compare a simulation against measured data or to fit a model
to a log.

- Reads `.csv`, `.npz`, `.mat` and whitespace-delimited `.txt`.
- `data_file` is the path; `time_col` and the signal column may be given as a
  name (CSV header / NPZ / MAT key) or as a 0-based numeric index.
- Samples are interpolated onto the simulation time grid (`linear`, `zoh` or
  `nearest`), with a hold-or-loop choice for times past the end of the file.
- The file is parsed once and cached; the cache is rebuilt when the simulation
  re-initialises or the path changes, so editing the path takes effect on the
  next run.

`LookupTable1D` / `LookupTable2D` read static tables the same way for
interpolated static maps.

---

## The Function Block

`Function` (category *Math*) evaluates a Python expression of its inputs and
time - the escape hatch for one-off nonlinearities.

- Inputs are exposed as the 0-indexed list `u[0]`, `u[1]`, ... and as the
  1-indexed aliases `u1`, `u2`, ...; the current simulation time is `t`.
- Add or remove input ports from the property editor's port spinner.
- Expressions go through a hardened AST evaluator: numpy math (`sin`, `cos`,
  `exp`, `sqrt`, `np.tanh`, ...) is allowed; imports, attribute escapes and
  statements are rejected.
- A list expression yields a vector output.
- Diagrams containing a `Function` block run on the interpreter.

Example: `sin(u[0]**2) + u[1]`

---

## PDE Blocks

DiaBloS solves PDEs by the **method of lines**: the spatial derivatives are
discretised on a uniform grid and the resulting ODE system is handed to the
solver, so PDE blocks run on the compiled fast path like everything else.

| Block | Equation |
|-------|----------|
| HeatEquation1D / 2D | `∂T/∂t = α∇²T + q` |
| WaveEquation1D / 2D | `∂²u/∂t² = c²∇²u` |
| AdvectionEquation1D / 2D | `∂u/∂t + v·∇u = 0` |
| DiffusionReaction1D | `∂u/∂t = D∇²u + R(u)` |

Boundary conditions are **Dirichlet**, **Neumann**, **Robin** (1D and 2D, with
per-edge convective coefficients) and **Periodic** (per axis). The Robin ambient
temperature arrives on the `bc_*` input ports, and the convective coefficient
`h` has optional input ports too, so both can vary in time; leave a port
unconnected to use the static parameter. Initial conditions can be a number, an
explicit list, or one of the presets (`sine`/`sinusoidal`, `gaussian`,
`uniform`, `linear`, `step`, `random`, plus `checkerboard`, `radial` and
`hot_spot` in 2D); `random` takes a `seed` for reproducibility.

Feed the field output into a `FieldScope` / `FieldScope2D` block to visualise
it (see [Viewing Results](#viewing-results)).

Full parameter tables: [PDE block reference](wiki/PDE.md). Roadmap:
[PDE_ROADMAP.md](PDE_ROADMAP.md).

---

## Optimization Blocks

Two complementary families:

**Optimization primitives** build an algorithm *as a diagram*, where one
simulation step is one iteration - `StateVariable` holds `x(k)`,
`ObjectiveFunction` evaluates `f(x)` from an expression, `VectorPerturb` +
`NumericalGradient` assemble a finite-difference gradient, and `VectorGain` /
`VectorSum` close the update loop. `Momentum`, `Adam`, `LinearSystemSolver`,
`RootFinder` and `ResidualNorm` round out the set. This is the teaching-friendly
way to show gradient descent as `x_{k+1} = x_k - α∇f(x_k)`.

**The optimizer family** (`Parameter`, `CostFunction`, `Constraint`,
`Optimizer`) instead wraps `scipy.optimize`: `Parameter` blocks declare the
tunable variables with bounds, `CostFunction` evaluates the objective, and
`Optimizer` configures the method (`L-BFGS-B` by default), iteration limit and
tolerance.

Details: [Optimization](wiki/Optimization.md) and
[Optimization Primitives](wiki/Optimization-Primitives.md).

---

## Subsystems

Subsystems allowed you to group blocks together to simplify large diagrams.

### Creating a Subsystem
1. Select multiple blocks using the selection rectangle or Shift+Click.
2. Press **Ctrl+G** (Group).
3. Selected blocks are replaced by a single "Subsystem" block.

### Editing a Subsystem
- **Double-click** a subsystem to enter it.
- **Press Esc** or click "Up" in the breadcrumb bar to exit.

### Adding Ports (MIMO)
- Inside a subsystem, drag **Inport** and **Outport** blocks from the palette.
- When you exit the subsystem, the parent block automatically updates to match the number and order of internal ports.
- Top-to-bottom order of internal ports corresponds to top-to-bottom order of external pins.

### Masking a Subsystem

A **mask** turns a subsystem into a component with its own small parameter
surface: a display name, an icon, a description, and the handful of values a
user of the block should be allowed to change. Everything else stays hidden
inside.

1. Select the subsystem and choose **Edit → Edit Mask...** (or right-click the
   block → *Edit mask...*).
2. On the **Parameters** tab, add one row per exposed parameter: a name, a type
   (`float`, `int`, `string`, `choice`, `list`, `bool`), a default, optional
   comma-separated choices, and a short description. Use **Move up** /
   **Move down** to set the order they appear in.
3. On **Icon & Appearance**, set the display name, the palette category the
   block is listed under, an icon label (a short formula or an emoji) and the
   outline shape (`rect`, `triangle`, `circle`, `tag`).
4. On **Documentation**, write the description shown in the property panel and
   in the palette tooltip.

Inside the subsystem, reference a mask parameter **by name** in any block value.
A Gain with `gain = K`, or a Transfer Function with `denominator = [m, b]`,
picks up the mask's values when the diagram runs. Names resolve against the
mask parameters first and the diagram's workspace variables second, so a mask
parameter shadows a workspace variable of the same name. Mask parameter values
are themselves expressions, so `b = m / 30` works, and a mask nested inside
another mask sees the outer mask's parameters when *its* values are evaluated.

The expression is never rewritten in your diagram: `denominator` still reads
`[m, b]` after a save, a reload and any number of runs. Only the copy the
engine executes carries the numbers.

Selecting a masked subsystem shows the **mask parameters** in the property
panel -- not the container's internals. To get at the contents, use **Look
Under Mask** (Edit menu or the context menu), or double-click as usual.

### Library blocks

A masked subsystem can be published as a reusable palette block.

1. Select it and choose **Edit → Save as Library Block...**.
2. Pick a file name. The default folder is your user library folder (or the
   first folder in `DIABLOS_LIBRARY_PATH` when that variable is set).
3. The block appears in the palette under the mask's category (**User Library**
   by default). **Edit → Refresh Block Library**, or the ⟳ button in the
   palette header, re-scans for new files.

Library folders are searched in this order, and the first copy of a given block
wins:

1. every folder listed in the `DIABLOS_LIBRARY_PATH` environment variable
   (separated by `:` on macOS/Linux, `;` on Windows);
2. a `library/` folder next to the diagram you have open;
3. your per-user library folder.

Drag a library block onto the canvas and DiaBloS drops in a **copy** of its
contents. Diagrams are therefore self-contained: deleting or renaming a library
file never breaks a diagram that used it. Each instance keeps its own mask
parameter values, so two "Vehicle" blocks can have different masses.

When the library file changes, right-click an instance and choose **Reload from
library** to pull in the new contents. Your instance's parameter values are
kept -- only the internals and the mask definition are refreshed.

### Try it

```bash
DIABLOS_LIBRARY_PATH=examples/library python diablos_modern.py
```

Open `examples/library_block_demo.diablos`: a proportional controller closes a
loop around a masked **Vehicle** block, `v(s)/F(s) = 1 / (m s + b)`, with the
mass `m` and damping `b` exposed on the mask. The same block is published as
`examples/library/vehicle.diablos`, so it also shows up in the palette under
*User Library*.

---

## MIMO Support

DiaBloS supports vector and matrix signals:

### Vector Signals
- Constant block: Set `value = [1, 2, 3]`
- Connections automatically carry vector data
- Thicker lines indicate vector signals (during simulation)

### Matrix Gain
- Gain block supports matrix multiplication: `gain = [[1,0],[0,1]]`
- Output: `y = K @ u` (matrix-vector multiplication)

### Selector Block
- Extract specific elements from vectors
- Use indices like `0,2` or `1:3`

---

## Wires

A wire always runs from an output port to an input port; an output may feed
any number of inputs, an input accepts one wire.

### Routing modes

- **Bezier (curved)** wires are smooth curves that leave and enter ports
  horizontally. They stay curved when you move blocks.
- **Orthogonal (Manhattan)** wires are laid out by an obstacle-avoiding
  router that keeps clear of blocks and their names. Auto-routed wires are
  re-routed whenever a block they touch is moved or resized.

Pick the mode for new wires in **View → Default Connection Routing**, or
right-click a wire → **Routing** to change one wire.

### Bending wires by hand

- **Drag a segment** to move it; dragging a straight or curved wire turns it
  into a three-segment bend first.
- **Drag a bend handle** (the small circles on a selected wire) to move that
  corner. Handles snap to the grid when grid snapping is on.
- **Double-click a bend handle** to remove it.
- Hand-bent wires keep their bends when blocks move; only the end segments
  follow the ports.
- Right-click → **Reset routing** discards the bends; **Auto-route wire**
  lets the router lay the wire out again.
- **Auto-route wires** (toolbar, or right-click the canvas) routes every wire
  at once.

Crossing wires show a small gap where one passes over the other. Bend and
move edits are undoable with **Ctrl+Z**.

---

## Alignment Tools

Keep your diagrams tidy with alignment and distribution tools.

### Aligning Blocks
1. Select 2 or more blocks (Shift+Click or rectangle selection)
2. Right-click → **Align** submenu
3. Choose alignment option:
   - **Align Left/Right**: Align to leftmost/rightmost block
   - **Align Center (Horizontal)**: Align to horizontal center
   - **Align Top/Bottom**: Align to topmost/bottommost block
   - **Align Center (Vertical)**: Align to vertical center

### Distributing Blocks
With 3+ blocks selected:
- **Distribute Horizontally**: Equal horizontal spacing
- **Distribute Vertically**: Equal vertical spacing

---

## Single-Step Simulation

Debug your simulations one timestep at a time.

### How to Use
1. Press **F8** to start stepping (no need to press F5 first)
2. Simulation initializes at t=0 and advances one step
3. Press **F8** again to advance another timestep
4. Check Scope plots after each step
5. Press **F7** to stop when done

### Use Cases
- Debug unexpected behavior
- Understand signal flow
- Verify initial conditions
- Step through short simulations

---

## Minimap

Navigate large diagrams with the minimap overview.

### Enabling the Minimap
- **View → Minimap** or press **Ctrl+Shift+M**

### Features
- Shows scaled overview of entire diagram
- Blue rectangle shows current viewport
- Click anywhere on minimap to pan canvas
- Drag on minimap for continuous panning
- Dock on left or right side of window

---

## Appearance

Everything here lives in the **View** menu. The first three are written to
`user_preferences.json` in the user data directory, so they survive a restart.

- **Toggle Theme** (**Ctrl+T**) switches between the dark and light themes.
- **Block Palette** picks the block colour scheme: *Solarized*, *Tailwind* or
  *Catppuccin Frappé*.
- **Solid Block Fills** switches blocks between tinted and solid fills.
- **UI Scale** offers 100%, 125% and 150% for high-DPI screens.
- **Default Connection Routing** chooses *Bezier (Curved)* or
  *Orthogonal (Manhattan)* for new connections. Right-click an existing
  connection → **Routing** to change just that one (see [Wires](#wires)).
- **Live overlay → Output value chips** shows each port's current value on the
  canvas during a run (on by default).
- **Show Grid** (**Ctrl+Shift+G**) toggles the canvas grid.

## Changing the Language

**View ▸ Language** switches the interface language. The menu lists every
translation shipped with DiaBloS (each named in its own language) plus
**System default**, which follows your operating system's locale. Spanish
(*Español*) ships with the application; anything without a translation falls
back to English, so nothing ever shows up blank.

The choice is remembered between sessions. Menus, the toolbar, the block
palette, panel and dock titles and the status bar switch immediately; windows
and dialogs that are already open keep the language they were opened with, so
close and reopen them (or restart) to see them translated.

Block names stay in English on purpose — they are what a saved diagram
records, so a `.diablos` file opens identically whatever language you use. The
palette's category headings and each parameter's help text *are* translated.

---

---

## Autosave and Recovery

- The diagram is autosaved every **2 minutes** to `.autosave.diablos`, and again
  immediately before every simulation run as `<name>_AUTOSAVE.diablos`. Both
  live in the application's user data directory:

  | Platform | Location |
  |----------|----------|
  | macOS | `~/Library/Application Support/DiaBloS/` |
  | Windows | `%APPDATA%\DiaBloS\` |
  | Linux | `~/.local/share/DiaBloS/` (or `$XDG_DATA_HOME/DiaBloS/`) |

  The 2-minute autosave is in `config/` there; the pre-run snapshot and the
  **Export** block's data files are in `saves/`. Running from a source checkout
  instead uses the project's own `config/` and `saves/` folders.
- If an autosave cannot be written, the status bar says so -- it never
  interrupts you with a dialog, so check there if recovery matters to you.
- If the application exits abnormally, the next start asks
  **"Recover Auto-save?"**. Answer **Yes** to reload that session; answering
  **No** deletes the autosave file.
- Closing the window with unsaved changes asks **Save / Discard / Cancel**.
  Cancel keeps the window open; the autosave file is removed only once the
  diagram has been saved or the changes explicitly discarded.
- A clean exit removes the autosave file, so the prompt only appears when there
  is genuinely something to recover.

Autosave is a safety net, not a substitute for **Ctrl+S**.

---

## Tips

- **Flip blocks**: Right-click → Flip
- **Resize blocks**: Drag corner handles
- **Toggle routing**: Right-click connection → Routing
- **Straighten a wire**: Right-click connection → Reset routing
- **Test incrementally**: Build and test small sections first
- **Use minimap**: For large diagrams, enable minimap for quick navigation
- **Step through**: Use F8 to debug simulations step-by-step
- **Search, don't hunt**: Ctrl+P finds any block, command or example by name
- **Reuse a run**: pin a parameter to the tuning panel instead of re-running by
  hand for every value
