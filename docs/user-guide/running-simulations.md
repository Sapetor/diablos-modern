# Running Simulations

## Simulation Parameters

Configure simulation parameters in the toolbar:

- **Simulation Time**: Total duration of the simulation (seconds)
- **Time Step (dt)**: Integration time step

## Running a Simulation

1. Ensure your diagram is valid (all ports connected)
2. Set simulation parameters
3. Click "Run" or press F5
4. Wait for completion
5. View results in Scope/Display blocks

## Simulation Engine

DiaBloS uses SciPy's `solve_ivp` for ODE integration with:

- RK45 (Runge-Kutta 4th/5th order) by default
- Adaptive step size for accuracy
- Algebraic loop detection

## Viewing Results

### Scope Plots

Scope blocks show time-series data:

- Multiple inputs shown as separate traces
- Zoom with mouse wheel
- Pan by dragging

### Field Visualizations

For PDE blocks:

- FieldScope: 1D field as heatmap over time
- FieldScope2D: 2D field with time slider

### Exporting Results

Right-click on plot windows to:

- Save as image (PNG, JPG)
- Export animation (GIF, MP4) for field plots
- Copy to clipboard

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
