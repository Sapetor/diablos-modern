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

## Troubleshooting

### "Algebraic Loop Detected"

Your diagram has a feedback loop without a delay element. Add an Integrator or TransportDelay block to break the loop.

### "Unlinked Port"

All required ports must be connected. Check the error message for which block has unconnected ports.
