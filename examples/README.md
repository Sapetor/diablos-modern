# Example Diagrams

This folder contains example diagrams demonstrating DiaBloS features.

---

## heat_equation_demo.diablos

**1D Heat Equation Simulation using Method of Lines (MOL)**

### Overview

This example demonstrates the new PDE (Partial Differential Equation) blocks by simulating 1D heat diffusion in a rod. The heat equation is:

```
∂T/∂t = α ∇²T + q
```

Where:
- `T(x,t)` = temperature at position x and time t
- `α` = thermal diffusivity (0.05 m²/s)
- `q` = heat source term (0 in this example)
- `∇²T` = second spatial derivative (∂²T/∂x²)

### Diagram Structure

```
[HeatSource_0] ──────────────────┐
   (q = 0)                       │ port 0: heat source
                                 ↓
[LeftBC_100] ──────────────────→ [HeatEquation1D] ──→ [Probe_x05] ──→ [Scope]
   (T = 100°)                    │ port 1: left BC  │    (x=0.5)       │
                                 │                  │                   │
[RightBC_0] ────────────────────→┘ port 2: right BC ├→ [Probe_x025] ──→┘
   (T = 0°)                                         │    (x=0.25)
                                                    │
                                                    └→ [FieldScope]
                                                         (2D heatmap)
```

### Block Parameters

| Block | Parameter | Value | Description |
|-------|-----------|-------|-------------|
| HeatEquation1D | alpha | 0.05 | Thermal diffusivity [m²/s] |
| | L | 1.0 | Domain length [m] |
| | N | 20 | Number of spatial nodes |
| | bc_type_left | Dirichlet | Fixed temperature at left |
| | bc_type_right | Dirichlet | Fixed temperature at right |
| | init_conds | [20.0] | Initial temperature (uniform 20°) |
| LeftBC_100 | value | 100.0 | Left boundary temperature |
| RightBC_0 | value | 0.0 | Right boundary temperature |
| Probe_x05 | position | 0.5 | Probe at center (x = 0.5) |
| Probe_x025 | position | 0.25 | Probe at quarter (x = 0.25) |

### Simulation Settings

- **Duration:** 5 seconds
- **Time step:** 0.01 s
- **Solver:** ODE solver with Method of Lines discretization

### Expected Results

1. **Initial Condition:** Temperature is 20° everywhere
2. **Boundary Conditions:** Left side held at 100°, right side at 0°
3. **Transient Behavior:** Heat diffuses from left (hot) to right (cold)
4. **Steady State:** Linear temperature gradient from 100° to 0°

### What to Observe

**Scope (Temperature vs Time):**
- T(x=0.5): Starts at 20°, rises toward ~50° (midpoint of gradient)
- T(x=0.25): Starts at 20°, rises toward ~75° (closer to hot end)

**FieldScope (2D Heatmap):**
- X-axis: Position (0 to 1 m)
- Y-axis: Time (0 to 5 s)
- Color: Temperature (hot = red/yellow, cold = dark)
- Shows the heat wave propagating from left to right

### How It Works

The **Method of Lines (MOL)** converts the PDE into a system of ODEs:

1. Space is discretized into N=20 nodes
2. Spatial derivatives (∇²T) computed using central finite differences
3. This gives 20 coupled ODEs: dTᵢ/dt = f(T₀, T₁, ..., T₁₉)
4. The ODE solver (solve_ivp) integrates these in time

The **HeatEquation1D** block:
- Stores N state variables (temperature at each node)
- Computes dT/dt at each time step
- Applies boundary conditions (Dirichlet: fixed values)

### Modifying the Example

Try changing these parameters:

| Change | Effect |
|--------|--------|
| Increase `alpha` to 0.2 | Faster heat diffusion |
| Decrease `alpha` to 0.01 | Slower diffusion, more visible transient |
| Change `bc_type_left` to "Neumann" | Fixed heat flux instead of temperature |
| Increase `N` to 50 | Finer spatial resolution |
| Add heat source (HeatSource_0 = 100) | Internal heating |

### Troubleshooting

**Simulation won't run:**
- Check that all connections are made (8 lines total)
- Verify block names match exactly

**Results look wrong:**
- Check boundary condition values
- Verify alpha is reasonable (0.01 - 0.5 typical)
- Ensure simulation time is long enough to see transient

---

## Creating Your Own PDE Examples

### Available PDE Blocks

| Block | Equation | Use Case |
|-------|----------|----------|
| HeatEquation1D | ∂T/∂t = α∇²T + q | Heat conduction, diffusion |
| WaveEquation1D | ∂²u/∂t² = c²∇²u | Vibrating string, acoustics |
| AdvectionEquation1D | ∂c/∂t + v∂c/∂x = 0 | Transport, convection |
| DiffusionReaction1D | ∂c/∂t = D∇²c - kc | Chemical reactions |

### Field Processing Blocks

| Block | Function |
|-------|----------|
| FieldProbe | Extract value at position x |
| FieldIntegral | Integrate field over domain |
| FieldMax | Find maximum value and location |
| FieldScope | 2D heatmap visualization |
| FieldGradient | Compute ∂field/∂x |
| FieldLaplacian | Compute ∇²field |

### Tips

1. Always connect boundary condition inputs (even if 0)
2. Use FieldProbe to extract scalar values for Scope
3. FieldScope shows full spatiotemporal evolution
4. Start with coarse grid (N=20), refine if needed
5. Check stability: smaller α or larger N may need smaller dt
