# PDE Blocks

DiaBloS includes blocks for solving Partial Differential Equations using the Method of Lines. These blocks discretize spatial derivatives while scipy's ODE solver handles time integration.

## 1D PDE Equations

### HeatEquation1D

Solves the 1D heat/diffusion equation:
```
∂T/∂t = α ∂²T/∂x²
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.01 | Thermal diffusivity [m²/s] |
| `L` | float | 1.0 | Domain length [m] |
| `N` | int | 50 | Number of spatial nodes |
| `bc_type_left` | string | "Dirichlet" | Left BC: "Dirichlet" or "Neumann" |
| `bc_type_right` | string | "Dirichlet" | Right BC type |
| `init_temp` | string | "sine" | Initial condition: "sine", "gaussian", "uniform" |

**Inputs:** q_source, bc_left, bc_right
**Outputs:** Temperature field T(x), total heat

---

### WaveEquation1D

Solves the 1D wave equation:
```
∂²u/∂t² = c² ∂²u/∂x² - γ ∂u/∂t
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | float | 1.0 | Wave speed [m/s] |
| `damping` | float | 0.0 | Damping coefficient γ |
| `L` | float | 1.0 | Domain length [m] |
| `N` | int | 50 | Number of nodes |
| `init_displacement` | string | "sine" | Initial u(x,0) |
| `init_velocity` | float | 0.0 | Initial ∂u/∂t(x,0) |

**Outputs:** Displacement field u(x), velocity field v(x)

---

### AdvectionEquation1D

Solves the 1D advection (transport) equation:
```
∂c/∂t + v ∂c/∂x = 0
```

Uses upwind finite difference scheme.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity` | float | 1.0 | Advection velocity [m/s] |
| `L` | float | 1.0 | Domain length [m] |
| `N` | int | 50 | Number of nodes |
| `bc_type` | string | "Dirichlet" | Boundary: "Dirichlet" or "Periodic" |
| `init_conds` | string | "gaussian" | Initial: "gaussian", "step", "sine" |

**Inputs:** inlet_value
**Outputs:** Concentration field c(x)

---

### DiffusionReaction1D

Solves diffusion with reaction term:
```
∂c/∂t = D ∂²c/∂x² - k·cⁿ + S(x)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `D` | float | 0.01 | Diffusion coefficient [m²/s] |
| `k` | float | 0.1 | Reaction rate constant |
| `n` | int | 1 | Reaction order (1=linear, 2=quadratic) |
| `L` | float | 1.0 | Domain length [m] |
| `N` | int | 30 | Number of nodes |
| `init_conds` | string | "uniform" | Initial: "uniform", "gaussian", "linear", "sine" |

**Inputs:** source, bc_left, bc_right
**Outputs:** Concentration field c(x), total mass

---

## 2D PDE Equations

### HeatEquation2D

Solves 2D heat equation on rectangular domain:
```
∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.01 | Thermal diffusivity |
| `Lx`, `Ly` | float | 1.0 | Domain dimensions [m] |
| `Nx`, `Ny` | int | 20 | Grid nodes in each direction |
| `bc_type_left/right/bottom/top` | string | "Dirichlet" | Boundary condition type |
| `init_temp` | string | "sinusoidal" | Initial: "sinusoidal", "gaussian", "hot_spot" |

**Inputs:** q_src, bc_left, bc_right, bc_bottom, bc_top
**Outputs:** 2D temperature field T(x,y), T_avg, T_max

---

### WaveEquation2D

Solves 2D wave equation:
```
∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²) - γ ∂u/∂t
```

Converts to first-order system with displacement u and velocity v = ∂u/∂t.
Uses 2×Nx×Ny state variables.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `c` | float | 1.0 | Wave speed [m/s] |
| `damping` | float | 0.0 | Damping coefficient γ |
| `Lx`, `Ly` | float | 1.0 | Domain dimensions [m] |
| `Nx`, `Ny` | int | 20 | Grid nodes in each direction |
| `bc_type_left/right/bottom/top` | string | "Dirichlet" | Boundary condition type |
| `init_displacement` | string | "0.0" | Initial: number, "sinusoidal", "gaussian", "radial" |
| `init_velocity` | string | "0.0" | Initial velocity field |
| `init_amplitude` | float | 1.0 | Amplitude for non-uniform ICs |

**Inputs:** force, bc_left, bc_right, bc_bottom, bc_top
**Outputs:** Displacement field u(x,y), velocity field v(x,y), total energy

---

### AdvectionEquation2D

Solves 2D advection-diffusion equation:
```
∂c/∂t = -vx ∂c/∂x - vy ∂c/∂y + D (∂²c/∂x² + ∂²c/∂y²) + S
```

Uses upwind scheme for advection (stability) and central differences for diffusion.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vx`, `vy` | float | 1.0, 0.0 | Velocity components [m/s] |
| `D` | float | 0.0 | Diffusion coefficient (0 = pure advection) |
| `Lx`, `Ly` | float | 1.0 | Domain dimensions [m] |
| `Nx`, `Ny` | int | 30 | Grid nodes in each direction |
| `bc_type_left/right/bottom/top` | string | varies | "Dirichlet", "Neumann", or "Outflow" |
| `init_concentration` | string | "0.0" | Initial: number, "gaussian", "step", "pulse" |

**Inputs:** source, bc_left, bc_right, bc_bottom, bc_top
**Outputs:** Concentration field c(x,y), c_avg, c_max

---

## Field Processing Blocks

### FieldProbe

Extracts scalar value at a specific location from a field.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | float | 0.5 | Probe position |
| `position_mode` | string | "normalized" | "normalized" (0-1) or "absolute" (meters) |
| `L` | float | 1.0 | Domain length for absolute mode |

**Inputs:** field array, (optional) dynamic position
**Outputs:** scalar value at probe location

---

### FieldScope

Visualizes 1D field evolution over time.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `L` | float | 1.0 | Domain length [m] |
| `colormap` | string | "viridis" | Matplotlib colormap |
| `display_mode` | string | "heatmap" | "heatmap" (space-time) or "slider" (animated) |
| `title` | string | "Field Evolution" | Plot title |

**Inputs:** field array from PDE block

**Export:** Click the "Export" button on the slider figure to save as GIF or MP4.

---

### FieldScope2D

Visualizes 2D field with interactive time slider.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Lx`, `Ly` | float | 1.0 | Domain dimensions |
| `colormap` | string | "viridis" | Matplotlib colormap |
| `title` | string | "2D Field" | Plot title |

**Export:** Click the "Export" button on the figure to save as animated GIF or MP4.

---

### FieldIntegral

Computes spatial integral ∫ field(x) dx.

### FieldMax

Finds maximum (or minimum) value and its location.

### FieldGradient

Computes spatial derivative ∂field/∂x.

### FieldLaplacian

Computes second derivative ∂²field/∂x².

---

## Verification Examples

Each PDE has a verification example comparing numerical vs analytical solutions:

| Example | Analytical Solution |
|---------|---------------------|
| `heat_equation_1d_verification.diablos` | T = sin(πx/L)·exp(-απ²t/L²) |
| `wave_equation_1d_verification.diablos` | u = sin(πx/L)·cos(πct/L) |
| `advection_equation_1d_verification.diablos` | Traveling Gaussian pulse |
| `diffusion_reaction_1d_verification.diablos` | c = sin(πx/L)·exp(-(Dπ²/L²+k)t) |
| `heat_equation_2d_verification.diablos` | T = sin(πx)·sin(πy)·exp(-2απ²t) |

---

## Tips

1. **CFL condition**: For stability, ensure dt < dx²/(2α) for heat, dt < dx/c for waves
2. **Resolution**: More nodes (N) = better accuracy but slower
3. **Boundary conditions**: Use "Neumann" for insulated boundaries, "Dirichlet" for fixed values
4. **Visualization**: Use `display_mode: "slider"` in FieldScope to see animated evolution
5. **Animation Export**: Click "Export" on FieldScope/FieldScope2D figures to save GIF (requires Pillow) or MP4 (requires ffmpeg)
