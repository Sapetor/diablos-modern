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
| `alpha` | float | 1.0 | Thermal diffusivity [m²/s] |
| `L` | float | 1.0 | Domain length [m] |
| `N` | int | 20 | Number of spatial nodes |
| `bc_type_left` | string | "Dirichlet" | Left BC: "Dirichlet", "Neumann", "Robin", or "Periodic" |
| `bc_type_right` | string | "Dirichlet" | Right BC type (same choices) |
| `h_left`, `h_right` | float | 10.0 | Robin convective coefficient per end |
| `k_thermal` | float | 1.0 | Thermal conductivity used by Robin |
| `init_conds` | list/string | `[0.0]` | Number, list of N values, or "sine", "gaussian", "uniform", "step", "linear", "random" |
| `seed` | int | 0 | Seed for the "random" IC (0 = not reproducible) |

**Inputs:** `q_src`, `bc_left`, `bc_right`, `h_left` *(optional)*, `h_right` *(optional)*
**Outputs:** Temperature field T(x), average temperature

`bc_left` / `bc_right` carry the Dirichlet value, the Neumann flux, or the Robin
ambient temperature `T_inf`, depending on the edge's BC type -- so the ambient
temperature is time-varying out of the box. The `h_left` / `h_right` ports are
optional: connect one to vary that Robin coefficient in time (a fan switching
on), or leave it unconnected to use the static param. See
[Boundary conditions](#boundary-conditions) below.

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
| `bc_type_left` | string | "Dirichlet" | Left BC: "Dirichlet", "Neumann", or "Periodic" |
| `bc_type_right` | string | "Dirichlet" | Right BC type (same choices) |
| `init_displacement` | list/string | `[0.0]` | Number, list, or "sine", "gaussian", "uniform", "step", "linear", "random" |
| `init_velocity` | list/string | `[0.0]` | Initial ∂u/∂t(x,0), same named patterns |
| `seed` | int | 0 | Seed for the "random" IC (0 = not reproducible) |

**Inputs:** `force`, `bc_left`, `bc_right`
**Outputs:** Displacement field u(x), velocity field v(x), total energy

With `"Periodic"` the string closes into a ring, so a pulse leaving one end
re-enters at the other instead of reflecting. There is no Robin option for the
wave family.

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
| `bc_type_left/right/bottom/top` | string | "Dirichlet" | "Dirichlet", "Neumann", "Robin", or "Periodic" |
| `h_left`, `h_right`, `h_bottom`, `h_top` | float | 10.0 | Per-edge Robin convective coefficient |
| `k_thermal` | float | 1.0 | Thermal conductivity used by Robin |
| `init_temp` | string | "0.0" | Number, or "sinusoidal", "gaussian", "hot_spot", "radial", "linear", "step", "random", "checkerboard" |
| `init_amplitude` | float | 1.0 | Amplitude for non-uniform ICs |
| `seed` | int | 0 | Seed for the "random" IC (0 = not reproducible) |

**Inputs:** `q_src`, `bc_left`, `bc_right`, `bc_bottom`, `bc_top`,
`h_left`, `h_right`, `h_bottom`, `h_top` *(the four h ports are optional)*
**Outputs:** 2D temperature field T(x,y), T_avg, T_max

Each edge gets its own `h`, so a forced-convection edge and a still-air edge can
coexist on one plate. As in 1D, a Robin edge's `bc_*` port is the ambient
temperature `T_inf`, and the matching `h_*` port (when connected) makes the
coefficient itself time-varying.

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
| `bc_type_left/right/bottom/top` | string | "Dirichlet" | "Dirichlet", "Neumann", or "Periodic" |
| `init_displacement` | string | "0.0" | Number, or "sinusoidal", "gaussian", "radial", "linear", "step", "random", "checkerboard" |
| `init_velocity` | string | "0.0" | Initial velocity field, same named patterns |
| `init_amplitude` | float | 1.0 | Amplitude for non-uniform ICs |
| `seed` | int | 0 | Seed for the "random" IC (0 = not reproducible) |

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

## Boundary conditions

| Type | Meaning | Available on |
|------|---------|--------------|
| `Dirichlet` | Field value is prescribed at the boundary | all |
| `Neumann` | Normal derivative ∂/∂n is prescribed (0 = insulated) | all |
| `Robin` | Convective exchange: `-k ∂T/∂n = h (T − T_inf)` | HeatEquation1D, HeatEquation2D |
| `Periodic` | The axis wraps; the domain has no boundary there | Heat 1D/2D, Wave 1D/2D, AdvectionEquation1D |

The `bc_*` **input port** supplies whichever value the chosen type needs: the
Dirichlet value, the Neumann flux, or the Robin ambient temperature `T_inf`.
Because it is a port, all three are time-varying — drive it from a Step, a Ramp,
or any other block.

### Periodic

Set `"Periodic"` on **either** end of an axis and that whole axis wraps; the
opposite edge's type and value are ignored. In 2D the axes are independent:

| left / right | bottom / top | Domain |
|--------------|--------------|--------|
| Periodic | Periodic | torus |
| Periodic | Dirichlet or Neumann | channel, wrapping in x |
| Dirichlet or Neumann | Periodic | channel, wrapping in y |

The `N` nodes wrap as a **ring**, so the two endpoints remain distinct grid
points and the effective period is `N·dx`, not `L = (N−1)·dx`. Keep that in mind
when you set up a travelling wave: a pulse returns to its starting point after
`N·dx / c`, not `L / c`.

With no source, a periodic heat domain **conserves total heat exactly** (the
wrapped Laplacian is circulant, so its columns sum to zero) — a useful sanity
check that the solver is behaving.

### Robin

The sign convention is the outward normal, so a positive `h` always cools a body
that is hotter than ambient, on every edge. Larger `h` means stronger coupling:
`h → 0` degenerates to an insulated (zero-flux Neumann) edge, and large `h`
approaches a Dirichlet edge held at `T_inf`. A Robin-cooled body with no source
relaxes to `T_inf`.

To make the coefficient itself time-varying, connect the matching `h_*` input
port; leave it unconnected to use the static param of the same name. Both the
interpreter and the compiled fast solver re-read these ports every step, so
there is no compilability restriction and no accuracy caveat.

---

## Initial conditions

Named patterns accepted by `init_conds` / `init_temp` / `init_displacement` /
`init_velocity`. Anything not recognised is parsed as a number; you can also
pass an explicit list, which is interpolated or subsampled to the grid.

| Pattern | 1D | 2D | Shape |
|---------|----|----|-------|
| `sine` / `sinusoidal` | ✅ | ✅ | Laplacian eigenmode; decays cleanly |
| `gaussian` | ✅ | ✅ | Bump at the domain centre |
| `uniform` | ✅ | — | Constant 1.0 |
| `hot_spot` / `radial` | — | ✅ | Bump at the (0,0) corner |
| `step` | ✅ | ✅ | 1 over the first quarter in x, 0 elsewhere |
| `linear` | ✅ | ✅ | Ramp from 1 at x=0 down to 0 at x=L |
| `random` | ✅ | ✅ | Uniform noise, seeded (see below) |
| `checkerboard` | — | ✅ | ±amplitude on adjacent nodes |

2D patterns are scaled by `init_amplitude`.

`checkerboard` is the highest spatial frequency the grid can represent, so it
decays fastest under diffusion — handy as a stiffness or stability probe.

### Seeding `random`

`random` draws from a `seed` param that follows the same convention as
`blocks/noise.py`: **`seed = 0` means entropy** (a different field every run),
and any non-zero value is reproducible. The interpreter and the compiled solver
build the field through the same helper, so a seeded run matches on both paths.

The wave blocks build two fields (displacement and velocity) from that one
`seed`; the velocity field is offset internally so the two are independent
rather than the identical array.

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

### FieldProbe2D

Extracts scalar value at a specific (x,y) location from a 2D field.

Uses bilinear interpolation for positions between nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x_position` | float | 0.5 | X probe position |
| `y_position` | float | 0.5 | Y probe position |
| `position_mode` | string | "normalized" | "normalized" (0-1) or "absolute" (meters) |
| `Lx`, `Ly` | float | 1.0 | Domain dimensions for absolute mode |

**Inputs:** 2D field array, (optional) dynamic x_pos, y_pos
**Outputs:** scalar value at probed location

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
3. **Boundary conditions**: "Neumann" for insulated boundaries, "Dirichlet" for fixed values, "Robin" for convective cooling to an ambient temperature, "Periodic" for a domain with no boundary. See [Boundary conditions](#boundary-conditions).
6. **Reproducibility**: set a non-zero `seed` whenever you use a `"random"` initial condition, or the run will differ every time
4. **Visualization**: Use `display_mode: "slider"` in FieldScope to see animated evolution
5. **Animation Export**: Click "Export" on FieldScope/FieldScope2D figures to save GIF (requires Pillow) or MP4 (requires ffmpeg)
