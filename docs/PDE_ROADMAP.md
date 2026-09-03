# PDE Blocks Enhancement Roadmap

> Future development plan for extending PDE simulation capabilities in DiaBloS.

## Current State

DiaBloS PDE blocks use **Method of Lines (MOL)**:
- Spatial discretization → system of ODEs → scipy `solve_ivp`
- **Domains**: Rectangular only (`[0,Lx] × [0,Ly]`)
- **BCs**: Dirichlet, Neumann, Robin (1D and 2D, per-edge `h`), Periodic (per axis)
- **Meshes**: Structured uniform grids only

The spatial discretisation and BC math for every family live in
`lib/engine/pde_ops.py`, which both the interpreter blocks (`blocks/pde/`) and
the compiled kernels (`lib/engine/compiler_kernels/pde.py`) call. Initial
conditions are single-sourced the same way through `lib/engine/pde_helpers.py`.
Adding a BC or an IC in one place therefore reaches both execution paths.

### Existing PDE Blocks

| Block | Equation | Dimensions |
|-------|----------|------------|
| HeatEquation1D/2D | `∂T/∂t = α∇²T + q` | 1D, 2D |
| WaveEquation1D/2D | `∂²u/∂t² = c²∇²u` | 1D, 2D |
| AdvectionEquation1D/2D | `∂u/∂t + v·∇u = 0` | 1D, 2D |
| DiffusionReaction1D | `∂u/∂t = D∇²u + R(u)` | 1D |

---

## Enhancement Phases

### Phase 1: Quick Wins (Low Effort) — ✅ DONE

**1.1 Periodic Boundary Conditions** — ✅ done
- Added to HeatEquation1D/2D and WaveEquation1D/2D, matching the
  `AdvectionEquation1D` pattern.
- Selecting `'Periodic'` on **either** end wraps that whole axis; the opposite
  edge's BC type and value are ignored (`pde_ops.is_periodic`). In 2D the axes
  are independent, so x-periodic + Dirichlet top/bottom (a channel) is valid.
- The N nodes wrap as a **ring**: the two endpoints stay distinct degrees of
  freedom, so the effective period is `N·dx`, not `L = (N-1)·dx`. This matches
  the pre-existing advection convention and makes the discrete Laplacian
  circulant — which is what conserves total heat exactly.
- Implemented in `lib/engine/pde_ops.py`, so the interpreter and compiled paths
  are equivalent by construction. The non-periodic branches were left
  bit-identical so the pinned golden traces did not move.

**1.2 Dynamic BC Coefficients** — ✅ done
- The Robin **ambient temperature** `T_inf` was already an input port (`bc_*`),
  hence already time-varying. What was static was `h`.
- HeatEquation1D gained optional input ports `h_left` (3), `h_right` (4);
  HeatEquation2D gained `h_left` (5), `h_right` (6), `h_bottom` (7),
  `h_top` (8). An unconnected port falls back to the matching param, so
  diagrams saved with the old port count load and run unchanged.
- Both paths re-read the ports on every RHS evaluation, so no compilability
  restriction was needed — the compiled solver supports input-driven `h`
  fully (the ports are ordinary signals, evaluated per solver step).

**1.3 More Initial Condition Templates** — ✅ done
- 1D: `'linear'`, `'step'`, `'random'` (plus the existing `'sine'`,
  `'gaussian'`, `'uniform'`).
- 2D: `'linear'`, `'step'`, `'random'`, `'checkerboard'`, `'radial'` (plus
  `'sinusoidal'`, `'gaussian'`, `'hot_spot'`).
- `'random'` is seeded by a new `seed` param following the `blocks/noise.py`
  convention (`0` = entropy / not reproducible). The wave blocks build two
  fields from one seed, so the velocity field uses `companion_seed()` to stay
  independent of the displacement field.
- The block `get_initial_conditions()` / `get_initial_state()` methods now
  delegate to `pde_helpers.parse_pde_*_initial_condition`, the same helpers the
  compiler uses. Previously each block carried its own copy; HeatEquation1D's
  copy recognised fewer patterns and used a narrower `'gaussian'`
  (`exp(-100 r²)` vs the compiled path's `exp(-50 r²)`), so the interpreter and
  the compiled path started from different fields.

**1.4 Robin BC for 2D** — ✅ done
- HeatEquation2D supports `'Robin'` on any edge, with per-edge `h_left`,
  `h_right`, `h_bottom`, `h_top` and a shared `k_thermal`.
- Convention is the outward normal, `-k ∂T/∂n = h (T − T_inf)`, so a positive
  `h` cools an over-ambient plate on every edge.
- The 2D form folds the convective flux into the **Neumann ghost-node stencil**
  rather than the penalty formulation the 1D block uses. The two agree in
  steady state (`T → T_inf` for `h > 0`) and the flux form is as stable as a
  Neumann edge, so it survives the interpreter's Forward-Euler step where a
  penalty term would not. 1D keeps its penalty/hold form because the
  compiled-golden traces pin it.

**Verification**: `tests/unit/test_pde_phase1.py` (52 cases: operator-level
periodic/Robin/IC behaviour, conservation, seeded reproducibility) and
`tests/regression/test_equiv_pde_phase1.py` (compiled-vs-interpreted
equivalence for periodic heat 1D/2D, periodic wave 1D, and a Robin plate driven
by a ramped ambient temperature).

---

### Phase 2: Geometric Abstraction (Medium Effort)

**Goal**: Decouple PDE solvers from grid topology

**2.1 MeshBase Abstract Class**
```
blocks/pde/mesh/
├── __init__.py
├── mesh_base.py          # Abstract interface
├── rectangular_mesh.py   # Current behavior
└── curvilinear_mesh.py   # Future: mapped coordinates
```

Interface:
```python
class MeshBase:
    def get_laplacian_matrix(self) -> sparse_matrix
    def get_boundary_nodes(self, edge: str) -> list[int]
    def get_node_coords(self, idx: int) -> tuple[float, float]
    def get_neighbors(self, idx: int) -> list[int]
```

**2.2 Refactor 2D PDE Blocks**
- Replace hardcoded `(i,j)` loops with mesh interface calls
- Keep `RectangularMesh` as default (backward compatible)
- Files affected:
  - `blocks/pde/heat_equation_2d.py`
  - `blocks/pde/wave_equation_2d.py`
  - `blocks/pde/advection_equation_2d.py`

**2.3 Curvilinear Mesh Support**
- Mapped coordinates for non-rectangular but structured grids
- Examples: Annular domains, tapered channels
- Uses coordinate transformation Jacobians

---

### Phase 3: Unstructured Meshes (High Effort)

**Goal**: Arbitrary 2D domain geometries

**3.1 Mesh Input/Output Blocks**
```
blocks/mesh/
├── mesh_loader.py        # Read .msh, .vtk, .stl
├── mesh_generator.py     # Simple shapes (circle, L-shape)
└── mesh_exporter.py      # Write VTK for ParaView
```

**3.2 UnstructuredMesh Class**
- Triangle/quad element storage
- Sparse connectivity matrix
- Boundary edge markers
- Uses `scipy.sparse` for Laplacian assembly

**3.3 FEM-based Spatial Operators**
- Local element assembly (P1 triangles)
- Precompute stiffness/mass matrices at compile time
- Store in block params for fast ODE evaluation

**3.4 Boundary Condition Infrastructure**
- Boundary markers from mesh file
- Map BC types to edge groups
- Support multiple BCs on same domain

**3.5 Field Processing Updates**
- `FieldProbe2D`: Barycentric interpolation for unstructured
- `FieldScope2D`: Triangulation-aware rendering
- New: `FieldExportVTK` for external visualization

---

### Phase 4: Advanced Features (Future)

**4.1 Absorbing Boundary Conditions**
- Sommerfeld radiation BCs for wave equations
- Perfectly Matched Layers (PML)
- Files: `wave_equation_1d.py`, `wave_equation_2d.py`

**4.2 Adaptive Mesh Refinement**
- Error estimator block
- Local refinement triggers
- Mesh coarsening for efficiency

**4.3 Domain Decomposition**
- Multi-block domains with interface conditions
- Flux matching at subdomain boundaries
- Enables parallel solving (future)

**4.4 3D Support**
- HeatEquation3D, WaveEquation3D
- Tetrahedral unstructured meshes
- VTK volume visualization

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  (Block Palette, Canvas, Property Editor)               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Block Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ PDE Blocks  │  │ Field Proc  │  │ Mesh Blocks │     │
│  │ (Heat,Wave) │  │ (Probe,Scope│  │ (Load,Gen)  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼────────────────┼────────────────┼─────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────┐
│                   Mesh Layer (NEW)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ MeshBase    │  │ Rectangular │  │ Unstructured│     │
│  │ (abstract)  │  │ Mesh        │  │ Mesh        │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 Simulation Engine                        │
│  SystemCompiler → ODE System → solve_ivp → Replay       │
└─────────────────────────────────────────────────────────┘
```

---

## Key Files to Modify

| Phase | Files |
|-------|-------|
| 1 | `blocks/pde/heat_equation_*.py`, `wave_equation_*.py`, `advection_*.py` |
| 2 | New: `blocks/pde/mesh/`, modify all 2D PDE blocks |
| 3 | New: `blocks/mesh/`, `lib/engine/system_compiler.py` (sparse support) |
| 4 | New blocks, major engine changes |

---

## Effort Estimates

| Phase | Complexity | New Files | Modified Files |
|-------|------------|-----------|----------------|
| 1 ✅  | Low        | 2 (tests) | 8              |
| 2     | Medium     | 4-5       | 4-6            |
| 3     | High       | 8-10      | 10+            |
| 4     | Very High  | 15+       | Many           |

---

## Verification Strategy

**Phase 1** (done):
- `tests/unit/test_pde_phase1.py` -- operator-level checks: wrapped end-node
  stencils, exact conservation on a periodic ring, Robin steady state reaching
  ambient on all four edges, per-edge `h` independence, `h = 0` degenerating to
  an insulated edge, and seeded-`'random'` reproducibility.
- `tests/regression/test_equiv_pde_phase1.py` -- compiled-vs-interpreted
  equivalence per new BC, plus a dynamic-ambient Robin plate.
- `tests/regression/test_compiled_golden.py` unchanged: the non-periodic
  branches were deliberately left bit-identical, so no golden data moved.

**Phase 2**:
- Curvilinear mesh: Solve on annulus, compare to analytical
- Backward compatibility: All existing .diablos files work unchanged

**Phase 3**:
- Circular domain heat equation (known solution)
- L-shaped domain (benchmark problem)
- Mesh convergence study

---

## Summary

**Can you simulate heat equation on any 2D domain today?** No, only rectangles.

**What would it take?**
- Phase 1: Easy BC extensions (days of work)
- Phase 2: Mesh abstraction layer (1-2 weeks)
- Phase 3: Full unstructured support (weeks to months)

**Recommended starting point**: Phase 1 is complete. Phase 2's mesh
abstraction is the next step, and it inherits a useful precondition from Phase
1: all spatial-operator and IC math is already funnelled through
`lib/engine/pde_ops.py` and `lib/engine/pde_helpers.py`, so the mesh interface
has exactly two seams to replace rather than one per block per execution path.

---

*Last updated: September 2026 (Phase 1 completed)*
