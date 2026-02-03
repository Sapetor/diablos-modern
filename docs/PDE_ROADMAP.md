# PDE Blocks Enhancement Roadmap

> Future development plan for extending PDE simulation capabilities in DiaBloS.

## Current State

DiaBloS PDE blocks use **Method of Lines (MOL)**:
- Spatial discretization → system of ODEs → scipy `solve_ivp`
- **Domains**: Rectangular only (`[0,Lx] × [0,Ly]`)
- **BCs**: Dirichlet, Neumann (Robin partial in 1D)
- **Meshes**: Structured uniform grids only

### Existing PDE Blocks

| Block | Equation | Dimensions |
|-------|----------|------------|
| HeatEquation1D/2D | `∂T/∂t = α∇²T + q` | 1D, 2D |
| WaveEquation1D/2D | `∂²u/∂t² = c²∇²u` | 1D, 2D |
| AdvectionEquation1D/2D | `∂u/∂t + v·∇u = 0` | 1D, 2D |
| DiffusionReaction1D | `∂u/∂t = D∇²u + R(u)` | 1D |

---

## Enhancement Phases

### Phase 1: Quick Wins (Low Effort)

**1.1 Periodic Boundary Conditions**
- Pattern already exists in `AdvectionEquation1D`
- Copy to: HeatEquation1D/2D, WaveEquation1D/2D
- Add `bc_type: 'Periodic'` option to params

**1.2 Dynamic BC Coefficients**
- Current: Fixed `h_left`, `h_right` params for Robin
- Change: Add optional input ports for time-varying coefficients
- Files: `heat_equation_1d.py`, `heat_equation_2d.py`

**1.3 More Initial Condition Templates**
- Add: `'linear'`, `'step'`, `'random'`, `'checkerboard'`
- Location: `get_initial_state()` methods

**1.4 Robin BC for 2D**
- Currently only in 1D blocks
- Extend to HeatEquation2D with per-edge h coefficients

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
| 1     | Low        | 0         | 6-8            |
| 2     | Medium     | 4-5       | 4-6            |
| 3     | High       | 8-10      | 10+            |
| 4     | Very High  | 15+       | Many           |

---

## Verification Strategy

**Phase 1**:
- Unit tests for periodic BCs (compare to analytical solutions)
- Existing verification examples still pass

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

**Recommended starting point**: Phase 1 quick wins, then Phase 2 mesh abstraction to enable future flexibility without breaking existing functionality.

---

*Last updated: February 2026*
