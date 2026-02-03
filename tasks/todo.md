# DiaBloS Modern - Consolidated TODO

> Single source of truth for all pending work items.
> Last updated: February 2026

---

## High Priority

### Testing & Quality
- [x] **Regression test suite** - Created `tests/regression/test_regression_suite.py` with 26 tests
  - Numerical accuracy tests (integrators, TFs, state-space, transport delay, PID)
  - Bug fix regression tests (Step/Scope params, External block, StateVariable)
  - PDE block tests (Heat1D conservation, Heat2D initialization)
  - Optimization primitives tests (ObjectiveFunction, VectorGain, VectorSum)
- [x] **Performance profiling** - Profiled simulation engine (`tests/profiling/profile_simulation.py`)
  - Block-level simulation: 1000 steps in 0.03s (excluding imports)
  - PDE simulation: 100 steps × 100 nodes in 6ms
  - Main bottleneck: Import overhead (scipy), not execution
  - Recommendations documented in profiling script

### Block Implementation
- [x] **External block** - Returns proper error dict with message (was stub returning None)
  - File: `blocks/external.py`
  - Status: Properly handles missing file with informative error

### Known Issues
- [x] **Legacy test files** - Marked as skipped with proper documentation
  - `tests/test_blocks.py` - DSim requires GUI
  - `tests/test_sine_params.py` - Legacy DBlock API
  - `tests/test_transfer_function_exec.py` - DSim requires GUI

---

## Medium Priority

### Refactoring (from REFACTORING_TODO.md)
- [ ] **Extract SubsystemManager from lib.py** - Complex extraction, needs careful testing
  - File: `lib/lib.py` (1,948 lines)
  - Risk: High - core functionality
- [ ] **Standardize block error returns** - All blocks should return consistent error signals
  - Currently 14 blocks return empty `{}` on execution
  - Target: `{0: value, 'E': False}` or `{'E': True, 'error': msg}`

### PDE Phase 1: Quick Wins (from PDE_ROADMAP.md)
- [ ] **Periodic BCs** - Add to HeatEquation1D/2D, WaveEquation1D/2D
  - Pattern exists in `AdvectionEquation1D`
- [ ] **Dynamic BC coefficients** - Time-varying Robin BC via input ports
  - Files: `heat_equation_1d.py`, `heat_equation_2d.py`
- [ ] **More initial conditions** - Add 'linear', 'step', 'random', 'checkerboard'
  - Location: `get_initial_state()` methods
- [ ] **Robin BC for 2D** - Extend from 1D to HeatEquation2D
  - Per-edge h coefficients

---

## Low Priority

### Refactoring (optional)
- [ ] **Split modern_canvas.py** - Further modularization (currently 2,267 lines)
  - Optional - already has extracted managers

### Documentation
- [ ] **API documentation** - Generate docs from docstrings
- [ ] **Video tutorials** - Demo videos for key features

---

## Future / Roadmap

### PDE Phase 2: Mesh Abstraction
- [ ] Create `MeshBase` abstract class
- [ ] Create `blocks/pde/mesh/` directory structure
- [ ] Refactor 2D PDE blocks to use mesh interface
- [ ] Add curvilinear mesh support

### PDE Phase 3: Unstructured Meshes
- [ ] Mesh loader block (read .msh, .vtk, .stl)
- [ ] Mesh generator block (circle, L-shape)
- [ ] Mesh exporter block (VTK for ParaView)
- [ ] FEM-based spatial operators (P1 triangles)
- [ ] Update FieldScope2D for triangulation rendering
- [ ] Add FieldExportVTK block

### PDE Phase 4: Advanced Features
- [ ] 3D PDE support (HeatEquation3D, WaveEquation3D)
- [ ] Absorbing BCs / PML for wave equations
- [ ] Adaptive mesh refinement
- [ ] Domain decomposition for parallel solving

---

## Completed

### February 2026
- [x] **Animation export for FieldScope** - GIF/MP4 export with dialog
  - Files: `lib/plotting/animation_exporter.py`, `modern_ui/widgets/animation_export_dialog.py`
- [x] **Unit tests for 14 untested blocks** - 160 new tests added
  - TransportDelay, DiscreteTranFn, External, Assert, FFT, Subsystem, Inport, Outport, Abs, Terminator
- [x] **Test coverage improvement** - 44% → 57% of blocks tested
- [x] **Total tests** - 346 → 573 (421 unit + 152 integration)

### January 2026
- [x] **Optimization Primitives** - 11 blocks for visual algorithm building
- [x] **PDE 2D blocks** - HeatEquation2D, WaveEquation2D, AdvectionEquation2D
- [x] **FieldScope2D** - Interactive time slider visualization

### Previous
- [x] All Priority 1-7 refactoring tasks (see REFACTORING_TODO.md)
- [x] StateSpaceBaseBlock consolidation
- [x] Circular import fixes
- [x] Canvas and MainWindow modularization

---

## Change Log

| Date | Change |
|------|--------|
| 2026-02-02 | Created consolidated TODO from REFACTORING_TODO.md, PDE_ROADMAP.md, CLAUDE.md |
| 2026-02-02 | Added animation export feature to completed |
| 2026-02-02 | Added 160 new unit tests to completed |

---

## References

- `docs/REFACTORING_TODO.md` - Detailed refactoring history
- `docs/PDE_ROADMAP.md` - Full PDE enhancement roadmap with architecture diagrams
- `CLAUDE.md` - Project overview and recent work
