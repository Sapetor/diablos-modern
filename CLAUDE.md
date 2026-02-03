# DiaBloS Modern - Project Notes

## Overview
DiaBloS is a Simulink-like block diagram simulation tool built with Python and PyQt5. It allows users to create, connect, and simulate dynamic systems using a visual block diagram interface.

## Architecture
- **modern_ui/**: PyQt5-based GUI (main_window.py, modern_canvas.py, widgets/)
- **lib/**: Core simulation engine (lib.py, engine/, plotting/, services/)
- **blocks/**: Block implementations organized by category (sources/, sinks/, math/, pde/, etc.)
- **examples/**: Example diagram files (.diablos)
- **tests/**: Unit and integration tests

## Recent Work (February 2026)

### First-Simulation Performance Fix
Fixed 3-second delay on first simulation caused by `scipy.signal.cont2discrete()` lazy initialization.

- **Root cause**: Transfer Function blocks call `cont2discrete()` which has ~3s lazy init on first use
- **Solution**: Background preload thread in `diablos_modern.py` calls `cont2discrete()` with dummy system at startup
- **Result**: First simulation now runs instantly (preload happens in background during app startup)

### Animation Export for FieldScope Visualizations
Added ability to export FieldScope (1D) and FieldScope2D (2D) time-series visualizations as animated GIF or MP4 files.

- **New files**: `lib/plotting/animation_exporter.py`, `modern_ui/widgets/animation_export_dialog.py`
- **Export button** on slider figure windows (next to time slider)
- **Format options**: GIF (via Pillow) or MP4 (via ffmpeg)
- **Configurable**: FPS, quality (72/100/150 dpi), output path
- **Threaded export** keeps UI responsive with progress bar

### Expanded Unit Test Coverage
Added 160 new unit tests for previously untested blocks:

| Test File | Tests | Block(s) |
|-----------|-------|----------|
| `test_transport_delay.py` | 14 | TransportDelay |
| `test_discrete_transfer_fn.py` | 16 | DiscreteTranFn |
| `test_external_block.py` | 10 | External (stub) |
| `test_assert_block.py` | 32 | Assert |
| `test_fft_block.py` | 19 | FFT |
| `test_subsystem_blocks.py` | 53 | Subsystem, Inport, Outport |
| `test_abs_block.py` | 8 | Abs |
| `test_terminator.py` | 8 | Terminator |

**Test coverage improved**: 44% → 57% of blocks now have dedicated unit tests.

### Optimization Primitives Implementation
New category of 11 blocks for building optimization algorithms visually using feedback loops:

- **Core Blocks**: ObjectiveFunction, VectorPerturb, NumericalGradient, StateVariable, VectorGain, VectorSum
- **Linear Algebra**: LinearSystemSolver (Ax=b), RootFinder (Newton step), ResidualNorm
- **Adaptive Methods**: Momentum, Adam

Key concept: Each simulation step = one optimization iteration. Build gradient descent as `X_{k+1} = X_k - α * ∇f(X_k)` using connected blocks.

**Files**: `blocks/optimization_primitives/` (11 blocks), 78 unit tests, 5 example diagrams
**Wiki**: `docs/wiki/Optimization_Primitives.md` with full block documentation

### Verification Examples (with Analytical Solutions)
- `gradient_descent_verification.diablos` - Compares numerical GD to x(k) = 0.8^k · [5,5]
- `newton_method_verification.diablos` - Newton converging to [1,0] with quadratic rate
- `linear_system_verification.diablos` - Solves Ax=b with ||x-x*|| and ||Ax-b|| metrics

### Teaching Examples
- `learning_rate_comparison.diablos` - Effect of α on convergence (0.1, 0.4, 0.6, 1.2)
- `optimizer_comparison.diablos` - GD vs Momentum vs Adam on Rosenbrock
- `convergence_rates.diablos` - Linear vs quadratic convergence visualization

### UI Improvements
- **Display block**: Dynamic width rendering for wider strings
- **Terminal output**: Simulation results summary printed after completion
- **Reduced logging**: Less verbose terminal output during simulation

### Test Suite Expansion
- **675 tests passing** across unit, integration, and regression tests
- Added tests for Integrator block (all 5 methods), Sink blocks (Scope, XYGraph, Display)
- Added tests for Noise and WaveGenerator source blocks
- End-to-end simulation tests: Constant→Scope, Step→Integrator→Scope, feedback loops
- FieldScope2D verification tests: snapshots, time slider, shape preservation
- Subsystem nesting tests: multi-level chains, split/merge, feedback patterns
- AnimationExporter tests: GIF/MP4 export, 1D/2D fields, writer availability
- **Regression test suite** (`tests/regression/`): Numerical accuracy, bug fix verification
- **Profiling tools** (`tests/profiling/`): Performance analysis for simulation engine

## Recent Work (January 2026)

### PDE Blocks Implementation
- **1D PDE Blocks**: HeatEquation1D, WaveEquation1D, AdvectionEquation1D, DiffusionReaction1D
- **2D PDE Blocks**: HeatEquation2D, WaveEquation2D, AdvectionEquation2D (Method of Lines with 5-point stencil Laplacian)
- **Field Processing**: FieldProbe, FieldScope (1D), FieldProbe2D, FieldScope2D, FieldSlice
- Blocks organized in palette categories: "PDE Equations" and "Field Processing"

### Key Features Added
- 2D heat equation solver with Dirichlet/Neumann boundary conditions
- Sinusoidal, Gaussian, and hot_spot initial conditions for 2D
- Interactive time slider for FieldScope2D visualization
- Verification summary with error metrics for comparison scopes
- Multi-input scope support in replay phase

### Verification Example
`examples/heat_equation_2d_verification.diablos` - Compares numerical solution against analytical solution for decaying sinusoidal mode: T(x,y,t) = A*sin(πx)*sin(πy)*exp(-2απ²t)

### Optimization Examples (January 2026)
- `examples/optimization_basic_demo.diablos` - Basic gain parameter optimization with ISE cost
- `examples/optimization_pid_tuning_demo.diablos` - PID auto-tuning with Kp, Ki, Kd parameters using ITAE cost
- `examples/optimization_constrained_demo.diablos` - Constrained optimization with overshoot limit using SLSQP
- `examples/optimization_data_fit_demo.diablos` - Model calibration against experimental data (with sample CSV)

### Additional PDE Examples
- `examples/pde_comparison_demo.diablos` - Side-by-side comparison of Heat, Wave, and Advection equations
- `examples/pde_neumann_bc_demo.diablos` - Insulated rod with Neumann BCs demonstrating energy conservation

## Key Files
- `lib/engine/simulation_engine.py` - Core simulation engine with replay phase
- `lib/engine/system_compiler.py` - Compiles diagrams to ODE systems for fast solving
- `lib/plotting/scope_plotter.py` - Plotting logic for Scope, FieldScope, FieldScope2D
- `lib/plotting/animation_exporter.py` - GIF/MP4 export for field animations
- `lib/diagram_validator.py` - Pre-simulation validation
- `blocks/pde/` - All PDE block implementations
- `blocks/optimization_primitives/` - Visual optimization algorithm building blocks
- `modern_ui/widgets/animation_export_dialog.py` - Export settings dialog
- `tasks/todo.md` - Consolidated TODO and roadmap

### Optimization Primitives Examples
- `examples/gradient_descent_demo.diablos` - Basic GD on f(x) = x₁² + x₂²
- `examples/momentum_demo.diablos` - Momentum optimizer on Rosenbrock
- `examples/adam_demo.diablos` - Adam optimizer on Rosenbrock
- `examples/newton_method_demo.diablos` - Newton's method for nonlinear system
- `examples/linear_system_demo.diablos` - Solving Ax = b with verification

## Pending Tasks

> **See [`tasks/todo.md`](tasks/todo.md) for the consolidated TODO list.**

### High Priority - Testing
- [x] **Comprehensive block testing**: 421 unit tests passing
  - Source blocks: Step, Ramp, Sine, Constant, Noise, WaveGenerator ✓
  - Math blocks: Gain, Sum, Product, MathFunction, Derivative, Abs ✓
  - Control blocks: Saturation, RateLimiter, PID, Hysteresis, Deadband, Switch, TransportDelay, DiscreteTranFn ✓
  - Sink blocks: Scope, XYGraph, Display, FFT, Assert, Terminator ✓
  - PDE blocks: Heat1D/2D, Wave1D/2D, Advection1D/2D, DiffusionReaction1D ✓
  - Integrator: All 5 methods (FWD_EULER, BWD_EULER, TUSTIN, RK45, SOLVE_IVP) ✓
  - Subsystem blocks: Subsystem, Inport, Outport ✓
  - Animation export: AnimationExporter ✓
- [x] **Integration tests**: Complete diagram simulations end-to-end ✓
- [x] **Subsystem nesting tests**: Signal flow patterns, nested processing ✓

### Medium Priority - Features
- [x] **FieldScope2D verification**: Tests for snapshots, time slider, shapes ✓
- [x] **2D PDE blocks**: WaveEquation2D, AdvectionEquation2D implemented ✓
- [ ] **3D PDE support**: Future consideration
- [x] **Regression tests**: `tests/regression/test_regression_suite.py` with 26 tests ✓

### Low Priority - Polish
- [x] **Documentation**: Added wiki pages (Optimization, PDE, Examples, Optimization_Primitives)
- [x] **Example diagrams**: Added optimization examples (4), PDE examples (2), and optimization primitives examples (5)
- [x] **Optimization Primitives**: 11 blocks for visual algorithm building (ObjectiveFunction, NumericalGradient, StateVariable, VectorGain, VectorSum, VectorPerturb, LinearSystemSolver, RootFinder, ResidualNorm, Momentum, Adam)
- [x] **Performance profiling**: `tests/profiling/profile_simulation.py` ✓

## Known Issues
- ~~Old test file (tests/test_blocks.py) crashes due to Qt initialization~~ → Marked as skipped
- External block (`blocks/external.py`) is a stub - returns error dict (not implemented)

## Dependencies
- `Pillow>=8.0.0` required for GIF animation export (added to requirements.txt)
- `ffmpeg` (external) required for MP4 export: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Ubuntu)

## Running Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/unit/ -v               # Unit tests only
pytest tests/integration/ -v        # Integration tests only
pytest tests/regression/ -v         # Regression tests only
python tests/profiling/profile_simulation.py  # Performance profiling
```

## Running the Application
```bash
python main.py
```


## Workflow Orchestration

  

### 1. Plan Mode Default

- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)

- If something goes sideways, STOP and re-plan immediately - don't keep pushing

- Use plan mode for verification steps, not just building

- Write detailed specs upfront to reduce ambiguity

  

### 2. Subagent Strategy to keep main context window clean

- Offload research, exploration, and parallel analysis to subagents

- For complex problems, throw more compute at it via subagents

- One task per subagent for focused execution

  

### 3. Self-Improvement Loop

- After ANY correction from the user: update 'tasks/lessons.md' with the pattern

- Write rules for yourself that prevent the same mistake

- Ruthlessly iterate on these lessons until mistake rate drops

- Review lessons at session start for relevant project

  

### 4. Verification Before Done

- Never mark a task complete without proving it works

- Diff behavior between main and your changes when relevant

- Ask yourself: "Would a staff engineer approve this?"

- Run tests, check logs, demonstrate correctness

  

### 5. Demand Elegance (Balanced)

- For non-trivial changes: pause and ask "is there a more elegant way?"

- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"

- Skip this for simple, obvious fixes - don't over-engineer

- Challenge your own work before presenting it

  

### 6. Autonomous Bug Fixing

- When given a bug report: just fix it. Don't ask for hand-holding

- Point at logs, errors, failing tests -> then resolve them

- Zero context switching required from the user

- Go fix failing CI tests without being told how

  

## Task Management

1. **Plan First**: Write plan to 'tasks/todo.md' with checkable items

2. **Verify Plan**: Check in before starting implementation

3. **Track Progress**: Mark items complete as you go

4. **Explain Changes**: High-level summary at each step

5. **Document Results**: Add review to 'tasks/todo.md'

6. **Capture Lessons**: Update 'tasks/lessons.md' after corrections

  

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.

- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.

- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.