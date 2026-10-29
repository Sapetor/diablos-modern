# Python API

`lib.api` runs diagrams from plain Python, with no window. Load a `.diablos`
file or build a diagram in code, change parameters, simulate, and get every
Scope signal back as a numpy array, together with the solver's diagnostics.
The Monte Carlo, parameter sweep, trim and linearization tools from the
**Analysis** menu are available on the same object.

DiaBloS installs no importable package, so run your script from the repository
root (or put the root on `sys.path`).

## Load and run

```python
from lib.api import load

sim = load("examples/first_order_step_response.diablos")
sim.blocks                         # ['step0', 'tranfn1', 'scope2']
sim.params("tranfn1")              # {'numerator': [1.0], 'denominator': [2.0, 1.0], ...}

res = sim.run()                    # the diagram's own duration, step and solver
res.time                           # 1-D array, 0 .. 10 s in 0.01 s steps
res.names                          # ['reference', 'output']
res["output"]                      # 1-D array aligned with res.time
res.engine                         # 'compiled'
res.diagnostics["method_used"]     # 'RK45'
res.save("out.csv")                # or .npz; same columns as the CLI
```

`run(t, dt)` overrides the duration and output step. Signal names are the
Scope's `labels` (default `<scope>-<j>`), the same column names
`diablos_modern.py run` writes.

## Change parameters

```python
sim.set_param("tranfn1", "denominator", [4.0, 1.0])   # every later run
sim.get_param("tranfn1", "denominator")                # [4.0, 1.0]

# One run only; the diagram is restored afterwards.
res = sim.run(params={("tranfn1", "numerator"): [2.0]})

sim.save("tuned.diablos")
```

A block inside a Subsystem is addressed as `"<Subsystem>/<block>"`; `sim.blocks`
lists every name. A value can also be an expression string, resolved at run time
exactly as in the parameter dialog.

Workspace variables are per diagram:

```python
sim.set_param("gain1", "gain", "K")
sim.set_variable("K", 2.0)
sim.variables                    # {"K": 2.0}
```

## Build a diagram in code

```python
from lib.api import from_builder
from lib.diagram_builder import DiagramBuilder

b = DiagramBuilder(sim_time=2.0, sim_dt=0.01)
u = b.add_block("Step", 0, 0, name="u", params={"value": 1.0, "delay": 0.0})
k = b.add_block("Gain", 100, 0, name="k", params={"gain": 3.0})
x = b.add_block("Integrator", 200, 0, name="x")
y = b.add_block("Scope", 300, 0, name="y", params={"labels": "x"})
b.connect(u, 0, k, 0)
b.connect(k, 0, x, 0)
b.connect(x, 0, y, 0)

sim = from_builder(b)
sim.run()["x"][-1]                 # 6.0
```

Blocks keep the names given to `add_block`, and any parameter you leave out
takes the block's default.

## Solver choice

| Argument | Default | Meaning |
|---|---|---|
| `solver` | `"compiled"` | `"compiled"` (scipy `solve_ivp`) or `"interpreter"` (fixed step) |
| `method` | saved setting | `"RK45"`, `"Radau"`, `"LSODA"`, ..., or `"auto"`, `"Euler"`, `"RK4"` |
| `rtol`, `atol` | saved setting | compiled-solver tolerances |
| `zero_crossing` | saved setting | locate switching instants exactly |

A diagram with a block the compiled path does not cover runs on the
interpreter. The result says so: `res.engine == "interpreter"` and
`res.fallback_reason` names the block. `res.diagnostics` is filled only by a
compiled run (method, backend, `nfev`, events located, a stiffness hint and
wall times). [Solver Semantics](../SOLVER_SEMANTICS.md) explains how the two
engines differ.

## Analysis

```python
ens = sim.monte_carlo(50, seed=1)                  # seeded ensemble
ens["signals"]["x"]["mean"], ens["signals"]["x"]["p95"]

sw = sim.sweep([{"block": "k", "param": "gain", "values": [1, 2, 4]}])
sw["signals"]["x"]["metrics"]["final"]             # one value per gain

trim = sim.find_operating_point(input_overrides={"u": 0.0})   # dx/dt = 0 with u held at 0
lin = sim.linearize(inputs=["u"], outputs=["x"],
                    operating_point=trim["operating_point"])
lin["A"], lin["B"], lin["C"], lin["D"]             # [[0.]], [[3.]], [[1.]], [[0.]]
```

Monte Carlo gives every block with a `seed` parameter its own per-run seed, so
the same `seed` reproduces the ensemble; `samplers={("k", "gain"): (1.0, 3.0)}`
also draws a parameter per run. Both runners restore the diagram's parameters
when they finish. For large studies, `workers=4` runs in four processes (results
are identical to `workers=1`) and `keep_traces=False` keeps only the streamed
statistics -- mean, std, min/max and per-run metrics, without the percentile
bands, which need every trace. The result dictionaries are the ones the Analysis windows show;
[Analysis & Experiments](analysis.md) describes them.

## Errors

| Raised | When |
|---|---|
| `FileNotFoundError` | `load` of a missing file |
| `lib.api.SimulationError` | the diagram will not load, or a run fails (the message says why) |
| `KeyError` | an unknown block, parameter or signal name |
| `ValueError` | bad arguments, or `linearize` on a diagram that does not compile or has no continuous states |

## Notes

- The first `load` creates an offscreen Qt application if the process has none.
  No window ever opens, so it works over SSH and in CI.
- Workspace variables belong to the diagram. `sim.set_variable("K", 2.0)` sets
  one, `sim.variables` lists them and `sim.workspace` is the underlying store
  (it can also `load_from_file`). Each `load` or `from_builder` starts with an
  empty workspace, and two diagrams in one process never see each other's
  variables.
- `sim.dsim` exposes the underlying simulator for anything not covered here. It
  is not part of the stable API.
