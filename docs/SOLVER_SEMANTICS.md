# Solver Semantics

What DiaBloS actually computes when you press Play: how a block's time behaviour
is decided, in what order blocks run, which of the two engines executes them,
and what each solver setting changes. Every statement below is annotated with
the file that implements it.

This is the semantics reference. [`FAST_SOLVER.md`](FAST_SOLVER.md) is the
compiled path's own document (block coverage, the zero-crossing machinery,
performance); [`ARCHITECTURE.md`](ARCHITECTURE.md) has the component map.

---

## 1. The two engines

Every run goes through exactly one of two engines. The choice is automatic:

```
DSim.execution_batch()                                      lib/lib.py
  ├─ use_fast_solver (View menu, default True)  and
  │  engine.check_compilability(blocks_list) == True
  │     → compiled path      lib/engine/compiled_runner.py
  └─ otherwise
        → interpreter path   lib/lib.py::_interpreter_step
```

|                       | Compiled path | Interpreter path |
| --- | --- | --- |
| Model form | one ODE `dy/dt = f(t, y)` for the whole diagram | per-block `execute()` calls, one pass per step |
| Integration | `scipy.integrate.solve_ivp`, adaptive, error-controlled | fixed step of `sim_dt` (or the block's own `Ts`) |
| Block coverage | the `@kernel` registry only (`lib/engine/compiler_kernels/`) | every block |
| Discrete rates | none — a `sampling_time > 0` anywhere sends the diagram here | honoured per block |
| Zero-crossing events | yes (`lib/engine/zero_crossing.py`) | no |
| Determinism | byte-deterministic for a given diagram + settings | depends on `sim_dt`; convergent as `sim_dt → 0` |
| Stochastic blocks | never (they are excluded from the allowlist) | yes, and only here are `seed` params honoured |

The compiled path is the numerically authoritative one — it is what
`tests/regression/test_analytic_solutions.py` checks against closed-form
answers. The interpreter exists so that *any* diagram runs. They are **not**
expected to agree sample-for-sample; `tests/regression/test_equiv_*.py` pins
where they do and characterises where they do not.

Which one ran is reported as `DSim.last_solver_type` (`"Fast (Compiled)"` /
`"Standard (Interpreter)"`), shown in the status bar at the end of a run.

---

## 2. Continuous, discrete, hybrid

DiaBloS has no separate "continuous" and "discrete" block libraries. A block's
time behaviour is the combination of three independent properties.

### 2.1 Does it hold state across steps?

A block is a **memory block** when its output at time *t* depends only on *past*
inputs, never on the input arriving this step. The classification is
`lib/engine/memory_blocks.py::is_memory_block`:

* unconditional, by function name — `OUTPUT_ONLY_SAFE_BLOCK_FNS`: `Integrator`,
  `StateVariable`, `TransportDelay`, `Delay`, `Adam`, `Momentum`,
  `FirstOrderHold`, `ZeroOrderHold`, `RateLimiter`, `RateTransition`, `Deriv`,
  `Hysteresis`, `PID`, `PRBS`;
* conditional, on resolved parameters — a `TranFn`/`DiscreteTranFn` is a memory
  block iff `deg(den) > deg(num)` (`is_strictly_proper_tf`), a
  `StateSpace`/`DiscreteStateSpace` iff `D == 0` (`is_zero_D_statespace`);
* plus, in the engine, any block whose class sets `requires_inputs = False`
  (sources), because those are always safe to run with no inputs
  (`SimulationEngine.identify_memory_blocks`).

The distinction matters twice: memory blocks are the ones the engine may run
with `output_only=True` before any source has driven them, and edges *into* a
memory block are removed when looking for algebraic loops, because the loop is
broken in time.

Feedthrough is tracked separately as `DBlock.b_type`. `set_block_type()`
(`lib/engine/simulation_engine.py`) sets it for the parameterised cases:
`b_type = 1` for a strictly proper `TranFn`/`DiscreteTranFn` or a `D == 0`
`DiscreteStateSpace`, `b_type = 2` when there is direct feedthrough. The
interpreter uses `b_type == 3` to mean "sink, do not propagate", and
`b_type == 0` (together with a zero *required* input count) to mean "source".

> `b_type` is a legacy per-block field with several producers and is **not** a
> reliable taxonomy on its own — `lib/diagram_builder.py`, for instance, gives
> `Gain` a `b_type` of 2. Only the three uses above are load-bearing.

### 2.2 Does it run every step, or on a rate of its own?

`DBlock.resolve_sample_time()` (`lib/simulation/block.py`) reads
`params['sampling_time']`, falling back to `params['sample_time']`:

| Declared value | Meaning |
| --- | --- |
| `-1` (default, or the key absent) | **continuous** — executes every solver step |
| `0` | **inherited** — takes the fastest discrete rate among its inputs |
| `> 0` | **discrete** — executes every `Ts` seconds and holds its output between |

Blocks that ship a `sampling_time` parameter: `derivative`,
`discrete_statespace`, `discrete_transfer_function`, `first_order_hold`,
`integrator`, `network_channel`, `packet_loss`, `pid`, `random_source`,
`rate_transition`, `statespace`, `statespace_base`, `transfer_function`,
`zero_order_hold` (see `blocks/`). A block declaring
`requires_sample_time = True` (`DiscreteTransferFunction`,
`DiscreteStateSpace`) is a pure recursion in the sample index and is
meaningless without a rate.

### 2.3 Does it switch?

A block whose output has a kink or a jump — `Saturation`, `Deadband`, `Switch`,
`Hysteresis`, `Abs`, `MathFunction(sign|abs)`, `Step`, `Ramp`,
`WaveGenerator`, `PRBS` — is *hybrid* in the compiled path: it contributes
zero-crossing event functions so the switching instant is located exactly
(section 7). In the interpreter it is just a function evaluated on the grid.

**A diagram is hybrid**, then, when it mixes any of these. A hybrid diagram with
a discrete rate always runs on the interpreter; a hybrid diagram that is purely
continuous plus switching runs compiled, with events.

---

## 3. Sample times and multirate

Rates are resolved once per run, in `SimulationEngine.propagate_sample_times()`
(`lib/engine/simulation_engine.py`), after memory-block identification and
before any block executes:

1. **Declare** — every block's `effective_sample_time` is set from
   `resolve_sample_time()`, and its schedule state is cleared
   (`reset_sample_time_state`).
2. **Inherit** — blocks declaring `0` repeatedly take the *smallest positive*
   `effective_sample_time` among their input sources, until nothing changes
   (bounded by the block count). A block with no discrete input resolves to
   `-1`, i.e. continuous.
3. **Mark wires** — `line.discrete_signal` is set for every wire leaving a block
   with a positive rate. This is presentation/analysis metadata; it does not
   gate execution.
4. **Warn** — a block with `requires_sample_time = True` that resolved to
   continuous gets a `logger.warning`: it will advance one sample per solver
   step, so its physical response would silently change with `sim_dt`.
5. **Stamp the step** — each block's `exec_params['dtime']` is set to
   `DBlock.execution_step(sim_dt)`, which is `Ts` for a discrete-rate block and
   `sim_dt` otherwise. This is the step `Integrator` / `TranFn` / `StateSpace` /
   `PID` integrate or discretise with, so a block gated to `Ts` advances `Ts`
   per execution rather than `sim_dt`.

During the run (`lib/lib.py::_interpreter_step`):

* `DBlock.should_execute(t)` is `True` for a continuous block always, and for a
  discrete block once `t >= _next_execution_time - 1e-9`.
* On a step where it does not fire, the engine propagates
  `DBlock.get_held_output(port)` to its consumers instead of executing it —
  this is the zero-order hold between sample instants.
* On a step where it does fire, `SimulationEngine.stamp_held_outputs()` records
  the outputs to hold and `DBlock.schedule_next_execution(t)` computes the next
  instant on the absolute grid (`(floor(t/Ts + 0.5) + 1) * Ts`), so the schedule
  cannot drift.
* A block declaring `output_is_post_update = True` returns the *advanced* state
  from `execute()`; for those the value held is the pre-update output captured
  by the `output_only` pass, or the staircase would lead the true sampled
  response by a whole period.

**Multirate is interpreter-only.** `SystemCompiler.check_compilability` rejects
any block with a declared `sampling_time > 0` outright
(`lib/engine/system_compiler.py`): a compiled ODE right-hand side has no notion
of a sample instant and would run the block as if it were continuous.

---

## 4. Execution order

### 4.1 Initialization (both paths)

`SimulationEngine.initialize_execution()` flattens subsystems
(`lib/engine/flattener.py`), checks connectivity, resolves parameters, then runs
three loops:

* **Loop 1 — sources and memory outputs** (`_init_execute_sources`). A block is
  a *source* when its number of **required** input ports (total minus
  `optional_inputs`) is zero and `b_type == 0`; it executes and is pinned to
  `hierarchy = 0`. Memory blocks are executed with `output_only=True` so
  `y[0] = h(x[0])` is available to their consumers, and pinned to hierarchy 0
  as well. Outputs are propagated.
* **Loop 2 — hierarchy fixpoint** (`_init_resolve_hierarchy`). Repeatedly
  execute every block whose required inputs have all arrived, assigning
  ascending `hierarchy` numbers, until every block is computed. If a pass
  computes nothing new, the remaining blocks are examined (section 4.4).
* **Loop 3 — advance memory state** (`_init_advance_memory_state`). Memory
  blocks are executed again, this time for real, using the inputs Loop 2
  produced. Without it the first simulation step would re-read `x[0]` and emit a
  duplicate sample.

`hierarchy` is what the interpreter steps through; the compiled path uses it
only as an input ordering that it then re-sorts (4.3).

### 4.2 Interpreter step order

Each `_interpreter_step` call:

1. Advance the clock and record a timeline sample (see 5.1 for the RK4 case).
2. **Memory pass** — every memory block runs `output_only=True` and propagates,
   so consumers see last step's state. A discrete block that is not due
   propagates its held output instead.
3. **Hierarchy passes** — for `hier` in `0 .. max_hier`, execute every
   uncomputed block at that level whose required inputs have arrived, repeating
   within the level until no progress is made, and repeating the whole sweep
   until no progress is made anywhere. (The inner repeat handles two blocks at
   the same level ordered producer-after-consumer in the list; the outer repeat
   handles memory blocks pinned to hierarchy 0 whose state-update inputs are
   produced at a higher level.)
4. A memory block **with** feedthrough (`b_type == 2`: `ZeroOrderHold`,
   `RateLimiter`, `PID`) re-propagates the value it just computed with
   `count=False`, overwriting the stale `output_only` value in its consumers'
   queues without re-counting the delivery. Strictly proper memory blocks
   (`b_type == 1`) are left alone — their earlier value was the correct one.

The consequence is the interpreter's characteristic **one-sample feedback delay
through strictly-proper memory blocks**, which the compiled path does not have.

### 4.3 Compiled execution order

`SystemCompiler.compile_system()` re-sorts the blocks into three groups
(`lib/engine/system_compiler.py`, and `tasks/lessons.md` for the two bugs that
produced this design):

1. **Sources** — `Step`, `Sine`, `Constant`, `From`, `Ramp`, `Noise`,
   `WaveGenerator`, `PRBS`, `Impulse`.
2. **Middle** — algebraic blocks **and** state blocks with `D != 0`, in
   topological order. A `D != 0` block's output is `C x + D u`, which depends on
   this instant's input, so it cannot be pre-populated; it must run where its
   input is available. `PID` is always in this group.
3. **`D = 0` state blocks** — strictly proper transfer functions, `StateSpace`
   with `D = 0`, `Integrator`, `RateLimiter`, PDE blocks. Their outputs are
   *pre-populated* as `C x` (exact, since `D u = 0`) before the sequence runs,
   which is what closes feedback loops; they execute last so their derivative
   `dx = A x + B u` sees the already-computed algebraic inputs.

The classification is `_is_d0_state_block`, which reads the `D` matrix the
compiler actually built rather than re-deriving it from parameters.

The **replay** that reconstructs Scope traces after the solve uses a different
order again: a topological sort over *feedthrough edges only*, because a state
block's replayed output is its known state and does not depend on this step's
inputs (`compiled_runner.replay_compiled_signals` / `replay_has_feedthrough`).

### 4.4 Algebraic loops

A cycle of blocks with no state in it has no execution order. Loop 2 detects the
stall and calls `graph_analysis.detect_algebraic_loops` (Kahn's algorithm with
edges into memory blocks removed). Two outcomes:

* a genuine cycle → `"Algebraic loop detected involving blocks: [...]"`, and
  initialization fails;
* no cycle but blocks still uncomputed → only memory blocks may legitimately
  remain (they run in Loop 3); any other uncomputed block is a hard
  `"Hierarchy resolution stalled with uncomputed non-memory blocks"` error
  rather than a block that silently never runs.

The fix is the usual one: put an `Integrator`, or a strictly proper `TranFn`, in
the loop.

---

## 5. The interpreter path in detail

### 5.1 Stepping

`sim_dt` is the step, full stop: every continuous block advances `sim_dt` per
call, every discrete block advances its own `Ts` (section 3). The timeline is
accumulated as `_timeline_list` and the run ends once the last sample that fits
inside `execution_time` has been produced, which is deliberately the same grid
the compiled path writes (`arange(0, T + dt, dt)` clipped to `<= T`).

Per-block integration strategy lives in the `Integrator`'s `method` parameter
(`blocks/integrator.py`): `FWD_EULER`, `BWD_EULER`, `TUSTIN`, `RK4`,
`SOLVE_IVP` (the default; a one-step `solve_ivp` over `[t, t+dtime]` with a held
input). `"RK45"` in a saved file is a legacy spelling of the fixed-step `RK4`
and is mapped by `resolve_method()`.

If **any** integrator resolves to `RK4`, `count_rk45_integrators()` sets
`rk45_len` and the whole loop runs four sub-steps per `sim_dt`, advancing the
clock `+0, +h/2, +0, +h/2` and recording one timeline sample per cycle. Sinks
are told to skip the intermediate stages via `exec_params['_skip_']`.

PDE blocks self-integrate their field with Forward Euler inside `execute()`,
persisting it in `params`, and reuse the same spatial operators as the compiled
kernels (`lib/engine/pde_ops.py`). FTCS stability limits apply.

### 5.2 Block state, `_init_start_`, and resets

All state that survives a time step must live in `params` / `exec_params`, never
on `self` (see the Block Contract in `CLAUDE.md`). The lifecycle:

* `_init_start_ = True` tells `execute()` to (re)build its state from
  parameters — allocate `mem`, discretise `A, B, C, D` at `dtime`, seed
  `_t_old_`, and so on — and to clear the flag afterwards.
* `SimulationEngine._resolve_block_params()` builds `exec_params` from `params`
  through the `WorkspaceManager`, copies every `_`-prefixed key across, and
  stamps `dtime`. It is skipped when `exec_params` is already current for this
  `dtime` and the parameter fingerprint is unchanged
  (`lib/engine/compile_cache.py::source_params_fingerprint`); on that cache hit
  the narrow set of engine-owned keys is still pushed down
  (`lib/engine/block_params.py::PUSH_DOWN_KEYS`).
* `SimulationEngine.reset_memblocks()` ends a run: it sets `_init_start_ = True`
  in both `params` and `exec_params` and deletes the stale accumulators
  `_prev`, `mem`, `output`. It never touches instance attributes — which is
  exactly why state on `self` leaks silently between runs.

---

## 6. The compiled path in detail

`compiled_runner.run_compiled_simulation()` does four things:

1. **Resolve and check.** Parameters are resolved for the flattened block list,
   then `SystemCompiler.check_compilability()` runs again on it.
2. **Compile.** `SimulationEngine._compile_system_cached()` builds (or reuses)
   `model_func`, `y0`, `state_map`, `block_matrices`. The cache is a single slot
   keyed on a fingerprint of the topology, the parameters and `dt`, because the
   hot path is re-running the same diagram after a scalar tweak.
3. **Solve** on `t_eval = arange(t0, t1 + dt, dt)` clipped to `<= t1`.
4. **Replay** the state trajectory through every block so Scope / FieldScope get
   their history — the solve itself only returns `y`.

### 6.1 Backends

The `backend` field of the diagnostics says which one ran:

| `backend` | When | Notes |
| --- | --- | --- |
| `algebraic` | `len(y0) == 0` | no ODE at all; the grid is generated and only the replay runs |
| `fixed_step` | method is `RK4` or `Euler` | in-house, steps on the output grid (`integrate_fixed_step`) |
| `scipy` | adaptive method, no events | one `solve_ivp` call, no `events=` argument |
| `scipy+events` | adaptive method and at least one event, detection on | segmented solve (`zero_crossing.solve_with_events`) |

### 6.2 What compiles

The allowlist is `SystemCompiler.COMPILABLE_BLOCKS`, plus
`ZERO_CROSSING_ONLY_BLOCKS` (`Hysteresis`) when detection is on. The per-family
implementations are the `@kernel`-decorated builders in
`lib/engine/compiler_kernels/`. [`FAST_SOLVER.md`](FAST_SOLVER.md) lists the
supported blocks by family.

Deliberate exclusions, each with a reason in the code:

| Block | Why it is not compiled |
| --- | --- |
| `Noise`, `PacketLoss`, `NetworkChannel`, `RandomSource` | `np.random` inside an ODE right-hand side is re-sampled on every solver stage and every rejected step, which destroys both the error estimate and reproducibility. Only the interpreter honours their `seed` params. |
| `Impulse`, and `Step` with `type == "impulse"` | modelled as a `dt * 1e-3`-wide rectangle, which an adaptive step can jump clean over — the response would silently vanish. |
| `StateVariable` | performs a discrete update inside the RHS, keyed on a monotonic-time assumption `solve_ivp` violates (repeated, rejected and out-of-order probe times). |
| `Hysteresis` **with detection off** | its latch is path-dependent, so it can only be part of an RHS if the mode is frozen inside each segment and flipped at a located instant. |
| anything with `sampling_time > 0` | section 3. |
| anything with no `@kernel` builder | including user blocks. |

### 6.3 Why a run fell back, and where to see it

`check_compilability` records the first blocking block on
`SystemCompiler.last_incompatibility`, formatted
`"<block name> (<block_fn>): <reason>"` and reachable as
`SimulationEngine.get_compile_fallback_reason()`. Three places surface it:

* **Property panel** — with nothing selected, the inspector's *Solver* section
  shows an `interpreter_because` row carrying that string
  (`modern_ui/widgets/property_editor.py::_last_run_rows`). The row is absent
  when the diagram compiles.
* **Status bar** — `"Simulation finished [Standard (Interpreter)]"` tells you
  *that* it fell back; for a compiled run the one-line diagnostics summary is
  appended instead (`DSim.last_solver_diagnostics_summary`).
* **Log** — a `DEBUG` line per rejected block, and `INFO`
  `"System not fully compilable. Using Interpreter Mode."`.

Failures *inside* the compiled run (as opposed to a refusal to compile) are
recorded in the diagnostics `failure_stage` field: `"initialize"`,
`"compilability"`, `"solve"` or `"exception"`.

### 6.4 Run diagnostics

Every compiled run stores a dict on `engine.last_solver_diagnostics`
(`lib/engine/solver_diagnostics.py::build_diagnostics`) with, among others:
`success`, `failure_stage`, `backend`, `method_requested`, `method_used`,
`fallback_reason`, `rtol`, `atol`, `dt`, `n_states`, `n_blocks`, `n_lines`,
`n_time_points`, `nfev` / `njev` / `nlu`, the compile-cache counters, the four
wall-time fields, `output_range`, the `zero_crossing` summary, and
`stiffness` / `stiffness_suspected` (section 8).
`format_diagnostics_for_log()` renders the one-line version.

---

## 7. Zero-crossing events

Full treatment in [`FAST_SOLVER.md`](FAST_SOLVER.md#zero-crossing-detection);
the semantics in brief:

* **Scope.** Compiled path only — the interpreter is fixed-step and has no event
  machinery. On by default; the setting lives in Simulation settings and in the
  file's `sim_data.zero_crossing`.
* **What is detected.** Each discontinuous block declares its switching surfaces
  next to its kernel through the `@events(...)` registry, as `EventSpec`s with a
  `func(t, y, signals) -> float`, a `direction` (+1 rising, −1 falling, 0
  either), an optional `on_event` discrete update (a relay latch flip), an
  optional `on_start` seeding, and a `monotonic` flag. The table of contributing
  blocks is in `FAST_SOLVER.md`.
* **How.** Every event is registered *terminal*; integration proceeds in
  segments, each ending at the earliest located root. Every event at that
  instant gets its update applied, then integration restarts a nudge
  (`1e-11` of the span) past the root, with the state advanced across that gap
  by one explicit Euler step of the post-event dynamics
  (`lib/engine/zero_crossing.py::solve_with_events`). *Both* coordinates have to
  move: nudging only `t` is enough for a guard written on time (a `Step` edge,
  `t - delay`) but leaves a guard written on the state — `Saturation`'s
  `u - max`, with `u` an integrator output — exactly zero at the new segment
  start, where scipy counts it as an active event and re-locates the same root.
  The returned `(t, y)` is exactly the array a single `solve_ivp` call would
  have produced.
* **Step cap.** `solve_ivp` only compares event signs at accepted step ends, so
  a state-dependent guard can cross and return inside one large step. Whenever
  any registered event is non-monotonic the step is capped at the output `dt`.
  Pure time events (`Step`, `Ramp`) are `monotonic` and skip the cap.
* **Chattering guard.** Two bounds: 20 consecutive events closer than `1e-9` of
  the span, or 10 000 events in a run (`DEFAULT_MAX_EVENTS`, overridable via
  `engine.zero_crossing_max_events`). Tripping either logs a warning naming the
  block and finishes the run with a **fixed step** (RK4 on the output grid),
  not with events merely switched off — an adaptive solver turned loose on a
  chattering system never terminates. Latching blocks are unfrozen at the same
  moment. The event summary in the diagnostics carries `guard_tripped` and
  `guard_reason`.
* **Per-block opt-out.** Any contributing block has a `zero_crossing` parameter
  (`"auto"` / `"off"`); `"off"` drops that block's events while the rest of the
  diagram keeps theirs (`SystemCompiler._collect_event_specs`).
* **Headless.** `python diablos_modern.py run diagram.diablos --no-zero-crossing`
  turns detection off for one run; without the flag the file's own setting is
  used (`lib/cli.py`).

Turning detection off also removes `Hysteresis` from the compilable set, so a
diagram containing one falls back to the interpreter.

---

## 8. Choosing a solver

The setting is *Simulation settings → Solver Method*
(`lib/dialogs.py::SimulationDialog`), stored as `sim_data.solver_method`.

| Method | Kind | Use it when |
| --- | --- | --- |
| `RK45` | explicit adaptive, order 5(4) | the default. Smooth, non-stiff dynamics. |
| `RK23` | explicit adaptive, order 3(2) | loose tolerances, cheap right-hand sides. |
| `DOP853` | explicit adaptive, order 8 | very tight tolerances on smooth problems. |
| `Radau` | implicit (Runge-Kutta) | stiff problems; good on the stiff *and* highly oscillatory ones. |
| `BDF` | implicit (multistep) | stiff problems with smooth solutions; cheapest per step of the implicit three. |
| `LSODA` | switches Adams ↔ BDF | you are not sure. It detects stiffness itself and swaps formulas. |
| `auto` | resolves to `LSODA` | you want that decision made for you. |
| `RK4` | fixed step, 4 stages | you need every sample on the `dt` grid with no adaptivity. |
| `Euler` | fixed step, 1 stage | comparison against a hand-written Euler model; teaching. |

`auto` is a DiaBloS-level setting, not a scipy method: it is resolved once, at
the top of each run, by `compiled_runner.resolve_solver_method()`, and both the
requested and resolved names are recorded (`method_requested` / `method_used`).
The default stays `RK45` so no existing diagram's numbers move.

**Tolerances.** `rtol` (default `1e-9`) and `atol` (default `1e-12`) are passed
straight to `solve_ivp` and apply to the adaptive methods only; the fixed-step
schemes ignore them. These defaults are tight — deliberately, since the compiled
path is the reference — but they cost steps: loosening to `1e-6` / `1e-9` is
usually free accuracy-wise on a well-conditioned model and noticeably faster.

**Symptoms and what to change:**

* Trace is fine but the run is slow → try `LSODA`, or loosen `rtol`.
* Trace is jagged at a switch → check that zero-crossing detection is on.
* Trace changes when you change `dt` → you are on the interpreter, or a block
  with `requires_sample_time` has no rate (look for the warning in the log).
* Solver reports failure or the trace blows up → the system is probably stiff;
  see below.

### 8.1 The stiffness indicator

After a compiled run with an **explicit** method (`RK45`, `RK23`, `DOP853`),
`lib/engine/solver_diagnostics.py::estimate_stiffness` decides whether an
implicit solver would have been the better tool. It measures two things, and
both gates must fire:

* **`work_ratio`** — estimated internal solver steps per *output* sample,
  `nfev / (stages * n_output_intervals)` with `stages` of 6 / 3 / 12 for
  RK45 / RK23 / DOP853. High means the solver stepped far finer than the output
  grid you asked for: the *cost* symptom. Threshold `STIFFNESS_WORK_RATIO = 4`.
* **`stiffness_index`** — `max|Re λ| · dt`, from a forward-difference Jacobian
  of the compiled right-hand side sampled at up to 5 points of the returned
  trajectory. High means the fastest *decaying* mode has a time constant far
  below one output step: the regime where an explicit method's step is bounded
  by its stability region rather than by accuracy, which is exactly what an
  implicit method fixes. Threshold `STIFFNESS_INDEX = 20`.

Neither alone is enough. A benign second-order oscillator written out every
0.5 s trips `work_ratio` (≈ 14 in the repo's test case) while its
`stiffness_index` is 0.3 — an implicit solver would buy nothing there. A fast
mode that has already died out trips nothing, because the solver has long since
stretched its step back out. Conversely the *real part* is what counts: a purely
imaginary spectrum (a fast oscillation) is not stiffness, and the index is 0
for it.

The classic max/min eigenvalue **ratio** is also recorded as `eig_ratio`, for
information, but it is not a gate: the textbook scalar stiff problem
`dy/dt = -10⁴(y - u(t))` has exactly one eigenvalue and a ratio of 1.

Costs are kept negligible. The Jacobian is only probed once `work_ratio` has
already cleared its gate, and never for more than
`STIFFNESS_MAX_STATES = 64` states (a 2-D PDE has thousands, and the probe is
`n + 1` right-hand-side evaluations per sample point). Anything that raises
degrades to the work-ratio-only result rather than failing the run. Implicit and
fixed-step methods are not probed at all — there is nothing to suggest to
someone already using one — so `stiffness` is `None` there.

Measured on the repo's own test diagram (`tests/unit/test_solver_semantics.py`),
a Van der Pol oscillator with `mu = 1000` over `[0, 1]` at `dt = 0.05`:

| Method | `nfev` | `work_ratio` | `stiffness_index` | flagged |
| --- | --- | --- | --- | --- |
| RK45 | 6428 | 53.6 | 150 | **yes** |
| Radau | 270 | — | — | not probed |
| LSODA | 131 | — | — | not probed |

— i.e. the suggestion is worth a 25–50× reduction in right-hand-side
evaluations. The same diagram with `mu = 100` is *not* flagged
(`stiffness_index` 15.0, below the gate), and there Radau only halves the work:
the heuristic stays quiet on the marginal cases by design.

When it fires, three things happen:

* `logger.warning` with the full diagnosis (`format_stiffness_for_log`);
* `stiffness_suspected: True` plus the numbers in the diagnostics dict;
* a **non-modal** status-bar line at the end of the run —
  *"This diagram looks stiff — RK45 took 54 solver steps per output sample. Try
  the LSODA solver in Simulation settings."*
  (`modern_ui/controllers/simulation_controller.py::_report_stiffness`, emitted
  on the same `status_changed` channel the run already reports through), and two
  rows in the property panel's *Solver* section (`stiffness`,
  `stiffness_detail`).

Nothing is changed automatically. Switching solver changes the numbers, and that
is the user's call.

---

## 9. What is saved

`FileService.serialize()` writes `sim_data` (`lib/services/file_service.py`):

| Key | Meaning |
| --- | --- |
| `sim_time` | run duration in seconds |
| `sim_dt` | base solver step / output step |
| `sim_trange` | scope plot window in samples (`plot_trange` in memory) |
| `solver_method` | `RK45` … `Euler`, or `auto` |
| `rtol`, `atol` | adaptive-solver tolerances |
| `zero_crossing` | compiled-path event detection on/off |
| `wind_width`, `wind_height`, `fps` | window metadata, not solver settings |

`apply_loaded_data()` reads them back with the same defaults a fresh diagram has
(`RK45` / `1e-9` / `1e-12` / detection on), so a file written before a setting
existed loads as if the user had never touched it. `DSim.deserialize()` copies
them onto the facade, and `run_tuning_simulation` / `execution_init` push them
into the engine via `update_sim_params()`.

Solver-relevant settings that live **per block**, in `params`, and are saved with
the block: `sampling_time` / `sample_time` (section 3), the `Integrator`'s
`method` (section 5.1), `zero_crossing` (`"auto"` / `"off"`, section 7), and
`init_conds`.

**Headless runs honour all of it.** `lib/cli.py::run_diagram` pushes the file's
`solver_method`, `rtol`, `atol` and `zero_crossing` onto the DSim before running;
`--method`, `--rtol`, `--atol` and `--no-zero-crossing` override individually,
and `--solver compiled|interpreter` picks the engine. (Before this, the CLI read
only `sim_time`/`sim_dt` from the file and silently integrated everything with
the built-in defaults.)

---

## 10. Known limitations

* **`RateLimiter` is an approximation when compiled.** The kernel is a stiff
  first-order chase `dy = clip((u − y)·K, falling, rising)` with `K = 1000`, not
  an exact per-step slew clamp (`lib/engine/compiler_kernels/state.py`). It also
  deliberately contributes **no** zero-crossing events: while slew-limited the
  chase pins its error right at the clip bound, so a guard on it grazes zero
  repeatedly and trips the chattering guard within milliseconds. Turn the fast
  solver off for exact rate-limiter parity.
* **Per-block `Integrator` methods are ignored by the compiled path.** One state
  vector cannot be integrated with Euler in one block and RK4 in another; the
  run uses the method from Simulation settings and logs a warning naming every
  block whose setting it discarded.
* **`ceil` / `floor` in `MathFunction` contribute no events** — an unbounded
  event train at every integer.
* **Compiled and interpreted transients differ** for state-heavy diagrams (the
  feedback-delay difference of section 4.2). Steady states agree; do not assert
  tight trajectory equality across the two.
* **The stiffness heuristic is a heuristic.** It cannot see stiffness that has
  not cost anything yet, it says nothing on implicit or fixed-step runs, and it
  is skipped above 64 states — a large PDE is never flagged even when it is
  stiff.
* **`blocks/external.py` is a stub** and returns an error dict on both paths.

---

## 11. File map

| Concern | File |
| --- | --- |
| Engine, init loops, sample-time propagation, diagnostics wrappers | `lib/engine/simulation_engine.py` |
| Compiled run: solve, replay, `auto` resolution, stiffness call | `lib/engine/compiled_runner.py` |
| Diagram → ODE, allowlist, three-group order, event collection | `lib/engine/system_compiler.py` |
| Per-block compiled kernels and their event declarations | `lib/engine/compiler_kernels/` |
| Event solve, restart, chattering guard | `lib/engine/zero_crossing.py` |
| Diagnostics dict, log line, stiffness heuristic | `lib/engine/solver_diagnostics.py` |
| Memory-block taxonomy | `lib/engine/memory_blocks.py` |
| Subsystem flattening | `lib/engine/flattener.py` |
| Interpreter loop | `lib/lib.py` (`_interpreter_step`) |
| Sample-time state, held outputs, per-block step | `lib/simulation/block.py` |
| Solver settings dialog | `lib/dialogs.py` |
| `.diablos` read/write | `lib/services/file_service.py` |
| Headless runs and export | `lib/cli.py` |
| Solver rows, fallback reason, stiffness rows | `modern_ui/widgets/property_editor.py` |
| End-of-run status messages | `modern_ui/controllers/simulation_controller.py` |
