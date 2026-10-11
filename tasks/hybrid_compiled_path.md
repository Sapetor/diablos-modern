# Hybrid compiled path -- scoped plan (2026-10-10)

Goal: a diagram with sampled-data blocks (ZOH, discrete TF/SS) stays on the
compiled path -- adaptive `solve_ivp` between sample instants, discrete
updates at the instants -- instead of dropping the whole diagram onto the
first-order fixed-step interpreter. Motivation: PathSim benchmark S1b
(`docs/BENCHMARK_PATHSIM.md`): DiaBloS needs 44 s for 2.4e-5, PathSim 1.2 s
for 2e-12. Acceptance: `scripts/benchmarks/pathsim/s1_sampled_data.py`.

## Design

1. **Sample instants are breakpoints, not events.** The schedule is known in
   advance: the sorted union of `k*Ts` over every discrete block's
   `effective_sample_time` (offset 0, as in `DBlock`), coincident hits merged
   within `1e-9*span`. `solve_with_events` (`lib/engine/zero_crossing.py`)
   gains a `breakpoints` list and an `on_breakpoint(t, y)` hook: each segment
   ends at `min(next event, next breakpoint)`; at a breakpoint the segment
   ends *exactly* at `t_k` and the next one starts *exactly* at `t_k` with
   the same `y` (no nudge, no Euler carry -- those exist for state roots).
   One driver, so zero-crossing events, the chatter guard, `mode_history`
   and the output index survive across sample intervals. The driver is used
   whenever there are breakpoints or event specs.
2. **Grid ownership.** A `t_eval` point equal to `t_k` is written by the
   segment ending at `t_k` (continuous state is continuous); the replay sees
   the *post-update* discrete output there, matching the interpreter, which
   samples at `t_k` and shows the new value at `t_k`.
3. **Ordered update at a hit.** Discrete blocks hitting at `t_k` update in
   hierarchy order, and the signals are re-evaluated after each update, so a
   ZOH -> DiscreteTF chain latches the ZOH's *fresh* sample (the one-period
   bug fixed in the interpreter on 2026-10-10, `68e8289`). All blocks are
   seeded at `t0` (k = 0) before the first segment.
4. **Holders.** Each discrete block gets a per-compile holder
   (`mode_registry` pattern): `{"mode": output, "params": <run-local copy of
   exec_params>, "init": ...}`. The RHS kernel is a pure read of
   `holder["mode"]`. The update calls the block's own `execute()` on the
   holder's params copy -- one source of truth for the discrete math, so the
   compiled samples equal the interpreter's bit for bit. Holders reset at the
   start of every run (`_reset_mode_states`), including on a compile-cache
   hit. Discrete outputs are snapshotted into `mode_history` at each hit, so
   `ModeHistoryReplayer` drives the replay unchanged.
5. **Gate.** `check_compilability` stops rejecting on the *declared* rate.
   In the runner, rates are resolved (`propagate_sample_times`), so the gate
   reads `effective_sample_time`: a rated block compiles only when its fn is
   in `DISCRETE_KERNEL_FNS`; every other rated block (TranFn/PID/Integrator/
   Derivative with `sampling_time > 0`, FirstOrderHold, RateTransition) and
   any `requires_sample_time` block whose rate stays unresolved keep the
   interpreter, each with a reason string.
6. **Linearizer / export keep refusing.** Both call
   `engine.check_compilability`; with frozen holders they would silently
   linearize a sampled loop as open. They get `allow_discrete=False` and the
   existing reason.
7. **Fixed-step solver settings (Euler/RK4).** v1: a diagram with discrete
   blocks under a fixed-step scheme fell back to the interpreter with a
   reason. Phase 2: `zero_crossing.solve_fixed_step_with_breakpoints`
   integrates on the output grid plus every hit (backend
   `fixed_step+hybrid`); the engine gate is gone.
8. **Diagnostics.** Backend label `scipy+hybrid`, `n_sample_hits` in the
   summary. `common.diablos_run` keys "compiled" off the summary, so the
   benchmark switches automatically.

## Phases

- **Phase 1** (me): ZeroOrderHold, DiscreteTranFn, DiscreteStateSpace.
  Acceptance: S1a compiled, error vs the expm reference <= 1e-8.
  Tests (new `tests/regression/test_hybrid_compiled_path.py`):
  - the staircase `[0, 0.2, 0.5, 0.85, 1.225]` from
    `test_discrete_block_sample_time.py` reproduced on the compiled path;
  - compiled vs interpreter equality at sample instants (S1a diagram);
  - multi-rate (0.1 and 0.15 s) hit merging;
  - discrete + Saturation (events and breakpoints together);
  - two consecutive runs on a compile-cache hit give identical scopes;
  - linearizer and export still refuse; unresolved rate and fixed-step
    fall back with their reasons;
  - mutation check: dropping the re-evaluation between updates must fail
    the staircase test.
- **Phase 2** (done 2026-10-10): rated continuous blocks -- TranFn,
  StateSpace, PID, Integrator, Derivative with a resolved rate -- compile as
  sampled holders (`discrete.RATED_FNS` / `is_sampled_block`, no ODE state,
  no events; an `output_is_post_update` Integrator holds its pre-advance
  `output_only` value and its `output` is synced after each advance, as in
  the interpreter); fixed-step schemes on a hit-aligned grid. Tests:
  `tests/regression/test_hybrid_rated_blocks.py` (source-fed traces equal
  the interpreter at 1e-12 for TranFn/StateSpace/Integrator/Derivative, PID
  too except at t = 0, where the interpreter reports 0 (tasks/todo.md); PID
  and Derivative also against an exact sampled oracle; loops
  with a continuous plant at 1e-7; inherited rate with `dtime = Ts`;
  fixed-step Euler/RK4 incl. off-grid instants), with a mutation check per
  mechanism. **Skipped:** FirstOrderHold and RateTransition. Both run every
  solver step with `sampling_time = -1` and sample on their own parameter;
  FOH ramps between samples (no held output) and RateTransition's Average
  mode averages over solver steps, so neither fits the holder pattern.
- **Phase 3** (done 2026-10-10): `Function` expression kernel (`19497e1`,
  S1b acceptance: 2.2e-13 in 1.4 s); algebraic loops inside the compiled RHS
  (`lib/engine/compiled_loops.py`): each cycle of middle-group blocks is one
  unit solved with `hybr` per RHS evaluation, the replay reads members from
  the compiled evaluation, and the linearizer works through loops. Python
  export still refuses loops (`lib/export/python_codegen.py` would need the
  same solver in the generated script).
- **Phase 4**: rerun the benchmark, update `docs/BENCHMARK_PATHSIM.md`,
  `docs/SOLVER_SEMANTICS.md`, `docs/FAST_SOLVER.md`.

## Known gaps after phase 1 (2026-10-10, `0e6afa2`)

- ~~**t0 seeding order.**~~ Closed 2026-10-10: `solve_with_events` seeds the
  latches, takes the t0 sample, then resets the latch holders and seeds again,
  so a ZOH behind a Hysteresis samples the seeded mode and a Hysteresis behind
  a ZOH ends up seeded from the first sample (it had kept the empty holder's
  mode for the whole run). Tests: `tests/regression/test_t0_initialization.py`.
- **Inherited rates inside a Subsystem.** The root-level gate in
  `DSim.execution_batch` sees the unflattened block, whose
  `effective_sample_time` is not resolved, so an inherited-rate DiscreteTF in a
  subsystem falls back as "unresolved". Conservative, undocumented elsewhere.

## Risks

- Per-segment `solve_ivp` overhead (~0.3-1 ms): Ts = 1 ms over 10 s is
  10^4 segments. Measure; if it bites, reuse the solver object across hits.
- Implicit methods (Radau/BDF/LSODA) lose their Jacobian at every restart.
- The block's own sample gating (`_next_sample_time_`) must fire at the
  breakpoint times computed here; float drift in `k*Ts` is the failure mode.
  Use the same rounding as `DBlock` and test long runs.
