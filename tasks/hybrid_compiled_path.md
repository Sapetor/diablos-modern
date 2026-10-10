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
   blocks under a fixed-step scheme falls back to the interpreter with a
   reason. (Phase 2: integrate on a grid that contains every hit.)
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
- **Phase 2**: rated continuous blocks (TranFn/PID/Integrator/Derivative
  with `sampling_time > 0`), FirstOrderHold, RateTransition; fixed-step
  schemes on a hit-aligned grid.
- **Phase 3**: `Function` expression kernel (S1b acceptance); algebraic
  loops inside the compiled RHS (which also unblocks linearize/export of
  diagrams with loops).
- **Phase 4**: rerun the benchmark, update `docs/BENCHMARK_PATHSIM.md`,
  `docs/SOLVER_SEMANTICS.md`, `docs/FAST_SOLVER.md`.

## Risks

- Per-segment `solve_ivp` overhead (~0.3-1 ms): Ts = 1 ms over 10 s is
  10^4 segments. Measure; if it bites, reuse the solver object across hits.
- Implicit methods (Radau/BDF/LSODA) lose their Jacobian at every restart.
- The block's own sample gating (`_next_sample_time_`) must fire at the
  breakpoint times computed here; float drift in `k*Ts` is the failure mode.
  Use the same rounding as `DBlock` and test long runs.
