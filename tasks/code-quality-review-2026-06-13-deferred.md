# Code Quality Review (2026-06-13) — Deferred Findings: Status & Rationale

Companion to `code-quality-review-2026-06-13.md`. Of the 334 confirmed findings,
**~292 were fixed** (287 in the automated wave + scaling read-side, dead-helper
removal, and the all-Neumann PDE corner fix in wave 2). This file accounts for
**every** finding that was *not* directly auto-fixed, so nothing is silently
dropped. Each is categorized with the reason it was deferred and a concrete
follow-up path.

All fixes verified: **1429 passed, 14 skipped** (unchanged from the pre-fix
baseline).

---

## A. Deferred during the automated wave, then fixed manually (wave 2)

| Finding | File | Resolution |
|---|---|---|
| Scaling factor written to user path but read from read-only bundle (HIGH) | `diablos_modern.py` / `main_window.py` | Read side now prefers `user_data_path(...)` with `resource_path` fallback. |
| All-Neumann 2D PDE corners frozen at initial value (HIGH) | `lib/engine/system_compiler.py` | Heat/Wave/Advection corners set to mean of edge neighbors; activates only in the all-Neumann case (zero regression to Dirichlet/Outflow/mixed). |
| Dead `sort_labels` / `sort_vectors` helpers | `lib/plotting/signal_plot.py` | Removed (confirmed unused repo-wide). |

## B. Already resolved or not a real defect (no action needed)

- **Hysteresis / Noise / StateVariable in the compiled ODE RHS** — fixed in the
  wave by removing them from `COMPILABLE_BLOCKS` so they use the interpreter
  path (the closure-mutated/stochastic RHS was the root cause).
- **`linearizer.time_constants` exact `np.isreal`** — fixed (tolerance check).
- **Nyquist plot-window retention** — already handled by the caller
  (`control_system_analyzer.py`); no leak.
- **FFT/Assert/Term/Outport return shape** — already returns `{'E': False}`,
  which matches the finding's own suggestion.
- **`Product` divide-by-zero → finite magic value** — intentional and pinned by
  `tests/unit/blocks/test_product_block.py::test_division_by_zero_handling`
  (requires a finite output). Uses `np.isinf`/`np.isnan`-aware handling already.

## C. Deliberate non-fix — architecture/design (large refactor, no behavior
   change, regression risk outweighs benefit)

These are valid observations but "fixing" them means structural refactors with
no functional payoff and real regression risk. Recorded as design debt, not bugs.

- `main_window.py` god-constructor; `modern_canvas.py` god object — manager
  wiring with ordering constraints; extraction risks reordering side effects.
- `lib/` ↔ `modern_ui/` circular-import risk worked around with function-local
  imports — the real fix is dependency inversion across many files.
- Broad `except Exception` at Qt event-handler boundaries
  (`connection_manager.py`, `modern_canvas.py`) — intentional: an unhandled
  exception there crashes the Qt event loop. High-value sub-cases were narrowed
  in the wave; the rest are deliberate.
- `DSim` facade / engine-state attributes re-copied onto `self` (`lib/lib.py`).
- Manager-layer fragmentation (~18 managers back-coupled to the parent).
- Duplicated logic: PDE stencils block-vs-`SystemCompiler`; `Export`/`Scope`
  accumulation; `blox_exporter`/`tikz_exporter` block content;
  `compute_derivatives` 1D-vs-2D signature. (Divergent guards were back-ported
  where safe; full single-sourcing is a dedicated refactor.)

> **Note — PDE kernel duplication is the highest-value item in this group.** The
> Robin BC differs between the block (convective) and the compiler (penalty
> Dirichlet), so interpreter vs compiled can compute different physics. Worth a
> focused task to single-source the finite-difference/BC kernels.

## D. Deferred performance (real, but perf-only and risky without perf tests)

Correct-but-slow; auto-vectorizing numerical kernels risks silent numeric drift
with no perf regression tests in place. Fix paths noted for a dedicated pass.

- Per-node Python loops in the compiled PDE RHS (1D & 2D heat/wave/advection) →
  vectorize with NumPy slicing.
- `Scope` / `Export` O(n²) per-step `np.concatenate` → accumulate to a list +
  concatenate once at sim end (needs a `BaseBlock` finalize hook).
- `canvas_renderer.draw_grid` O(W·H) per frame; `minimap` recomputes bounds per
  paint; `block_renderer` rebuilds `QColor` per frame (cache on `theme_manager`).
- `lib.py` O(blocks²) per-timestep re-iteration; diagnostic `np.gradient`/energy
  computed every RHS evaluation → move to a post-step pass.
- `integrator` solve_ivp path ignores the selected method / rebuilds closure.

## E. Deferred correctness (real; needs focused work or a feature, not a quick
   safe edit)

- **Impulse / Step-`impulse` spike too narrow for the adaptive solver (HIGH)** —
  pulse width `eps = dtime*1e-3` can fall between RK45 evaluations. Proper fix:
  force solver evaluations at the pulse edges (`t_eval`/`max_step`) or model the
  impulse analytically. (Stochastic/path-dependent blocks were already routed to
  the interpreter; impulse needs solver-grid coordination, deferred.)
- **`base_analyzer` PID transfer function ignores discrete `sampling_time`** —
  needs continuous→discrete (c2d) conversion (a feature, not a bug-patch).
- **`file_service` two divergent save/load paths (`.dat` vs `.diablos`)** —
  needs investigation of which path is canonical before unifying.
- **`connection.py` orthogonal routing assumes ports face right** — cosmetic
  routing; correct fix derives stub direction from actual port orientation.
- **`Step` `'impulse'` subtype duplicates `ImpulseBlock`** — cannot be removed
  (exercised by tests); a consolidation/deprecation decision, not a bug.

---

## Follow-up tests not yet added
- Parametrized compiled-vs-interpreted equivalence over each compiled stateful
  block (RateLimiter, PID, TransportDelay, Selector) — needs example diagrams;
  would validate the RateLimiter `rising_slew`/`falling_slew` key fix.
- Integration test for all-Neumann 2D PDE corners (compiled path) confirming
  corners evolve rather than staying frozen.
