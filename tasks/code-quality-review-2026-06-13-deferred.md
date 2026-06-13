# Code Quality Review (2026-06-13) — Deferred Findings: Final Status

Companion to `code-quality-review-2026-06-13.md`. This tracks every finding that
was not fixed in the initial automated wave, and its final disposition after the
follow-up fix passes.

**Outcome:** of the 334 confirmed findings, all concrete defects, correctness
issues, and performance problems with a safe fix have been **fixed**. What
remains is a small set of (a) genuine architectural refactors with no behavior
change and real regression risk, and (b) items that are deliberate/correct as-is.

All fixes verified green throughout: **1429 passed, 14 skipped** (unchanged from
the pre-fix baseline).

---

## Fixed after initial deferral (the follow-up passes)

| Finding | File(s) | Fix |
|---|---|---|
| Impulse / Step(`impulse`) skipped by adaptive solver (HIGH) | `system_compiler.py` | Excluded from the compiled path → interpreter fires a correct `value/dt` sample. |
| Per-node Python loops in PDE RHS (HIGH/med) | `system_compiler.py` | Vectorized every interior stencil (heat/wave/advection/diffusion 1D + 2D); numerically identical. |
| Compiled heat **Robin BC** = penalty-Dirichlet (wrong physics) | `system_compiler.py` | Drives the boundary node to the same Robin-consistent value the block computes. |
| Scope / Export O(n²) per-step concatenation (HIGH/med) | `scope.py`, `export.py` | Geometric-growth buffers, amortized O(1) append, `vector` exposed as a view. |
| Discrete PID transfer function ignored `sampling_time` | `analyzers/base_analyzer.py` | `cont2discrete` when `sampling_time > 0`. |
| Integrator ignored ODE method / rebuilt closure each step | `integrator.py` | Configurable `ivp_method`; module-level RHS (default RK45 → identical). |
| Routing assumed ports face +x | `connection.py` | Stub direction derived from port orientation (`block.flipped`). |
| `draw_grid` O(W·H) per frame | `canvas_renderer.py` | Skip small-dot grid below ~6px device spacing. |
| Per-frame QColor / category scan | `block_renderer.py` | Memoized theme-independent category→key lookup. |
| Minimap bounds recomputed per paint | `minimap_widget.py` | Cached, invalidated by a geometry fingerprint. |
| `compute_derivatives` 1D vs 2D signature mismatch | `blocks/pde/*_1d.py` | Unified to `(self, time, state, inputs, params)`. |
| Frozen-build scaling read path; all-Neumann PDE corners; Step `impulse` docs | `diablos_modern.py`, `system_compiler.py`, `step.py` | See git history (wave 2 + this pass). |
| Dead code, `_bounded_eye` branches, corner-fill duplication | `signal_plot.py`, `safe_eval.py`, `system_compiler.py` | Removed / consolidated. |
| MainWindow god-constructor (altitude) | `main_window.py` | Extracted `_init_core_managers()`. |

## Remaining — architectural refactors (tracked backlog, not defects)

Large structural changes with no behavior delta and real regression risk; best
done as dedicated, separately-tested work rather than blind churn in a quality
sweep. Listed in `todo.md`.

- **`lib.py` interpreter hot path**: O(blocks²) per-timestep re-iteration;
  duplicated multi-rate/hierarchy loops; `DSim` facade breadth; engine-state
  attributes re-copied onto `self`. (The *correctness* part of the duplication —
  shared param resolution incl. External/subsystem recursion — was already fixed.)
- **Full PDE kernel single-sourcing** between the blocks and `SystemCompiler`.
  (The concrete correctness divergence, the Robin BC, is now fixed; merging the
  two implementations into one shared kernel is the larger remaining refactor.)
- **`modern_canvas.py` god object** and its implicit connection-start state.
- **Manager-layer count** (~18 managers back-coupled to the parent). The
  MainWindow constructor altitude was improved; consolidating managers is a
  separate design decision.
- **`lib/` ↔ `modern_ui/` import layering** (theme_manager imported from `lib`).
  No actual runtime cycle exists; the proper fix is dependency inversion / moving
  shared theming into `lib`, not the function-local-import band-aid.

## Remaining — deliberate / correct as-is (not defects)

- **Broad `except Exception` at Qt event-handler boundaries** — intentional: an
  unhandled exception there crashes the Qt event loop. High-value sub-cases were
  already narrowed.
- **wave2d energy/avg/max diagnostics computed in the RHS** — inherent to the
  compiler's "signals are produced inside the RHS" design; deferring them needs a
  post-accepted-step output pass.
- **Export/Scope and blox/tikz/latex "duplication"** — these are distinct sinks /
  distinct output formats with different escape tables; not a single-source merge.
- **`Product` divide-by-zero → finite value** — deliberate and pinned by
  `test_product_block.py` (requires a finite output); already np-aware.
