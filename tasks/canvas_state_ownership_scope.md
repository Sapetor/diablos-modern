# §5 Scoping: Canvas State-Ownership Redesign

Scoped 2026-07-19. Data-driven follow-up to the "63 canvas_state property
proxies" item in `todo.md` (manager-layer consolidation, line ~220).

## Measured current state

- `ModernCanvas` owns a `CanvasState` dataclass aggregate
  (`modern_ui/widgets/canvas_state.py`) with 8 domain slices: zoom_pan, grid,
  selection(rect), hover, drag, resize, connection, validation.
- The canvas re-exposes it through **31 getter+setter proxy pairs** (62
  properties, `modern_canvas.py:1385` onward). One proxy
  (`source_block_for_connection`) is unused outside the canvas.
- **The facade is airtight**: zero direct `canvas_state` access exists outside
  `modern_canvas.py`. All external access goes through the flat proxy names,
  so the migration surface is exactly those 31 names.
- External proxy usage (non-test): managers 103, interactions 65, widgets 13,
  main_window ~5, builders 2, lib 1 (`diagram_service.py:254` persists
  `zoom_factor` — legit serialization read). Tests: 61 sites.
- Top consumer files: `interaction_manager.py` (59), `zoom_pan_manager.py`
  (25), `rendering_manager.py` (24), `connection_manager.py` (22),
  `view_actions_manager.py` (13).

## Read/write matrix (r = reads, w = writes, external to canvas)

| file                    | zoom_pan | grid | select | hover | drag  | resize | conn  | valid |
|-------------------------|----------|------|--------|-------|-------|--------|-------|-------|
| interaction_manager     | 4r4w     | 4r   | 3r4w   | –     | 12r5w | 13r11w | 4r1w  | –     |
| zoom_pan_manager        | 17r15w   | –    | –      | –     | –     | –      | –     | –     |
| rendering_manager       | –        | –    | –      | 7r7w  | –     | –      | –     | 2r8w  |
| connection_manager      | –        | –    | –      | –     | 2w    | –      | 13r8w | –     |
| view_actions_manager    | 9r1w     | 3r   | –      | –     | –     | –      | –     | –     |
| command_palette_manager | 8r       | –    | –      | –     | –     | –      | –     | –     |
| history_manager         | –        | –    | –      | 3w    | –     | –      | –     | –     |

Anomalies found:
- `rendering_manager.py:60-68` **mutates** `hovered_port` (save/None/restore
  around a draw pass) — a renderer writing interaction state.
- `rendering_manager.py:120-143` runs the diagram **validator** and writes all
  validation state — computation living in the display layer.
- `history_manager.py:153-158` clears hover state directly (hasattr-guarded).
- `main_window._init_state_management()` (line 229): 6 shadow attrs
  (`dragging_block`, `drag_offset`, `line_creation_state`, `line_start_block`,
  `line_start_port`, `temp_line`) set once to `None`, **never read anywhere**
  — dead legacy state, free deletion.

## Proposed ownership map

| Slice            | Owner              | Rationale |
|------------------|--------------------|-----------|
| ZoomPanState     | ZoomPanManager     | Already the de-facto owner (15 of 21 writes). Others route through its API. |
| GridState        | Canvas (stays)     | External access is already read-only; nothing to invert. |
| SelectionState   | InteractionManager | Transient rect-gesture state; interaction is the only writer. |
| HoverState       | InteractionManager | Rendering/history writes become `clear_hover()` / a draw-param instead of the save-restore hack. |
| DragState        | InteractionManager | Gesture pipeline state. |
| ResizeState      | InteractionManager | Interaction is the only writer. |
| ConnectionState  | ConnectionManager  | Already main writer; expose `cancel_connection()` for interaction/main_window. |
| ValidationState  | *decision needed*  | Option A: RenderingManager keeps it (status quo, owner = computer). Option B: new validation flow owns compute + state; rendering only reads. |

## Migration plan (increments, each gated on full suite + ruff)

0. **Freebie**: delete the dead main_window shadow state + the unused
   `source_block_for_connection` proxy. Zero risk.
1. **Per-domain ownership moves** (≈5 increments; grid stays, gesture slices
   [selection/hover/drag/resize] can go together since InteractionManager owns
   all four): construct the slice inside the owning manager; the canvas proxy
   delegates to the manager (external call sites untouched → behavior-
   preserving); cross-owner writes become semantic API calls.
2. **Repoint readers**: managers/interactions read owner state directly
   instead of via canvas proxies (~193 non-test sites, mechanical).
3. **Delete proxies + migrate the 61 test sites**; keep only `zoom_factor` (or
   route `diagram_service` persistence via `zoom_pan_manager`).
4. `CanvasState` dissolves; `reset_interaction_state()` becomes delegation to
   the owning managers.

## Effort & risk

- Effort: ~2–3 sessions of the size of the 2026-07-18 engine-split increments.
  Steps 0–1 are the valuable part; steps 2–3 are optional polish (the proxies
  become harmless delegations after step 1).
- Risk: moderate. Gesture pipeline has decent headless GUI coverage (61 test
  usages of these proxies; tests/modern_ui exercises mouse interactions), but
  hover/drag edge cases during live mouse use are the least-covered area.
  Mitigation: per-domain increments, delegating proxies keep every external
  call site byte-identical until step 2.

## Open decisions (need user input before implementing)

1. ValidationState home — Option A (rendering keeps it) or B (extract a
   validation flow)? B is cleaner but grows the diff.
2. Stop after step 1 (ownership fixed, proxies remain as delegations) or go
   all the way through proxy deletion (steps 2–3)?
