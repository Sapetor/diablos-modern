# DiaBloS Modern — Whole-App Code Quality Review

**Date:** 2026-06-13  
**Method:** Two-pass multi-agent audit. Every source file in `lib/`, `modern_ui/`, and `blocks/` (~48.6k LOC, ~180 files) was partitioned into review units; each unit was reviewed by a dedicated agent that read the full source, then every finding was re-checked by an independent *adversarial verifier* agent that re-read the cited code and refuted hallucinations/misreads. 25 findings were rejected as false-positives and are excluded below.  
**Confirmed findings:** 334  
**Severity breakdown:** critical: 2, high: 42, medium: 75, low: 215  

> Findings were produced by automated review and adversarial verification. Each cites a concrete file:line and is independently confirmed, but you should still confirm against current `main` before acting — line numbers reflect the reviewed snapshot.

### Findings by category

| Category | Count |
|---|---|
| correctness | 121 |
| error-handling | 80 |
| performance | 28 |
| consistency | 25 |
| dead-code | 22 |
| maintainability | 15 |
| duplication | 12 |
| resource-leak | 11 |
| design | 10 |
| security | 7 |
| testing | 1 |
| design-smell | 1 |
| contract-violation | 1 |

### Files with the most findings

| File | Findings |
|---|---|
| `lib/engine/system_compiler.py` | 23 |
| `lib/engine/simulation_engine.py` | 17 |
| `lib/lib.py` | 16 |
| `modern_ui/widgets/modern_canvas.py` | 13 |
| `lib/analysis/linearizer.py` | 10 |
| `modern_ui/widgets/property_editor.py` | 9 |
| `lib/export/blox_exporter.py` | 8 |
| `lib/simulation/block.py` | 7 |
| `lib/engine/symbolic_engine.py` | 7 |
| `modern_ui/main_window.py` | 6 |
| `modern_ui/widgets/variable_editor.py` | 6 |
| `blocks/export.py` | 6 |
| `lib/export/tikz_exporter.py` | 6 |
| `modern_ui/renderers/canvas_renderer.py` | 6 |
| `modern_ui/renderers/block_renderer.py` | 6 |
| `blocks/step.py` | 5 |
| `lib/engine/flattener.py` | 5 |
| `lib/managers/subsystem_manager.py` | 5 |
| `lib/services/file_service.py` | 5 |
| `lib/engine/optimization_engine.py` | 4 |

## Critical severity (2)


### `lib/analysis/linearizer.py`

- **linearize_at_point returns a fabricated (all-zero) A matrix from a stub derivative function** — `231-259` _(correctness)_  
  Verified: compute_derivatives (lines 231-256) only handles Integrator, and even that sets dx[idx:idx+n_states]=0.0 with a literal '# Placeholder' comment; '# Add other block types...' is the only other content. dx is therefore always the zero vector, so A = compute_jacobian_numerical(...) (line 259) is always the zero matrix regardless of the system.  
  **Fix:** Drive the actual block.execute()/engine step at the perturbed state to obtain real derivatives, or raise NotImplementedError rather than returning a result dict that looks valid but is meaningless.
- **B, C, D returned as placeholder zero matrices presented as valid results** — `264-290` _(correctness)_  
  Verified: lines 268-270 hardcode B=np.zeros((n,m)), C=np.zeros((p,n)), D=np.zeros((p,m)) with a '# Placeholder B, C, D matrices' comment and no input/output perturbation. They are returned (lines 278-290) in the same dict as eigenvalues/is_stable as if computed, so downstream TF/controllability/observability get nonsense.  
  **Fix:** Compute B by perturbing inputs and C by perturbing states/measuring outputs; until implemented, mark the result partial/invalid instead of returning silent zeros.


## High severity (42)


### `blocks/demux.py`

- **Demux divides by output_shape with no zero guard (uncaught ZeroDivisionError)** — `80` _(error-handling)_  
  At blocks/demux.py:80 and :83, `n_elements / output_shape` runs OUTSIDE any try/except (the try at lines 71-77 wraps only parsing). `output_shape` is an int param (default 1, no min). A user setting output_shape=0 raises ZeroDivisionError that propagates out of execute() and crashes the simulation instead of returning {'E': True}.  
  **Fix:** After parsing output_shape (line 73), validate `if output_shape < 1: return {'E': True, 'error': '...'}`, or use integer floor-division with a guard, before the comparisons at lines 80/83.

### `blocks/pde/field_processing.py`

- **FieldScope rebuilds full history array into params['vector'] every timestep (O(n^2) memory churn)** — `465-471` _(performance)_  
  FieldScopeBlock.execute (per-timestep sink hot path) appends the new snapshot to params['_field_history_'] then immediately rebuilds the entire history into a fresh NumPy array (np.array(params['_field_history_'])) and stores it in params['vector'] on EVERY step (lines 469-470). With T steps and N spatial points this is O(T) allocations each of O(T*N) size => O(T^2 * N) total work/churn. The history list also grows unbounded with no sampling cap, unlike FieldScope2D which has sample_interval. For long PDE runs this dominates runtime and inflates transient memory.  
  **Fix:** Build params['vector'] = np.array(history) lazily in plot_field() / at simulation end instead of every step, or keep a preallocated growable buffer. Optionally add a sample_interval like FieldScope2D to bound memory for long runs.

### `blocks/scope.py`

- **Scope buffer built with O(n^2) per-timestep np.concatenate** — `139` _(performance)_  
  blocks/scope.py:139 does `aux_vector = np.concatenate((aux_vector, new_sample))` on every execute() call, reallocating and copying the entire growing buffer each timestep -> O(N^2) total in sample count, with large memory churn on long/fine-step runs.  
  **Fix:** Append per-sample arrays to a Python list stored in params and np.concatenate once at simulation end (mirroring the fast-path replay in simulation_engine.py), or grow a pre-sized buffer geometrically.

### `blocks/step.py`

- **Step 'pulse' advances boundaries with if not while; skips toggles when dt > delay** — `84` _(correctness)_  
  The pulse branch (line 84) advances _step_old by a single 'if delay' per call, so a timestep spanning more than one pulse half-period (large/variable dt) misses toggles and desyncs the pulse. PRBS correctly uses a while loop (prbs.py:135) for the same boundary-advance pattern.  
  **Fix:** Use a while loop to advance _step_old and toggle across every boundary crossed within the step, mirroring PRBS's 'while time >= params[_next_flip]' logic.

### `blocks/transport_delay.py`

- **TransportDelay appends to its history buffer on every execute call, including output_only init calls** — `74-76` _(correctness)_  
  Confirmed. TransportDelay is listed in OUTPUT_ONLY_SAFE_BLOCK_FNS (lib/engine/memory_blocks.py:49), so the engine's Loop-1 init invokes execute with empty/output_only inputs. blocks/transport_delay.py:74-76 unconditionally does time_buffer.append(float(time)); value_buffer.append(current_input.copy()), with current_input falling back to initial_value when 0 not in inputs (line 72). Unlike DelayBlock/Derivative/Hysteresis/RateLimiter, there is no '0 not in inputs'/output_only early-return, so a spurious sample (and possibly a duplicate-time entry on the subsequent full execute) is inserted into the interpolation time history.  
  **Fix:** Add an early-return guard: if output_only (kwargs.get('output_only')) or 0 not in inputs, return the interpolated/held output without appending, mirroring the output_only branches in hysteresis.py:73 and rate_limiter.py:65.

### `blocks/zero_order_hold.py`

- **ZeroOrderHold crashes on vector/array inputs** — `74-76, 86-89` _(correctness)_  
  At lines 73-76 and 86-89, ZOH coerces input via val.item() then float(val). For a multi-element ndarray, ndarray.item() raises ValueError ('can only convert an array of size 1 to a Python scalar') and float() likewise fails. Inputs are declared type 'any', so any genuine vector signal crashes the block. Single-element arrays work (item() succeeds), so failure is specifically for >1-element arrays.  
  **Fix:** Use get_vector(inputs, 0) (as Saturation/RateLimiter do) and store the held value as an array to preserve vector shape rather than forcing a scalar.

### `lib/analysis/analyzers/nyquist.py`

- **Discrete Nyquist plot is permanently broken: freqresp rejects discrete systems** — `27-35` _(correctness)_  
  Verified empirically against scipy 1.17.1: for dt>0 the code builds signal.TransferFunction(num,den,dt=dt) (a TransferFunctionDiscrete) then calls the MODULE-level signal.freqresp(sys,w=w) at line 35, which raises AttributeError('freqresp can only be used with continuous-time systems.'). The bare except at line 67-69 swallows it, so discrete Nyquist silently returns None with no user feedback. (Note: sys.freqresp instance method exists but the code uses the module function.)  
  **Fix:** In the dt>0 branch call signal.dfreqresp(sys, w=w) instead of signal.freqresp; verified dfreqresp returns correct complex response for discrete systems. Keep freqresp only for the dt==0 continuous branch.

### `lib/analysis/linearizer.py`

- **WaveEquation1D state size inconsistent between get (50) and set ('N')** — `113-124, 166-170` _(correctness)_  
  Verified: get_state_vector (lines 114-119) reads u/v via params.get('u', np.zeros(50)) and uses their actual flattened lengths, while set_state_vector (lines 167-170) uses n=params.get('N',50). WaveEquation1DBlock defines param N (default 50). If the actual u/v arrays differ in length from N, get and set disagree and idx misaligns all subsequent states.  
  **Fix:** Derive n from the actual array length in both get and set (e.g. n=np.atleast_1d(params['u']).size) so round-tripping is consistent.
- **is_stable computed from a degenerate A is always False (all eigenvalues 0)** — `273-276` _(correctness)_  
  Verified: line 273 eigenvalues=np.linalg.eigvals(A) and line 276 is_stable=np.all(np.real(eigenvalues)<0). Because A is the stub zero matrix (Finding 1), all eigenvalues are 0 and the strict <0 test is never satisfied, so is_stable is always False. This is a direct consequence of the stubbed derivatives.  
  **Fix:** Fix the derivative computation first; also distinguish marginal stability for purely-imaginary/zero eigenvalues rather than reporting unstable.
- **RateLimiter/DiffusionReaction1D detected as state blocks but never read/written in state vector** — `54-66, 78-125` _(correctness)_  
  Partially corrected: 'RateLimiter' IS in state_block_types (line 56) but has NO branch in get_state_vector/set_state_vector/compute_derivatives, so it is detected (counted in self.state_blocks) yet contributes no states, drifting the idx pointer. However the finding's DiffusionReaction1D claim is WRONG: DiffusionReaction1D IS handled in get_state_vector (lines 105-106) and set_state_vector (lines 156-163 via the else: params['c'] branch). The real defect is RateLimiter only.  
  **Fix:** Add an explicit RateLimiter branch (or remove it from detection). Make the detection type set match exactly the types handled in get/set/derivative to keep idx alignment consistent.

### `lib/app_paths.py`

- **user_data_path crashes on bare filename (no directory component)** — `54-56` _(correctness)_  
  user_data_path() at line 55 unconditionally calls os.makedirs(os.path.dirname(full), exist_ok=True). When relative_path is a bare filename (no separator), os.path.dirname(full) is non-empty in dev mode only because get_user_data_dir() prepends the project root; but two real callers pass bare names: diablos_modern.py:126 and appearance_manager.py:59 both call user_data_path('user_preferences.json'). In dev mode dirname returns the project-root dir (exists, OK), but in frozen mode get_user_data_dir() returns the DiaBloS app-data dir (also already created), so dirname is non-empty there too. The genuine crash case is any relative_path with NO directory component combined with a base dir — verified os.makedirs('') raises FileNotFoundError [WinError 3]. Risk is real but the current callers happen to always yield a non-empty parent because get_user_data_dir() always returns a non-empty directory; dirname of join(dir, 'file.json') is dir, never ''. So in practice the crash does NOT trigger today. The defensive guard is still good hygiene.  
  **Fix:** Guard the makedirs call defensively: parent = os.path.dirname(full); if parent: os.makedirs(parent, exist_ok=True). Harmless and future-proofs against a future caller passing an absolute or root-level relative path.

### `lib/engine/optimization_engine.py`

- **scipy constraint functions are non-functional placeholders** — `345-358` _(correctness)_  
  create_constraints_scipy builds constraint dicts whose 'fun' always returns 0.0: the inner loop over compute_constraints() has body `pass` and the function returns `0.0  # Placeholder` (lines 345-352). Any constrained optimization silently ignores all Constraint blocks. Note penalty-based handling exists (compute_penalty, use_penalty) as a partial alternative, but the scipy hard-constraint path is non-functional.  
  **Fix:** Track a block->constraint mapping and return the real constraint value for this block's index, or raise NotImplementedError so callers do not assume constraints are enforced.

### `lib/engine/simulation_engine.py`

- **propagate_outputs raises KeyError if block omits a wired output port** — `1037` _(error-handling)_  
  Line 1037 `mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]` directly indexes the returned dict with no `in`/`.get()` guard. If a block omits a port that has a downstream wire (sparse/partial output), this raises KeyError. The only catch is the broad try/except at line 402-407 in initialize_execution, which surfaces it as a generic init failure. During the per-step propagate path it is uncaught.  
  **Fix:** Guard with `if tuple_child['srcport'] in out_value:` and log a diagnostic naming source block and missing port, instead of letting a raw KeyError surface as an opaque init failure.
- **Replay loop re-scans all lines/blocks every timestep (O(steps*blocks*lines))** — `1296` _(performance)_  
  Within `for i in range(num_steps)` (1265) and `for block in sorted_blocks` (1285), inputs are gathered by `for line in current_lines: if line.dstblock == b_name` (1296-1307), rescanning every connection for every block on every timestep -> O(steps*blocks*lines). For long simulations this dominates cost.  
  **Fix:** Precompute a dst_name -> list of (srcblock, srcport, dstport) map once before the timestep loop and reuse it instead of rescanning current_lines per block per step.
- **PID replay indexes current_states without guard, KeyError swallowed** — `1466` _(error-handling)_  
  Line 1466 does `states = current_states[b_name]` unconditionally, then indexes states[0]/states[1] (1467-1468), unlike Integrator (1313 `if b_name in current_states`) and StateSpace/TransferFcn (1318) which guard. If the compiler did not register the PID state (naming mismatch), this raises KeyError that the broad except at 1919 converts into a silent full-simulation failure (return False).  
  **Fix:** Guard with `if b_name in current_states:` and fall back to out_val=0.0 otherwise, matching the other state-block branches; also guard states having >=2 elements.
- **StateVariable replay state never reset between simulation runs** — `1546-1555` _(correctness)_  
  Lines 1546-1555 init _replay_state_/_replay_pending_ in block.params only when '_replay_state_' not in params. reset_memblocks (687-701) resets only _init_start_ (params+exec_params) and deletes exec_params['_prev']; it never clears _replay_state_/_replay_pending_. So a second run keeps the final state from the previous run instead of re-initializing to initial_value. The Hysteresis branch (1739) shows the correct pattern: re-init when _init_start_ is True.  
  **Fix:** Re-initialize _replay_state_ when block.exec_params/params _init_start_ is True OR the key is missing (mirroring line 1739), then clear _init_start_, so reset_memblocks takes effect across runs.

### `lib/engine/system_compiler.py`

- **Vectorize nested per-node Python loops in 2D PDE RHS hot path** — `1126-1170` _(performance)_  
  Confirmed: Heatequation2D (1126-1130), Waveequation2D (1230-1235) and Advectionequation2D (1346-1361) compute the interior stencil with an explicit double Python for-loop over every interior grid node inside the solve_ivp RHS closure, which is invoked many times per timestep (plus intermediate stages). For Nx=Ny=20/30 this is hundreds of pure-Python iterations per evaluation. Boundary loops are also pure-Python. Numpy slicing would be 1-2 orders of magnitude faster.  
  **Fix:** Replace interior loops with vectorized slicing, e.g. dT_dt[1:-1,1:-1] = alpha*((T[1:-1,2:]-2*T[1:-1,1:-1]+T[1:-1,:-2])/dx_sq + (T[2:,1:-1]-2*T[1:-1,1:-1]+T[:-2,1:-1])/dy_sq)+q_src, and vectorize the edge updates similarly.
- **All-Neumann 2D PDE corners never assigned (frozen derivative)** — `1136-1170` _(correctness)_  
  Confirmed in Heatequation2D: interior loop is range(1,_Ny-1) x range(1,_Nx-1) (1126-1130). The Neumann edge loops are left/right range(1,_Ny-1) (1137,1147) and bottom/top range(1,_Nx-1) (1157,1167), all excluding their endpoints. So when ALL four sides are Neumann, the four corner nodes (0,0),(0,Nx-1),(Ny-1,0),(Ny-1,Nx-1) are written by no branch; dT_dt is zero-initialized (1120) so corners stay frozen at initial value. Same structure in Waveequation2D (1230-1283) and Advectionequation2D Neumann branches (1346-1422). Note: if ANY side is Dirichlet, its loop uses range(_Ny)/range(_Nx) and covers the corners, so the bug only manifests for fully-Neumann (or Neumann+Outflow, since Outflow loops use range(_Ny)/range(_Nx) and DO cover corners) boundary configs.  
  **Fix:** Add explicit corner handling for the all-Neumann case (corner ghost-node stencil or average of adjacent edge derivatives), or validate against / document the all-Neumann configuration.
- **Impulse/Step-impulse spike height uses a fixed default dtime and is too narrow for adaptive solvers** — `300-331` _(correctness)_  
  Step(type='impulse') and Impulse build a rectangular pulse width eps=dtime*1e-3, height val/eps, dtime defaulting to 0.01 if absent. With an adaptive RK45 step the ~1e-5 s active window can fall entirely between two evaluation times so the impulse is skipped (response silently wrong); if dtime mismatches the actual solver dt the impulse area (val) is also scaled wrong.  
  **Fix:** Make eps relative to the solver's max_step / actual sampling dt and force solver evaluations at the pulse edges (t_eval / max_step), or model the impulse analytically; do not rely on a hard-coded 0.01 fallback for area scaling.
- **StateVariable mutates closure state inside the RHS keyed on monotonic-time assumption** — `595-621` _(correctness)_  
  exec_statevariable holds state={'current':initial,'prev_t':-1.0} in a closure and, inside the RHS, does if t > _state['prev_t']+0.5: _state['current']=copy(signals[_src]); _state['prev_t']=t. solve_ivp calls the RHS at non-monotonic, repeated and rejected-step times, so the discrete 0.5-threshold update fires on arbitrary internal stages; state also lives outside params, surviving resets invisibly.  
  **Fix:** Do not perform discrete state updates inside the continuous ODE RHS. Drive StateVariable updates from the accepted-step loop / interpreter path, or model it as a proper ODE state held in y.
- **Compiled RateLimiter reads wrong param keys ('rising'/'falling'), ignoring user slew rates** — `652-653` _(consistency)_  
  Confirmed. Lines 652-653 read params.get('rising', 1.0) and params.get('falling', -1.0). The interpreted RateLimiterBlock (blocks/rate_limiter.py:79-80) and slew_rate_params (blocks/param_templates.py:127-138) define rising_slew/falling_slew with default np.inf. The 'rising'/'falling' keys never exist, so the compiled solver always clamps dy to hardcoded [-1.0, 1.0] regardless of user configuration. Additionally line 1811 reads sparams.get('init_cond', 0.0) for the initial output, but RateLimiterBlock has no init_cond param (its params are slew_rate_params + init_flag), so the IC silently defaults to 0.0.  
  **Fix:** Read params.get('rising_slew', np.inf) and params.get('falling_slew', np.inf) and apply abs()/clip(delta, -max_dec, max_inc) semantics matching RateLimiterBlock.execute. Remove or correct the nonexistent 'init_cond' key at line 1811.
- **Noise block injects np.random.randn() into the ODE RHS** — `671-677` _(correctness)_  
  exec_noise sets signals[b_name]=mu+sigma*np.random.randn() on every RHS call. Adaptive solvers evaluate the RHS multiple times per step (stages, error estimate, rejected steps), so a non-deterministic RHS breaks step-size error control and yields an irreproducible/meaningless trajectory rather than a band-limited noise input.  
  **Fix:** Generate noise on a fixed sampling grid (precompute a t-indexed sequence and interpolate, like PRBS does) or exclude Noise from the compiled solver path so it only runs on accepted time steps.
- **Hysteresis latch stores cross-timestep state in a closure list mutated inside the ODE RHS** — `794-817` _(correctness)_  
  exec_hysteresis mutates state_holder=[low_val] inside the RHS: val>=upper sets _state[0]=high_val, val<=lower sets low_val, output is _state[0]. Adaptive RK45 evaluates the RHS at multiple internal stages and rejected steps, so the latch flips on transient stage values, not accepted steps, giving wrong path-dependent output. The latch also lives in a closure list, outside params, surviving reset_memblocks() invisibly.  
  **Fix:** Hysteresis is path-dependent and cannot be a pure function of (t,y); exclude it from the compiled fast path (interpreter fallback), or model it as a smooth ODE state so the latch is part of y and reproducible across RHS re-evaluations.
- **Compiled Hysteresis stores switching state in a closure list mutated on every solve_ivp RHS evaluation** — `794-817` _(correctness)_  
  exec_hysteresis (lines 804-817) mutates _state[0]=state_holder inside the closure, and this executor is run from model_func (line 2016) which is the RHS passed to solve_ivp. solve_ivp calls the RHS many times per accepted step (RK stages, rejected/retried steps) at non-monotonic trial t, so the latched hysteresis state depends on call count/order rather than physical time, corrupting output and reproducibility.  
  **Fix:** Exclude Hysteresis from the compilable set (fall back to interpreted path) or implement it as an event/accepted-step state update; do not mutate latched state inside an adaptive ODE RHS.
- **Compiled Hysteresis relay holds state in a mutable closure mutated by every solve_ivp RHS probe** — `794-817` _(correctness)_  
  Confirmed at lines 794-817: exec_hysteresis closes over state_holder=[low_val] and mutates _state[0] (high_val/low_val) inside the ODE RHS based on the input signal at the current probe point. solve_ivp (RK45) evaluates the RHS at intermediate stages, rejected/trial steps, and not in monotonically increasing accepted-time order, so the latched relay state is corrupted by non-accepted probes, diverging from the interpreted history-dependent path in blocks/hysteresis.py.  
  **Fix:** History-dependent non-differentiable relay logic is unsafe inside a continuous ODE RHS. Route Hysteresis through the interpreted/discrete path, or implement it with solve_ivp event detection (zero-crossing on upper/lower) and only latch state on accepted events, rather than mutating a closure on every RHS call.

### `lib/export/blox_exporter.py`

- **Feedback signal labels inserted into blox macros without LaTeX escaping** — `156-181` _(security)_  
  fb_label = line.label (line 157) is interpolated raw into \bXReturn at line 173 (label_opt) and line 180 ({fb_label}), with no _escape_latex, while _find_label (line 397) escapes via f'${_escape_latex(label)}$'. A label containing _ & # % { or } breaks/injects LaTeX. Note one branch (173) is currently unreachable (see _find_feedback_block always None), but line 180 is live.  
  **Fix:** Mirror _find_label: if not math-delimited, wrap as f'${_escape_latex(fb_label)}$' before interpolating into both \bXReturn forms.

### `lib/lib.py`

- **export_data writes to hardcoded relative 'saves/' path and assumes 4-char extension** — `1193` _(error-handling)_  
  Line 1193 reads exactly np.savez('saves/' + self.filename[:-4], t=self.timeline, **vec_dict). The 'saves/' prefix is a relative path with no os.makedirs(exist_ok=True) guard (FileNotFoundError if the directory is absent or cwd differs), and self.filename[:-4] blindly strips 4 trailing characters assuming a '.xxx' extension. A filename without that exact extension length corrupts the output basename. Log line 1194 reuses the same slice.  
  **Fix:** Resolve the saves directory via lib/app_paths, create it with os.makedirs(path, exist_ok=True), and derive the basename with os.path.splitext(self.filename)[0] instead of [:-4].
- **Massive duplication between run_tuning_simulation/execution_loop_headless and execution_init/execution_loop** — `639-823` _(duplication)_  
  run_tuning_simulation (639-721) and execution_loop_headless (723-823) re-implement param resolution, engine init, goto/from linking, counter setup, RK45 sub-step bookkeeping (731-740 vs 876-888), and the full nested hierarchy iteration (759-812 vs 940-1000) from execution_init/execution_loop. The bodies are near-verbatim copies, so any fix to the stepping logic must be applied in two places, inviting silent divergence.  
  **Fix:** Extract the shared param-resolution and inner hierarchy-iteration into helpers (e.g. _resolve_blocks() and _run_hierarchy_step()) consumed by both UI and headless paths.
- **headless resolve loop omits External reload and Subsystem recursion present in execution_init** — `667-671` _(correctness)_  
  run_tuning_simulation resolves params with a flat loop over root_blocks only (667-671), skipping the reload_external_data() call (476-477) and the recursion into block.sub_blocks (486-488) that resolve_recursive performs. Tuning a diagram with subsystems or External blocks runs with stale/unresolved nested params, producing wrong results silently.  
  **Fix:** Reuse the same resolve_recursive helper (or factor it out) so subsystem and External handling stay consistent across both code paths.

### `lib/plotting/scope_plotter.py`

- **Matplotlib figures in ScopePlotter are never closed, leaking figures across runs** — `148-157, 213-222, 305-347, 368-433, 499-578, 621-689, 762-770, 802-821` _(resource-leak)_  
  Every matplotlib plotting method in ScopePlotter creates a figure via plt.subplots(...) and shows it with plt.show()/plt.show(block=False) but never calls plt.close(fig). Confirmed at: _plot_xygraph (148 fig, 157 plt.show()), _plot_fft (213 fig, 222 plt.show()), _plot_field_scope (305 fig, 343 plt.show(block=False)), _plot_field_scope_slider (368 fig, 429 plt.show(block=False), with widgets stored on fig._time_slider/fig._export_btn at 405/426), _plot_field_scope_2d (499 fig, 574 plt.show(block=False)), _plot_agent_scope (621 fig, 686 plt.show(block=False)). Because pyplot keeps a global registry of figures created via plt.subplots, each re-run/re-plot creates a new figure while old ones are retained, accumulating figures, their Qt windows, the Slider/Button widgets, and the large field_data/positions arrays captured in the update/on_export closures. Confirmed via grep that the only plt.close in the entire package is animation_exporter.py:349. The pyqtgraph SignalPlot path is separately managed via close()/deleteLater() in pyqtPlotScope.  
  **Fix:** Track created figures (e.g. self._open_figs list) and close stale ones at the start of plot_again, or register mpl_connect('close_event', ...) handlers that call plt.close(fig). At minimum close the figure when its window is dismissed so re-running the sim does not accumulate figures and the arrays captured in their closures.

### `lib/plotting/signal_plot.py`

- **SignalPlot subscribes to theme_manager singleton but never disconnects, leaking the widget and risking RuntimeError** — `116` _(resource-leak)_  
  In __init__ (signal_plot.py:116) SignalPlot connects the module-level theme_manager singleton's theme_changed signal to self._apply_theme. Grep confirms there is no theme_changed.disconnect and no closeEvent override anywhere in the file. theme_manager is a long-lived singleton (imported at line 21), so it retains a bound-method reference to every SignalPlot instance, preventing GC of closed windows. Additionally ScopePlotter.pyqtPlotScope (scope_plotter.py:1132-1141) closes and deleteLater()s the previous plotty; if the user then changes the theme, theme_manager still holds a reference to the deleted SignalPlot and would invoke _apply_theme on a deleted underlying C++ QWidget, raising RuntimeError.  
  **Fix:** Override closeEvent (and/or connect self.destroyed) to call theme_manager.theme_changed.disconnect(self._apply_theme) wrapped in try/except (TypeError, RuntimeError). This frees the widget for GC and prevents _apply_theme from firing on a deleted widget.

### `lib/simulation/block.py`

- **__deepcopy__ swallows all exceptions and sets attributes to None, silently corrupting copies** — `696-702` _(error-handling)_  
  In __deepcopy__ (lines 696-702), the catch-all `except Exception` for any per-attribute deepcopy failure logs a warning and sets that attribute to None on the copied block. A copied DBlock can silently end up with params/rect/b_color/etc. = None, producing AttributeErrors or wrong behavior far from the real cause during copy/paste or subsystem flatten.  
  **Fix:** Handle known-unpicklable Qt types explicitly (e.g. QColor->QColor(v), QRect->QRect(v)) and let unexpected failures raise; at minimum re-raise for load-bearing keys like params/rect/b_color instead of nulling them.

### `lib/simulation/menu_block.py`

- **MenuBlocks hardcodes ./lib/icons/ path, breaking frozen/packaged builds and non-cwd launches** — `23` _(correctness)_  
  Line 23 reads pixmap = QPixmap(f'./lib/icons/{self.block_fn.lower()}.png'). This is a cwd-relative literal. In a PyInstaller frozen build the asset lives under sys._MEIPASS, and launching from any other working directory resolves wrong, so every menu icon falls back to a null pixmap (silently handled by the isNull() check at line 24).  
  **Fix:** Import lib.app_paths.resource_path (confirmed to exist and resolve sys._MEIPASS when frozen, project root in dev) and use QPixmap(resource_path(f'lib/icons/{self.block_fn.lower()}.png')).

### `modern_ui/main_window.py`

- **UI scaling factor is written to user-data path but read from read-only bundled resource (frozen mode)** — `247-262` _(correctness)_  
  _set_scaling() (main_window.py line 250) persists scaling_factor to user_data_path('config/default_config.json'), while setup_application() in diablos_modern.py (lines 93-96) reads it from resource_path('config/default_config.json'). Per lib/app_paths.py, resource_path resolves to sys._MEIPASS (read-only extracted bundle) when frozen, while user_data_path resolves to %APPDATA%/DiaBloS (~/Library/Application Support on macOS, ~/.local/share on Linux). The written file is never read back on next launch in a packaged build, so the 'restart to take effect' dialog (lines 264-265) promises behavior that never occurs. In dev mode both functions resolve to project root (lib/app_paths.py lines 20 and 38 are identical when not frozen), masking the bug.  
  **Fix:** Make read and write use the same path. In setup_application(), read scaling via user_data_path('config/default_config.json'), falling back to the bundled default via resource_path when the user file is absent. lib/config_manager.py reportedly already implements a user-then-bundled fallback; reuse that pattern.

### `modern_ui/managers/history_manager.py`

- **_restore_state does not bounds-check ports and leaves the diagram half-cleared on failure** — `204-226` _(error-handling)_  
  In _restore_state, blocks_list and line_list are cleared at lines 140-141. The connection-restore loop (lines 205-226) indexes src_block.out_coords[line_data['srcport']] (line 210) and dst_block.in_coords[line_data['dstport']] (line 211) with NO bounds check and NO inner try/except. If a snapshot references a port index no longer present on the rebuilt block, this raises IndexError, which propagates to the outer `except Exception` at line 232 returning False -- after the canvas has already been wiped (140-141) and only partially repopulated. The result is a destroyed diagram from an undo/redo. The block-restore loop above (159-201) DOES have an inner try/except (160/199), and the paste path in clipboard_manager (lines 235-240) DOES guard these port indices, confirming this is an oversight here.  
  **Fix:** Wrap the per-line restore (205-226) in its own try/except (continue on error) and validate srcport < len(src_block.out_coords) and dstport < len(dst_block.in_coords) before indexing, mirroring the guards in ClipboardManager.paste_blocks (235-240). This prevents one bad connection from destroying the entire restored diagram.

### `modern_ui/managers/project_manager.py`

- **ProjectManager.autosave_path derived from module name string, not a real path — recovery/cleanup never touch the actual autosave file** — `18` _(correctness)_  
  Line 18: `self.autosave_path = os.path.join(os.path.dirname(os.path.abspath(main_window.__module__)), 'autosave.json')`. `main_window.__module__` is the class's module-name STRING (e.g. "modern_ui.main_window" or "__main__"), not a filesystem path. `os.path.abspath` of that string resolves it relative to CWD, and `os.path.dirname` then yields the project root / CWD, so `self.autosave_path` becomes `<cwd>/autosave.json`. I confirmed this empirically: `os.path.dirname(os.path.abspath('modern_ui.main_window'))` returned the repo root. Meanwhile the real autosave is written by MainWindow to `user_data_path('config/.autosave.diablos')` (main_window.py line 192, used by `_auto_save` at line 802). MainWindow's `_check_autosave_recovery` (line 814) and `_cleanup_autosave` (line 817) delegate to ProjectManager's `check_autosave_recovery`/`cleanup_autosave`, which both operate on the bogus `<cwd>/autosave.json`. Therefore recovery never finds the real autosave, the recovery prompt for the real file never appears, and cleanup never removes the real file. Confirmed serious correctness bug.  
  **Fix:** Use the canonical path the window uses. Either set `self.autosave_path = main_window.autosave_path` (after the window assigns it) or import `user_data_path` and set `self.autosave_path = user_data_path('config/.autosave.diablos')`. Never derive a filesystem path from `__module__`.

### `modern_ui/themes/theme_manager.py`

- **get_color() corrupts RGB channels for 8-digit #RRGGBBAA colors; shadow/fill colors render wrong** — `483-491` _(correctness)_  
  At theme_manager.py:484-491 get_color() does `c = QColor(color_hex)` on the full 9-char string, then only patches alpha via setAlpha(). Runtime test (Qt 5.15.2) confirms PyQt5 QColor mis-parses 9-char `#RRGGBBAA`: `QColor('#00000066')` yields r=0,g=0,b=102,a=0 (a dark BLUE, not black), and `QColor('#60A5FA1A')` yields r=165,g=250,b=26,a=96 (yellow-green, not light blue). The subsequent setAlpha() fixes only alpha; the RGB is already corrupted. block_shadow IS consumed by block_renderer.py:71-72, though note line 72 immediately overrides the alpha with setAlpha(80) — so for that consumer the alpha-patch is moot but the corrupted RGB (b=102) still renders a blue-tinted shadow on every block. selection_rectangle_fill (#60A5FA1A) and connection_shadow are defined but not yet consumed, making this a latent trap for future callers. The light theme is also affected (block_shadow '#00000015', selection_rectangle_fill '#2563EB1A').  
  **Fix:** Parse RGB and alpha explicitly: `if len(color_hex)==9 and color_hex.startswith('#'): r,g,b,a = (int(color_hex[i:i+2],16) for i in (1,3,5,7)); c = QColor(r,g,b,a)` wrapped in try/except ValueError falling back to QColor(color_hex[:7]). Add a unit test asserting get_color('block_shadow') == QColor(0,0,0,0x66).

### `modern_ui/widgets/toast_notification.py`

- **Toast fade-out connects finished->hide every call; never disconnected, hides subsequent fade-ins** — `98` _(correctness)_  
  Verified: line 98 'self.fade_animation.finished.connect(self.hide)' lives in _start_fade_out, which runs every time the hide_timer fires (connected line 38). No disconnect exists anywhere. The same self.fade_animation object is reused by show_message() for the fade-IN (lines 83-87). After one fade-out completes, the finished->hide connection persists, so the next show_message's fade-IN will emit 'finished' and immediately hide the freshly shown toast. Connections also accumulate across repeated toasts (signal leak / multiple hide calls).  
  **Fix:** Connect finished->hide once in __init__, or disconnect before each use. Concretely, in show_message() before starting the fade-in do: try: self.fade_animation.finished.disconnect(self.hide) except TypeError: pass — and connect it inside _start_fade_out as now. Better: gate hide on endValue==0.0 so the fade-in's finished signal is a no-op.

### `modern_ui/widgets/variable_editor.py`

- **Raw exec() on user-controlled code bypasses safe_eval** — `235` _(security)_  
  Line 235 is exactly `exec(code, {}, local_vars)` with `code = self.editor.toPlainText()` (line 225). This runs arbitrary user-typed Python with full builtins, bypassing the project's lib/safe_eval.py which the sibling WorkspaceEditor correctly uses (safe_literal/safe_expr). The placeholder text on line 116 literally suggests `import math`, confirming arbitrary code is the intended UX.  
  **Fix:** Parse assignments with ast and evaluate each RHS via safe_eval (safe_literal/safe_expr) as WorkspaceEditor already does, instead of raw exec. If exec must remain, the design intent (running imports/math) makes pure safe_eval insufficient, so at minimum restrict builtins and gate behind explicit consent.
- **exec with empty globals still exposes full builtins including __import__** — `235` _(security)_  
  Passing `{}` as the globals dict to exec does not sandbox: CPython auto-injects `__builtins__` into an empty globals mapping, so user code can call open(), __import__('os').system(...), eval(), etc. The placeholder encourages `import math`, signaling imports are intended.  
  **Fix:** If exec must stay, set restricted builtins via {'__builtins__': safe_subset}; better, route assignments through lib/safe_eval which has a strict node allowlist and no exec/eval. Note this is the same root cause as the prior finding (same line); treat as one fix.

### `modern_ui/widgets/waveform_inspector.py`

- **CSV export reads stale active_indices that _on_trace_changed never updates** — `258` _(correctness)_  
  _export_csv selects traces via `self.active_indices` (line 258), but active_indices is only assigned in __init__ (line 32) and _rebuild_traces (line 139, `set(self.active_traces)`). _on_trace_changed (lines 186-191) mutates only self.active_traces when checkboxes toggle, never active_indices. So after the user toggles individual trace visibility, the exported CSV reflects the selection at the last run-list rebuild, diverging from the plot/readout (which use active_traces).  
  **Fix:** Use self.active_traces in _export_csv (the same set the plot and readout use) and remove the redundant active_indices entirely.

### `tests/regression/test_interpreter_compiled_equivalence.py`

- **Compiled vs interpreted equivalence is only tested on one tank-feedback example; stateful compiled blocks untested** — `85-128` _(testing)_  
  Confirmed. The file contains only test_tank_feedback_compiled_matches_interpreter and test_initial_sample_matches, both exercising the single c01_tank_feedback.diablos diagram (Step->Sum->Gain->TranFn->Scope). RateLimiter, Hysteresis, PID, TransportDelay and Selector each have separate compiled RHS implementations in system_compiler.py with independent state/param handling, none of which are compared to the interpreted block. tests/unit/test_control_blocks.py only instantiates the interpreted RateLimiterBlock/HysteresisBlock directly (with correct rising_slew keys), so it cannot catch the compiled-path 'rising'/'falling' key bug or the Hysteresis closure bug.  
  **Fix:** Parametrize the equivalence test over diagrams containing each compiled stateful block (RateLimiter, Hysteresis, PID, TransportDelay, Selector) and assert compiled scope trace == interpreted trace within tolerance, which would surface the wrong-key and closure-state divergences.


## Medium severity (75)


### `blocks/deadband.py`

- **Deadband uses inputs[0] (KeyError) instead of inputs.get(0)** — `75` _(error-handling)_  
  Line 75 reads inputs[0] directly via np.array(inputs[0], dtype=float). If port 0 is absent (e.g. an output_only probe), this raises KeyError. Peer blocks Hysteresis (line 73) and RateLimiter (line 65) guard with 'if 0 not in inputs'. Deadband has no such guard.  
  **Fix:** Use get_vector(inputs, 0) or inputs.get(0, 0.0) and handle the missing-input/output_only case gracefully, matching the peer blocks.

### `blocks/export.py`

- **Export vector grows by full re-concatenation every timestep (O(n^2))** — `103` _(performance)_  
  export.py:102-104 each non-init execute does aux_vector = params['vector']; aux_vector = np.concatenate((aux_vector, [inputs[0]])); params['vector'] = aux_vector — reallocating and copying the entire accumulated history every simulation step, giving O(n^2) total work and memory churn for long runs.  
  **Fix:** Append to a Python list stored in params and convert to np.array once at finalize/save, or use a preallocated growable buffer, to make accumulation amortized O(1).

### `blocks/fft.py`

- **FFT buffer stores inconsistent scalar/array element types** — `86` _(correctness)_  
  fft.py:86 appends u[0] (scalar) when len(u)==1 but the full array u otherwise. If the input signal's vector width varies across timesteps, np.array(buffer) in _plot_fft / plot_spectrum builds a ragged object array, after which signal[:,0] indexing and the FFT window multiply misbehave or raise. Latent edge case (most signals keep constant width), so medium-to-low impact.  
  **Fix:** Append a consistent representation every step (e.g. always u, or always the analyzed channel u[0]) so np.array(buffer) yields a regular 1-D/2-D array.

### `blocks/first_order_hold.py`

- **FirstOrderHold extrapolates instead of interpolating, contradicting its doc and overshooting** — `129-150` _(correctness)_  
  Lines 129-150 compute slope=(v_curr-v_prev)/dt_samples and output_val=v_curr+slope*dt_from_curr where dt_from_curr=time-t_curr (>=0 and growing until next sample). This extrapolates forward past the latest sample value, producing a ramp that can overshoot beyond v_curr, whereas the doc (lines 53, 60) states 'linear interpolation between samples' and 'Output = linear interpolation from previous sample to current sample'. Behavior mismatches documented contract.  
  **Fix:** Either interpolate between v_prev and v_curr over [t_prev, t_curr] clamping alpha to [0,1], or rename/redocument as a slope-extrapolating predictor.

### `blocks/impulse.py`

- **Impulse area is wrong when actual solver dt differs from fallback 0.01** — `75` _(correctness)_  
  dt falls back to 0.01 (line 75) if neither kwargs['dtime'] nor params['dtime'] is supplied; the emitted value/dt (line 85) then yields an impulse whose integral is value*(dt_real/0.01), silently mis-scaling the Dirac approximation. The same fallback exists in step.py:90.  
  **Fix:** Require dt from the engine and return {'E': True, 'error': ...} if unavailable rather than guessing 0.01, so the impulse area stays equal to value.

### `blocks/integrator.py`

- **Integrator multi-stage 'add_in_memory=False' branches are unreachable dead code; next_add_in_memory kwarg never consumed** — `100` _(dead-code)_  
  Line 100 captures next_add_in_memory = kwargs.get('next_add_in_memory', True) but it is never assigned into params['add_in_memory'], which is set True at line 118 and never updated thereafter. The engine only passes next_add_in_memory=False together with output_only=True (simulation_engine.py lines 437-439), and output_only returns early at lines 120-122. Therefore params['add_in_memory'] is always True when the method branches run, making the else/predictor branches at lines 142/148/154 unreachable dead code and the kwarg effectively dead.  
  **Fix:** Either wire params['add_in_memory'] = next_add_in_memory near the top of execute (if the predictor stages are intended), or delete the unused kwarg and the unreachable else-branches.

### `blocks/math_function.py`

- **MathFunction swallows all errors and silently returns 0.0** — `114-116` _(error-handling)_  
  math_function.py:114-116 catches the broadest `except Exception as e`, logs at warning level only (logger.warning), and returns {0: 0.0} — a bare scalar, dropping any vector shape of the input u. This wraps the entire execute body including the Python-syntax fallback at line 112 (safe_expr), so a SafeEvalError from a malformed user expression is swallowed and surfaced only as a log warning plus a silent 0.0, not the contract's {'E': True, 'error': msg}. Both consequences verified: (1) a bad expression produces wrong-but-plausible 0.0 output instead of a surfaced error, and (2) the scalar 0.0 mismatches an expected vector shape (input u may be array-valued), potentially corrupting downstream blocks. The except also masks genuine numeric/other errors.  
  **Fix:** Catch SafeEvalError specifically around the line 112 fallback and return {'E': True, 'error': ...} so malformed expressions are surfaced. At minimum return an output shaped like the input (np.zeros_like(u)) rather than scalar 0.0, and narrow the broad `except Exception` to the exceptions actually expected.

### `blocks/matrix_gain.py`

- **MatrixGain advertises workspace-variable gains but execute() cannot resolve them** — `84-90` _(correctness)_  
  block_name 'MatrixGain', doc (lines 41-52: 'Accepts workspace variable names (e.g. K)', example 'K (workspace variable)'), and params doc (line 60: 'scalar, vector, matrix, or workspace variable name') all advertise workspace-variable gains. But execute() (lines 84-90) only parses literals: for a string like 'K', ast.literal_eval('K') raises ValueError, the except catches (ValueError, SyntaxError) and falls back to np.array([float('K')]) which also raises ValueError; that ValueError is caught by the outer (ValueError, TypeError) handler (line 114) and returned as {'E': True, 'error': ...}. Net effect confirmed: entering the documented workspace-variable name yields a simulation error, not the advertised resolution. There is no workspace/variables lookup anywhere in execute(). The feature is unimplemented relative to the docs.  
  **Fix:** Either implement workspace-variable resolution (resolve the name against a workspace/variables dict before parsing, e.g. via safe_expr(K_raw, variables=workspace)), or remove the 'workspace variable name' claims from the doc/params text to avoid misleading users about a capability the code does not provide.

### `blocks/mux.py`

- **Mux drops port ordering when iterating inputs.values()** — `66` _(correctness)_  
  blocks/mux.py:66 iterates `for val in inputs.values()` to build the combined vector, relying on dict insertion order, whereas Scope.execute uses `sorted(inputs.keys())` (scope.py:111,129). If the engine ever passes inputs out of port-index order, Mux concatenates in the wrong order silently and inconsistently with Scope.  
  **Fix:** Iterate `for p in sorted(inputs.keys()): combined.extend(np.atleast_1d(inputs[p]).flatten())` to guarantee deterministic port-index order consistent with Scope.

### `blocks/noise.py`

- **Noise block uses global np.random with no seed in params; non-reproducible across resets** — `73` _(correctness)_  
  NoiseBlock.execute (line 73) calls global np.random.randn() with no RNG state stored in params, so output is unaffected by reset_memblocks and is non-reproducible. The doc (line 35) advertises a 'Seed' parameter that does not exist; params (lines 42-45) only define mu/sigma. Doc/params mismatch confirmed.  
  **Fix:** Add a 'seed' param and store a numpy Generator (np.random.default_rng(seed)) under the _init_start_ flag in params so reset_memblocks re-seeds it; draw via that generator instead of global np.random. Downgraded to medium: noise non-reproducibility is a quality/correctness concern but not a hard simulation breakage.

### `blocks/optimization/cost_function.py`

- **CostFunction logs at INFO level on essentially every timestep, spamming logs during optimization** — `243-244` _(performance)_  
  Confirmed: lines 243-244 are `if time < 1.1 or int(time * 10) % 10 == 0:` then `logger.info(...)`. With dt=0.01 this logs on every step for the first ~110 steps and then on roughly every step where int(time*10) is a multiple of 10. CostFunctionBlock.execute() runs per timestep and the whole simulation is re-run for every optimizer function evaluation (optimization_engine.create_objective re-runs execution_init + execution_batch per eval), so this emits a flood of INFO records with f-string formatting in the hottest part of the optimization loop.  
  **Fix:** Demote to logger.debug() (off by default) or gate behind a verbose param, and fix the cadence to log only when crossing each integer second using the already-stored params['_prev_time_'].

### `blocks/optimization/data_fit.py`

- **Spline interpolation rebuilds a UnivariateSpline on every timestep (per-step hot path)** — `172-178` _(performance)_  
  Confirmed: in DataFitBlock.execute(), when interpolation=='spline' (lines 172-178), a fresh scipy UnivariateSpline(t_data, y_data, s=0) is constructed over the entire dataset on every execute() call, then evaluated at one point. execute() runs once per simulation timestep, and t_data/y_data (params['_time_data_']/'_signal_data_') are loaded once in _load_data() and never change, so the spline is recomputed redundantly each step. Inside the optimization loop the cost is multiplied by (timesteps x optimizer evaluations). Note: this only affects users who explicitly set interpolation='spline'; the default is 'linear' (np.interp, cheap), so impact is conditional, hence severity lowered to medium.  
  **Fix:** Build the spline once (e.g. in _load_data() or lazily on first use) and cache the fitted callable in exec_params (e.g. exec_params['_spline_']), reusing it each timestep; fall back to np.interp if construction fails. Do not store it where it would be persisted/pickled to a project file.

### `blocks/pde/field_processing.py`

- **matplotlib figure created in plot_field is never closed (resource leak)** — `501-516` _(resource-leak)_  
  FieldScopeBlock.plot_field creates fig via plt.subplots (line 501), calls plt.show() (line 514) and returns fig (line 516) but never calls plt.close(fig). The figure stays registered with pyplot's global figure manager. In the long-lived Qt GUI process, repeated runs/replots accumulate figures and memory.  
  **Fix:** Close the figure when the caller is done (e.g. have the caller plt.close(fig)), or route figure creation through the project's plotting service rather than the global pyplot state to prevent unbounded figure accumulation across runs.

### `blocks/pde/heat_equation_1d.py`

- **PDE discretization logic is fully duplicated between blocks and SystemCompiler (drift risk, already diverged)** — `215-258` _(duplication)_  
  Each PDE block's execute()/compute_derivatives() re-derives finite-difference stencils and BCs that are independently reimplemented inline in lib/engine/system_compiler.py (heat1D kernel at ~857-890). The compiled path (production solve_ivp path) never calls the block methods (Grep: zero compute_derivatives references in system_compiler.py), so the two copies can and already do diverge: Robin BC is a real convective relation in heat_equation_1d.execute (lines 237-244) but is collapsed to a penalty Dirichlet in the compiler (lines 872-874, 883-884). Maintaining two copies guarantees subtle interpreter-vs-compiled mismatches.  
  **Fix:** Have the compiler delegate to each block's compute_derivatives (after unifying the signature), or extract the FD kernels/BCs into shared functions used by both the block and the compiler so they cannot drift.
- **No CFL / stability check or guidance for explicit Forward-Euler interpreter path** — `260-262` _(correctness)_  
  The interpreter-mode execute() advances diffusion with explicit Forward Euler (T_new = T + dT_dt*dtime, lines 260-262). The FTCS stability limit is dtime <= dx^2/(2*alpha). With confirmed defaults alpha=1.0 (line 189), L=1.0 (line 190), N=20 (line 191) -> dx≈0.0526 -> stable limit ≈ 0.00138, far below the confirmed default dtime=0.01 (line 193). The interpreter path will diverge silently with no warning. Same unconditioned Forward Euler appears in diffusion_reaction_1d (line 263) and wave_equation_1d (lines 272-273, CFL dtime <= dx/c).  
  **Fix:** Compute the stability-limited step (dx^2/(2*alpha) for diffusion, dx/c for waves) and sub-cycle Forward Euler within one dtime, or emit a warning / return {'E': True, 'error': ...} when dtime exceeds the CFL limit rather than producing silently divergent output.
- **Inconsistent compute_derivatives signatures between 1D and 2D PDE blocks** — `269` _(consistency)_  
  1D PDE blocks define compute_derivatives with no time and inputs/params swapped relative to 2D: heat_equation_1d:269 (self, T, params, inputs); wave_equation_1d:298 (self, state, params, inputs); diffusion_reaction_1d:275 (self, c, params, inputs); advection_equation_1d:286 (self, c, params, inputs). 2D blocks use (self, time, state, inputs, params): heat_equation_2d:196, wave_equation_2d:255, advection_equation_2d:259. Both the argument set (1D lacks time) and order (inputs/params swapped) differ. A generic caller not special-casing dimension passes args to wrong parameters. Tests exercise only the 2D form (test_regression_suite.py:347 calls (0.0, state, {}, params); test_pde_blocks.py:1019,1050 likewise), so the 1D form is untested via that path.  
  **Fix:** Standardize on one signature (e.g. (self, time, state, inputs, params)) across all PDE blocks and update 1D blocks, or document the divergence prominently. Add tests that call the 1D compute_derivatives with its actual signature.

### `blocks/selector.py`

- **Selector range bounds not validated; negative/descending ranges yield bad indices** — `112` _(correctness)_  
  At blocks/selector.py:112 `indices.extend(range(start, min(end, max_len)))` only clamps `end` to max_len; `start` is never normalized or bounds-checked. The single-index path (lines 116-118) handles negatives (max_len+idx) and filters 0<=idx<max_len, but the range path does not. A negative start (e.g. '-2:0' or '-1:') yields negative ints fed to `u[indices]`, which numpy interprets as from-the-end indices (wrong selection); a descending range silently yields an empty list.  
  **Fix:** Normalize negative start/end like the single-index path, then filter range output with `[v for v in range(start, end) if 0 <= v < max_len]`, mirroring the single-index validation.

### `blocks/statespace.py`

- **StateSpace reads params['dtime'] at init without default (KeyError risk)** — `79` _(error-handling)_  
  Line 79 `dtime = params['dtime']` direct access; params property (lines 33-43) has no 'dtime' and it is not pulled from kwargs. Same situation as TransferFunction: safe in the normal engine flow (dtime injected at simulation_engine.py 198/862) but raises an uncaught KeyError for callers that don't inject it.  
  **Fix:** Read defensively via kwargs.get('dtime', params.get('dtime', 0.01)) and validate before cont2discrete.

### `blocks/step.py`

- **Step 'pulse' with delay==0 toggles every timestep and never advances _step_old** — `84` _(error-handling)_  
  With delay (pulse half-period) == 0, 'time - _step_old >= 0' is always true, '_step_old += 0' never advances, and the pulse flips every timestep — meaningless output with no validation. PRBS guards bit_time<=0 (prbs.py:82-83); the step pulse path has no equivalent guard.  
  **Fix:** Validate delay > 0 for pulse type and return {'E': True, 'error': ...}, or treat a non-positive period as constant output.
- **Step 'impulse' subtype duplicates ImpulseBlock and is undocumented in params** — `89` _(duplication)_  
  The step_type=='impulse' branch (lines 89-95) reimplements ImpulseBlock's value/dt one-step logic verbatim, but the 'type' param doc (line 47) only lists 'up, down, pulse, constant'. This is a hidden mode and duplicated maintenance. It also relies on params.get('_impulse_fired') without explicit init (see related step.py:71 finding).  
  **Fix:** Remove the duplicated 'impulse' branch and direct users to ImpulseBlock, or document it and factor the shared value/dt logic; if kept, init _impulse_fired in the _init_start_ block.

### `blocks/transfer_function.py`

- **TransferFunction reads params['dtime'] at init without default or kwargs fallback (KeyError risk)** — `108` _(error-handling)_  
  Line 108 `dtime = params['dtime']` is a direct dict access; the params property (lines 36-42) defines numerator/denominator/_init_start_/sampling_time but no 'dtime', and unlike IntegratorBlock (which uses kwargs.get('dtime', params.get('dtime', 0.01)) at line 101) it is not read from kwargs. In the normal engine flow this is safe because the engine injects exec_params['dtime']=sim_dt before every execute (simulation_engine.py lines 198, 439, 862), so a KeyError only occurs for alternate/direct/test callers that omit dtime. Real inconsistency and a latent crash path, but not high severity since the production path always populates dtime.  
  **Fix:** Match IntegratorBlock: `dtime = kwargs.get('dtime', params.get('dtime', 0.01))`, or guard with params.get('dtime') and return {'E':True,'error':...} if missing.

### `lib/analysis/analyzers/base_analyzer.py`

- **_extract_system_model swallows all exceptions returning None with no diagnostics** — `95-96, 106-107, 117-118` _(error-handling)_  
  Verified: three blocks 'except Exception: return None' at lines 95-96 (TF), 106-107 (StateSpace), 117-118 (PID) discard the real error (bad coefficients, singular SS, scipy failure) with no log line. Users get a silent 'no model' with no way to diagnose why extraction failed.  
  **Fix:** Log the exception (logger.debug/warning with exc_info=True) before returning None so failures are diagnosable.

### `lib/analysis/analyzers/root_locus.py`

- **Bare except swallows all analysis errors, hiding real failures** — `121-123` _(error-handling)_  
  Confirmed in all three plot analyzers: root_locus.py:121-123, nyquist.py:67-69, bode.py:80-82 each wrap the whole body in 'except Exception as e: logger.error(...); return None'. Genuine failures (the discrete freqresp AttributeError, the improper-PID np.array inhomogeneous-shape ValueError, bad matrices, plotting errors) are reduced to a log line and a silent None — the user sees nothing happen. LQR (lqr.py:_show_error) by contrast surfaces a QMessageBox, showing the project has a better pattern available.  
  **Fix:** Surface failures to the user via a QMessageBox like LQRAnalyzer._show_error, use logger.exception() to capture stack traces, and narrow the except to expected exception types so unexpected bugs are not masked.

### `lib/config_manager.py`

- **Loaded config is not merged with defaults; missing sections silently absent** — `34-46` _(correctness)_  
  _load_config (lines 37-39) sets self._config = json.load(f) — a full replace with no deep-merge against _get_default_config(). A partial on-disk config (missing e.g. 'performance' or 'logging') leaves those sections entirely absent from _config; callers depend on per-call get(key, default) defaults that can drift from _get_default_config and are easy to omit (e.g. get_all() returns the partial dict; apply_to_dsim relies on inline defaults).  
  **Fix:** Deep-merge the loaded dict over a copy of _get_default_config() in _load_config so every section/key has a baseline regardless of file contents.

### `lib/dialogs.py`

- **SimulationDialog.get_values raises ValueError on invalid numeric input** — `175-184` _(error-handling)_  
  SimulationDialog.get_values() (lines 175-184) calls float() directly on six free-text QLineEdit fields (sim_time, sim_dt, plot_trange, rtol, atol) with no validation or try/except. Any non-numeric or empty entry raises an uncaught ValueError. Since accept_btn.clicked is wired to QDialog.accept (line 167) and the dialog does not override accept(), the dialog closes on OK and the caller invokes get_values() afterward, where the exception propagates into the caller's accept flow.  
  **Fix:** Override accept() to validate all fields (catching ValueError per field, showing a QMessageBox and keeping the dialog open on invalid input), or wrap each float() conversion and surface a user-facing error rather than letting ValueError escape.

### `lib/engine/flattener.py`

- **Cycle detected during flattening silently drops the connection** — `112-115` _(error-handling)_  
  Verified: lines 113-115 log 'Cycle detected during flattening' and return None; flatten()'s `if driver:` guard at line 59 then omits the line. This is inconsistent with the hardened branches at lines 168-175 and 219-233 which raise RuntimeError on unresolved boundaries. A malformed feedback loop is silently mis-wired (zeros downstream) instead of failing loudly.  
  **Fix:** Raise RuntimeError (or surface a user-visible error) on cycle detection for consistency with the other failure paths, including the offending node path in the message.
- **Inport-to-parent-port resolution can map to wrong port via trailing-digit regex** — `150-163` _(correctness)_  
  Verified: lines 154-156 use `re.search(r'(\d+)$', my_name)` and set found_idx = int(group(1)) - 1, then line 162 uses it directly as the parent input port index. A name like 'Sensor3' maps to port 2 regardless of actual ordering. The Subsystem/Outport branch has a case-insensitive child scan third fallback (lines 205-217); the Inport branch has no equivalent positional/validated fallback.  
  **Fix:** Restrict the regex fallback to conventional names (require In/Inport prefix), validate found_idx against the subsystem's known input count, and prefer an order-based mapping from sub_blocks when ports_map is absent.
- **Top-level Inport returned as connection source but excluded from primitive blocks** — `47, 178-179` _(correctness)_  
  Verified: line 47 excludes 'Inport'/'Outport'/'Subsystem' from primitives; lines 177-179 return `driver` (the top-level Inport name as src) when an Inport has no '/' in its name. flatten() (line 62) then emits a DLine whose src is that Inport name, which was never appended to flat_blocks/primitives -> dangling source reference. Edge case (top-level Inports driving primitives are atypical) but real.  
  **Fix:** Either keep top-level Inports as passthrough primitives, or continue resolving the top-level Inport to whatever drives it, or treat the top-level Inport as an explicit unsupported error so emitted lines never reference dropped blocks.

### `lib/engine/simulation_engine.py`

- **Broad except Exception swallows all replay/solver errors as generic failure** — `1919` _(error-handling)_  
  Lines 1919-1921 wrap the entire run body in `except Exception as e: logger.error(...); return False`, so any internal bug (KeyError, AttributeError, shape mismatch) is reported only as a generic failure, masking the root cause and giving the caller/UI no actionable error. Compounds findings 2 and 8.  
  **Fix:** Narrow the catch (e.g. only solver/numpy errors) or surface the exception type/message into self.error_msg so callers can distinguish solver failure from internal bugs.
- **Algebraic-loop break condition can exit hierarchy resolution leaving blocks uncomputed without error** — `351` _(correctness)_  
  Lines 346-352: when computed_count stalls and detect_algebraic_loops returns False, the code `break`s out of the while loop without verifying that the remaining uncomputed blocks are all legitimate (memory) blocks. Real uncomputed non-memory blocks would silently never execute at init; max_hier/propagation then proceed. No error_msg is set in this path.  
  **Fix:** After break, verify all remaining uncomputed blocks are memory blocks (the only legitimate stall); otherwise set error_msg and return False.

### `lib/engine/symbolic_engine.py`

- **Sum block symbolic_execute mismatch: 'sign' param length may not match input count** — `253-262` _(correctness)_  
  Sum (lines 253-262) iterates `for i, sign in enumerate(signs)` pulling input_exprs.get(i, 0). Any character other than '+' is treated as subtraction (else branch), and inputs at indices >= len(signs) are silently dropped. If signs contains spaces, a count spec, or fewer chars than connected inputs, the symbolic result is wrong.  
  **Fix:** Validate that sign chars are only '+'/'-', skip/ignore other chars, and ensure all connected input ports are covered (or warn on length mismatch).
- **extract_transfer_function divides traced expr by U without verifying linear dependence** — `357-364` _(correctness)_  
  extract_transfer_function sets U=Symbol('U'), traces Y via _get_block_output, then G = simplify(Y / U) (lines 357-364). It assumes Y is linear in U. With other independent sources (Constant returns sympy.Float per lines 328-330) or nonlinear blocks (Saturation returns u, line 335), Y/U produces a U-dependent or incorrect 'transfer function' rather than failing or warning.  
  **Fix:** Verify Y depends only linearly on U (e.g. diff(Y,U) is U-free) or zero out all other inputs before dividing; warn otherwise.

### `lib/engine/system_compiler.py`

- **Wave2D/Advection2D crash if force/source is a 1D array** — `1220-1234` _(error-handling)_  
  Confirmed: in Wave2D, force is only collapsed to scalar when size==1 (1220-1221); for any other ndarray it is indexed as force[j,i] (1234,1246,1258,1270,1282) assuming shape (Ny,Nx). A connected source producing a 1D field (e.g. length Nx) yields wrong indexing or IndexError inside the RHS. Advectionequation2D source has the identical pattern (1337-1338 collapse, then source[j,i] at 1360,1373,1389,1405,1421). Heatequation2D avoids this by always coercing q_src to a scalar via float(q_src.flat[0]) (1117-1118).  
  **Fix:** Before the loop, validate/broadcast force/source to shape (Ny,Nx) (or coerce a scalar), raising a clear error on incompatible shapes instead of blindly indexing.
- **Advection2D Neumann diffusion omits boundary flux term (inconsistent with Heat2D/1D)** — `1371` _(correctness)_  
  Confirmed: Advection2D left-Neumann diffusion uses d2c_dx2 = (c[j,1]-c[j,0])/dx_sq*2 (1371) and right uses (c[j,_Nx-2]-c[j,_Nx-1])/dx_sq*2 (1387), with analogous bottom/top (1404,1420). These drop the prescribed-gradient flux term that Heat2D's ghost-node Laplacian includes, e.g. d2Tdx2 = (2*T[j,1]-2*T[j,0]-2*_dx*bc_left)/dx_sq (1138). Note the advection branch does use bc_left in the advective dc_dx term (1369), so only the diffusive Laplacian ignores the BC gradient, effectively imposing zero-gradient diffusion regardless of bc_left/bc_right. Inconsistent with Heat2D/1D treatment.  
  **Fix:** Include the bc flux term in the Neumann ghost-node Laplacian (matching Heat2D), or explicitly document that advection Neumann diffusion is intentionally zero-gradient.
- **MatrixGain 1-D path silently truncates on length mismatch** — `216-219` _(error-handling)_  
  For 1-D K with K.size>1, when len(K)!=len(u) the executor elementwise-multiplies only the overlapping prefix (K[:min]*u[:min]) and drops the remainder, yielding a wrong-dimension result with no diagnostic.  
  **Fix:** Treat a dimension mismatch as a configuration error (return {'E':True,...} at compile time or log a clear error) instead of silently truncating; a vector gain mismatched against the input is almost always a user mistake.
- **Sum drops connected inputs that exceed the sign string length** — `227-241` _(correctness)_  
  ops is built by iterating only over the sign characters (for i,char in enumerate(signs)). If more input ports are connected than sign chars (e.g. signs='++' but 3 inputs wired), the extra ports are never added to ops and are silently ignored rather than defaulting to '+' or erroring.  
  **Fix:** Iterate over the number of connected input ports and default missing sign chars to '+', or validate len(signs) == number of connected inputs and emit an error if they disagree.
- **Product divide-by-zero replaces the result with magic 1e308** — `579-593` _(correctness)_  
  On a '/' op with val==0 the code does res = res / val if val != 0 else 1e308, substituting an arbitrary huge constant for a true division by zero rather than producing inf/NaN or an error, corrupting the result silently. NOTE: the original finding's claim that it 'discards all previously accumulated factors' is INACCURATE — res is the running accumulator and on the non-zero path it correctly becomes res/val; only the val==0 case overwrites res with the magic constant.  
  **Fix:** On val==0, preserve sign/magnitude semantics (e.g. res = res/(val or eps) or res *= np.inf with sign) or emit an error dict for a true division by zero, instead of substituting a constant. Lowered to medium since accumulation is otherwise preserved.
- **Compiled Noise re-samples np.random.randn() on every RHS evaluation** — `671-677` _(correctness)_  
  exec_noise (675-677) calls np.random.randn() inside the closure run from model_func/the solve_ivp RHS. Noise is therefore redrawn multiple times per accepted step and at non-physical trial t, making noise injected into an integrator ill-defined and non-reproducible versus the interpreted path (blocks/noise.py also unseeded).  
  **Fix:** Document that white noise is unsupported under the adaptive compiled solver, or precompute a time-indexed noise sequence sampled only at fixed step times so the RHS is deterministic in t.
- **Compiled MathFunction expression fallback swallows all per-step errors to 0.0** — `735-740` _(error-handling)_  
  Confirmed. In exec_mathfunc_expr (system_compiler.py:731-740) any exception from evaluating the compiled user expression is caught by 'except Exception as _e' and replaced with signals[b_name] = 0.0, logged only at logger.debug. A consistently-failing expression (typo NameError, KeyError, type error) yields a plausible all-zero signal with no user-visible error. Note the expression IS compiled once at compile time (line 729 compile_expr) via the project's safe-eval path, so this is not a raw-eval security issue, only a swallowed-error/diagnosability issue.  
  **Fix:** Evaluate the compiled expression once on a representative sample input at compile time and raise a hard error if it cannot evaluate (catching NameError/KeyError/SyntaxError), reserving the per-step 0.0 fallback for genuine numeric domain errors (ValueError/ZeroDivisionError/FloatingPointError) as the np_func branch at line 747 already does.
- **Per-timestep PDE interior loops use Python for-loops instead of vectorized NumPy** — `861-863, 933-935, 981-1007` _(performance)_  
  Heat (861-863), Wave (933-935) and Advection (981-998) 1D compute interior finite differences with element-by-element Python loops over N nodes inside the ODE RHS, which the solver calls many times per accepted step. This is O(N) Python overhead per RHS call and dominates runtime for larger N, defeating the compiled fast path.  
  **Fix:** Vectorize with slicing, e.g. dT_dt[1:-1]=_alpha*(T[2:]-2*T[1:-1]+T[:-2])/dx_sq + q_src[1:-1]; analogous slicing for the wave and (second-order upwind) advection stencils, removing the Python loop from the hot RHS.

### `lib/export/blox_exporter.py`

- **_find_feedback_block always returns None, making the entire intermediate-feedback-block branch dead code** — `288-294` _(dead-code)_  
  _find_feedback_block (lines 288-294) unconditionally `return None` (line 294, docstring calls it a 'simplification'), so fb_block is always falsy and the `if fb_block:` branch at lines 161-174 (\bXBranchy/\bXBlocrL/\bXReturn) is unreachable; only the else simple-feedback path (175-181) runs.  
  **Fix:** Implement intermediate-block detection or delete the unreachable branch (161-174) and the stub to reduce misleading maintenance surface.
- **_block_content duplicates tikz_exporter._get_block_content with divergent logic** — `306-356` _(duplication)_  
  _block_content (blox 306-356) copies per-block-type rendering from TikZExporter._get_block_content (tikz_exporter.py 563-641): Gain/TranFn/DiscreteTranFn/Integrator/Deriv/StateSpace/PID/Subsystem are near-verbatim. They already diverge: tikz handles Step/Constant and _name_to_math/_tf_symbols/_gain_symbols fallbacks; blox omits them. Fixes must be applied twice and will drift.  
  **Fix:** Extract shared block-to-LaTeX-content logic into a free function (e.g. in tikz_exporter) and call it from both exporters; pass symbol-map fallbacks as params.
- **Sum 'short string' branch uses bXSumb regardless of actual signs, losing minus signs** — `360-365` _(correctness)_  
  Only '+-' (line 360, \bXComp) and '-+' (line 362, \bXComp*) map to comparators; any other <=2-char combo ('++','--','+','-') hits line 364-365 returning \bXSumb, a plain summing junction with no per-port +/- rendering, silently dropping subtraction semantics for e.g. '--'.  
  **Fix:** Handle '--','++' and single-sign cases explicitly, or route them through \bXCompSum with explicit signs so the comparator matches configured signs.
- **_emit_sum general-case sign computation is incorrect and partly redundant** — `364-375` _(correctness)_  
  In the len>2 fall-through (lines 366-375): the guard `sign_str[0] if len(sign_str) > 2 else ''` (line 368) is always true here, n_sign and w_sign both = sign_str[0] (lines 368,370) duplicating the port-0 sign, s_sign = sign_str[1], e_sign = ''. Signs at index >=2 are dropped, so 3+ input Sum blocks render wrong/missing signs.  
  **Fix:** Map signs to blox compass slots deliberately (e.g. w=sign[0], s=sign[1], n=sign[2]) with documented port-to-anchor order; add a 3-input Sum test.

### `lib/export/latex_exporter.py`

- **Files written without encoding and without error handling/atomicity** — `272-275, 302-305` _(error-handling)_  
  open(filename,'w') at lines 273-274 (export_document) and 303-304 (export_equations_only) use the platform default encoding (cp1252 on Windows); non-Latin1 chars in equations/title raise UnicodeEncodeError. No try/except, so a bad path raises raw OSError to the GUI caller.  
  **Fix:** Use open(filename,'w',encoding='utf-8') and wrap writes in try/except logging a clean error (consider returning a success flag).

### `lib/export/tikz_exporter.py`

- **Constant/Step block values emitted into LaTeX without escaping** — `622-628` _(correctness)_  
  For Step and Constant blocks, the raw 'value' param is interpolated directly into LaTeX: line 625 `return f'Step({val})'` and line 628 `return f'${val}$'`, with no _escape_latex or numeric coercion. If 'value' is a string containing _, %, {}, #, & or expression text, the produced TikZ is malformed/uncompilable, unlike the Sinks branch (line 636) which escapes.  
  **Fix:** Coerce to float and format with :.4g when numeric; for non-numeric values wrap in \text{} and run through _escape_latex as the Sinks branch does.
- **Explicit line label with backslash bypasses LaTeX escaping (injection vector)** — `793-801` _(security)_  
  _format_explicit_label (line 794) returns any user label verbatim when it contains a backslash (lines 799-800: `if '\\' in label: return label`). line.label is a user-supplied connection label (callers at lines 826, 938). A label like '\input{...}' or with unbalanced braces is emitted raw into the generated .tex, a LaTeX injection/breakage vector since the output is meant to be compiled.  
  **Fix:** Only pass through raw LaTeX when fully wrapped in $...$ math mode (the existing first branch at line 797); otherwise always run through _escape_latex. Alternatively restrict raw passthrough to a whitelist of known-safe macros.

### `lib/lib.py`

- **resolve_recursive swallows external-data errors and returns ambiguous False** — `474-483` _(error-handling)_  
  In resolve_recursive, a missing external file (missing_file_flag==1 -> return False at 478-480) and any exception during reload (except Exception -> logger.error + return False at 481-483) both abort init via the caller's return False at 497-498, without setting self.error_msg or calling self.execution_failed. The user sees init fail silently with only a log line.  
  **Fix:** On these failures set self.error_msg (and/or call self.execution_failed(...)) before returning False so the UI can report why init aborted.
- **execution_init top-level except returns False but swallows the real exception** — `525-527` _(error-handling)_  
  The broad except at 525-527 logs only str(e) (no exc_info/traceback) and returns False without setting self.error_msg or calling execution_failed, so caller and UI get no actionable message and the stack trace is lost.  
  **Fix:** Use logger.exception(...) (or exc_info=True) and set self.error_msg = str(e) / call execution_failed so the failure is visible to the user.
- **Block execution exception discarded: execution_failed() called with no message** — `922-925` _(error-handling)_  
  The per-block try/except (922-925) logs str(e) then calls self.execution_failed() with no argument. execution_failed(self, msg='') (line 1038) sets self.error_msg = msg, so error_msg becomes '' and the real cause never reaches the user-facing error panel/status; only the log carries it.  
  **Fix:** Pass the detail: self.execution_failed(f'Error in {block.name}: {e}') so the UI surfaces the actual cause.
- **execution_loop swallows real exception, calls execution_failed() with empty message** — `924` _(error-handling)_  
  Lines 922-924: the per-block except logs the message but calls self.execution_failed() with no argument. execution_failed(self, msg="") (line 1038) sets self.error_msg = msg, so error_msg is cleared to empty. Tuning's run_compiled/interpreter path returns (False, self.error_msg) at line 714-715, so it would report an empty failure cause. Note the outer handler at line 1036 has the same defect.  
  **Fix:** Pass the message through: self.execution_failed(f"Error executing block {block.name}: {e}"); apply the same fix to the outer handler at line 1036.

### `lib/managers/subsystem_manager.py`

- **Subsystem port Y-positions divided by total boundary-line count instead of per-direction count** — `383, 443` _(correctness)_  
  Lines 383 and 443 both use denominator len(boundary_lines)+1, where boundary_lines mixes 'in' and 'out' entries, but inport_idx/outport_idx are independent per-direction counters. With 2 in + 2 out, divisor is 5 instead of 3, so input ports are placed at height/5*1, height/5*2 rather than evenly distributed. Real layout/correctness bug.  
  **Fix:** Compute per-direction counts (num_in = sum(1 for _,d in boundary_lines if d=='in'); num_out symmetrically) and divide by the matching count+1.
- **Unconnected-port spacing denominator double-counts both directions** — `504-505, 565-566` _(correctness)_  
  Line 504 total_inputs = len(boundary_lines) + len(unconnected_inputs) adds ALL boundary lines (including 'out' direction) to the input denominator; line 565 symmetrically inflates the output denominator. Generated Inport/Outport positions are spaced with an inflated, wrong count.  
  **Fix:** Use only same-direction boundary count plus unconnected count: num_in_boundary + len(unconnected_inputs) (+1) for inputs and num_out_boundary + len(unconnected_outputs) (+1) for outputs.

### `lib/models/simulation_model.py`

- **link_goto_from virtual line uses sid=len(line_list), risking duplicate line names/sids** — `295-296` _(correctness)_  
  Line 296 sets sid=len(self.line_list) for the hidden virtual DLine. DLine.name is 'Line'+str(sid) (connection.py:46). After any line deletion, len(line_list) can equal an existing line's sid, producing two lines named 'LineN'; name-based lookups become ambiguous. add_line uses max(sids)+1 instead, confirming the inconsistency.  
  **Fix:** Compute sid = max([l.sid for l in self.line_list]+[-1])+1 mirroring add_line, instead of len(self.line_list).

### `lib/plotting/animation_exporter.py`

- **AnimationExporter.export leaks the figure on the failure path** — `320-358` _(resource-leak)_  
  export() creates the figure via fig, anim = self.create_animation(...) at line 322 inside the try block, and only calls plt.close(fig) at line 349 on the success path after anim.save. If anim.save (line 341 or 347) raises (common for MP4 with ffmpeg errors, disk full, codec issues, or a throwing progress_callback), control jumps to the except at line 354, which logs and returns False at 358 without ever closing fig. The figure (and its colorbar, FuncAnimation, and captured field_data) leaks on every failed export.  
  **Fix:** Use try/finally: assign fig = None before the try, then in a finally block call `if fig is not None: plt.close(fig)` to guarantee cleanup on both success and failure. Remove the success-path-only plt.close at line 349.
- **Matplotlib figure leaked when animation export fails mid-save** — `320-358` _(resource-leak)_  
  fig is created at line 322 via create_animation, but plt.close(fig) only runs on the success path (line 349). If anim.save (341/347) raises, control jumps to except (354-358) which logs and returns False without closing fig, leaking the figure and its agg buffers.  
  **Fix:** Use try/finally so plt.close(fig) always runs, or close fig in the except branch before returning False.

### `lib/safe_eval.py`

- **safe_eval exposes mutable numpy array constructors enabling memory-exhaustion DoS despite Pow guard** — `69-71` _(security)_  
  Lines 69-71 expose np.zeros/ones/eye/identity/full plus arange/linspace (71). The only allocation guard is the Pow exponent cap (lines 299-303, abs(right)>10000). A call like np.zeros(10**9) or np.ones((100000,100000)) passes all checks and triggers unbounded allocation, since the large value is a Call argument not a Pow exponent. safe_expr/CompiledExpr evaluate user-controlled parameter strings, so this is a real memory-exhaustion DoS vector.  
  **Fix:** Wrap allocating constructors (zeros, ones, eye, identity, full, arange, linspace) to cap requested element count against a max-size threshold before delegating to numpy, or validate integer Constant size arguments at the Call site.

### `lib/services/file_service.py`

- **Two divergent save/load code paths with mismatched file extensions (.dat vs .diablos)** — `185-267` _(consistency)_  
  FileService.save forces a .dat extension (lines 206-207) and uses filter 'Data Files (*.dat)' (line 200); FileService.load uses the same .dat filter (line 250). DiagramService (the live UI path) saves/loads .diablos via dsim.serialize/deserialize (diagram_service.py:209, 267, 225/290). Two parallel serialization stacks coexist (FileService.serialize vs dsim serialize/deserialize) and risk drifting apart; the .dat extension/filters confuse which is canonical.  
  **Fix:** Pick one canonical save/load path and extension; if FileService.save/load is dead/legacy, remove it, otherwise unify extension and filters to .diablos. Note DiagramService.load_diagram already accepts *.dat among its filters, so they are not fully consistent.
- **Untrusted .diablos load performs no schema/version validation or structural checks** — `257-267` _(security)_  
  load() (lines 257-267) does json.load(fp) and returns data with no version gate or structural validation. _construct_block accesses required keys with [] not .get(): block_data['block_fn'] (355), ['coords_left'/'coords_top'/'coords_width'/'coords_height'] (356-361), block_data['sid'] (391), ['in_ports']/['out_ports'] (394-395). _construct_line uses line_data['points'] (507), ['sid'] (513), ['srcblock'] (509), ['srcport'] (514), ['dstblock'] (510), ['dstport'] (516). A malformed/hostile file therefore raises KeyError mid-reconstruction (after model.clear_all() at 283, leaving a half-cleared model) rather than failing gracefully. Severity lowered to medium: this is a local desktop tool loading user-chosen files, not a network-facing parser, so the practical threat is robustness/crash rather than RCE.  
  **Fix:** Validate top-level shape and a version field after json.load; use block_data.get(...) with defaults or per-record try/except so one bad block/line is skipped (logged) rather than crashing the whole load and leaving the model half-cleared.

### `lib/simulation/connection.py`

- **__deepcopy__ swallows all exceptions and silently sets attributes to None, masking copy failures** — `461-465` _(error-handling)_  
  Lines 463-465 catch every Exception, bind it to 'e' but never use or log it, and replace the attribute with None. A failed deepcopy of any real attribute (points, color, label, etc.) yields a corrupted DLine with None fields during copy/paste/undo with zero diagnostics.  
  **Fix:** Catch narrowly or at minimum logger.warning the attribute name k and the exception, so silent data loss is observable. Do not swallow without logging.

### `modern_ui/interactions/interaction_manager.py`

- **Broad except Exception in mouse-move/press/release handlers swallows errors without traceback** — `223-224` _(error-handling)_  
  Confirmed. handle_mouse_move wraps its entire body in try/except (lines 130-224) ending with `except Exception as e: logger.error(f"Error in handle_mouse_move: {str(e)}")` at 223-224 with no exc_info. handle_mouse_release does the same at 251-252. Mouse-move fires on every motion event during a drag, so any exception (e.g. an IndexError in the segment logic at 188-215, or operating on a stale dragging_item) is logged once per event with no stack trace, leaving the canvas mid-drag and the root cause hard to find.  
  **Fix:** Add exc_info=True to the logger.error calls, narrow the try to the operations that can fail, and reset interaction state to IDLE on exception so the canvas is not left mid-drag.
- **InteractionManager initializes a full set of instance attributes that are never read (dead state, shadows real canvas state)** — `30-50` _(dead-code)_  
  Confirmed. __init__ (lines 30-50) sets drag_start_pos, last_mouse_pos, connection_start_block/port, connection_end_block/port, temp_connection_points, dragging_block, dragging_line, dragging_segment_index, dragging_point_index, drag_offset, resizing_block, resize_handle, original_block_rect on self. A grep of the whole file shows none of these are ever read in any handler; drag_start_pos/last_mouse_pos are also written in handle_mouse_press (lines 74-75) but never read. The handlers instead read the canvas-owned state: self.canvas.dragging_block (line 150/164/169), self.canvas.drag_offset (lines 152-153), self.canvas.drag_offsets (167-172), self.canvas.dragging_item (180/186/243), self.canvas.resizing_block (177). modern_canvas.py defines these as the real source of truth (dragging_block at line 100; drag_offset/drag_offsets/resizing_block as properties at lines 1393-1422). So the manager's instance attributes are dead duplicates and a genuine maintainability trap.  
  **Fix:** Delete the unused instance attributes so canvas state is the single source of truth, or consolidate ownership into the manager. If any are kept intentionally, add a comment; removal is cleaner.

### `modern_ui/main_window.py`

- **Pervasive lib/ <-> modern_ui/ circular-import risk worked around with function-local imports** — `66-100, 348-374, 623` _(design)_  
  MainWindow __init__ performs 10+ function-local manager imports (66-100) and methods import widgets lazily (e.g. tikz_export_dialog at 352, WaveformInspector path at 373-375). A genuine cycle exists: lib imports modern_ui at module scope (signal_plot.py:21 'from modern_ui.themes.theme_manager import ...') and lazily in scope_plotter.py, diagram_service.py, simulation_model.py, menu_block.py. The deferred-import pattern masks the dependency direction and the lib->modern_ui violation.  
  **Fix:** Establish strict direction (lib must not import modern_ui), lift imports to module scope where no cycle exists, and break remaining cycles via dependency inversion (pass collaborators in) rather than scattering local imports.
- **Auto-save silently fails in frozen mode due to unnecessary relative os.makedirs('config')** — `784-811` _(correctness)_  
  _auto_save() (main_window.py line 792) calls os.makedirs('config', exist_ok=True) with a relative path (resolved against the cwd), but the actual save target is self.autosave_path (line 802), set at line 192 to user_data_path('config/.autosave.diablos') whose parent directory is already created by user_data_path (lib/app_paths.py line 55). The relative 'config' dir is never used. In frozen mode where cwd may be read-only (e.g. '/' on macOS launched from Finder), os.makedirs('config') raises OSError, caught by the broad except Exception at line 810 and merely logged, aborting the entire auto-save. Net effect: the 2-minute autosave (timer at lines 188-190) silently never completes in packaged builds, defeating crash recovery.  
  **Fix:** Remove the os.makedirs('config', exist_ok=True) line; self.autosave_path comes from user_data_path() which already creates its parent. If a guard is still wanted, base it on os.path.dirname(self.autosave_path).

### `modern_ui/managers/`

- **Manager layer is fragmented: ~18 managers each tightly back-coupled to a single parent (distributed god object)** — `connection_manager.py:24-26; simulation_actions_manager.py:23-24` _(design)_  
  modern_ui/managers contains 18 manager classes (glob confirmed). ConnectionManager.__init__(self,canvas) stores self.canvas/self.dsim (24-26). SimulationActionsManager.__init__(self,main_window) stores self.window and reaches into window.canvas/window.error_panel/window.status_message via hasattr/getattr (23-44). modern_canvas.py is 1529 lines, window_setup_manager 94 lines. This is method-extraction off ModernCanvas/MainWindow with full back-references, not narrow-interface decomposition; managers are hard to test in isolation.  
  **Fix:** Pass narrow protocols/collaborators each manager actually needs instead of the whole canvas/window; consolidate trivially small managers and move shared state into explicit models rather than reaching through the parent.

### `modern_ui/managers/clipboard_manager.py`

- **ClipboardManager reads dsim.connections_list while every other manager uses line_list — relies on a fragile alias** — `78-81, 244, 262` _(maintainability)_  
  copy_selected_blocks reads self.dsim.connections_list (lines 78, 81) and paste_blocks reads (line 244) and appends to it (line 262), whereas connection_manager, rendering_manager, selection_manager and history_manager all use self.dsim.line_list (verified via grep). connections_list is only an alias assigned via self.connections_list = self.line_list in lib/lib.py (line 90) and must be manually re-synced after every reassignment of line_list (re-sync lines exist at lib/lib.py 275, 379, 552, 687). tasks/lessons.md (section 'Copy/Paste Connections Lost Due to Stale Alias', Feb 2026) documents a production data-loss bug caused by exactly this: after remove_block_and_lines/clear_all rebound line_list, connections_list pointed at a stale empty list and copy silently produced zero connections. ClipboardManager being the sole alias consumer keeps the codebase one missed re-sync away from regressing.  
  **Fix:** Switch ClipboardManager to use self.dsim.line_list consistently (lines 78, 81, 244, 262), removing dependency on the connections_list alias and eliminating the recurring stale-alias hazard.

### `modern_ui/managers/history_manager.py`

- **Undo/redo snapshot stores params with a shallow copy, aliasing mutable parameter values across states** — `107` _(correctness)_  
  _capture_state stores 'params': block.params.copy() at line 107 -- a SHALLOW copy. Blocks commonly hold mutable values in params (numpy arrays, lists, dicts) per the state-in-params contract. The shallow copy shares nested objects with the live block. On _restore_state, block_data['params'] is passed positionally into DBlock at line 186; DBlock.__init__ then does self.params = params.copy() (block.py line 90) -- again shallow -- so nested mutables remain shared between snapshots and the restored block. Consequences: in-place edits to a block parameter can mutate historical snapshots, and editing a restored block can corrupt redo state. ClipboardManager uses copy.deepcopy(block.params) at line 49 for the same data, confirming the inconsistency.  
  **Fix:** Use copy.deepcopy(block.params) when capturing state (matching ClipboardManager line 49) so undo/redo snapshots are fully independent of live block state and of each other.

### `modern_ui/managers/project_manager.py`

- **Two independent recent-files stores both clear and repopulate the same recent_files_menu** — `61-92` _(consistency)_  
  ProjectManager.update_recent_files_menu (lines 61-81) clears `self.window.recent_files_menu` and repopulates it from a QSettings-backed list (`self.settings.value("recent_files")`). It is invoked via add_recent_file (line 92), which is called from open_diagram (line 43). open_diagram is the handler wired to the user's Open action (main_window.py line 338 routes `open_diagram` -> `project_manager.open_diagram()`). Separately, RecentFilesManager.update_menu (recent_files_manager.py lines 92-115) clears the SAME `self.window.recent_files_menu` and repopulates it from a different JSON store (`config/recent_files.json`); this is what MenuBuilder calls via `_update_recent_files_menu` (menu_builder.py line 44-45 -> main_window line 773 -> recent_files_manager.update_menu). Both target the identical menu widget (`recent_files_menu`, created once in menu_builder.py line 43). So after Open, the QSettings list is shown; any RecentFilesManager refresh overwrites it with the JSON list, and vice versa — nondeterministic, store-dependent menu contents. RecentFilesManager's own docstring (lines 13-16) claims ProjectManager's helpers "are not wired into the menu", which is contradicted by update_recent_files_menu writing directly into that menu widget. Confirmed.  
  **Fix:** Pick a single source of truth. Either delete ProjectManager's QSettings recent-files code (update_recent_files_menu, add_recent_file) and route open_diagram/save_diagram through `window.recent_files_manager.add(...)`, or remove RecentFilesManager. Two stores writing one menu is a latent bug; also correct/remove the now-false docstring claim.

### `modern_ui/renderers/canvas_renderer.py`

- **draw_grid mutates painter pen/brush without save/restore** — `17-60` _(correctness)_  
  draw_grid (lines 27-60) wraps its body only in try/except, with no painter.save() at the top nor painter.restore(). It sets painter.setPen(Qt.NoPen) (line 45) and painter.setBrush(...) (lines 46, 52) and leaves them set, so subsequent draws in the same paintEvent inherit NoPen and the dot brush. This is inconsistent with draw_selection_rect (67-88) and draw_temp_line (98-134), which both save()/restore().  
  **Fix:** Wrap the body in painter.save() ... finally: painter.restore() like the sibling renderers so leaked pen/brush state cannot affect later draws.
- **draw_grid is an unbounded O(W*H) per-frame hot path** — `47-57` _(performance)_  
  Lines 47-50 and 55-57 nest two range() loops over the entire visible scene rect at 20px (then 100px) steps, calling painter.drawEllipse per dot. When zoomed out the visible scene rect is large, so dot count grows with rect_area/400, producing many primitive calls per repaint and stalling pan/zoom.  
  **Fix:** Cap small-dot density at low zoom (skip them when the rect is large), or batch with painter.drawPoints/QPolygon or a cached tiled QPixmap brush instead of per-dot drawEllipse.

### `modern_ui/styles/qss_styles.py`

- **Duplicate, divergent menubar/menu QSS in get_menubar_style() vs AppearanceManager.update_menubar_colors()** — `426-469` _(duplication)_  
  ModernStyles.get_menubar_style() (qss_styles.py:426-469) defines a full QMenuBar/QMenu stylesheet using @surface_elevated for selected items, padding 6px 10px, QMenu border-radius 8px, and QMenu::item[role="danger"] red-text rules (lines 462-464). appearance_manager.py:128-180 update_menubar_colors() hand-rolls a SECOND, divergent menubar+menu stylesheet (accent at 30% alpha for selected, padding 4px 10px, border-radius 4px, NO danger rules) and applies it DIRECTLY to window.menuBar() (line 180) on every theme change (called from appearance_manager.on_theme_changed:94, triggered by toolbar theme_toggled -> main_window.on_theme_changed). Because a widget-specific stylesheet on the menuBar takes precedence over the cascaded QMainWindow stylesheet from get_complete_stylesheet, the appearance_manager version effectively wins for the menubar after any theme switch. The danger styling is genuinely reachable and lost: menu_builder.py:53-54 sets exit_action.setProperty('role','danger'), and the danger QSS exists ONLY in get_menubar_style. So the Exit/Close 'danger' red styling and elevated-surface look are silently dropped on theme change.  
  **Fix:** Single-source the styling: have AppearanceManager.update_menubar_colors() call ModernStyles.get_menubar_style() (which already resolves theme variables) instead of building a parallel stylesheet, or remove update_menubar_colors and rely solely on the per-window stylesheet application, ensuring the menubar-specific stylesheet is not overridden.

### `modern_ui/widgets/modern_toolbar.py`

- **set_status status-pill logic has operator-precedence bug; 'paused' never wins over 'running'** — `559-567` _(correctness)_  
  In set_status (compatibility shim) the condition `if 'run' in m or 'simulat' in m and 'pause' not in m:` parses by Python precedence (and binds tighter than or) as `'run' in m or ('simulat' in m and 'pause' not in m)`. The `and 'pause' not in m` guard only applies to the 'simulat' half. Any message containing the substring 'run' (e.g. 'Run paused') is classified as 'running', so the elif `'paus' in m` paused branch is unreachable for those messages. Note also a token mismatch: the guard tests 'pause' while the paused branch tests 'paus'.  
  **Fix:** Check paused first and parenthesize: `if 'paus' in m: set_state('paused')` then `elif 'run' in m or 'simulat' in m: set_state('running')`.

### `modern_ui/widgets/property_editor.py`

- **Uncaught TypeError when numeric field contains a nested or non-numeric list** — `1057-1060` _(error-handling)_  
  _validate_and_submit_numeric: ast.literal_eval parses '[[1,2]]' to a nested list, then line 1057 does [float(x) for x in val]; float([1,2]) raises TypeError, but line 1060 catches only (ValueError, SyntaxError). The TypeError escapes the editingFinished handler unhandled.  
  **Fix:** Add TypeError to the except tuple: except (ValueError, SyntaxError, TypeError), so malformed list elements surface as an inline validation error.

### `modern_ui/widgets/variable_editor.py`

- **Raw exec() of user-entered code in VariableEditor with non-isolating sandbox** — `235` _(security)_  
  update_workspace runs editor text via exec(code, {}, local_vars) at line 235. Empty globals do not strip __builtins__ (Python re-injects it), so arbitrary code including imports and OS/file access executes. This bypasses the project-provided lib/safe_eval.py for user input.  
  **Fix:** Use safe_eval/AST-validated assignments, or at minimum pass globals with {'__builtins__': {}} plus a curated whitelist, and gate the feature as trusted-only.

### `modern_ui/widgets/waveform_inspector.py`

- **CSV export crashes on column_stack when a trace is shorter than the timeline** — `262-270` _(error-handling)_  
  _export_csv builds data_cols starting with the full self.timeline (line 262) then appends each trace y truncated only on the long side via `y = y[:len(self.timeline)]` (line 266). If a trace is shorter than the timeline, truncation is a no-op and `np.column_stack(data_cols)` (line 269) raises ValueError on unequal column lengths. _rebuild_traces (lines 126-127) picks the longest run's timeline, so traces from shorter runs are routinely shorter; _refresh_curves (lines 177-180) explicitly handles `len(x) < len(y)` and `len(x) > len(y)`, confirming shorter traces are anticipated. The export has no try/except, so the exception escapes.  
  **Fix:** Align all columns to a common length (truncate timeline and each y to the min length, or right-pad short traces with NaN) before column_stack, and wrap the stack/savetxt in try/except to show a QMessageBox on failure.


## Low severity (215)


### `blocks/agent_scope.py`

- **AgentScope/XYGraph collect unbounded per-timestep history with no cap** — `89` _(resource-leak)_  
  blocks/agent_scope.py:89-90 appends u.copy() and time to _pos_history_/_time_history_ every timestep with no cap; XYGraph (xygraph.py:94-95) does the same for _x_data_/_y_data_. trail_length only limits the rendered trail, not stored raw history, so long runs grow memory unboundedly. This is recorder-style behavior intrinsic to a history-capturing sink (Scope behaves similarly), so it is a documentation/design note rather than a true leak; downgraded from medium to low.  
  **Fix:** Document the intended unbounded-history behavior, and optionally add a max-samples/decimation cap for very long runs to bound memory.

### `blocks/assert_block.py`

- **Assert error message formats time with :.4f, crashes on non-float time** — `93` _(error-handling)_  
  assert_block.py:93 uses f"...time={time:.4f}". The engine passes a scalar float time in normal operation, but if time were ever a numpy array/multi-element value the format spec would raise TypeError, masking the real assertion failure. Defensive nit, low likelihood.  
  **Fix:** Coerce defensively, e.g. time={float(np.atleast_1d(time)[0]):.4f}, so the assertion message is always produced.

### `blocks/bodemag.py`

- **BodeMag.execute returns inconsistent result shape ({'E': False}) versus sibling analysis blocks ({})** — `37-41` _(consistency)_  
  bodemag.py:41 returns {'E': False}, whereas rootlocus.py:71, nyquist.py:39, and bodephase.py:39 all return {}. All four declare no output ports (outputs == []) and are no-ops during simulation, so neither form supplies a port-0 value; the difference is a pure consistency wart. Note {'E': False} is harmless — execute_block (simulation_engine.py:467) only treats the dict as an error when 'E' is truthy, so 'E': False is equivalent to no error.  
  **Fix:** Return {} from BodeMag.execute to match RootLocus/Nyquist/BodePhase, or standardize all four analysis blocks on a single return convention.
- **Inconsistent error-dict contract across analysis blocks** — `41` _(consistency)_  
  BodeMag.execute returns {'E': False} (bodemag.py:41) while sibling analysis blocks RootLocus (rootlocus.py:71), Nyquist (nyquist.py:39), and BodePhase (bodephase.py:39) return a bare {}. Note Export (export.py:78,105) also returns {0:..., 'E': False}, so the {'E': False} shape is not unique to BodeMag, but the analysis-block family is genuinely inconsistent (3 return {} vs BodeMag returns {'E': False}).  
  **Fix:** Standardize the no-op analysis-block return value (all {} or all {'E': False}) so the engine treats this block family uniformly.

### `blocks/derivative.py`

- **Derivative block divides by dt with only an exact-equality guard against zero** — `99-104` _(error-handling)_  
  Confirmed code-wise: blocks/derivative.py:99 guards 'if time == params["_t_old_"]' with exact equality, then line 102 computes dt = time - _t_old_ and line 104 didt = di/dt with no tolerance or zero-check. If time were ever extremely close to but not equal to _t_old_, dt would be a tiny denominator producing a huge spurious spike. In practice the engine advances time on a fixed dt grid so this is largely theoretical; severity lowered from medium to low.  
  **Fix:** Optionally harden with a tolerance: 'if abs(time - params["_t_old_"]) < 1e-12: return {0: np.array(params["_didt_old_"])}' before dividing, to be robust to any near-equal repeated timestamps.

### `blocks/display.py`

- **Display swallows formatting errors silently and returns raw value** — `88` _(error-handling)_  
  blocks/display.py:88-89 `except (ValueError, TypeError): formatted = str(input_value)` falls back with no logging, hiding malformed format strings or unexpected input types; the displayed value silently becomes the raw repr. (Aside: the default format is printf-style '%.3f' applied via `fmt % v`, while the docstring example shows '{:.2f}' Python-format syntax -- a separate doc mismatch.)  
  **Fix:** Add a logger.debug/warning on the fallback path so format/type problems are diagnosable; consider also reconciling the docstring's '{:.2f}' example with the actual printf-style formatting.

### `blocks/exponential.py`

- **Exponential block accesses inputs[0] with bare indexing, raising unhandled KeyError on missing input** — `47-48` _(error-handling)_  
  execute() (line 48) is a one-liner: return {0: params['a'] * np.exp(np.clip(params['b'] * inputs[0], -700, 700))}. It reads inputs[0] with hard subscription and params['a']/params['b'] with hard keys, and there is no try/except. If port 0 is missing from the inputs dict this raises a raw KeyError that propagates out of execute() instead of returning the contract-mandated {'E': True, 'error': msg} dict. Confirmed that sibling blocks differ: Abs (abs_block.py:45) uses inputs.get(0, 0); MathFunction (math_function.py:64) uses inputs.get(0, ...) plus try/except; Sum/Product wrap their body in try/except returning an error dict. So the inconsistency and contract deviation are real. Severity lowered to low: Exp has a single required input port (inputs = [{'name':'x'}]) with requires_inputs defaulting True, so in normal connected operation port 0 is present every timestep; the KeyError only manifests on a misconfigured/unconnected diagram, making the practical risk moderate rather than a likely runtime crash.  
  **Fix:** Use inputs.get(0, 0.0) and params.get('a', 1.0)/params.get('b', 1.0) (mirroring Abs/MathFunction), and/or wrap the body in try/except returning {'E': True, 'error': str(e)} to match the BaseBlock error-return convention used by Sum/Product/Gain.

### `blocks/export.py`

- **Export else-branch assumes params['vector'] exists; KeyError if _init_start_ is loaded as False** — `102` _(error-handling)_  
  export.py:102 does aux_vector = params['vector'] with a direct subscript, only safe if the init-branch (export.py:81-100, which sets params['vector']) ran first. The init-branch is entered when params.get('_init_start_', True) is truthy. The Export param default is _init_start_=True (export.py:30), and reset_memblocks (simulation_engine.py:693-698) forces _init_start_ back to True (in both params and exec_params) whenever the key is present, which it always is. So under normal operation the init-branch always runs first and 'vector' is set. A KeyError requires a persisted/hand-edited param set with _init_start_==False and no 'vector' key. No example .diablos file persists _init_start_, and FileService does not special-case it, so this is a contrived/defensive concern rather than a normal-path bug.  
  **Fix:** Use params.get('vector') and fall back to re-initialization when it is missing or not a valid 1-D/2-D ndarray, consistent with ScopeBlock's defensive handling.
- **Export does not guard against per-sample dimension changes between steps, producing ragged/invalid arrays** — `103` _(correctness)_  
  export.py:102-103 does aux_vector = params['vector']; aux_vector = np.concatenate((aux_vector, [inputs[0]])) with no check that the new sample's dimension matches the first sample that set vec_dim. Scope (blocks/scope.py:134-139) explicitly guards this: it computes expected_dim = params.get('vec_dim', ...) and re-initializes the buffer when new_sample.size != expected_dim. Export has no such guard. If the upstream signal width changes between time steps, np.concatenate on mismatched-shape 2-D arrays raises ValueError, or for a 1-D vs 2-D mix produces a buffer whose layout no longer matches vec_dim, breaking lib/lib.py:1190-1191 (vector[:, i] requires a 2-D (N, vec_dim) array). Note this only manifests if signal width actually varies mid-run, which is uncommon in practice; hence low rather than medium.  
  **Fix:** Mirror Scope's guard at scope.py:134-139: compare the new sample's size to params['vec_dim'] and re-init (or return an error dict) on mismatch instead of blindly concatenating. Coercing inputs through a helper like Scope's _coerce() would also make scalar-vs-array inputs consistent.
- **Duplicated sink accumulation logic between Export and Scope blocks** — `71-105` _(duplication)_  
  ExportBlock.execute (export.py:71-105) and ScopeBlock.execute (scope.py:76-141) implement the same shape of algorithm: the _skip_ RK45 guard, _init_start_ vector initialization, label parsing/padding, and per-step concatenation into params['vector']. They were clearly copied, and Scope has since accreted fixes Export never received (the stale-ndarray re-init guard at scope.py:103-105, the dimension-change guard at scope.py:134-139, and None/0-d coercion via _coerce at scope.py:86-98). The dimension-change divergence (finding #2) and the missing-guard divergence are real consequences. Correction: the stale-Python-list crash attributed to this duplication (finding #1) is NOT a real consequence for Export, because the fast-path replay loop never writes a list to Export's vector.  
  **Fix:** Extract the shared skip-guard + init + label-parsing + concatenation logic into a common helper or mixin used by both ScopeBlock and ExportBlock so future fixes apply uniformly.
- **Export reads inputs[0] without guarding missing/None input** — `82` _(error-handling)_  
  On init, export.py:82 `np.array([inputs[0]])` and line 84 `len(inputs[0])` index port 0 directly. The try/except only wraps `len(inputs[0])` catching TypeError (line 85); a missing key (KeyError) at line 82 propagates uncaught, never converted to the {'E': True, 'error': ...} contract. Realistically the engine wires an input before execute for a connected sink, so KeyError is an edge case (e.g. dangling port).  
  **Fix:** Validate inputs.get(0) is not None and return {'E': True, 'error': 'Export has no input'} otherwise, instead of letting an unguarded KeyError propagate.
- **Label backfill produces wrong/duplicate names (off-by index)** — `93` _(correctness)_  
  export.py:92-94: when fewer labels than vec_dim are supplied, the backfill suffix uses str(params['vec_dim'] + i - 1). E.g. vec_dim=3, labels=['x']: loop i in range(2) appends suffixes '2' and '3', yielding ['x','name-2','name-3'] — the appended columns are indices 1 and 2 but get labels 2 and 3, mismatched and potentially misleading.  
  **Fix:** Use the true column position, e.g. str(len(labels) + i) captured before the loop, so labels map 1:1 to vector components.

### `blocks/fft.py`

- **FFT/Assert/Term/Outport declare empty outputs but execute returns port 0** — `89` _(consistency)_  
  Confirmed inconsistency across sinks: assert_block.py returns {0: np.array([0.0])} (plus error dict on failure) while declaring outputs==[]; fft.py:89 returns {'E': False}; terminator.py:48 returns {}. Returning a port-0 value from a zero-output block is dead data and inconsistent with Terminator. Harmless but untidy.  
  **Fix:** Standardize sink returns: zero-output sinks should return {} (or {'E': False}); drop the spurious {0: ...} from Assert.

### `blocks/from_block.py`

- **From/Goto execute returns scalar 0 default instead of array on missing input** — `57` _(consistency)_  
  from_block.py:57 (and goto.py:55) return {0: inputs.get(0, 0)} with a Python int 0 default rather than a numpy array. If the virtual routing line is absent, downstream receives a bare int. These are intentional pass-through routing blocks so passing the value untouched is defensible, making this a minor consistency nit.  
  **Fix:** Return {0: np.atleast_1d(inputs.get(0, 0.0)).astype(float)} for shape/dtype consistency, or document that routing blocks pass values through untouched.

### `blocks/gain.py`

- **Gain imports safe_literal but uses raw ast.literal_eval in execute()** — `144-149` _(consistency)_  
  gain.py imports safe_literal/SafeEvalError from lib.safe_eval (line 5) and uses safe_literal in symbolic_execute (line 104). But execute() at lines 144-149 does `import ast` inside the function and parses the user gain string with ast.literal_eval(K_raw), catching (ValueError, SyntaxError). ast.literal_eval only evaluates Python literals so it is not a security hole (no arbitrary eval), confirming the finding's own caveat. The two real points stand: it is inconsistent with the project's safe_eval wrappers used elsewhere in the same module, and it performs a per-call `import ast` inside execute() (though after first import this is a cheap sys.modules dict lookup, so the perf cost is negligible). matrix_gain.py uses a module-level `import ast` (line 3) instead.  
  **Fix:** Use the already-imported safe_literal(K_raw) (catching SafeEvalError) in execute() for consistency with symbolic_execute and the rest of the codebase, and drop the inline `import ast`. Note this is purely a consistency/style cleanup, not a security fix, since ast.literal_eval is already safe.
- **Dead/unreachable branch in Gain vector-gain logic (len(K) == 1 inside len(K) > 1)** — `164-173` _(dead-code)_  
  At gain.py:164 the branch is guarded by `elif K.ndim == 1 and len(K) > 1`. Inside it (lines 166-173) the chain tests `if len(K) == len(u)`, then `elif len(K) == 1` (line 168) — which can never be true because the enclosing guard already requires len(K) > 1 — then an else (lines 170-173) that silently truncates both K and u to min(len(K), len(u)) on a dimension mismatch. Both claims verified: the elif is unreachable dead code, and the else silently produces a wrong-but-finite result with only no warning/error on a true vector/input length mismatch.  
  **Fix:** Remove the unreachable `elif len(K) == 1` branch, and on the size-mismatch else emit a logger.warning or return {'E': True, 'error': ...} instead of silently truncating, so dimension errors are surfaced rather than masked.

### `blocks/hysteresis.py`

- **Hysteresis output-only path can return undefined state before init** — `73-74, 78-86` _(correctness)_  
  Line 73-74: when 0 not in inputs, returns params.get('_state', low) with no init. The _init_start_ initialization (lines 78-86) only runs on the input-present path. So an output-only probe occurring before the first real input step returns the 'low' baseline regardless of what the first input would have established, which can be a wrong baseline.  
  **Fix:** Initialize _state independently of the input-present branch (e.g. set a default in params or run init unconditionally) so the held state is always defined.
- **Hysteresis init branch duplicates threshold logic and forces low state in the ambiguous band** — `78-91` _(duplication)_  
  Confirmed. blocks/hysteresis.py:78-86 (the _init_start_ branch) computes u>=upper / u<=lower / else, and lines 88-91 immediately repeat the same u>=upper / u<=lower comparison. The only non-redundant effect of the init branch is setting the dead-band default _state=low (lines 84-85); the high/low cases are recomputed unconditionally right after. This is harmless but redundant and obscures intent.  
  **Fix:** In the init branch set only the dead-band default (params['_state'] = float(params['low'])) and let the single comparison block at lines 88-91 handle high/low transitions, removing the duplicated comparison.

### `blocks/impulse.py`

- **Impulse fires on first step with time >= delay, losing exact edge timing** — `83` _(correctness)_  
  When delay falls between grid points, the impulse fires on the first sample at-or-past delay (line 83 'time >= delay'), so the spike lands late by up to one dt; combined with value/dt this can place energy at a slightly wrong instant for response identification.  
  **Fix:** Consider firing on the sample nearest delay (|time-delay| < dt/2) or document the half-open right-edge convention, keeping it consistent with step.py:91.

### `blocks/integrator.py`

- **Integrator SOLVE_IVP path ignores selected ODE method and re-creates closure per timestep (per-step overhead)** — `181` _(performance)_  
  Lines 181-184 define a fresh `fun` closure and call scipy.integrate.solve_ivp over a single [time, time+dtime] interval every timestep, even though `fun` returns a constant (np.atleast_1d(inputs[0]).flatten()) over the step, reducing analytically to forward Euler (mem += dtime*input). SOLVE_IVP is the block default (line 39), so this runs on the per-timestep hot path for every Integrator using the default method, adding real overhead.  
  **Fix:** For constant-over-step input integrate analytically (mem += dtime*input) and reserve solve_ivp for genuinely intra-step-varying inputs; at minimum avoid recreating the closure each step.
- **Integrator mem_list trimming uses hardcoded -5 slice instead of mem_len param** — `194` _(maintainability)_  
  Line 110 sets params['mem_len']=5.0 and line 193 checks `if len(aux_list) > params['mem_len']`, but line 194 trims with the hardcoded `aux_list[-5:]`. If mem_len were ever changed, the guard and the trim would silently disagree.  
  **Fix:** Use `aux_list = aux_list[-int(params['mem_len']):]` so the trim length tracks the parameter.

### `blocks/logical_operator.py`

- **LogicalOperator XOR over >2 inputs computes parity, not exclusive-or** — `96-99` _(correctness)_  
  logical_operator.py:96-99 chains np.logical_xor across N inputs, yielding odd-parity semantics for N>2. This is the standard cascaded-XOR result and matches most tools, but it differs from a possible 'exactly one true' expectation and is undocumented for N>2. Debatable/documentation issue rather than a clear bug.  
  **Fix:** Document the parity semantics in the block doc/, or restrict XOR to two inputs to avoid surprising multi-input results.

### `blocks/matrix_gain.py`

- **Dead/unreachable branch in MatrixGain vector-gain logic (duplicated from Gain)** — `102-109` _(dead-code)_  
  matrix_gain.py:102 guards with `elif K.ndim == 1 and len(K) > 1`, and line 105 contains the unreachable `elif len(K) == 1`, with lines 107-109 silently truncating to min length on mismatch — identical to gain.py:164-173. Verified that MatrixGain.execute (lines 80-116) is a near-verbatim copy of GainBlock.execute (gain.py:135-181): same input coercion, same matrix/vector/scalar dispatch, same pad/truncate logic, same (ValueError, TypeError) handler. So both the unreachable branch and the duplication claim hold.  
  **Fix:** Remove the unreachable `elif len(K) == 1` branch and warn/error on mismatch. Extract the shared scalar/vector/matrix gain application into a single helper (e.g. in input_helpers) reused by GainBlock and MatrixGainBlock to eliminate the duplicated logic and the duplicated dead branch.

### `blocks/noise.py`

- **Noise block documents a Seed parameter that does not exist and is never used** — `35, 41-45, 70-73` _(maintainability)_  
  doc (line 35) advertises 'Seed: Random seed for reproducibility (0 = random).' but params (41-45) defines only mu and sigma, and execute (70-73) calls np.random.randn() with no seeding. Runs are never reproducible despite the documented contract.  
  **Fix:** Either add a 'seed' param and seed a local np.random.Generator on init (stored in params), or remove the Seed line from the doc.

### `blocks/optimization/constraint.py`

- **Constraint equality constraint value ignores configured tolerance** — `222-225` _(consistency)_  
  Confirmed: get_constraint_value() (lines 222-225) returns `('eq', signal - bound)` for '==' constraints with no tolerance band, while _compute_violation() (lines 179-181) uses `max(0, abs(signal - bound) - tolerance)` which honors params['tolerance']. The two paths that interpret the same '==' constraint disagree about whether the tolerance applies, so the penalty-based path (get_penalty -> _compute_violation) can report a constraint satisfied within tolerance while the scipy 'eq' path treats the same point as a strict equality violation.  
  **Fix:** Pick one tolerance semantics and apply it in both get_constraint_value() and _compute_violation(); e.g. when tolerance > 0 relax the eq constraint to two inequalities |signal-bound| <= tolerance, or drop the tolerance band from _compute_violation for consistency.

### `blocks/optimization/data_fit.py`

- **Unused dtime variable read in DataFit.execute (dead code)** — `148` _(dead-code)_  
  Confirmed: line 148 `dtime = float(params.get('dtime', 0.01))` is read in DataFitBlock.execute() but never used anywhere in the method body (the MSE/MAE/RMSE/R2 metrics are point averages divided by n_points, not dtime-weighted integrals). Misleading leftover that implies time-integration semantics the block does not implement.  
  **Fix:** Remove the unused dtime read, or if integral-style error was intended, actually weight the accumulation by dtime.
- **CSV loader assumes signal column is always index 1 when time_col is numeric** — `251-255` _(correctness)_  
  Confirmed: in _load_data() lines 251-253, when time_col.isdigit() the code sets `signal_idx = int(signal_col) if signal_col.isdigit() else 1`. So a numeric time_col paired with a NAMED signal_col (e.g. time_col='0', signal_col='voltage') silently reads positional column index 1 instead of the named column, loading the wrong data with no error. The both-numeric and both-named branches are handled, but the mixed numeric-time/named-signal case is wrong.  
  **Fix:** Resolve time and signal columns independently: for each spec, use positional indexing if it is a digit, otherwise use the DictReader name lookup, rather than coupling signal_idx resolution to whether time_col is numeric.

### `blocks/optimization_primitives/vector_sum.py`

- **Empty __init__ overrides that only call super() are no-op dead code** — `26-27` _(dead-code)_  
  Confirmed at the cited location: VectorSumBlock defines `def __init__(self): super().__init__()` (vector_sum.py lines 26-27), and NumericalGradientBlock has the identical no-op override (numerical_gradient.py lines 27-28). BaseBlock declares no __init__, so super().__init__() resolves to object.__init__ and the override adds nothing, inconsistent with the other primitives that omit __init__. Correction to the finding's description: StateVariableBlock does NOT have such an override (verified), so only VectorSum and NumericalGradient are affected -- the 'StateVariableBlock-adjacent classes' wording is inaccurate, but the cited file/line and the core issue are correct.  
  **Fix:** Delete the empty __init__ overrides in VectorSumBlock and NumericalGradientBlock for consistency with the other primitive blocks.

### `blocks/pde/advection_equation_2d.py`

- **advection_equation_2d outflow BC copies a possibly-uninitialized neighbor derivative at corner overlaps** — `351-353` _(correctness)_  
  Outflow boundaries copy the neighbor's dc_dt (left: dc_dt[j,0]=dc_dt[j,1] for all j, lines 351-353). The interior loop (lines 303-304) only fills dc_dt for j,i in 1..N-2, so for j=0 and j=Ny-1 the neighbor dc_dt[0,1]/dc_dt[Ny-1,1] is set only if the bottom/top pass ran first. But left/right passes (lines 332-375) execute BEFORE bottom/top (lines 377-419). So with left edge Outflow and bottom edge non-Outflow, dc_dt[0,0] is copied from dc_dt[0,1] which is still 0 at that moment. Corner derivatives can be silently zero/stale depending on BC-pass ordering.  
  **Fix:** Handle corners explicitly, or compute interior + edge-normal derivatives for all rows/cols first and resolve corners last, so outflow corners never depend on the incidental order of the four boundary passes.

### `blocks/pde/field_processing.py`

- **FieldLaplacian boundary formulas are not true one-sided second derivatives (just shifted-stencil copies)** — `697-699` _(correctness)_  
  The comment 'use one-sided differences' (line 697) is misleading: laplacian[0] = (field[2]-2*field[1]+field[0])/dx^2 (line 698) is the standard central second-difference centered at node 1, reused for node 0 (it returns the 2nd derivative at x1, not x0). laplacian[N-1] (line 699) reuses the stencil centered at node N-2. These are not genuine one-sided second-derivative formulas; the endpoint Laplacian is effectively a copy of the nearest interior value, first-order wrong where the field has boundary curvature.  
  **Fix:** Use a proper one-sided 3-point (or 4-point) second-derivative formula, e.g. laplacian[0] = (2*field[0] - 5*field[1] + 4*field[2] - field[3])/dx^2 for N>=4, or document that endpoints simply copy the nearest interior second derivative.

### `blocks/pde/field_processing_2d.py`

- **FieldScope2D drops the t=0 frame and stores at the wrong phase due to sample_count logic** — `304-311` _(correctness)_  
  _sample_count_ initializes to 0 (line 295) and is incremented BEFORE the test (line 306: get(...,0)+1), so the condition '_sample_count_ >= sample_interval' (line 308, default sample_interval=5) is first true on the 5th call. The t=0 initial field is never recorded; the first stored frame is at step sample_interval, not step 0. This omits the initial condition / earliest transient for 2D PDE visualization. FieldScope 1D stores every frame and does not share this issue.  
  **Fix:** Capture the t=0 frame: e.g. record when count % interval == 0 with count starting at 0 (so first call stores), or initialize _sample_count_ to sample_interval. Then reset the counter as today.

### `blocks/pde/heat_equation_1d.py`

- **Robin boundary in interpreter execute() reads T[1] before this step's interior update, mixing time levels** — `237-244` _(correctness)_  
  In execute(), T = params.get('T', ...) (line 196) is the same object stored in params. The Robin BC sets T[0] = (k*T[1]/dx + h*bc)/(k/dx + h) in place (line 243) and dT_dt[0]=0 (line 244). The interior loop (lines 219-221) already used the old T[0] when computing dT_dt[1], and the subsequent Euler update T_new = T + dT_dt*dtime (line 261) adds 0 to the just-overwritten T[0]. The BC update thus depends on in-place mutation ordering of the stored state array. Combined with the compiler treating Robin as a penalty Dirichlet (system_compiler.py 866-884), Robin physics differs between interpreter and compiled solver.  
  **Fix:** Avoid order-dependent in-place mutation (operate on a copy of the state for BC algebra), and reconcile the Robin implementation with the compiled path so both solvers produce identical physics.

### `blocks/prbs.py`

- **PRBS doc describes Amplitude/Clock Period params that do not exist** — `36` _(maintainability)_  
  The doc (lines 36-37) lists 'Amplitude' and 'Clock Period', but actual params (lines 45-50) are high/low/bit_time/order/seed. Misleads users configuring the block.  
  **Fix:** Update the doc to describe high/low/bit_time/order/seed (and the LFSR semantics).

### `blocks/product.py`

- **Product block replaces division-by-zero inf with 1e308 and NaN with 0, silently corrupting results** — `133-142` _(correctness)_  
  product.py:133-142 confirmed: inside `with np.errstate(divide='ignore', invalid='ignore')` it divides, checks np.any(np.isinf/isnan), logs a warning if either, then replaces inf with np.sign(result)*1e308 (line 140) and NaN with 0.0 (line 142). This converts an invalid result (5/0 -> inf, 0/0 -> nan) into a finite number that looks valid to downstream blocks/solver, masking the singularity. 1e308 is near double max (~1.8e308) and will overflow to inf again if multiplied further. The behavior is logged (line 138), which partially mitigates, but the magic finite substitution can hide real modeling errors and yields misleading plots. Note the substitution is at least a deliberate, documented-in-code choice rather than an accident.  
  **Fix:** Consider leaving inf/nan in place (let the solver/scope flag them) or returning {'E': True, 'error': 'division by zero'} on a true singularity, instead of substituting magic finite values (1e308/0.0) that silently corrupt the signal. If substitution must remain, use np.inf/np.nan-aware handling consistently rather than a near-overflow constant.

### `blocks/ramp.py`

- **Ramp returns dict without 'E' key, inconsistent contract shape** — `63` _(consistency)_  
  RampBlock.execute (lines 66-71) returns {0: ...} with no 'E' key, while sibling sources (step, impulse, prbs) return 'E': False. The slope==0 branch (line 67) returns a 0-d np.array(0) rather than np.atleast_1d, diverging from the atleast_1d convention used by step/impulse/prbs. SineBlock and NoiseBlock also omit atleast_1d/'E', so this is a pre-existing inconsistency rather than ramp-specific, but the cited divergence is real.  
  **Fix:** Return np.atleast_1d outputs and include 'E': False consistently; the sign branches can collapse to a single sign-aware max/min.

### `blocks/rate_limiter.py`

- **RateLimiter relies on dtime living in params rather than kwargs (fragile convention)** — `69` _(consistency)_  
  Line 69 reads dt only from params: float(params.get('dtime', 0.01)). It ignores any dtime passed via kwargs. The engine injects exec_params['dtime'] so it works in normal runs, but a unit test (or any caller) passing dtime as a kwarg silently gets 0.01, mismatching the more robust kwargs.get('dtime', params.get('dtime', 0.01)) convention used by other timestep blocks.  
  **Fix:** Use dt = float(kwargs.get('dtime', params.get('dtime', 0.01))) to honor the dtime kwarg and match the convention.
- **RateLimiter reads dtime from params only, not the execute kwarg used elsewhere** — `69` _(consistency)_  
  Confirmed. blocks/rate_limiter.py:69 reads dt = float(params.get('dtime', 0.01)), without consulting kwargs. Other stateful blocks (e.g. integrator) use kwargs.get('dtime', params.get('dtime', 0.01)). It works in normal runs because the engine writes exec_params['dtime'], but a direct execute(..., dtime=...) call (as several unit tests pass) would be silently ignored. Note test_control_blocks.py works around this by putting dtime in the params dict (line 21), masking the inconsistency.  
  **Fix:** Use the same precedence as other blocks: dt = float(kwargs.get('dtime', params.get('dtime', 0.01))).

### `blocks/rate_transition.py`

- **RateTransition Average mode resets shared sample buffer only in Average branch** — `184-189, 227-237` _(performance)_  
  Lines 184-189 append (time,val) to _sample_buffer_ on every non-output_only call and trim it to the last 50 once it exceeds 100. Only the Average branch (line 235) clears/consumes it. Filter mode uses _filter_state_ (line 248), not the buffer, and ZOH/Sample/Linear never read it. So for all non-Average modes the per-step append plus the 100->50 reslice is dead bookkeeping.  
  **Fix:** Only maintain the sample buffer when mode == 'Average' so the other modes avoid the per-timestep append and periodic reslice.

### `blocks/rootlocus.py`

- **Dead '_init_start_' param declared on stateless analysis blocks** — `20-23` _(dead-code)_  
  RootLocus (rootlocus.py:20-23), Nyquist (nyquist.py:20-23), BodePhase (bodephase.py:20-23), and BodeMag (bodemag.py:20-23) each declare params = {'_init_start_': {'default': True, 'type': 'bool'}}, but their execute() methods are pure no-ops (return {} or {'E': False}) that never read _init_start_. The flag's sole purpose is to trigger first-call init, which these blocks do not perform, so it is dead state copied across four files. LQR (lqr.py:42-52) correctly does NOT declare _init_start_, consistent with the finding's scope.  
  **Fix:** Drop the unused _init_start_ param from these analysis blocks, or add a brief comment if it is intentionally retained (e.g. so reset_memblocks/serialization treats them uniformly).

### `blocks/step.py`

- **Step 'impulse' branch does not init _impulse_fired under _init_start_** — `71` _(correctness)_  
  The _init_start_ block (lines 71-74) only sets _step_old/_change_old; the impulse branch (line 91) reads params.get('_impulse_fired', False) but it is never reset on init, unlike ImpulseBlock which sets it explicitly (impulse.py:79-81). Re-init relies on params being cleared rather than explicit reset.  
  **Fix:** Set params['_impulse_fired'] = False inside the _init_start_ block to make re-initialization explicit and reset-safe, mirroring ImpulseBlock.
- **Step prints to stdout on bad 'type' instead of returning error message** — `99` _(error-handling)_  
  On an unrecognized type the block calls print(...) (line 99) and returns {0: 0.0, 'E': True} (line 100) with no 'error' message and a bare float (not atleast_1d), unlike the structured {'E': True, 'error': msg} contract used by PRBS (prbs.py:83,87,120).  
  **Fix:** Return {'E': True, 'error': f"unknown step type {step_type}"} and drop the print.

### `blocks/switch.py`

- **Switch reads inputs[0] without get(), KeyError if control unconnected** — `91` _(error-handling)_  
  blocks/switch.py:91 `ctrl = float(np.atleast_1d(inputs[0])[0])` uses direct subscription; an unconnected/missing control port raises KeyError that escapes execute() (Switch has no surrounding try), whereas data inputs use the defensive `inputs.get(sel + 1, 0.0)` at line 101.  
  **Fix:** Use `inputs.get(0, 0.0)` for the control input to match the defensive handling of the data inputs.
- **Switch threshold-mode comment/contract mismatch on selected port** — `98` _(consistency)_  
  blocks/switch.py doc (lines 29-30) states 'u2 >= Threshold: Output = u1 (Top port)' and 'u2 < Threshold: Output = u3 (Bottom port)', but the control port is actually port 0 ('ctrl', placed top via get_inputs), and line 98 computes sel=0 (->in0) when ctrl>=threshold else sel=1 (->in1), reading inputs.get(sel+1). The u1/u2/u3 / top/bottom naming in the doc does not map cleanly to the 0-based ctrl/in0/in1 port scheme, risking confusion about which port is selected.  
  **Fix:** Rewrite the docstring using the actual port names (ctrl, in0, in1) so the threshold-to-port mapping (ctrl>=threshold -> in0, else in1) is unambiguous.

### `blocks/transport_delay.py`

- **TransportDelay documents a Buffer Size parameter that does not exist** — `35, 41-48` _(maintainability)_  
  The doc (line 35) advertises '- Buffer Size: Max history length (increase if simulation is long/fast).' but the params property (lines 41-48) defines only delay_time, initial_value, _time_buffer_, _value_buffer_, _init_start_. There is no buffer_size key, so the documented parameter cannot be set by users.  
  **Fix:** Either add a buffer_size param that caps the deque length, or remove the misleading line from the doc.

### `blocks/zero_order_hold.py`

- **ZeroOrderHold reads params['sampling_time'] / params['_held_value_'] without defaults** — `79, 82` _(error-handling)_  
  Lines 79 and 82 index params['_held_value_'] and params['sampling_time'] directly (not .get). _held_value_ is only set inside the _init_start_-guarded init block (line 76); if a reset/load path leaves _init_start_ False but _held_value_ unset, line 79 raises KeyError. sampling_time has a params default but is still read by direct index, so a missing key raises rather than falling back.  
  **Fix:** Use params.get('sampling_time', 0.1) and params.get('_held_value_', 0.0) for graceful fallback, consistent with the .get-based access elsewhere in the block.
- **ZeroOrderHold returns output dict mixing the 'E' error key with port output** — `95-97` _(design-smell)_  
  Confirmed. blocks/zero_order_hold.py returns {0: params['_held_value_'], 'E': False} at line 95 and {0: held_val, 'E': False} at line 97, embedding the error-signaling key into a normal output dict. The BaseBlock contract is to return either {port_idx: value} or {'E': True, 'error': msg}. This works only because downstream code treats only truthy 'E' as an error, but it is inconsistent with sibling blocks (e.g. hysteresis/rate_limiter return plain {0: value}).  
  **Fix:** Return plain {0: value} without the 'E' key from both return paths, matching the BaseBlock contract and other blocks.

### `lib/analysis/analyzers/base_analyzer.py`

- **PID transfer-function extraction ignores discrete sampling_time, producing a continuous TF for discrete blocks** — `110-118` _(correctness)_  
  Verified: PID branch (lines 110-118) returns np.array([kd,kp,ki]), np.array([1,0]), 0.0 — always continuous (dt=0.0), ideal-form (Kd s^2+Kp s+Ki)/s, ignoring any sampling_time and any derivative filter coefficient. For a discretized PID this yields an incorrect frequency response. Severity lowered to low: it is unconfirmed whether the PID block supports a discrete/filtered mode, so impact may be narrow.  
  **Fix:** Honor params['sampling_time'] and any derivative-filter term; emit the discrete-equivalent (or matching s-domain) TF and propagate dt.
- **PID transfer function for root locus/bode is improper (degree-2 numerator over degree-1 denominator)** — `110-118` _(correctness)_  
  Confirmed the improper TF at line 116: returns num=np.array([kd,kp,ki]) (degree 2 when kd!=0) over den=np.array([1,0]) (degree 1). Correction to the finding's mechanism: signal.TransferFunction does NOT warn or error on this improper TF (verified scipy 1.17.1). The real failure is in root_locus.py: padding makes padded_den=[0,1,0]; at K=0 np.roots gives 1 root but at K>0 gives 2 roots, so np.array(all_roots) raises ValueError (inhomogeneous shape), caught by the bare except -> silent None. So a PID with kd!=0 silently produces no root-locus plot rather than 'spurious roots'.  
  **Fix:** Use a realizable filtered-derivative PID form, e.g. den=[Tf,1,0], so the TF is proper; or in root_locus guard/skip when len(num)>len(den) and surface a clear message instead of letting the inhomogeneous np.array crash be swallowed.
- **_position_window wraps a broad try/except Exception around all positioning logic** — `197-220` _(error-handling)_  
  Verified: lines 197-220 wrap the entire cascade computation (including count=len(self.parent.analyzer.plot_windows)) in try/except Exception that silently falls back to window.move(100,100), masking real bugs (e.g., attribute typos) as a benign default position.  
  **Fix:** Narrow the except to AttributeError, or compute count defensively (hasattr/getattr) without a broad catch so unexpected errors surface.

### `lib/analysis/analyzers/lqr.py`

- **LQR stability check uses strict < 0 and ignores marginal/Q-PSD assumptions** — `77-78` _(correctness)_  
  Confirmed: line 78 'stable = all(e.real < 0 for e in eig_cl)' uses a strict zero threshold with no tolerance; an eigenvalue numerically at ~-1e-15 vs +1e-15 flips the verdict (verified). solve_continuous_are also presumes (A,B) stabilizable and Q PSD / R PD with no precondition validation, so an ill-posed input could display a misleading P/K. Low severity: for a well-posed LQR solution from solve_continuous_are the closed-loop eigenvalues are strictly in the LHP by construction, so the practical flip risk is small.  
  **Fix:** Use a small negative tolerance (e.g. e.real < -1e-9) for the stability verdict, and optionally validate Q symmetric PSD and R symmetric PD before solving to reject ill-posed inputs cleanly.

### `lib/analysis/analyzers/nyquist.py`

- **Plot QWidget windows have no parent and are only kept alive by caller's return value** — `41-65` _(resource-leak)_  
  Confirmed: nyquist.py line 42 'plot_window = QWidget()' has no parent; it is shown (line 64) and returned (65), kept alive only via the caller's stored reference. Same pattern in bode.py (38/77-78) and root_locus.py (87/118-119). If the caller does not retain the returned reference, the top-level window is garbage-collected and vanishes. _position_window reads self.parent.analyzer.plot_windows, implying a retention list exists, but these analyzers do not append to it themselves — retention depends entirely on the facade caller.  
  **Fix:** Have each analyzer append the created window to a persistent list (e.g. self.parent.analyzer.plot_windows) before returning, or assign a parent, so lifetime does not depend on the caller storing the return value.

### `lib/analysis/analyzers/root_locus.py`

- **Root locus only sweeps positive gains and a narrow log range (1e-2..1e2)** — `27-28` _(correctness)_  
  Confirmed: line 27 k_values=np.logspace(-2,2,500) (range 1e-2..1e2), line 28 inserts K=0. Only the positive-gain locus over a narrow 4-decade range is swept; high-gain asymptotes (K>100) and low-gain detail (K<0.01) are truncated, and the complementary (negative-K) locus is omitted. This is a fidelity/completeness limitation, not a crash. Severity lowered to low since the plot is still valid within the swept range.  
  **Fix:** Widen the gain range (e.g. 1e-3..1e4) and/or scale relative to system gain; optionally add the negative-K branch. Document that only the positive-gain locus over the fixed range is shown.

### `lib/analysis/linearizer.py`

- **Forward-difference Jacobian with fixed eps and no dtype guard loses precision / can truncate** — `172-198` _(correctness)_  
  Verified: compute_jacobian_numerical uses one-sided forward difference with fixed absolute eps=1e-6 (line 192 x_plus[j]+=eps; line 196 (f_plus-f0)/eps), no relative scaling. x0 originates from np.array(states) in get_state_vector and could be integer dtype, making x_plus[j]+=1e-6 truncate to 0 and yield a zero column. Real but academic given A is currently always zero (Finding 1) — severity lowered to low.  
  **Fix:** Cast x0 to float, use a relative step eps*max(1,|x0[j]|), and prefer central differences (f(x+h)-f(x-h))/(2h).
- **ss2tf only returns first output row; multi-output systems silently truncated** — `307-309` _(correctness)_  
  Verified: line 308 tf=signal.ss2tf(A,B,C,D); line 309 returns tf[0][0], tf[1]. signal.ss2tf returns num as a 2D array (one row per output), so tf[0][0] discards all but the first output's numerator, silently reducing MIMO/MISO to SISO. Severity lowered: this linearizer is currently only exercised with stub SISO data, limiting real-world impact.  
  **Fix:** Return the full numerator array (tf[0]) or explicitly document/enforce SISO; for multi-output handle each row.
- **time_constants uses exact np.isreal which misses numerically-complex real eigenvalues** — `340-342` _(correctness)_  
  Verified: lines 340-342 use 'if np.isreal(ev) and np.real(ev)<0'. eigvals of a real matrix can return tiny nonzero imaginary parts for genuinely real eigenvalues, making np.isreal(ev) False and excluding those poles from time-constant reporting.  
  **Fix:** Use a tolerance: if abs(np.imag(ev))<tol and np.real(ev)<0.
- **oscillatory_modes counts each conjugate eigenvalue pair twice** — `345-354` _(correctness)_  
  Verified: loop at lines 346-354 appends a mode for every eigenvalue with np.imag(ev)!=0, so a complex-conjugate pair produces two identical entries (omega_n and zeta are equal for the pair), double-counting oscillatory modes.  
  **Fix:** Only add modes for eigenvalues with positive imaginary part (np.imag(ev)>0) to count each conjugate pair once.
- **find_state_blocks crashes when blocks is None and no dsim provided** — `48-60` _(error-handling)_  
  Verified: lines 48-49 only assign blocks from dsim when self.dsim is not None; if both are None, blocks stays None, self.blocks=None (line 51), and 'for block in blocks' (line 60) raises TypeError: 'NoneType' is not iterable. The primary caller linearize_at_point guards self.dsim first, but find_state_blocks is public and directly callable, so the ungraceful crash is reachable. Severity lowered to low (a clear error vs a confusing TypeError).  
  **Fix:** Guard at top: if blocks is None: raise ValueError('No blocks or dsim provided') before iterating.

### `lib/block_loader.py`

- **Block-loader import failures use print() instead of logger** — `114-115` _(error-handling)_  
  _collect_block_classes (lines 114-115) catches all exceptions during importlib.import_module and prints to stdout (print(f"Error loading block {module_name}: {e}")) rather than logging. A missing/broken block module is silently dropped with no log record and no traceback — problematic in frozen mode where stdout may be unavailable, and inconsistent with the module's own logger (defined at line 9 and used elsewhere in load_blocks).  
  **Fix:** Replace print with logger.error(f"Error loading block {module_name}: {e}", exc_info=True) so failures and tracebacks land in the configured log.

### `lib/config_manager.py`

- **Broad 'except Exception' swallows errors throughout ConfigManager** — `44, 119, 148, 164, 196, 237` _(error-handling)_  
  ConfigManager wraps method bodies in broad 'except Exception as e' that logs and returns a default/False: _load_config (44), get (119-121), set (148-150), save (164-166), apply_to_dsim (196-197), validate_config (237-239). This masks programming errors (e.g. a non-dict _config) behind generic log lines, hindering diagnosis.  
  **Fix:** Narrow to the expected exception types per method (e.g. OSError/json.JSONDecodeError for load/save, leave get/set largely exception-free since dot-walk already guards with isinstance checks).

### `lib/diagram_builder.py`

- **DiagramBuilder.connect divides by (in_ports + 1) without guarding against ports being recomputed; port_spacing can be 0 causing all ports stacked** — `189-190` _(correctness)_  
  Lines 189-190: port_spacing = dst_b['coords_height'] // (dst_b['in_ports'] + 1); dst_y = dst_b['coords_top'] + port_spacing * (dst_port + 1). With small height and many in_ports, integer floor division yields 0, collapsing all dst_y to coords_top. No division-by-zero (the +1 prevents it). This is cosmetic only — the points are approximate line endpoints used for programmatic diagram generation, not simulation logic.  
  **Fix:** Clamp port_spacing to a minimum of 1 (e.g. max(1, height // (in_ports + 1))) so multi-port destinations get visually distinct endpoints.

### `lib/diagram_validator.py`

- **DiagramValidator._check_goto_from_tags assumes block.params and block.block_fn exist without hasattr guards used elsewhere** — `219-225` _(consistency)_  
  Lines 220-221 access block.block_fn and block.params.get('tag', ...) directly without hasattr/getattr guards. Most other checks (e.g. _check_disconnected_inputs lines 131-141, _check_isolated_blocks 194-196) defensively use hasattr/getattr. A block lacking block_fn or params would raise AttributeError and abort validate(). Note: the finding overstates 'every other check' — _check_rate_mismatches also accesses block.params.get (355) and block.block_fn (367) directly, so this pattern is not unique to _check_goto_from_tags.  
  **Fix:** Use getattr(block, 'block_fn', None) and getattr(block, 'params', {}) for consistency and robustness against malformed blocks. (Same applies to _check_rate_mismatches.)
- **DiagramValidator._check_duplicate_connections does O(n) linear scan of blocks_list per duplicate, and message reads block.username without None guard** — `296-304` _(correctness)_  
  Line 304: message=f"Block '{block.username if block else block_name}' ...". When the block is found, it uses block.username unconditionally; if username is None/empty the message prints 'None'. Every other message in this file uses the 'block.username or block.name' fallback (lines 150, 182, 209, 397-399, 421-423). The fallback block_name is only used when block is None. The inner linear scan (297-300) over blocks_list per duplicate is minor since duplicates are rare.  
  **Fix:** Use 'block.username or block.name' when block is found, and block_name otherwise, to match the rest of the file and avoid printing 'None'.

### `lib/engine/flattener.py`

- **import re inside hot recursion / loop body** — `153` _(maintainability)_  
  Verified: `import re` at line 153 sits inside the `if found_idx is None:` try block within the `while curr:` loop of _resolve_driver, which is called per primitive input port. Cached after first import so negligible cost, but it is a code smell that obscures module dependencies.  
  **Fix:** Move `import re` to the module-level imports at the top of flattener.py.
- **input_drivers silently overwrites on duplicate destination port** — `70-72` _(correctness)_  
  Verified: line 72 `self.input_drivers[(dst, line.dstport)] = (src, line.srcport)` performs an unconditional dict assignment with no collision check, so a second line targeting the same (dst, dstport) silently overwrites the first and one connection vanishes.  
  **Fix:** Check for an existing key and log a warning (or raise) when a destination port already has a driver, since one input port cannot legally have two sources.

### `lib/engine/optimization_engine.py`

- **set_parameters does not transform initial guess consistently with bounds for log scale** — `168` _(correctness)_  
  _transform_from_optimizer ends with np.clip(value, lower, upper) (line 168) with no validation that lower < upper. If a Parameter block has lower==upper or lower>upper, clip silently collapses the value to a degenerate bound. get_parameter_info (lines 108-115) reads lower/upper directly from params with no ordering check. The finding's reasoning about log/normalized scale interaction is muddled, but the unvalidated-clip concern is real.  
  **Fix:** Validate lower < upper when building params_info (get_parameter_info) and guard against degenerate bounds before clipping.
- **objective() appends full x.copy() and grows history unboundedly** — `318-322` _(performance)_  
  objective() appends {'n', 'x': x.copy(), 'cost'} to self.history on every evaluation (lines 318-322); for differential_evolution or many-iteration runs this grows without bound and the full list is returned in results (line 484).  
  **Fix:** Cap history length, make it opt-in via config, or store only improvements (cost < best_cost).
- **differential_evolution ignores x0 initial guess and tol semantics differ** — `426-448` _(correctness)_  
  When method is differential_evolution (lines 426-436) the computed x0 (lines 401-402) is unused, and tol is passed (line 432) but DE's tol means relative population convergence, semantically different from minimize's tol (line 447). Users tuning tol get inconsistent behavior across methods.  
  **Fix:** Document the divergence, optionally pass x0 via DE's x0 kwarg (recent scipy), and use a DE-appropriate tol default.

### `lib/engine/pde_helpers.py`

- **parse_pde_2d_initial_condition crashes on array IC instead of handling like 1D** — `145-146` _(error-handling)_  
  Verified: line 146 `return np.full((Ny, Nx), float(ic_spec))` in the non-string else branch with no array path and no try/except; a list/ndarray ic_spec raises TypeError on float(). The 1D parser handles arrays (lines 80-95). NOTE: the type hint (line 99) restricts ic_spec to Union[str,int,float], and the sole caller (system_compiler.py:1895) passes pde_params.get('init_temp','0.0') (a string/scalar param), so an array reaching here is currently out-of-contract and not exercised. Real robustness gap, but not triggerable on the known code path.  
  **Fix:** Add an array-handling branch (reshape/broadcast to (Ny, Nx)) or guard with a clear error/log mirroring the 1D parser, if array ICs are intended to be supported.
- **Robin BC reduces to Dirichlet penalty, ignoring h, k, dx flux physics** — `238-239` _(correctness)_  
  Verified: apply_robin_1d docstring (line 224) advertises Robin BC k*dT/dx = h*(T_inf - T) and accepts h, k, dx (lines 217-219), but the body (lines 238-239) applies only `dc_dt[idx] = penalty * (bc_val - c_val)` (a Dirichlet penalty), leaving h, k, dx entirely unused. Physics is wrong for non-trivial convective cases. The docstring does flag it as 'Simplified', mitigating the surprise.  
  **Fix:** Implement the convective flux ghost-node formula using h, k, dx, or clearly document the limitation and remove/flag the unused parameters.
- **ensure_field_array silently fills mismatched array with first element** — `312-314` _(correctness)_  
  Verified: lines 312-314 return `np.full(N, arr[0] if len(arr) > 0 else default)` when len(arr) is neither N nor 1, with no warning. Contrast parse_pde_initial_condition (lines 87-95) which interpolates/subsamples. A wrong-sized field is silently degraded to a constant, masking likely user error.  
  **Fix:** Log a warning on size mismatch and consider interpolating/subsampling to match N for consistency with parse_pde_initial_condition.

### `lib/engine/simulation_engine.py`

- **propagate_outputs indexes out_value[srcport] and logs on hot path** — `1030` _(error-handling)_  
  propagate_outputs emits logger.info per call (1030) and does `out_value[tuple_child['srcport']]` (1037) which KeyErrors if a block did not produce that port. logger.info is noisy/costly when called repeatedly.  
  **Fix:** Use out_value.get(srcport, 0.0) (or validate the port exists) at 1037 and downgrade the 1030 log to logger.debug.
- **Kahn's algorithm uses list.pop(0), O(n^2) queue operations** — `1245` _(performance)_  
  Line 1245 `u = queue.pop(0)` on a Python list is O(n), making the topological sort O(n^2) in block count. Additional `next((b for b in current_blocks if ...))` linear scans at 1214 and 1252 add O(blocks) lookups. Built once per simulation (not per timestep), so low impact.  
  **Fix:** Use collections.deque with popleft(), and build a name->block dict once instead of the `next(... for b in current_blocks)` linear scans at 1214/1252.
- **Product block returns magic sentinel 1e308 on divide-by-zero** — `1538` _(correctness)_  
  Line 1538 `res = res / float(val) if val != 0 else 1e308` substitutes a finite magic constant on divide-by-zero instead of np.inf/error, producing a misleading finite value that diverges from numpy inf semantics used elsewhere and loses sign information.  
  **Fix:** Return np.inf (preserving sign of res) or signal an error consistent with the block's runtime execute(), rather than a magic finite constant.
- **Scope labels parsing assumes string, crashes (caught) if labels is a list** — `1846-1847` _(correctness)_  
  Line 1845 reads labels_raw from params; 1846-1847 call labels_raw.replace(' ','').split(',') without ensuring it is a string. If 'labels' holds a list/dict (not 'default'), this raises AttributeError that is silently swallowed by the broad except at 1919, aborting the whole simulation as a generic failure.  
  **Fix:** Branch on isinstance(labels_raw, str) (or coerce via str(labels_raw)) before calling string methods at 1847.
- **hasattr(self,'sim_dt') guard is dead code; attribute always set in __init__** — `190` _(dead-code)_  
  sim_dt is unconditionally assigned at line 100 in __init__ (`self.sim_dt: float = 0.01`), so the guard at line 190 `self.sim_dt if hasattr(self, 'sim_dt') else 0.01` can never take the fallback branch.  
  **Fix:** Replace with `current_dt = self.sim_dt`.
- **Redundant duplicated parameter-resolution loop in initialize_execution and run_compiled_simulation** — `191` _(duplication)_  
  The exec_params resolution block (resolve_params + copy '_' keys + set dtime + dtime-cache skip) is duplicated at lines 191-198 (initialize_execution), 1108-1114 (run_compiled_simulation), and largely at 850-862 (prepare_execution; the 850-862 variant lacks the cache-skip and adds set_block_type/reload_external_data).  
  **Fix:** Extract a single _resolve_block_params(block, dt) helper and call it from all sites to keep cache-skip logic consistent.
- **Verbose per-block logger.info in hierarchy resolution hot path** — `302` _(performance)_  
  Line 302 emits logger.info with an f-string for every block on every hierarchy pass inside the while/for nesting (also lines 337, 1030). f-strings are eagerly formatted regardless of effective log level, costing roughly O(blocks * passes) string work at INFO. These are init-time (not per-timestep) but still wasteful and noisy.  
  **Fix:** Demote to logger.debug or guard with logger.isEnabledFor(logging.INFO) before building the string.
- **execute_block swallows all exceptions including the External-stub path returning False (ambiguous with valid output)** — `472` _(error-handling)_  
  Lines 472-475 catch bare Exception, log only `e` (no traceback) and return False — unlike initialize_execution at line 405 which logs traceback.format_exc(). Diagnosability of real block bugs is reduced. The 'ambiguous with valid output' framing is overstated: callers (lines 242, 255, 307) check `is False` by identity and a falsy dict {} would not be confused with False; the substantive issue is the missing traceback.  
  **Fix:** Log traceback.format_exc() here for parity with initialize_execution so block bugs are diagnosable; the bool overloading is acceptable as-is given identity checks.
- **reset_memblocks does not clear integrator/state accumulator ('mem','output'), relying on _init_start_ side effect** — `687` _(correctness)_  
  reset_memblocks (687-701) resets `_init_start_` in params/exec_params and deletes `_prev`, but leaves stale `exec_params['mem']` and `exec_params['output']` (set at lines 318, 378). Correctness depends on every memory block fully re-initializing mem when `_init_start_` is True. This is an established convention (the block contract), so it is a documentation/defensiveness gap rather than a live bug — hence low.  
  **Fix:** Also clear known accumulator keys ('mem','output') in reset_memblocks, or document/enforce that memory blocks must fully re-init when _init_start_ is set.
- **set_block_type for DiscreteStateSpace D-matrix uses np.all(D==0) on possibly ragged/object array** — `911` _(correctness)_  
  Lines 911-912: `D = np.array(block.exec_params.get('D', [[0.0]]))` then `np.all(D == 0)`. If D is a ragged list, np.array yields an object array and elementwise `== 0` behaves unexpectedly / may warn, potentially misclassifying b_type. Edge case requiring malformed user input.  
  **Fix:** Coerce with dtype=float and validate shape, or wrap in try/except to surface a clear param error for malformed D.
- **identify_memory_blocks instantiates every block's class purely to probe requires_inputs (per-init cost + side effects)** — `930` _(performance)_  
  Line 930 `instance = block_class()` constructs a fresh instance solely to read `requires_inputs` (a class-level attribute) on every initialize_execution for every block. Instantiation is unnecessary and risks constructor cost/side effects. Mitigated by the surrounding try/except (934) so failures are non-fatal, hence low rather than medium.  
  **Fix:** Read off the class: `getattr(block_class, 'requires_inputs', True)` instead of instantiating.

### `lib/engine/symbolic_engine.py`

- **StateSpace symbolic forces single-input u even for MIMO B** — `285-311` _(correctness)_  
  StateSpace symbolic (lines 285-311) computes the full transfer matrix G = C(sI-A)^-1 B + D, then multiplies by a single scalar input symbol u = input_exprs.get(0) (lines 307-311). For multi-input (MIMO) systems G has multiple columns, so G*u (scalar) is dimensionally wrong, silently producing a matrix*scalar result instead of multiplying by a per-input vector.  
  **Fix:** Build an input vector from all connected ports (input_exprs by column index) and multiply G by that vector rather than a single scalar.
- **PID/Gain symbolic: numeric params used directly in sympy without Float conversion** — `313-326` _(correctness)_  
  PID (lines 313-326) uses raw params Kp/Ki/Kd/N directly in the symbolic expression C_pid = Kp + Ki/s + Kd*N*s/(s+N), unlike Gain (lines 247-249) which wraps K in sympy.Float. numpy floats/arrays could yield inconsistent or ndarray-typed sympy expressions, and N is not guarded against 0 (default 20.0, but a user N=0 gives s/(s+0)).  
  **Fix:** Coerce Kp/Ki/Kd/N to sympy.Float and guard N>0 (special-case Kd=0 or N=0) before composing the transfer function.
- **linearize_at_point catches broad Exception and returns None without distinguishing failure modes** — `428-430` _(error-handling)_  
  linearize_at_point has a broad `except Exception as e` that logs a warning and returns None (lines 428-430) for any failure: non-rational G, float() on symbolic coeffs (lines 420-421), or scipy tf2ss errors. The docstring promises a (A,B,C,D) tuple but None is undocumented, so callers cannot distinguish success from failure or learn the cause.  
  **Fix:** Narrow the except (e.g. catch specific Poly/conversion errors) or return a structured error, and document the None contract for callers.
- **Bare/blanket except swallows LaTeX export errors silently** — `494-495` _(error-handling)_  
  export_equations_latex wraps per-equation LaTeX conversion in `except Exception:` / `pass` (lines 494-495), silently dropping equations that fail to render with no logging.  
  **Fix:** Log the exception (logger.debug/warning including the block name) so silently omitted equations are diagnosable.
- **File write in export_equations_latex not using explicit encoding and no error handling** — `501-503` _(resource-leak)_  
  export_equations_latex writes with open(filename, 'w') (lines 501-503) using platform-default encoding and no try/except. On Windows non-ASCII block names would raise UnicodeEncodeError and any IO error propagates uncaught. (The file handle itself is properly closed by the with-statement, so it is not literally a leak; the issue is encoding/error-handling.)  
  **Fix:** Use open(filename, 'w', encoding='utf-8') and handle or deliberately propagate IO errors.

### `lib/engine/system_compiler.py`

- **check_compilability dead branches and ignored external/custom-script check** — `113-131` _(dead-code)_  
  The b_type.title() and b_type.upper() branches (115-119) both just `pass` with no effect on the return, and the block.external branch (125-129) is a no-op `pass` despite a comment that custom-script user blocks should be rejected. The intended rejection of non-standard external/custom blocks never happens, so such blocks may be wrongly admitted to the compiled path.  
  **Fix:** Remove the no-op branches or implement the intended logic: reject blocks whose external/custom flag indicates a user script while not in COMPILABLE_BLOCKS, returning False so they fall back to the interpreter.
- **Expensive diagnostic np.gradient/energy computed every RHS evaluation** — `1287-1292` _(performance)_  
  Confirmed: Wave2D computes two full np.gradient passes plus sum-of-squares to populate signals[b_name+'_energy'] (1288-1292) on every model_func evaluation, including discarded intermediate solver stages. Heat2D computes np.mean/np.max diagnostics per call (1174-1175); Advection2D likewise (1428-1429). These are diagnostic-only outputs not needed at intermediate stages.  
  **Fix:** Defer diagnostic-only signals (energy, avg, max) to a post-step output pass rather than computing them inside the derivative RHS on every (intermediate) evaluation.
- **Compiled RateLimiter silently replaces exact slew limiting with a stiff gain-1000 approximation** — `655-669` _(consistency)_  
  Compiled RateLimiter models slew limiting as dy = clip((u - y)*K, falling, rising) with hard-coded K=1000.0 (655, 663-667), a first-order stiff chase, rather than exact per-step slew clamping. This diverges from the interpreted block and the magic gain is undocumented (a comment merely calls it 'Stiffness gain').  
  **Fix:** Document the approximation and K choice, expose it as a parameter, or exclude RateLimiter from compilation to guarantee fast/interpreted parity.
- **MathFunction reciprocal lambda double-evaluates np.where mask and silently returns 0 for 1/0** — `718-719` _(correctness)_  
  The reciprocal lambda computes np.where(x!=0, 1.0/np.where(x!=0,x,1.0), 0.0) for arrays — the x!=0 mask is built twice per call in the hot RHS — and both array and scalar branches return 0.0 for zero input, silently masking a divide-by-zero rather than propagating inf.  
  **Fix:** Compute the mask once (m = x != 0) and reuse it; document or reconsider whether silently returning 0 for 1/0 is the intended contract versus propagating inf.
- **Broad except Exception in MathFunction expr executor swallows all errors to 0.0 every step** — `731-740` _(error-handling)_  
  exec_mathfunc_expr catches bare `except Exception` and returns 0.0 on any failure, logging only at debug level. A persistently broken user expression (NameError/TypeError failing every step) is silently treated as 0.0 for the whole run, so the user sees a flat-zero signal with no obvious cause.  
  **Fix:** Distinguish structural/compile-time failures (surface a one-time WARNING or raise) from transient per-step domain errors; log at WARNING the first time an expression fails so persistent breakage is visible.
- **Selector range parsing can raise ValueError, killing compilation** — `757-768` _(error-handling)_  
  For a ':' range, int(parts[0])/int(parts[1]) at 761-762 are NOT wrapped in try/except, unlike the scalar idx branch (765-768) which catches ValueError. A malformed range like 'a:b' raises ValueError out of compile and aborts the whole simulation instead of degrading gracefully.  
  **Fix:** Wrap the range int() conversions in try/except like the scalar branch, defaulting to a safe range (e.g. 0:None) and logging a warning on parse failure.

### `lib/export/blox_exporter.py`

- **Redundant duplicate source branch in export() chain emission** — `103-111` _(dead-code)_  
  Lines 103-106 (`if i == 0 and self._is_source(block)`) emit \bXInput, set prev_node, continue — identical to lines 108-111 (`if self._is_source(block)`). The first block is fully subsumed by the second; i==0 adds nothing.  
  **Fix:** Collapse the two identical source branches into one `if self._is_source(block):`.
- **Unused import _name_to_math in blox_exporter** — `15-20` _(maintainability)_  
  _name_to_math is imported from tikz_exporter (line 18) but never referenced elsewhere in blox_exporter (grep shows the sole occurrence is the import). _poly_to_latex/_escape_latex/_sanitize_node_id are used; this one is a dead import.  
  **Fix:** Remove the unused _name_to_math import.
- **Inconsistent \bXReturn argument arity between feedback branches** — `172-181` _(correctness)_  
  The (currently unreachable) branch emits \bXReturn{label_opt}{fb_nid}{dst_nid} (line 173) passing label as optional [..] plus 2 mandatory args, while the live simple path emits \bXReturn[fb_dist]{src_nid}{dst_nid}{fb_label} (line 180) passing label as a 3rd positional brace. Different label mechanisms; if the dead branch is revived, rendering will be inconsistent.  
  **Fix:** Pick one \bXReturn calling convention matching the blox package signature and use it in both branches; add a feedback-arc rendering test.

### `lib/export/latex_exporter.py`

- **export_document interpolates title/description/equation values without escaping** — `227-266` _(correctness)_  
  content['title'] via %s (line 228), content['description'] raw (line 232), and equation values via f'y = {eq}' (line 264) are inserted unescaped; only equation keys are escaped (line 262 name.replace('_','\_')). An underscore or ampersand in title/description breaks the build, inconsistent with key escaping.  
  **Fix:** Escape free-text fields (title/description) with a LaTeX escape helper, or document that callers must pre-escape; be consistent with the key escaping.
- **MathematicaExporter.array_to_mma assumes ndarray, breaks on None/scalar D** — `398-409` _(error-handling)_  
  array_to_mma accesses arr.ndim (line 404); if A/B/C/D is a Python list, scalar int/float, or None (common for D=0), this raises AttributeError. Note numpy scalars/0-d arrays do have .ndim, but plain Python scalars/None/list do not. Unlike matrix_to_latex which guards None, this path does not.  
  **Fix:** Coerce with arr = np.atleast_2d(np.asarray(arr)) (handle None first) before accessing .ndim.

### `lib/export/tikz_exporter.py`

- **Inconsistent option access: opts['include_sinks'] vs opts.get(...)** — `165-166` _(consistency)_  
  include_sinks is read via subscript opts['include_sinks'] (lines 165 and 197) while nearly every other option uses opts.get(...) (e.g. lines 171-172). It works only because the local default dict at line 150 guarantees the key, but the mismatch is fragile if options handling is refactored.  
  **Fix:** Use opts.get('include_sinks', True) consistently to match the rest of the method.
- **_poly_to_latex assumes numeric coefficients; non-numeric params raise/garble output** — `21-37` _(error-handling)_  
  _poly_to_latex (line 17) calls abs(c) (line 23) and f'{c:.4g}' (line 30) on each coefficient. TranFn/DiscreteTranFn numerator/denominator come from user-editable block.params. A non-numeric/string entry raises TypeError mid-export, aborting the snippet rather than falling back to the symbol form. The isinstance guard at line 589 only checks list/tuple, not element types.  
  **Fix:** Validate that all coeffs are real finite numbers before calling _poly_to_latex; on failure fall back to the '${sym}(s)$' / '${sym}(z)$' symbol path (lines 593-594, 616-617).
- **BFS layout uses list.pop(0) — O(n^2) queue** — `340-341` _(performance)_  
  The topological BFS at lines 340-341 pops from the front of a Python list (`name = queue.pop(0)`), which is O(n) per pop, making the layout O(n^2) in block count. Export is one-shot so not a hot path, but trivially fixable.  
  **Fix:** Use collections.deque with popleft() for O(1) dequeue.
- **Gain integer-check fails for numpy scalar / raises for non-numeric coercion** — `578-581` _(correctness)_  
  Line 580: `return f'${gain:.4g}$' if gain != int(gain) else f'${int(gain)}$'`. int(gain) raises ValueError on NaN and OverflowError on inf, so a degenerate NaN/inf gain param crashes the whole export. (The numpy-subclass concern in the original note is moot: numpy.float64 passes isinstance(float) and int() works on finite values; the real bug is the non-finite case.)  
  **Fix:** Guard with math.isfinite(gain) before calling int(gain); fall back to :.4g (or symbol) formatting for non-finite values.

### `lib/improvements.py`

- **validate_simulation_parameters duplicates SimulationConfig.validate logic, risking divergence** — `399-424, 30-50` _(duplication)_  
  validate_simulation_parameters (412-419) re-implements the same positivity and step<time checks as SimulationConfig.validate (34-41): both have 'time must be positive', 'step must be positive', and 'step cannot be larger than simulation time'. validate_simulation_parameters additionally caps total steps at 1M (421-422), which the dataclass version lacks. The two validators can drift out of sync.  
  **Fix:** Have one validator delegate to the other (e.g. SimulationConfig.validate calls validate_simulation_parameters) so the rules, including the step-count cap, live in one place.
- **safe_execute_block_function catches bare Exception and discards traceback, returning only str(e)** — `443-446` _(error-handling)_  
  Lines 443-446: 'except Exception as e: error_msg = f"Exception in block function: {str(e)}"; logger.error(error_msg); return False, None, error_msg'. Uses logger.error with only the formatted message, so no stack trace is captured. Block-function failures lose their traceback, making them harder to diagnose.  
  **Fix:** Use logger.exception(error_msg) or logger.error(error_msg, exc_info=True) to capture the stack trace.

### `lib/lib.py`

- **Repeated getattr-with-default for attributes always set in __init__** — `299-301` _(consistency)_  
  solver_method/rtol/atol are unconditionally assigned in __init__ (102-104), yet save (299-301), serialize (327-329), execution_init_time (406-408) and execution_init (506-508) read them via getattr(self,...,default) with inconsistent literal defaults (e.g. atol 1e-12, rtol 1e-9). This dead defensiveness risks default drift.  
  **Fix:** Access self.solver_method/self.rtol/self.atol directly (guaranteed by __init__), or centralize the defaults in one place.
- **Dead variable: check_loop assigned but never used** — `537` _(dead-code)_  
  check_loop = self.count_computed_global_list() at line 537 stores a return value that is never read again (grep confirms single occurrence). count_computed_global_list (engine line 1061-1063) is a pure query: sum(...) over global_computed_list, no side effect. So the assignment AND the call are dead.  
  **Fix:** Remove the dead assignment and the call entirely, since the query has no side effect.
- **Stale comment references non-existent line numbers / removed second init call** — `560-562` _(maintainability)_  
  Comment at 560-562 says 'The first call at line 842 already initialized the engine' but the actual initialize_execution call is at line 514; the absolute line reference is wrong/misleading. A similar obsolete 'Legacy?' comment sits at line 519.  
  **Fix:** Drop the brittle absolute line-number reference (reference the method name instead) and remove the obsolete duplicate-init explanatory comment.
- **Engine-state attributes re-copied onto self after init (drifting source of truth)** — `564-569` _(maintainability)_  
  execution_init copies max_hier/rk45_len/rk_counter/execution_time_start/execution_initialized from engine onto self at 564-569. execution_initialized (198-204) and execution_time_start (225-231) are already properties delegating to engine (so those copies are redundant no-ops), while max_hier/rk45_len/rk_counter are plain attributes, creating two sources of truth that can drift from engine.rk45_len/rk_counter during the loop.  
  **Fix:** Make max_hier/rk45_len/rk_counter properties delegating to the engine like the others, removing the redundant copies.
- **DSim is a god/facade object with dozens of pass-through delegations** — `62-256` _(design)_  
  DSim mixes UI screen geometry, button construction (main_buttons_init), file save/load delegation, subsystem navigation, and the full simulation orchestration loop, plus ~10 property getter/setter pairs forwarding to self.engine (timeline/time_step/global_computed_list/execution_initialized/execution_stop/error_msg/execution_time_start/memory_blocks at 170-240). Tight coupling complicates testing and obscures state ownership.  
  **Fix:** Split UI-shell concerns from simulation orchestration, or have callers use self.engine directly to shrink the forwarding surface.
- **Duplicated multi-rate/hierarchy execution logic across three loops** — `764` _(duplication)_  
  The outer/inner hierarchy re-iteration with should_execute/held-output/propagate logic is duplicated between execution_loop_headless (lines 764-812) and execution_loop (lines 940-1000); the block bodies are near-identical. A comment at line 762-763 explicitly notes 'See execution_loop() for the rationale ... Same logic applies here.' Fixes (e.g. the empty-message execution_failed) must be applied in two places and can drift. Title says 'three loops' but only two duplicate hierarchy loops exist in this region; the count is overstated.  
  **Fix:** Extract the hierarchy re-iteration into a shared private helper invoked by both execution_loop and execution_loop_headless.
- **Headless loop block iteration lacks per-block try/except unlike interactive loop** — `764` _(error-handling)_  
  execution_loop wraps the memory-block pre-pass in per-block try/except (lines 895-925) attributing failures to block.name. The headless equivalent (execution_loop_headless lines 744-760, and the hierarchy loop 764-812) has no inner try/except; any block exception bubbles to the outer handler at lines 821-823 which only logs 'Headless loop error' and calls self.execution_failed(str(e)), losing which block failed.  
  **Fix:** Mirror the interactive loop's per-block try/except in the headless loop so failures are attributed to a specific block name, and pass that context into execution_failed.
- **Inconsistent termination attribute: headless uses sim_time, interactive uses execution_time** — `814` _(consistency)_  
  Headless loop terminates on self.time_step > self.sim_time (line 814); interactive loop terminates on self.time_step > self.execution_time (line 1006). They are distinct attributes. However the interactive init sets self.execution_time = self.execution_init_time() (line 444), and execution_init_time returns self.sim_time (lines 421/424), so execution_time always equals sim_time at init; the headless path sets sim_time (line 660) but never reads execution_time. No path currently runs the wrong duration, so the described 'never terminate' failure mode is not currently reachable -- this is a latent maintainability/consistency smell, not a live correctness bug. Severity lowered from medium to low.  
  **Fix:** Standardize on one duration attribute in both loops (e.g. set self.execution_time = self.sim_time in headless init and use execution_time everywhere) to prevent future divergence.
- **Outer/inner re-iteration loop is O(blocks^2) or worse per timestep in hot path** — `940` _(performance)_  
  Lines 940-1000: each timestep runs while(outer) -> for hier in range(max_hier+1) -> while(inner) -> for block in current_blocks, and inside the innermost loop it reconstructs set(block.block_instance.optional_inputs) via hasattr/getattr on every pass (lines 947-951). The nesting is super-linear in block count and the reflection/set construction is repeated work. Real but bounded (each block fires once per timestep) and only material for large diagrams; severity lowered from medium to low.  
  **Fix:** Precompute each block's required_ports / optional_inputs set once at init, and group blocks by hierarchy into a dict keyed by hier, so per-timestep passes iterate only the relevant level without rebuilding sets or scanning all blocks each pass.

### `lib/logging_config.py`

- **logging fallback also prints instead of logging the config load failure** — `37-39` _(consistency)_  
  setup_logging (lines 37-39) catches (json.JSONDecodeError, ValueError, KeyError) and prints warnings before falling through to _setup_default_logging(). Using print here is defensible since logging isn't configured yet. However the except tuple is narrow: an OSError on open() (e.g. permission denied, or filename validation) is NOT caught and would propagate out of setup_logging, defeating the fallback. The original exception detail is also only printed, never re-logged after default logging is set up.  
  **Fix:** Add OSError to the except tuple (the open() at line 28 can raise it) so a read failure still falls back to default logging, and optionally re-log the original error via the fallback logger after _setup_default_logging() runs.

### `lib/managers/subsystem_manager.py`

- **create_subsystem_from_selection returns None implicitly on empty selection (inconsistent contract)** — `212-213` _(contract-violation)_  
  Lines 212-213: 'if not selected_blocks: return' yields None, while the docstring (lines 209-210) states it returns a Subsystem block instance. Callers expecting a Subsystem could dereference None.  
  **Fix:** Use explicit 'return None' and document the empty-selection case, or have callers guard the None result.
- **Redundant duplicated nested membership check on connections_list** — `295-297` _(dead-code)_  
  Lines 295-297 nest 'if l in self.dsim.connections_list:' inside an identical check, so the inner guard is always true (dead). Moreover connections_list is aliased to line_list (enter at line 64, exit at line 103), and l was already removed from line_list at lines 293-294, so the entire outer block is effectively a no-op (l is no longer present). Harmless but redundant/misleading.  
  **Fix:** Remove the duplicated inner if; since connections_list aliases line_list and the element is already removed at lines 293-294, the whole block can be dropped.
- **Broad/bare-pass exception swallowing during trajectory recompute** — `436-437, 556-557` _(error-handling)_  
  Lines 436-437 and 556-557 are 'except Exception: pass', silently discarding create_trajectory failures, whereas the parallel input-direction branches (lines 375-376, 495-496) logger.warning. A failed trajectory leaves the line with a stale/default path and no diagnostic.  
  **Fix:** logger.warning(...) the exception in both bare-pass branches to match the input-direction branches.

### `lib/models/simulation_model.py`

- **add_line/add_block ID derivation O(n) string-parses all names each call** — `189-192, 336-337` _(performance)_  
  Line 336 id_list = [int(line.name[4:]) for line in self.line_list] and lines 189-191 int(b_elem.name[len(b_elem.block_fn):]) scan all existing lines/blocks and int-parse their numeric suffix on every add. O(n) per add (O(n^2) over n adds), and int() raises if any name deviates from '<prefix><int>'. Minor for interactive use; degrades on large diagrams and is fragile.  
  **Fix:** Maintain a monotonic per-type counter, or wrap the int() parse in a guard that skips/handles non-conforming names.

### `lib/plotting/animation_exporter.py`

- **check_writers swallows all exceptions via a redundant bare Exception in the except tuple** — `386` _(error-handling)_  
  check_writers (line 386) catches (FileNotFoundError, subprocess.TimeoutExpired, Exception) and passes. Including Exception in the tuple makes the first two entries redundant and turns the clause into a catch-all that silently swallows any error (programming errors, permission errors, etc.) while leaving available['mp4']=False. A genuine bug in availability detection would be invisible.  
  **Fix:** Narrow to expected failures: except (FileNotFoundError, subprocess.TimeoutExpired, OSError): pass. Drop the redundant Exception so unexpected errors surface or are at least logged.

### `lib/plotting/scope_plotter.py`

- **plot_again leaves stale plot windows open when no scope data is present** — `50-55` _(correctness)_  
  In plot_again (scope_plotter.py:50-55), pyqtPlotScope() -- which is responsible for closing the previous SignalPlot window via the disconnect+close+deleteLater block at lines 1132-1141 -- is only invoked when valid_vectors is non-empty (line 50). If a re-run produces no scope data, the else branch (lines 54-55) only logs and the previous SignalPlot window from the prior run is never closed, leaving a stale window on screen showing old data. The XY/FFT/Field plots have the same property but are independent figures; the SignalPlot case is the one with explicit close-on-rerun semantics that gets skipped.  
  **Fix:** When there is no scope data, still close/clear any existing self.plotty window (factor the disconnect+close+deleteLater used in pyqtPlotScope into a helper and call it in the else branch) so stale windows from a previous run do not remain.
- **Dead code: five unused plotting/helper methods** — `734-823` _(dead-code)_  
  _plot_field_snapshot (scope_plotter.py:734) and _plot_field_animation (scope_plotter.py:772), plus plot_config (signal_plot.py:244), sort_labels (signal_plot.py:413), and sort_vectors (signal_plot.py:422) have no callers anywhere in the repository (verified by grep across the project, which returned only the definition sites). _plot_field_animation builds a FuncAnimation and calls a blocking plt.show() (line 821), so it is both unreachable and a hazard if wired up. plot_config is a no-op stub (return). This dead code adds confusion and maintenance burden.  
  **Fix:** Remove the unused methods (or move _plot_field_animation behind a real entry point if intended as future functionality). Delete the no-op plot_config and the unused sort_labels/sort_vectors helpers.

### `lib/plotting/signal_plot.py`

- **Mutable default argument in plot_config** — `244` _(correctness)_  
  plot_config (signal_plot.py:244) uses a mutable default argument settings_dict={}. The body is currently a no-op (return), so there is no concrete bug today, but a shared mutable default is a known Python footgun: a single dict instance is reused across calls, so any future mutation would leak between calls. Combined with the method being dead code, the cleanest fix is removal.  
  **Fix:** Remove the method (it is unused and a no-op), or if kept change the signature to settings_dict=None and create a fresh dict inside the body.
- **Mutable default argument settings_dict={} in plot_config** — `244` _(correctness)_  
  def plot_config(self, settings_dict={}) at line 244 uses a mutable default. Body is just 'return' (245) so it is harmless today, but it is a latent bug if the method is implemented and mutates the default; the parameter/method is currently dead.  
  **Fix:** Use settings_dict=None and assign {} in the body, or remove the dead parameter/method.
- **Unusual non-PEP8 indentation in SignalPlot.loop makes the hot-path method hard to maintain** — `249-280` _(maintainability)_  
  The body of loop() (the per-update plotting method, signal_plot.py:247-280) uses non-standard indentation: the try block at line 249 (8-space indent) is followed by body lines at 13-space indent (e.g. line 250 ` self.timeline = new_t`), with deeper nesting at 17/21/25/29 spaces and a stray single-leading-space blank line at 252. The code parses and runs, but is fragile: re-indenting any line to a normal 4-space stop would silently change block membership, and auto-formatters will fight the existing style. Since loop() is the central data-update method, readability matters.  
  **Fix:** Re-indent the loop() body to standard 4-space increments to match the rest of the file and avoid accidental block-boundary changes on future edits.

### `lib/profiling.py`

- **profile_block decorator almost always records 'unknown'** — `132-134` _(correctness)_  
  profile_block wrapper (line 133) reads block_name = kwargs.get('block_name', 'unknown'). Per the block contract, execute(self, time, inputs, params, **kwargs) is generally called positionally, so block_name is rarely present as a kwarg, causing nearly all timings to aggregate under 'unknown' and making the per-block report useless. Verified no usages of @profile_block exist in the codebase, so the decorator is effectively dead/unused infrastructure today; the defect is latent rather than active.  
  **Fix:** Derive the name from the bound instance (e.g. getattr(args[0], 'block_name', type(args[0]).__name__)) or accept an explicit name argument at decoration time. Also consider removing the decorator if it remains unused.

### `lib/safe_eval.py`

- **Bare-except masking via broad Exception catch in _visit_Call wraps all errors as SafeEvalError** — `373-374` _(error-handling)_  
  Lines 373-374: 'except Exception as exc: raise SafeEvalError(f"Call error: {exc}") from exc'. Any internal/programming error from an allowlisted callable is rewrapped as a SafeEvalError with a generic 'Call error' prefix. The 'from exc' chaining does preserve the original cause in the traceback, slightly mitigating the diagnostic loss, but real defects are still presented to callers as evaluation errors.  
  **Fix:** Narrow to expected arithmetic/type exceptions where possible, or log the original exception at debug level before re-raising so genuine bugs in allowlisted callables are not silently reclassified.
- **compile_expr validation skips per-element Subscript/Call structural depth but does NOT re-validate at call time, while __call__ trusts pre-validated tree — verify Name resolution gap** — `499-505, 470-478` _(correctness)_  
  _validate_tree (lines 508-544) only checks _FORBIDDEN_NODE_NAMES, Attribute, and Call structure via ast.walk. It does NOT verify that every node type has a corresponding _visit_<Name> handler on _Walker. So an unsupported-but-not-forbidden node (e.g. a future ast node type with no _visit_ method) passes compile_expr and only raises 'Unsupported AST node' at first CompiledExpr.__call__ (via _Walker.visit line 205-207). The docstring claims compile_expr 'Raises SafeEvalError at compile time for any disallowed construct,' which is not fully true.  
  **Fix:** In _validate_tree, also reject any node whose type lacks a _visit_<type-name> handler on _Walker (mirror the check in _Walker.visit), so compile_expr fully validates structure upfront and matches its docstring.

### `lib/services/diagram_service.py`

- **QFileDialog objects created but never explicitly disposed** — `205-216, 263-274` _(resource-leak)_  
  _create_styled_file_dialog builds a QFileDialog parented to main_window (line 15); in save_diagram (205-216) and load_diagram (263-274) it is exec_()'d but never deleteLater()'d. With a parent it is eventually cleaned up, but repeated save/load cycles accumulate child dialog widgets (each with a large stylesheet) until the parent is destroyed.  
  **Fix:** Call dialog.deleteLater() after exec_() returns, or wrap the dialog in a context that disposes it promptly.
- **DiagramService writes/saves on a service object but pops modal QMessageBox dialogs on error** — `244-247, 304-307` _(design)_  
  DiagramService docstring states it 'Decouples file I/O from the Main Window UI' (line 163) yet directly raises modal QMessageBox.critical on save error (246) and load error (306), recoupling it to the GUI and making it unusable headlessly (tests/CLI/autosave).  
  **Fix:** Return error info (bool/raise) and let the MainWindow layer present the dialog, or guard QMessageBox behind a 'has GUI' check.

### `lib/services/file_service.py`

- **Autosave path-slicing assumes 4-char extension and breaks on .diablos** — `212-215` _(correctness)_  
  Line 213 builds f'saves/{self.filename[:-4]}_AUTOSAVE.dat'. [:-4] strips the last 4 characters, which only matches a 3-char-plus-dot extension like '.dat'. For a filename ending in '.diablos' (8 chars), [:-4] removes 'blos', producing e.g. 'foo.dia_AUTOSAVE.dat' rather than 'foo_AUTOSAVE.dat'. The original finding said it truncates 'diab'; it actually strips the trailing 'blos', but the underlying bug (hardcoded 4-char assumption) is real. The default filename is 'data.dat' so the common path is unaffected; this is also the legacy FileService path.  
  **Fix:** Use os.path.splitext(self.filename)[0] instead of self.filename[:-4] to strip the extension robustly.
- **Block-class instantiation error silently swallowed during load** — `380-387` _(error-handling)_  
  At lines 380-387, menu_block.block_class() is wrapped in 'except Exception:' that only falls back to b_type from block_data with no logging, hiding constructor errors that may indicate a genuinely broken block during reload.  
  **Fix:** Log the exception (logger.warning/debug including block_fn) inside the except so silent block-construction failures during load are diagnosable.
- **Subsystem update_Block failure during load swallowed (only warned), leaving ports unpositioned** — `469-472` _(error-handling)_  
  In _construct_subsystem, block.update_Block() (which recomputes external port positions) is wrapped in try/except at 469-472 that only logs logger.warning without the subsystem name/sid; the partially-constructed subsystem is still returned (473) and appended by the caller, so a reloaded diagram may contain a subsystem with stale/empty port geometry that silently misroutes connections.  
  **Fix:** Log at error level including the subsystem name/sid (e.g. block.name/sid), and consider surfacing the failure to the caller so the user knows the reloaded subsystem may be malformed.

### `lib/services/run_history_service.py`

- **record_run stores timeline/traces by reference without copying** — `121-130` _(correctness)_  
  run_entry (lines 121-128) stores the caller's timeline ndarray and traces list/dicts directly with no copy. If the engine reuses/mutates those buffers on a later run, previously recorded history entries could be corrupted. Severity lowered to low: this is conditional on callers reusing buffers (not demonstrated in the cited code), and save_history later converts via .tolist() for persistence; the in-memory aliasing is a latent risk rather than a confirmed live bug.  
  **Fix:** Defensively copy on record: timeline=np.array(timeline) and per-trace y=np.array(tr['y']) before storing, or document that callers must hand over owned buffers.
- **Run history JSON opened without explicit encoding** — `24, 86` _(correctness)_  
  load_history opens with open(self.persist_path, 'r') (line 24) and save_history with open(self.persist_path, 'w') (line 86), neither passing encoding=, so they use the platform default (cp1252 on this Windows env). User-supplied run names with non-ASCII characters could round-trip inconsistently across platforms; note json.dump defaults to ensure_ascii=True so written output is ASCII-escaped, mitigating the write side, but the read side still decodes with the platform default.  
  **Fix:** Add encoding='utf-8' to both open() calls for cross-platform consistency; matches file_service.py which already uses encoding='utf-8'.
- **set_persist swallows file-deletion errors with bare except: pass** — `97-101` _(error-handling)_  
  In set_persist, when disabling persistence and unlinking the history file (98-99), any failure (permission error, locked file) is caught by 'except Exception: pass' (100-101) and silently ignored, so a stale history file may remain with no log trace.  
  **Fix:** Log the exception (logger.warning) inside the except instead of pass so a failed cleanup is observable.

### `lib/simulation/block.py`

- **Broad bare-ish except Exception blocks hide real errors in port resolution** — `126-127, 273-274, 281-284, 351-352` _(error-handling)_  
  Multiple get_inputs/inputs/outputs accesses are wrapped in broad `except Exception` that either log-and-continue (126-127) or silently fall back to generic names (273-274, 281-284) or empty port_defs (351-352). A genuine bug in a block's get_inputs() (e.g. TypeError from a wrong params shape) is masked as a harmless fallback, hiding the real cause.  
  **Fix:** Narrow caught types to the expected AttributeError/KeyError, and let unexpected exceptions propagate (or log at error level with exc_info/stack trace) so real bugs surface during development.
- **Dead/unreachable code after return in calculate_min_size()** — `319-326` _(dead-code)_  
  calculate_min_size() unconditionally returns at line 319 (return max(self.height_base, required_height)). Lines 322-326 (self.ls_width=5, self.l_width=5, self.rectf=QRect(...), logging.debug(...)) are unreachable and never execute. The misplaced body looks like copy-pasted __init__ code that landed after the return.  
  **Fix:** Delete unreachable lines 322-326. If ls_width/l_width are genuinely required by rendering, initialize them explicitly in __init__ instead.
- **in_ports cast to int(new_io[...]) without validation can crash or create invalid port counts** — `513-531` _(error-handling)_  
  change_port_numbers casts dialog values directly: self.in_ports=int(new_io['inputs']) / self.out_ports=int(new_io['outputs']) (lines 513-514, 522, 530) with no validation. A non-numeric value raises an uncaught ValueError, and a negative value would set a negative/zero port count that update_Block (line 537) iterates over, yielding inconsistent geometry.  
  **Fix:** Validate/clamp the parsed values to a non-negative range (try/except ValueError; max(0,...)) and reject invalid input before assigning, instead of letting int() raise mid-update.
- **loading_params() mutates the caller's dict in place** — `566-573` _(correctness)_  
  loading_params iterates new_params.keys() and reassigns new_params[key]=np.array(...) on the passed-in dict (lines 567-569), mutating the caller's object as a side effect while also returning it. A caller that reuses the original dict (e.g. saved JSON params) will find list values silently replaced by numpy arrays. The method name and docstring ('Returns Dictionary with list values converted') imply a normalized copy, not in-place mutation.  
  **Fix:** Build and return a new dict: out=dict(new_params); for k,v: out[k]=np.array(v) if list; return out.
- **Duplicated dynamic-port + TranFn b_type logic across change_params/update_params/__init__** — `599-626 and 645-669` _(duplication)_  
  The 'get_inputs -> recompute in_ports -> update_Block' block and the TranFn numerator/denominator b_type+_init_start_ logic are near-identical in change_params (599-626) and update_params (645-669), with the dynamic-port portion partly mirrored in __init__ (115-127). A fix to one path will not propagate to the others (divergence risk).  
  **Fix:** Extract private helpers (e.g. _refresh_dynamic_ports() and _refresh_tranfn_btype()) and call them from change_params, update_params, and __init__.
- **DBlock.dirty is a write-only attribute, never initialized or read** — `629, 672` _(dead-code)_  
  change_params() (line 629) and update_params() (line 672) set self.dirty=True on the DBlock, but DBlock.__init__ never initializes self.dirty and no code reads block-level .dirty. A repo-wide grep shows every meaningful .dirty read/consume is on the model/dsim (simulation_model.dirty, dsim.dirty), never on a block instance. These two assignments are dead write state and misleading.  
  **Fix:** Remove the two self.dirty=True assignments, or route the dirty flag to the owning model (dsim.dirty / simulation_model.dirty) which is what other code actually consumes.

### `lib/simulation/connection.py`

- **Orthogonal/feedback routing assumes ports always face right (start.x > finish.x heuristic)** — `107, 212` _(correctness)_  
  Lines 107 and 212 both compute is_feedback = start.x() > finish.x() and the feedback/forward stubs always extend +20/-20 in x (e.g. lines 120-124, 170-182, 222-225). For mirrored/rotated blocks or vertically stacked layouts where a port faces left or up, this yields visually overlapping/wrong routes. Cosmetic only, not a simulation-correctness bug.  
  **Fix:** Derive stub direction from actual port orientation/side rather than the relative x of endpoints.
- **collision() uses inconsistent distance metrics across its two branches, mis-detecting clicks on near-vertical/horizontal segments** — `396-404` _(correctness)_  
  In the zero-length branch (line 398) dist_sq = u.x()**2 + u.y()**2 is squared Euclidean distance; in the segment branch (line 402) dist_sq = (m_coords - projection).manhattanLength() is L1 distance. Both are compared to the same line_threshold (line 404), so the units are incompatible and the threshold is inconsistent. The zero-length branch is largely dead anyway since the bounding-box contains() check at line 389 already gates degenerate segments, so practical impact is limited to the L1 overestimation on diagonal offsets.  
  **Fix:** Use a single consistent metric (e.g. math.hypot of the projection delta for true Euclidean distance) in both branches and rename dist_sq to dist since it is not squared in the main branch.
- **segments deepcopy fallback silently produces empty collision list** — `455-459` _(error-handling)_  
  Lines 455-459: if deepcopy of the segments list raises, the fallback sets segments to [] with no log, leaving the copied line with no collision geometry so it becomes unclickable with no indication why.  
  **Fix:** Log the failure (logger.warning). Segments can also be regenerated from points via create_trajectory rather than dropped to an empty list.

### `lib/simulation/wire_router.py`

- **route_all_lines / route_line do repeated O(n) block lookups per line, O(n*m) overall** — `168-179` _(performance)_  
  route_line scans all blocks twice via next(...) (lines 135-136) to find src and dst, then route_all_lines scans all blocks twice more for the same src/dst (lines 174-175). With L lines and B blocks this is 4*L*B linear scans plus an A* per line. Routing is a user-triggered batch op, not a per-timestep hot path, so impact is bounded to large-diagram autoroute latency.  
  **Fix:** Build a {block.name: block} dict once in route_all_lines and pass it down (or have route_line return the resolved start/end) to avoid re-scanning the block list 4x per line.

### `modern_ui/builders/menu_builder.py`

- **Examples menu reads directory on os.listdir without guarding against unreadable dir; misleading log/no-handling** — `61-78` _(error-handling)_  
  Confirmed. _populate_examples_menu (lines 61-65) checks os.path.exists(examples_dir) then calls os.listdir with no try/except. os.path.exists being True does not guarantee listability (permission denied, or the path is a file), so an OSError from os.listdir would propagate up through _create_file_menu -> setup_menubar, aborting menu construction. It is an edge case but it is on the startup path.  
  **Fix:** Wrap the listdir in try/except OSError and fall back to a disabled 'No examples found' / 'directory not found' action, mirroring the existing else branch.

### `modern_ui/controllers/simulation_controller.py`

- **format_val function redefined on every loop iteration in verification report** — `229-238` _(performance)_  
  Confirmed. format_val is defined inside the `for name, info in scope_convergence.items():` loop (def at line 229, loop at 223), so the closure is rebuilt each iteration despite not depending on the loop variable. Functionally harmless; minor repeated work. This runs once per simulation completion (not in the per-timestep hot path), so impact is negligible, but the smell is real.  
  **Fix:** Hoist format_val to a module-level or method-level helper defined once outside the loop.
- **Misleading log message references non-existent 'execution_initialize' while the code probes 'execution_init'** — `55-79` _(maintainability)_  
  Confirmed. start() guards with hasattr(self.dsim, 'execution_init') at line 55, but the else branch at line 79 logs "DSim does not have execution_initialize method". The mismatched name would send a debugger grepping for the wrong attribute.  
  **Fix:** Change the log message to reference 'execution_init' to match the actual attribute checked at line 55.
- **QMessageBox created without a parent in simulation start failure path** — `70-76` _(maintainability)_  
  Confirmed. Lines 70-76 construct msgBox = QMessageBox() with no parent and exec_() it. A parentless modal dialog does not center on / stay attached to the main window, may not inherit the app stylesheet/theme, and on some platforms can appear behind the main window. Elsewhere the app parents dialogs (e.g. menu_builder._show_about uses QMessageBox.about(self.window, ...)).  
  **Fix:** Parent the dialog (e.g. QMessageBox.critical(parent, ...)) by threading a parent widget reference through, or emit a signal the UI layer handles, keeping the dialog parented and themed.

### `modern_ui/controllers/tuning_controller.py`

- **TuningController applies slider value to indexed list param but silently no-ops on out-of-range / non-list** — `109-118` _(error-handling)_  
  Confirmed, with context that narrows the impact. Lines 109-118: indexed params only update when isinstance(base_val,(list,tuple)) and idx<len(base_val); otherwise the change is silently dropped (no log, no status). The else branch writes block.params[param_name]=value unconditionally. CONTEXT: the tuning panel's add_parameter (modern_ui/widgets/tuning_panel.py:372-387) only creates an indexed slider when base_val is already a list/tuple and idx is in range, and only creates a non-indexed slider when the param exists and is numeric — so the 'spurious key' and 'numpy array is common' framings are overstated at slider-creation time. The real residual risk is that params can change type (e.g. to np.ndarray) between creation and apply, after which the indexed apply silently no-ops with stale params re-simulated.  
  **Fix:** Also accept np.ndarray in the indexed branch and emit a logged warning + status message when an indexed update is skipped, so silent stale-tuning is visible. The unconditional else write is low-risk given panel gating but a defensive existence check is cheap.

### `modern_ui/interactions/interaction_manager.py`

- **State.DRAGGING_BLOCK / CONNECTING / CONFIGURING enum members are defined but never used** — `12-16` _(dead-code)_  
  Partially as described, with one correction. DRAGGING_BLOCK (line 12) is never assigned or compared anywhere in the codebase (grep returns only its definition). CONFIGURING (line 16) of THIS State enum is unused; the only other 'CONFIGURING' hit is an unrelated separate enum in main_window.py line 211 (auto()-based), so this enum's member is genuinely dead. CORRECTION: the finding lumps CONNECTING in as unused, but State.CONNECTING IS used — connection_manager.py line 59 sets self.canvas.state = State.CONNECTING (and a test asserts it). So CONNECTING is NOT dead. Only DRAGGING_BLOCK and CONFIGURING are unused here.  
  **Fix:** Remove DRAGGING_BLOCK and CONFIGURING from this State enum (block drags use State.DRAGGING). Keep CONNECTING — it is actively used by connection_manager.
- **Line-segment drag: segment_index can become stale / out of range after inserting a split point** — `193-215` _(correctness)_  
  Confirmed in part, with severity lowered. Lines 193-200: when len(line.points)==2, the midpoint computation correctly branches on is_horizontal, but the re-derived segment index is `1 if pos.y() > p1.y() else 0` unconditionally (line 199) — it always uses pos.y() even in the vertical (!is_horizontal) branch where the relevant axis is X. This is a genuine axis-mismatch heuristic that can pick the visually wrong bend index when splitting a vertical 2-point line. CORRECTION to the IndexError concern: in the len==2 split path the list becomes 3 points and segment_index is 0 or 1, so segment_index+1 is 1 or 2 — always in range; the broad-except IndexError scenario the finding hypothesizes does not actually arise on this specific path. The general lack of a `0 <= segment_index < len(points)-1` guard remains a fair defensive-coding note for non-split paths where dragging_item came from elsewhere.  
  **Fix:** Use pos.x() for the index heuristic in the !is_horizontal branch (mirror the insert axis), and add an explicit bounds check before the setX/setY block. Add unit tests for splitting both a horizontal and a vertical 2-point line.

### `modern_ui/main_window.py`

- **Duplicated module docstring at top of main_window.py** — `1-9` _(dead-code)_  
  main_window.py lines 1-4 and lines 6-9 contain identical triple-quoted string literals. The first is the module docstring; the second (lines 6-9) is a no-op expression statement and dead code, a copy/paste or merge artifact.  
  **Fix:** Delete the duplicate string literal (lines 6-9), keeping a single module docstring.
- **Logging configured twice with conflicting setups at import vs main()** — `38` _(design)_  
  main_window.py line 38 calls LoggingHelper.setup_logging(level='INFO', log_file='diablos_modern.log') at module import time. LoggingHelper.setup_logging (lib/improvements.py lines 301-337) calls logging.basicConfig(..., force=True), which removes and replaces all existing root handlers. diablos_modern.py imports main_window first (line 64), triggering that import-time config, then calls lib.logging_config.setup_logging() at line 70 (which uses logging.config.dictConfig from config/logging.json, or a basicConfig fallback). The second call overrides the first, so the import-time LoggingHelper config is wasted work and is order-dependent; it also runs for any importer including tests. Which config 'wins' is non-obvious.  
  **Fix:** Pick a single logging owner (the entry point diablos_modern.py via lib.logging_config) and remove the import-time LoggingHelper.setup_logging() call from main_window.py. Modules should only do logger = logging.getLogger(__name__) at import time.
- **__init__ is a god-constructor instantiating ~10 managers and doing all wiring inline** — `45-201` _(design)_  
  ModernDiaBloSWindow.__init__ spans lines 45-201 (~156 lines): it constructs ten managers via inline local imports (lines 66-94), runs state init, builds the menu builder, window/menubar/toolbar/layout/statusbar, creates and wires four dock widgets (variable editor, workspace editor, minimap, tuning panel), the tuning controller, toast, command palette, a global Ctrl+K shortcut, two QTimers (update and autosave), and three QTimer.singleShot callbacks. The constructor concentrates all coupling and ordering constraints in one place (note the 'Must be before state init' comment at line 65), making startup ordering bugs easy to introduce. The per-manager inline 'from modern_ui.managers... import' is repetitive.  
  **Fix:** Extract grouped initialization into private helpers (e.g. _init_managers(), _init_docks(), _init_timers(), _wire_signals()) called in sequence from __init__, and hoist manager imports to module top (or a single registry loop) so ordering is explicit and the constructor scannable.

### `modern_ui/managers/appearance_manager.py`

- **on_theme_changed sets theme_status text that is immediately overwritten** — `87-91` _(dead-code)_  
  on_theme_changed (appearance_manager.py) sets `window.theme_status` to "Dark Theme"/"Light Theme" (lines 87-88), then immediately calls `self.update_statusbar_colors()` (line 91), which unconditionally re-sets `window.theme_status` to the `f"{theme_label} · {palette_label}"` format (lines 116-121, e.g. "Dark · Default"). The first setText is dead from the user's perspective and documents an inconsistent intended label. Confirmed.  
  **Fix:** Remove the redundant `theme_status.setText(theme_name)` block (lines 87-88) and the now-unused `theme_name` local, since update_statusbar_colors() (line 121) is the authoritative place that sets the pill text.

### `modern_ui/managers/clipboard_manager.py`

- **Copy/paste failures are silently swallowed, returning no result with only a log line** — `112-113, 287-288` _(error-handling)_  
  copy_selected_blocks (except at 112-113) and paste_blocks (except at 287-288) wrap their entire body in `except Exception as e: logger.error(...)` with no re-raise and no user feedback. A mid-operation failure leaves the clipboard partially populated or the paste partially applied while the user sees nothing. Notably paste_blocks pushes an undo state at line 124 before the bulk of work, so a mid-way failure can leave a half-pasted diagram with no error surfaced. This broad-swallow pattern recurs across these managers (connection_manager 153/178/205/243, etc.).  
  **Fix:** Surface failures to the user (e.g. via self.canvas.simulation_status_changed.emit(...)) so a failed paste/copy is visible, and narrow the except to the operations that can realistically fail rather than wrapping the whole method.
- **Paste shares deep-copied params across multiple pastes via shallow .copy()** — `168, 196` _(correctness)_  
  On copy, params are deep-copied once into the clipboard (line 49: copy.deepcopy(block.params)). On each paste the new block receives block_data['params'].copy() -- shallow -- at line 168 (Subsystem branch) and line 196 (DBlock branch). DBlock.__init__ also only shallow-copies (block.py line 90). Pasting the same clipboard entry multiple times yields blocks whose params share the SAME nested mutable objects (the single clipboard deepcopy). Editing one pasted block's nested param (list/array/dict) mutates the clipboard entry and sibling pastes. Per the state-in-params contract, mutable params are common.  
  **Fix:** Use copy.deepcopy(block_data['params']) at paste time (lines 168 and 196) so each pasted block gets an independent params dict and repeated pastes do not alias the stored clipboard copy.

### `modern_ui/managers/connection_manager.py`

- **check_line_clicks calls handle_line_click with wrong number of arguments (latent TypeError in dead code)** — `169-178, 247` _(dead-code)_  
  check_line_clicks() (defined at line 169) calls self.handle_line_click(line, pos) at line 175 with two positional arguments, but handle_line_click is defined at line 247 as (self, line, collision_result, pos) requiring three. Any invocation raises TypeError, which is swallowed by the surrounding `except Exception` at line 177-178 (logged only). A codebase grep confirms check_line_clicks has NO callers anywhere; the live click path uses get_clicked_line (line 158, via line.collision) and routes through interaction_manager to handle_line_click with the correct 3 args. So this is dead code containing a guaranteed bug if ever re-wired.  
  **Fix:** Delete check_line_clicks(), point_near_line() and point_to_line_distance() if they are unused (the live path uses get_clicked_line via line.collision). If kept, compute collision_result from line.collision(pos) and call self.handle_line_click(line, result, pos).
- **ConnectionManager wraps every method in broad except Exception that returns a falsy default** — `32-47 (10 occurrences in file)` _(error-handling)_  
  check_port_clicks (32-47) and 9 other methods (10 'except Exception as e:' occurrences total, count verified) catch all exceptions, log, and return False/None. This masks real programming errors (e.g. a bug in port_collision) as 'no port clicked', making connection bugs hard to diagnose and silently dropping user interactions.  
  **Fix:** Narrow caught exception types to expected ones, or re-raise unexpected errors (at least in debug builds) so genuine logic bugs are not hidden behind a benign False/None.

### `modern_ui/managers/history_manager.py`

- **Snapshot captures block.height while clipboard captures height_base, producing inconsistent restored geometry** — `99` _(consistency)_  
  _capture_state stores coords as (block.left, block.top, block.width, block.height) at line 99, whereas ClipboardManager.copy_selected_blocks stores QRect(block.left, block.top, block.width, block.height_base) at clipboard line 41. In lib/simulation/block.py, height is the rendered/auto-expanded height (update_Block grows it to min_height via calculate_min_size at lines 365-373 for multi-port blocks) while height_base is the configured base (line 79). On restore, coords = QRect(*block_data['coords']) (history line 161) feeds the auto-expanded height into DBlock, where line 78-79 sets self.height = rect.height() then self.height_base = self.height -- baking the expanded height into the base. Repeated undo/redo on multi-port blocks can therefore drift the height monotonically.  
  **Fix:** Use height_base consistently for serialized geometry in both the history snapshot (line 99) and clipboard copy (line 41), since height_base is the configured value used by add/paste paths.

### `modern_ui/managers/project_manager.py`

- **update_recent_files_menu silently purges recent entries on any os.path.exists transient miss and persists the deletion** — `70-71` _(error-handling)_  
  update_recent_files_menu filters to currently-existing paths (`recent_files = [f for f in recent_files if os.path.exists(f)]`, line 70) then immediately persists the pruned list via `self.settings.setValue("recent_files", recent_files)` (line 71). If a file is temporarily unavailable (network/removable drive offline, transient lock), the entry is permanently dropped from QSettings on the next passive menu refresh, with no recovery when the drive returns. Confirmed. Note: severity is mitigated because this whole QSettings recent-files path is the secondary/divergent store (see finding 2); the user-facing menu is primarily backed by RecentFilesManager, which prunes persistently only on a user-initiated open miss (recent_files_manager.py lines 143-154).  
  **Fix:** Filter for display only without rewriting the stored list on passive refreshes; prune persistently only when a user-initiated open fails, mirroring RecentFilesManager.open (lines 143-154). Or, per finding 2, retire this QSettings path entirely.

### `modern_ui/managers/view_actions_manager.py`

- **fit_to_window sets canvas zoom/pan directly, bypassing set_zoom and desyncing the toolbar zoom rocker** — `114-127` _(consistency)_  
  fit_to_window assigns `window.canvas.zoom_factor` (line 114) and `window.canvas.pan_offset` (line 123) directly and updates only `window.zoom_status` (line 127). Unlike set_zoom (line 29) which calls `canvas.set_zoom(factor)`, this path bypasses the canvas zoom API. The toolbar zoom rocker is driven by `toolbar.zoom_changed` (status_bar_manager.py line 155), so if set_zoom emits/routes through that signal while fit_to_window does not, the rocker keeps showing the pre-fit zoom and a subsequent rocker interaction can jump. The bypass of the public API is verified; the exact desync depends on whether canvas.set_zoom emits zoom_changed (runtime/canvas detail), so the consequence is plausible but not 100% confirmed from these files alone.  
  **Fix:** Route the computed zoom through `window.canvas.set_zoom(new_zoom)` instead of assigning `zoom_factor` directly (line 114), then apply the pan offset, so the toolbar rocker and any other zoom listeners stay in sync.
- **Zoom status text format diverges between ViewActionsManager and the status-bar baseline ("100%" vs "zoom 100%")** — `30` _(consistency)_  
  status_bar_manager.py initializes the zoom pill as "zoom 100%" (line 133) and wires the toolbar's zoom_changed signal to set `f"zoom {int(round(f*100))}%"` (line 156). ViewActionsManager set_zoom/zoom_in/zoom_out/fit_to_window all set the pill via `window.zoom_status.setText(f"{int(window.canvas.zoom_factor * 100)}%")` (lines 30, 36, 43, 127) — no "zoom " prefix. Whichever path fires last, the same widget alternates between "zoom 120%" and "120%". Cosmetic but a real, verified inconsistency.  
  **Fix:** Use one format everywhere. Either add the "zoom " prefix to all four ViewActionsManager setters (lines 30, 36, 43, 127) to match the status bar, or drop it from the status_bar_manager init (line 133) and the zoom_changed lambda (line 156).

### `modern_ui/platform_config.py`

- **PlatformConfig assumes QApplication.primaryScreen() is non-None** — `26-31` _(error-handling)_  
  _detect_platform() (platform_config.py lines 26-31) calls QApplication.primaryScreen() and immediately dereferences it (screen.devicePixelRatio() line 27, screen.availableGeometry() line 29) with no None check. QApplication.primaryScreen() returns None if no QApplication exists yet or in a headless/no-display environment, producing an AttributeError that is harder to diagnose than an explicit guard.  
  **Fix:** Guard for None: if QApplication.instance() is None or QApplication.primaryScreen() is None, raise a clear RuntimeError ('PlatformConfig requires an active QApplication with a display') or fall back to sane defaults so the failure mode is explicit.

### `modern_ui/renderers/block_renderer.py`

- **draw_icon exception swallowed to a warning, leaving path possibly inconsistent** — `147-156` _(error-handling)_  
  Lines 147-156: a failing block_instance.draw_icon is caught by a broad 'except Exception' and logged at warning (line 153), then flow falls through to _draw_legacy_icon (line 156). The fallback IS deterministic - _draw_legacy_icon guards each source-block branch on path.isEmpty() (lines 413,415,417), so an empty path correctly triggers the legacy icon. The genuine concern is that a broad except logging at warning every frame would spam logs and could mask real errors. The finding's claim that the path is 'left inconsistent' is overstated; the fallback recovers cleanly.  
  **Fix:** Narrow the except or rate-limit/log-once-per-block to avoid per-frame warning spam; consider downgrading repeated occurrences to debug after the first.
- **Stateless renderer mutates the block's persistent QFont in place** — `179-181` _(design)_  
  Lines 179-181: 'font = block.font; font.setWeight(400); painter.setFont(font)'. block.font is a persistent QFont stored on the model (lib/simulation/block.py:139), so the class documented as a 'Stateless renderer' (line 50) mutates shared model state. Currently harmless (Qt weight 400 == Normal, and it is re-applied identically each paint) but couples view to model and is fragile if the model's font weight default changes.  
  **Fix:** Copy first: font = QFont(block.font); font.setWeight(QFont.Normal); painter.setFont(font), so rendering never mutates the model's font object.
- **Per-frame imports inside hot paint methods** — `272, 672` _(performance)_  
  draw_port_labels runs 'from PyQt5.QtGui import QFont, QFontMetrics' (line 272) on every paint of every multi-port block, and _draw_sample_rate_indicator runs 'import math' (line 672) on every paint of discrete blocks with sample_time>0. These run in the per-frame render path. Python caches modules in sys.modules so the cost is a dict lookup + local binding, not a real re-import, so impact is minor.  
  **Fix:** Move QFont/QFontMetrics and math to module top-level imports so they are resolved once. Low effort, marginal benefit.
- **draw_block mutates painter state with no outer save/restore** — `56-200` _(correctness)_  
  draw_block (lines 56-200) sets brush/pen/font/transform repeatedly but never wraps the body in painter.save()/restore(). Only the small helpers (_draw_resize_handles line 348-367, _draw_sample_rate_indicator line 688-696) save/restore. Font weight/bold/italic mutations applied via helpers and the name-draw (line 180 setWeight(400)) can leak to the next block since not every block fully re-specifies font attributes. In practice each block re-sets pen/brush/color so color leakage is unlikely, making this defensive rather than an active bug.  
  **Fix:** Wrap the body of draw_block in painter.save() ... painter.restore() (try/finally) so font/pen/brush state cannot leak between blocks or back to the caller.
- **Font helper restoration is not exception-safe and _draw_corner_labels never resets bold/italic** — `617-625` _(correctness)_  
  _draw_corner_labels (lines 617-625) changes only point size (orig-1 then back to orig) and never sets or restores bold/italic, so it inherits whatever bold/italic state a prior icon helper left on the painter font. It is only used for the PID block (line 514, called after _draw_centered_text 'PID' bold=True which restores bold to False on line 604, so currently benign). Other helpers (_draw_centered_text 594-605, _draw_text_icon 573-592, _draw_corner_label 607-615) restore font fields manually but if painter.drawText raised, the mutated font would not be restored (no try/finally / save-restore).  
  **Fix:** Use painter.save()/restore() around each helper (or the whole draw_block) so restoration is guaranteed even on exception, and explicitly normalize bold/italic in _draw_corner_labels.
- **Per-frame QColor reconstruction and category substring scan** — `71-119, 207-208, 285-287` _(performance)_  
  theme_manager.get_color constructs a fresh QColor from a hex string on every call (theme_manager.py:484, no caching). draw_block calls it ~3-5x (lines 71, 101-102, 104-105, 108, 141, 177, 187), draw_ports 2x (207-208), draw_port_labels 2x (285-286), each going through _category_color_key substring scans (lines 41-45). For large diagrams this is repeated small allocations/parsing each frame.  
  **Fix:** Memoize resolved theme colors keyed by (theme, palette, color_name) and invalidate on theme/palette change; cache the category-key resolution per block. Note QColor allocation is cheap, so benefit only matters for very large diagrams.

### `modern_ui/renderers/canvas_renderer.py`

- **Broad try/except Exception silently swallows all render errors** — `195-196` _(error-handling)_  
  draw_hover_effects (195-196), draw_grid (59-60), draw_tag_hud (260-261), draw_validation_errors (285-286), and connection_renderer draw_port_value_chips (74-75) each catch Exception and only log, hiding programming errors (e.g. AttributeError) behind a log line so they never surface in tests.  
  **Fix:** Catch narrower expected exceptions, or at minimum log with exc_info=True and re-raise in debug builds so swallowed defects remain diagnosable.
- **_draw_error_indicator_on_block ignores block height for vertical placement** — `291-299` _(correctness)_  
  _draw_error_indicator_on_block reads left (291), top (292), width (293) but never height, then sets y = top - indicator_size/2 (297). The badge is pinned to the block's top-right corner regardless of block size. The docstring/comment says top-right, so this is likely intentional, but the partial-geometry read is inconsistent with draw_hover_effects (170-173), which fetches all four of left/top/width/height.  
  **Fix:** If top-right anchoring is intended, document it explicitly; otherwise fetch height for consistent anchoring and confirm the badge does not overlap the block title on small blocks.
- **QColor.setAlpha mutates returned theme color (mitigated by fresh QColor)** — `31-33` _(maintainability)_  
  canvas_renderer call sites mutate alpha in place on the QColor returned by theme_manager.get_color() (large_dot_color.setAlpha(180) at 33, fill_color.setAlpha(50) at 79, glow_color/center_color at 149/156, bg.setAlpha at 247, etc.). This is safe only because get_color currently returns a fresh QColor; it would corrupt shared theme state if get_color ever cached. Notably connection_renderer already defends against this with QColor(success) (33), QColor(active_connection_color) (147,175), so the two files are inconsistent.  
  **Fix:** In canvas_renderer, copy before mutating, e.g. c = QColor(theme_manager.get_color('grid_dots')); c.setAlpha(180), matching the defensive copies already used in connection_renderer.
- **draw_temp_line lacks painter.isActive guard used elsewhere** — `90-98` _(consistency)_  
  draw_line (connection_renderer:118) and draw_port_value_chips (connection_renderer:28) start with `if not painter or not painter.isActive(): return`, but canvas_renderer's draw_temp_line (90-98), draw_selection_rect, draw_hover_effects, draw_grid, and draw_tag_hud have no such guard before calling painter.save()/draw ops. An inactive painter would raise inside these (then be swallowed by the broad except in some).  
  **Fix:** Apply the `if not painter or not painter.isActive(): return` guard uniformly across all draw_* entry points.

### `modern_ui/styles/qss_styles.py`

- **Theme switch only re-applies stylesheet to QMainWindow; other top-level windows keep stale QSS** — `555-569` _(correctness)_  
  qss_styles.py:555-567: on_theme_changed re-applies the resolved stylesheet only to top-level widgets where isinstance(w, QMainWindow). The palette is re-applied app-wide (app.setPalette, line 556), but any non-QMainWindow top-level window open at switch time (free-floating dialogs, detached panels, editors that set their own QSS) keeps stale QSS colors until reconstructed. The comment (lines 558-560) explains why app.setStyleSheet is avoided (pyqtgraph/OpenGL segfault) but does not explain why ordinary dialogs are skipped. Note several widgets (error_panel, command_palette, workspace_editor, variable_editor, property_editor, etc.) connect to theme_manager.theme_changed and re-style themselves independently, which mitigates the impact for those known widgets — so the practical risk is limited to top-level windows that rely on inherited QSS and do not self-subscribe. Hence low severity is appropriate.  
  **Fix:** Document why only QMainWindow is restyled (current comment only justifies skipping pyqtgraph windows, not dialogs), or invert the filter to restyle all top-level widgets except pyqtgraph/GraphicsLayoutWidget/OpenGL types so open dialogs retheme live.

### `modern_ui/widgets/animation_export_dialog.py`

- **ExportWorker QThread is never deleted, leaking thread objects** — `291` _(resource-leak)_  
  Line 291 `self.worker = ExportWorker(...)` reassigns self.worker on each _start_export with no deleteLater() on a prior worker and no reset to None in _on_export_finished (lines 300-314). After a failed export the user can re-export (controls are re-enabled on failure), orphaning the previous QThread QObject. On success the dialog accepts but the worker is not cleaned up either.  
  **Fix:** Connect self.worker.finished to self.worker.deleteLater() (or call it in _on_export_finished) and set self.worker = None; ensure any prior worker has finished before reassigning.
- **FPS/quality controls not re-enabled on successful re-check, but format radios stay disabled after failure** — `310` _(correctness)_  
  On export failure (lines 309-314) the code re-enables export/browse/fps/quality and calls _check_writers(), but _check_writers() only ever DISABLES radios (never calls setEnabled(True)). The radios were force-disabled in _start_export (lines 282-283). So after a failed export, both gif/mp4 radios remain disabled even when their writers are available — a real UX regression, not merely cosmetic.  
  **Fix:** Before calling _check_writers() in _on_export_finished, explicitly re-enable both radios (self.gif_radio.setEnabled(True); self.mp4_radio.setEnabled(True)) so _check_writers() is the single source of truth for disabling unavailable ones.
- **Export file written/created via worker with no overwrite confirmation and partial-file risk on cancel** — `319` _(error-handling)_  
  progress_callback raises InterruptedError mid-export on cancel (lines 44-45); the partially written gif/mp4 is left on disk with no cleanup. closeEvent (lines 316-322) calls self.worker.wait(3000) then proceeds via super().closeEvent regardless of whether the worker actually finished, so the thread may still be running after the dialog closes.  
  **Fix:** On InterruptedError/cancel, delete the partial output file; and either block close until the worker truly finishes or call self.worker.terminate() after the wait timeout. Note overwrite confirmation is partly mitigated by QFileDialog.getSaveFileName (line 248) but typing/using the default path bypasses it.

### `modern_ui/widgets/breadcrumb_bar.py`

- **Module-level theme_manager import placed after first use** — `85` _(maintainability)_  
  Verified: theme_manager is referenced at lines 55-57 inside set_path, but the 'from modern_ui.themes.theme_manager import theme_manager' import is at the bottom of the file (line 85). It works only because set_path is invoked at instance construction (line 25, run after module import completes), not at module import time. Constructing a BreadcrumbBar during this module's own import would raise NameError. Fragile and surprising; the comment even hedges about a circular-dep concern that other top-importing widgets (e.g. toast_notification.py, minimap_widget.py) show does not apply.  
  **Fix:** Move the import to the top of the module alongside the other imports. toast_notification.py and minimap_widget.py both import theme_manager at top with no circular-dependency issue.

### `modern_ui/widgets/command_palette.py`

- **Bare except: pass swallows OS errors when creating recents directory** — `60-65` _(error-handling)_  
  _recent_path uses `except OSError: pass` (lines 63-64) around makedirs; _load_recents and _save_recents similarly swallow OSError (and json.JSONDecodeError) for best-effort recents persistence. The finding itself correctly notes these are narrow OSError handlers, not bare excepts, and the only concern is the absence of a debug log so a persistently unwritable ~/.diablos gives no diagnostic trace. Low impact since recents are non-essential.  
  **Fix:** Optionally add logger.debug(...) in the except blocks of _recent_path, _load_recents, and _save_recents while keeping behavior best-effort.

### `modern_ui/widgets/minimap_widget.py`

- **Minimap recomputes diagram bounds/scale on every paintEvent over all blocks** — `104` _(performance)_  
  Verified: paintEvent unconditionally calls self._diagram_bounds = self._calculate_diagram_bounds() (line 104) and self._scale, self._offset = self._calculate_scale_and_offset(...) (line 105) on every repaint. _calculate_diagram_bounds iterates all blocks four times (min/max over b.left/top/width/height, lines 57-60). Despite the 'Cached bounds' comment (line 38), no caching/invalidation exists, so this is repeated O(n) work per repaint in a frequently-repainted overlay widget.  
  **Fix:** Cache bounds/scale and recompute only when the diagram or widget size changes — invalidate from refresh()/resizeEvent via a dirty flag rather than recomputing in every paintEvent.
- **Minimap viewport rect divides by zoom_factor with no zero/negative guard** — `167` _(error-handling)_  
  Verified: _draw_viewport_rect divides by self.canvas.zoom_factor at lines 167-170 with no zero/negative guard; a zoom_factor of 0 would raise ZeroDivisionError during paint. _pan_to_minimap_pos guards self._scale <= 0 (line 198) and multiplies by zoom_factor at lines 210-211 (no div-by-zero there, but still assumes a sane value), yet neither method guards zoom_factor against 0.  
  **Fix:** Guard against zoom_factor <= 0 (early-return or clamp to a small positive value) before the divisions in _draw_viewport_rect, mirroring the existing _scale<=0 guard.

### `modern_ui/widgets/modern_canvas.py`

- **Duplicate-connection check is redundant with input-port-already-connected check** — `1088` _(duplication)_  
  Two back-to-back loops over existing_lines: 1089-1097 flags 'Connection already exists' (matches src+srcport+dst+dstport); 1100-1105 flags 'Input port already connected' (matches dst+dstport). An exact-duplicate connection satisfies both, producing two overlapping error messages and a second O(n) scan.  
  **Fix:** Merge into a single pass: report 'Input port already connected' when the destination port matches; the exact-duplicate case is a strict subset, so only one message is needed.
- **Dynamic class creation per validation call instead of a simple object** — `1114` _(performance)_  
  type('TempLine', (), {...})() at line 1114 builds a brand-new class object every connection-validation call just to hold four attributes (srcblock, srcport, dstblock, dstport). Wasteful and obscure versus a lightweight namespace.  
  **Fix:** Use types.SimpleNamespace(srcblock=..., srcport=..., dstblock=..., dstport=...) or a module-level lightweight class to avoid recreating a type each call.
- **ValidationHelper failure is silently downgraded to debug** — `1127` _(error-handling)_  
  The inner try at 1108-1128 catches all exceptions from ValidationHelper.validate_block_connections and logs at debug level as 'not available or failed'. A genuine bug in the helper therefore passes silently; validation_errors is not extended and the connection is treated as valid for that check (final return at 1131 depends only on errors accumulated elsewhere).  
  **Fix:** Distinguish import/availability errors from execution errors; log execution failures at warning/error level so a broken validator is not mistaken for a passing connection.
- **add_block_from_palette pushes undo even when downstream emit could fail** — `175-183` _(correctness)_  
  Lines 152-192: the whole add path is inside one broad `try/except Exception as e: ... return None`. self.dsim.add_block(menu_block, position) is at line 160; if a later step (_push_undo line 175, block_selected.emit line 178, or update line 181) raises, the block already exists in dsim but the method returns None, so caller and dsim disagree. Inconsistency window is real, though the post-add steps rarely raise in practice.  
  **Fix:** Narrow the try to wrap only dsim.add_block; on later failure either return the created block or roll back the add so dsim state and return value stay consistent.
- **Swallowed exception in paintEvent live-chips hot path** — `281-286` _(error-handling)_  
  Lines 281-286 wrap draw_port_value_chips in `try: ... except Exception: pass`, silently hiding any chip-render error on every paint while a simulation runs. Confirmed verbatim. Severity lowered to low: the chips are a non-critical live overlay explicitly hidden during drag/zoom, and the surrounding paint still completes; a silent failure here degrades only the overlay, not core function.  
  **Fix:** Replace `pass` with `logger.debug(...)` so genuine chip-render bugs are diagnosable without spamming warnings each frame.
- **_reroute_affected_lines sets dirty but does not request repaint; broad except** — `322-332` _(correctness)_  
  Lines 322-332: after route_all_lines(affected,...) it sets self.dsim.dirty=True but never calls self.update(); the broad `except Exception as e: logger.error(...)` catches all routing failures. The missing repaint is real but typically benign because this is invoked during a move where the caller's interaction flow triggers update() anyway. The broad except at least logs (not silent).  
  **Fix:** Optionally call self.update() after a successful reroute for safety; consider logging affected block names. Low priority.
- **Connection start state set via instance attrs then read implicitly by manager** — `543-545` _(maintainability)_  
  Lines 543-545 set self.line_start_block/self.line_start_port (properties proxying canvas_state.connection.start_block/start_port, defined at 1466-1480) then call self._finish_line_creation(target_block, target_port_index). _finish_line_creation(end_block, end_port) at line 591 delegates to connection_manager.finish_line_creation(end_block, end_port) which receives only the end params, so the manager must read the start from canvas_state. Implicit temporal coupling confirmed.  
  **Fix:** Pass source explicitly: finish_line_creation(start_block, start_port, end_block, end_port) rather than communicating through shared canvas_state.
- **Swallowed exception in mouseMoveEvent cursor emit** — `614-618` _(error-handling)_  
  Lines 614-618: screen_to_world(event.pos()) and cursor_moved.emit(...) are wrapped in `try: ... except Exception: pass`, silently dropping any coordinate-transform failure during mouse move. Confirmed verbatim. Impact limited to the status-bar cursor readout.  
  **Fix:** Add logger.debug(...) inside the except so a broken transform is observable.
- **Duplicate PerformanceHelper instantiation; self.performance is dead** — `62,80` _(dead-code)_  
  Line 62 creates self.perf_helper = PerformanceHelper() (used in paintEvent timers) and line 80 creates self.performance = PerformanceHelper(). Grep across the file shows self.performance has exactly one occurrence (the assignment at line 80) and is never read. Dead allocation confirmed.  
  **Fix:** Remove the unused self.performance assignment at line 80.
- **Paste pushes undo state before validating anything will be created** — `800` _(correctness)_  
  _push_undo('Paste') at line 800 runs before the block-creation loop (809-845). If add_block returns falsy for every clipboard entry or an exception is swallowed at 847, the undo stack gains a no-op 'Paste' entry, requiring a double-undo for no visible effect.  
  **Fix:** Track whether any new_block was actually created and only push undo when at least one block was added, or roll back the pushed state on failure.
- **Redundant 'if clipboard_blocks' guard in _paste_blocks** — `803` _(dead-code)_  
  Lines 796-797 already 'return' when clipboard_blocks is empty, so 'if clipboard_blocks:' at line 803 is always True - redundant code that adds nesting.  
  **Fix:** Remove the redundant inner 'if clipboard_blocks:' guard and dedent its body.
- **Broad exception handlers swallow errors and only log** — `847` _(error-handling)_  
  Multiple methods wrap large bodies in 'except Exception' that only logs: _paste_blocks (847-848), clear_canvas (923-924), auto_route_lines (940-941), dragEnterEvent (1006), dragMoveEvent (1021), dropEvent (1053), _find_menu_block_by_name (1065-1066), _validate_connection (1132-1134). These hide real bugs (KeyError on missing clipboard keys, AttributeError) and can leave partial state without user feedback. Note drag handlers at 1006/1021 do call event.ignore() and _validate_connection at 1132 does return an error tuple, so not all are fully silent.  
  **Fix:** Catch narrower exception types where possible, surface a user-visible error for paste/clear/route failures, and avoid wrapping the entire method body so unexpected programmer errors are not silently swallowed.
- **ModernCanvas is a god object wiring 10+ managers and many facade methods** — `88-112` _(design)_  
  Lines 88-112 construct 10 managers/controllers (interaction, history, menu, selection, clipboard, zoom_pan, connection, rendering, drag_resize, _sim_controller) plus analyzer and 3 renderers, and the class exposes many one-line delegating facade methods (e.g. _render_ports->rendering_manager, _finish_line_creation->connection_manager, start_drag->drag_resize_manager). God-object/facade smell is real. Severity lowered to low: this is a subjective design observation, not a defect, and the delegation pattern is already a deliberate decomposition into managers.  
  **Fix:** Optional: group managers behind a context/registry and drop trivial pass-throughs where callers can use managers directly; not a functional issue.

### `modern_ui/widgets/modern_palette.py`

- **Many block instances are constructed twice per palette row for tooltip + category + glyph** — `101-124` _(performance)_  
  _build_tooltip instantiates the block class (`inst = block_cls()`, line 105). _block_category_name instantiates it again (`inst = cls()`, line 297) and is invoked from _BlockGlyphLabel.paintEvent (line 271) on every repaint. _categorize_blocks instantiates it a third time during load (`inst = cls()`, line 701). For a library of dozens of blocks this multiplies constructor work at palette build and, more importantly, reconstructs a block on every glyph paint event.  
  **Fix:** Compute the category and tooltip data once per menu_block (e.g. during _categorize_blocks) and pass them into the row/glyph; cache the category in _BlockGlyphLabel so paintEvent does not call _block_category_name (and thus cls()) each repaint.
- **theme_manager.theme_changed connections are never disconnected when rows/sections are destroyed** — `99, 265, 559` _(resource-leak)_  
  CompactBlockRow (line 99), _BlockGlyphLabel (line 265), and _CategorySection (line 559) connect to the long-lived theme_manager.theme_changed singleton. refresh_blocks removes old sections via `w.setParent(None)` (line 799) with no deleteLater and no explicit disconnect. setParent(None) reparents but does not destroy the widget; the singleton's connection holds a reference to the receiver, so orphaned rows/glyphs/sections can survive and still receive theme_changed callbacks until GC, restyling/repainting dead widgets.  
  **Fix:** In refresh_blocks call w.deleteLater() (deleting the section deletes its child rows/glyphs and releases their connections) instead of bare setParent(None), or disconnect in a cleanup method.

### `modern_ui/widgets/property_editor.py`

- **Inconsistent block-name resolution between handlers** — `1136` _(consistency)_  
  _on_name_changed (line 1136) emits with self.block.name directly, whereas _on_property_changed (line 1123) and _on_port_count_changed (line 747) use getattr(self.block,'name','Unknown'). If a block lacked a name attribute, the name handler would raise AttributeError where the others degrade gracefully.  
  **Fix:** Use getattr(self.block, 'name', 'Unknown') in _on_name_changed for consistency.
- **Parameter with a legitimate None default can never be reset** — `1140-1143` _(correctness)_  
  _reset_param early-returns when default is None (lines 1141-1143), conflating a genuine None default with no recorded default. _defaults[key] is always populated in _add_param_row (line 792), so a param whose true default is None silently cannot be reset.  
  **Fix:** Distinguish missing keys from None defaults using a sentinel: if key not in self._defaults: return, then handle None explicitly.
- **_reset_param casts can raise on string/workspace-variable defaults** — `1149-1154` _(error-handling)_  
  For QSpinBox/SliderSpinBox/QDoubleSpinBox editors, _reset_param calls int(default)/float(default) (lines 1150,1152,1154). default comes from meta.get('default', value); if a numeric editor's recorded default is a non-numeric string (e.g. workspace-variable name), the cast raises an uncaught exception on reset-button click.  
  **Fix:** Wrap the numeric casts in try/except (or validate default type) and fall back to no-op/setText on failure.
- **Workspace read swallows all exceptions silently** — `420-423` _(error-handling)_  
  WorkspaceManager().variables is wrapped in bare 'except Exception: ws = {}' (420-423) with no logging, hiding a real WorkspaceManager init/read failure behind an empty-workspace display.  
  **Fix:** Log via self.logger.debug in the except so the cause is recoverable; ideally catch the specific expected exception type instead of bare Exception.
- **Silent broad except swallows validation errors without logging** — `456-461` _(error-handling)_  
  Validation block (456-461) catches Exception and resets errors=[] with no logging; the elif at 480 then renders the 'no issues detected' panel, so a raising/broken diagram_validator silently presents an 'all good' state. The Connections section at 601-602 does log via self.logger.debug, making the inconsistency clear.  
  **Fix:** Add self.logger.debug('validator failed: %s', e) in the except (mirroring line 601-602) so validator failures are diagnosable rather than masked as a passing state.
- **Repeated local imports in per-widget factory helpers** — `502-503, 513, 531-532` _(performance)_  
  _apply_label_color (502-503: imports QPalette/QColor/Qt), _mk_kv_label (513: import sys), and _mk_kv_value (531-532: import sys, QFont) re-import on every call. These helpers run once per label and the inspector builds many labels per rebuild, so the per-call import lookup is redundant work in a UI build path.  
  **Fix:** Hoist these standard PyQt/sys imports to module top level to avoid per-call import machinery; impact is minor (CPython caches modules in sys.modules) so severity stays low.
- **Integer spinbox range too narrow for large-count params** — `868-869` _(correctness)_  
  Integer params get a fixed QSpinBox range of -999999..999999 (line 869) then setValue(value) (line 870). QSpinBox clamps set values to range, so an int param whose default exceeds 999999 is silently clamped on display and larger values cannot be entered.  
  **Fix:** Derive range from metadata when present, or auto-expand based on incoming value (e.g. max(abs(value)*10, 999999)).
- **Slider hidden for float param of exactly 0.0** — `924-926` _(design)_  
  _should_show_slider (line 924) returns False when value==0, so a tunable float currently equal to 0.0 never gets a slider, making slider eligibility depend on the current value rather than the param's nature. This early return also masks a degenerate span in _get_slider_range (line 931: span=abs(value)*10), which would yield range [0,0] for value==0.  
  **Fix:** Use presence of meta['range'] / a sensible default span to decide eligibility; if value==0 and no range, fall back to a fixed default span (e.g. [-1,1]) so behavior is value-independent and _get_slider_range never produces a zero span.

### `modern_ui/widgets/tikz_export_dialog.py`

- **Broad except Exception swallows preview errors into the text body** — `283` _(error-handling)_  
  _update_preview (lines 279-285) catches `except Exception as e`, logs at error level, and renders the message as a TikZ comment in the preview pane, masking programming bugs (AttributeError, etc.) as user-facing preview text. _copy_to_clipboard (303) and _export (321) have similar broad catches but those surface via QMessageBox and log, so are less concerning.  
  **Fix:** Use logger.exception(...) (full traceback) in the broad catches so unexpected bugs are surfaced, or narrow to the exporter's expected exception types. Acceptable to keep broad catch for resilience but improve diagnostics.
- **TikZ export overwrites existing file without confirmation** — `314` _(correctness)_  
  _export (lines 306-323) opens filepath in 'w' mode unconditionally (line 314). The path defaults to ~/diagram.tex (line 169), so clicking Export without browsing silently overwrites any existing ~/diagram.tex. QFileDialog.getSaveFileName (used only in _browse_file) provides overwrite confirmation, but the default/typed path does not.  
  **Fix:** When the path did not come from the save dialog, check os.path.exists(filepath) and prompt via QMessageBox.question before overwriting.

### `modern_ui/widgets/tuning_panel.py`

- **TuningParameterRow auto-expand range can produce min>=max for typed value of zero or sign edge cases** — `169-172` _(correctness)_  
  _on_value_typed auto-expands via `if val < self._min: set_range(val*1.5 if val<0 else val*0.5, self._max)` / `elif val > self._max: set_range(self._min, val*1.5 if val>0 else val*0.5)` (lines 169-172). set_range returns early when `min_val >= max_val` (line 144). When that early-return triggers, the slider is repositioned (line 174) with the old _min/_max while value_changed still fires (line 176), so the slider thumb and the emitted/typed value disagree. The reviewer's specific zero/sign pathological scenarios are hard to reach in practice (initial ranges from _get_slider_range keep _min<=0 for positive starts), but the underlying bailout-yet-emit mismatch is real.  
  **Fix:** After set_range, verify the typed value lies within [_min,_max]; if set_range bailed, widen symmetrically around the typed value (e.g. _min=val-abs(val)-1, _max=val+abs(val)+1) before repositioning the slider so the thumb always tracks the entered value.

### `modern_ui/widgets/variable_editor.py`

- **Editor file load opens without explicit encoding** — `196` _(correctness)_  
  load_script line 196 `with open(filename, 'r') as f` uses the platform default encoding (cp1252 on Windows per env), which can raise UnicodeDecodeError or corrupt non-ASCII content. The error is caught (lines 202-205) so it degrades gracefully, but reads can still mis-decode silently.  
  **Fix:** Open with encoding='utf-8' for consistent cross-platform reads (matching how scripts are typically authored). Note: lib/workspace.py also opens without encoding (lines 32, 113) — same latent issue.
- **Greedy string highlighting regex over-matches across quotes** — `37` _(correctness)_  
  Lines 37-38 use greedy patterns QRegExp(r'".*"') and QRegExp(r"'.*'"). The greedy .* matches from the first quote to the last quote on a line, so on a line like `'a' + x + 'b'` the code between the two strings is also highlighted as a string. Cosmetic only.  
  **Fix:** Use non-greedy or negated character classes: "[^\"]*" and '[^']*' to match individual string tokens.
- **PythonHighlighter recompiles QRegExp on every highlightBlock call** — `53` _(performance)_  
  highlightBlock (lines 51-58) does `expression = QRegExp(pattern)` for every stored rule on every block (per line, per keystroke). The rules already store compiled QRegExp objects (lines 31-49 append QRegExp instances), so wrapping each in a fresh QRegExp(pattern) is redundant copy/construction work in a per-edit hot path.  
  **Fix:** Use the stored compiled QRegExp directly: `expression = pattern` (copy only if mutating index state matters; QRegExp.indexIn does not mutate the pattern, so direct reuse is safe).

### `modern_ui/widgets/waveform_inspector.py`

- **_refresh_curves can index x[-2] on an empty/length-1 timeline in step mode** — `175-176` _(correctness)_  
  In step mode `if step_mode and len(x) == len(y): x = np.append(x, x[-1] + (x[-1] - x[-2] if len(x) > 1 else 1.0))`. The inner `len(x) > 1` guard protects x[-2], but the outer branch is entered whenever step_mode and len(x)==len(y), including the empty case (0==0). With an empty timeline/trace, `x[-1]` raises IndexError before the inner guard helps. Note _refresh_curves (unlike _export_csv/_update_readout) has no `len(timeline)` guard, so an empty active step trace would crash the curve refresh.  
  **Fix:** Add a length guard: `if step_mode and len(x) == len(y) and len(x) > 0:` and skip empty traces.
- **_update_readout y-index uses scrub/timeline index against per-trace y of different length** — `231-241` _(correctness)_  
  idx is computed via np.searchsorted(self.timeline, t) and clamped to the global timeline length (lines 231-232), then used to index each trace y via `y[min(idx, len(y) - 1)]` (line 240). Because the shared timeline is the longest run's timeline but y arrays belong to individual (possibly shorter) runs, a cursor past a short run's data clamps to that run's last sample and reports it as the value at time t, which is misleading rather than a crash.  
  **Fix:** Resolve each trace value against its own run timeline (available via tr['run_idx']) instead of the global timeline, or display '-'/'n/a' when idx exceeds that trace's sample count.

### `modern_ui/widgets/workspace_editor.py`

- **WorkspaceManager singleton makes both editors share hidden global state** — `21` _(design)_  
  WorkspaceManager (lib/workspace.py lines 13-20) is a process-wide singleton via __new__ returning a shared _instance. Both WorkspaceEditor (line 21) and VariableEditor (variable_editor.py line 69) construct WorkspaceManager() and get the same instance. There is no change signal: VariableEditor emits variables_updated (its own pyqtSignal) but WorkspaceEditor does not subscribe to it, so table edits from the script editor are invisible until manual Refresh.  
  **Fix:** Add a changed signal to WorkspaceManager (or have WorkspaceEditor connect to VariableEditor.variables_updated / a shared signal) so the table auto-refreshes when the shared workspace mutates.
- **on_cell_changed leaves stale red error styling after a later valid edit** — `229` _(correctness)_  
  In on_cell_changed (lines 201-230) the failure branch sets the value item foreground to red and a tooltip (lines 229-230). The success branch (lines 211-223) updates the backend and type column but never resets foreground/tooltip, so a cell that previously failed parsing stays red after a subsequent valid edit.  
  **Fix:** In the success branch, reset the value item's foreground to the theme text color (theme_manager.get_color('text_primary')) and clear its tooltip (setToolTip('')).
