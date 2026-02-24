# DiaBloS Development Lessons Learned

> Patterns and anti-patterns discovered during development.
> Last updated: February 2026

---

## Performance Issues

### scipy.signal.cont2discrete Lazy Initialization (February 2026)

**Problem**: First simulation took ~3 seconds to show results, subsequent runs were instant.

**Root Cause**: `scipy.signal.cont2discrete()` has lazy initialization that takes ~3 seconds on first call. This affected Transfer Function blocks which use this function for discretization.

**Fix**: Added background preload thread in `diablos_modern.py` that calls `cont2discrete()` with a dummy system at app startup:
```python
def _preload_heavy_modules():
    from scipy import signal
    import numpy as np
    # Trigger cont2discrete lazy init with dummy system
    _dummy_A = np.array([[0, 1], [-1, -1]])
    _dummy_B = np.array([[0], [1]])
    _dummy_C = np.array([[1, 0]])
    _dummy_D = np.array([[0]])
    signal.cont2discrete((_dummy_A, _dummy_B, _dummy_C, _dummy_D), 0.01)

_preload_thread = threading.Thread(target=_preload_heavy_modules, daemon=True)
_preload_thread.start()
```

**Lesson**: When using scipy functions, test first-call performance - many have lazy initialization. Preload in background threads at app startup.

---

### Redundant Initialization (February 2026)

**Problem**: Simple diagrams took several seconds to show plots. Users reported "preparation stage" delay not present in earlier versions.

**Root Cause**: Triple initialization of the simulation engine:
1. `execution_init()` line 842: `engine.initialize_execution(root_blocks, root_lines)`
2. `execution_init()` line 885: `engine.initialize_execution(self.blocks_list)` - **REDUNDANT**
3. `run_compiled_simulation()` line 917: `initialize_execution(blocks, lines)` - **REDUNDANT**

Each initialization call performs expensive operations:
- Flattening check (iterating all blocks)
- Diagram integrity check
- Global computed list reset
- Memory block identification
- Source block execution loop
- Hierarchy resolution loop (O(n) per block)

**Fix**:
1. Removed duplicate call at line 885 in `lib/lib.py`
2. Added check in `run_compiled_simulation()` to skip init if `active_blocks_list` already populated

**Lesson**: When refactoring initialization code, audit all call sites to ensure no redundant calls. Use logging or profiling to detect repeated expensive operations.

---

## Bug Patterns

### Accessing Non-Existent Attributes (February 2026)

**Problem**: `'DSim' object has no attribute 'use_active'` error in `_print_terminal_verification()`

**Root Cause**: Code referenced `self.dsim.use_active` which doesn't exist. The intent was to check if the engine has an active blocks list.

**Fix**: Changed to proper pattern:
```python
has_engine = hasattr(self.dsim, 'engine') and self.dsim.engine is not None
use_active = has_engine and len(self.dsim.engine.active_blocks_list) > 0
```

**Lesson**: Always use `hasattr()` or `getattr()` with defaults when accessing attributes that may not exist. Never assume attribute existence without checking.

### Optional Parameters Without Defaults (February 2026)

**Problem**: `KeyError` when blocks accessed optional parameters directly via `params['key']`

**Fix**: Use `params.get('key', default)` for all optional parameters.

**Lesson**: Always use `.get()` with defaults for dictionary access to optional parameters.

---

## Testing Patterns

### Legacy Test Files (February 2026)

**Problem**: Old test files (test_blocks.py, test_sine_params.py) crash due to Qt initialization requirements.

**Fix**: Mark with `pytest.mark.skip` and document why:
```python
@pytest.mark.skip(reason="DSim requires GUI initialization - use unit tests instead")
```

**Lesson**: When test files become outdated, either update them or mark as skipped with clear documentation. Don't let broken tests accumulate.

---

## Subsystem Patterns

### Inport/Outport Block Naming (February 2026)

**Problem**: Simulation failed with "Could not find Outport Subsystem/outport2" when creating subsystems with multiple unconnected ports.

**Root Cause**: `DBlock.__init__` sets `self.name = block_fn.lower() + str(sid)`. When creating Inport/Outport blocks:
```python
outport = Outport(block_name=f"Out{outport_idx}")  # username set
outport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1  # sid set AFTER init
```
The `name` is computed during `__init__` with default `sid=1`, so ALL Outport blocks get `name="outport1"`. Setting `sid` afterwards doesn't update `name`. Multiple blocks with the same name overwrite each other in `block_map`.

**The Flattener Expectation**: The flattener in `lib/engine/flattener.py` looks for Inport/Outport blocks by convention:
- Port index 0 → looks for `Out1` or `outport1`
- Port index 1 → looks for `Out2` or `outport2`

**Fix**: Always set `name` explicitly after setting `sid` in `subsystem_manager.py`:
```python
inport = Inport(block_name=f"In{inport_idx}")
inport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
inport.name = f"inport{inport_idx}"  # CRITICAL: Update name to match port index

outport = Outport(block_name=f"Out{outport_idx}")
outport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
outport.name = f"outport{outport_idx}"  # CRITICAL: Update name to match port index
```

**Lesson**: When creating Inport/Outport blocks programmatically, always explicitly set the `name` attribute to match the 1-based port index (`inport1`, `inport2`, `outport1`, `outport2`). The `name` is used by the flattener to resolve signal paths through subsystems.

---

## Block Parameter Naming

### TranFn Parameter Names Must Match Block Definition (February 2026)

**Problem**: Optimization examples failed with "Algebraic loop detected" even though the feedback loop contained a Transfer Function block (which should break the loop).

**Root Cause**: The `.diablos` example files used abbreviated parameter names:
```json
"params": {
    "num": [1.0],
    "den": [1.0, 2.0, 1.0]
}
```

But the TranFn block definition (`blocks/transfer_function.py`) and the `identify_memory_blocks()` function in the simulation engine expect the full names:
```json
"params": {
    "numerator": [1.0],
    "denominator": [1.0, 2.0, 1.0]
}
```

The `identify_memory_blocks()` function checks:
```python
num = block.params.get('numerator', [])
den = block.params.get('denominator', [])
if len(den) > len(num):
    self.memory_blocks.add(block.name)
```

With wrong parameter names, `num` and `den` are empty lists, so the TranFn is NOT identified as a memory block. The algebraic loop detector then incorrectly flags the feedback loop.

**Fix**: Update example files to use correct parameter names (`numerator`/`denominator` not `num`/`den`).

**Lesson**: When creating or editing `.diablos` files manually, always use the exact parameter names from the block's `PARAMS` definition. The simulation engine's memory block detection depends on these exact names.

---

### Optional Inputs Not Handled in Interpreter Mode execution_loop (February 2026)

**Problem**: Blocks with optional inputs (like CostFunction with optional reference port) were not executing in interpreter mode. The ISE Cost plot was empty even though the block was properly connected.

**Root Cause**: In `lib/lib.py` `execution_loop()`, line 703 checked:
```python
block.data_recieved == block.in_ports or block.in_ports == 0
```

This required ALL input ports to receive data. CostFunction has 2 ports (signal and reference), but reference is optional (`optional_inputs = [1]`). With only signal connected: `data_recieved=1` but `in_ports=2`, so `1 != 2` and the block was skipped.

Note: The `initialize_execution()` method correctly handles optional inputs, but the main simulation loop did not.

**Fix**: Added optional input handling to match `initialize_execution()`:
```python
# Check if block has enough required inputs (accounting for optional inputs)
optional_inputs = set()
if hasattr(block, 'block_instance') and block.block_instance:
    if hasattr(block.block_instance, 'optional_inputs'):
        optional_inputs = set(block.block_instance.optional_inputs)
required_ports = block.in_ports - len(optional_inputs)
has_enough_inputs = block.data_recieved >= required_ports or block.in_ports == 0
```

**Lesson**: When adding `optional_inputs` support to a new code path (like `initialize_execution`), ensure ALL code paths that check input readiness are updated. The interpreter mode `execution_loop` was missed.

---

### Memory Block Feedback Not Applied (February 2026)

**Problem**: StateVariable blocks in optimization diagrams stayed at initial value - gradient descent didn't converge. The feedback computed in the diagram was never applied to update the state.

**Root Cause**: Two issues in `lib/engine/simulation_engine.py`:

1. **input_queue cleared between time steps** (lines 522, 530): `reset_execution_data()` cleared `input_queue = {}` for ALL blocks. Memory blocks receive their feedback at the END of time step t, but need that feedback at the START of time step t+1. Clearing the queue erased the feedback.

2. **Memory blocks un-computed after Loop 1** (lines 207-211): After memory blocks output their state in Loop 1 (with `output_only=True`), they were un-computed and would re-execute in Loop 2. With `optional_inputs = [0]`, StateVariable would execute IMMEDIATELY at the start of Loop 2 (before feedback was computed), with empty `input_queue`, so state never updated.

**The Execution Flow Bug**:
```
Time t=0:
- Loop 1: StateVariable outputs [5,5], input_queue empty
- Memory blocks UN-COMPUTED (bug!)
- Loop 2: StateVariable re-executes immediately (optional input), input_queue still empty
- Loop 2 continues: X_update computes [3,3], propagates to StateVariable
- End: input_queue = {0: [3,3]}

Time t=1:
- reset_execution_data() clears input_queue = {} (bug!)
- Loop 1: StateVariable executes with empty input, outputs [5,5] again
```

**Fix**:
1. Preserve `input_queue` for memory blocks in `reset_execution_data()`:
```python
if block.name not in self.memory_blocks:
    block.input_queue = {}
```

2. Don't un-compute memory blocks after Loop 1 - remove lines 207-211. Memory blocks stay computed and receive feedback via propagation.

**Lesson**: For memory blocks with feedback loops, the execution model must support:
- Memory blocks output CURRENT state in Loop 1
- Diagram computes NEXT state based on current state
- Feedback arrives via propagation (stored in `input_queue`)
- On NEXT time step, memory block uses preserved `input_queue` to update state

Never clear `input_queue` for memory blocks between time steps, and don't re-execute them in Loop 2.

---

### Copy/Paste Connections Lost Due to Stale Alias (February 2026)

**Problem**: When copying multiple connected blocks and pasting, the pasted blocks appeared but connections between them were missing. Error log showed "Skipping connection: Block index out of range".

**Root Cause**: In `lib/lib.py`, `connections_list` was set as an alias to `line_list` once at initialization (line 90):
```python
self.connections_list = self.line_list  # Alias
```

However, `line_list` was reassigned in multiple places WITHOUT updating the alias:
- `remove_block_and_lines()` - line 271
- `clear_all()` - line 371
- After `link_goto_from()` during simulation - line 523

After any of these operations, `connections_list` still pointed to the OLD (empty) list while `line_list` had the actual connections. The `ClipboardManager.copy_selected_blocks()` iterated `dsim.connections_list` which was empty, so no connections were copied.

**Fix**: Add `self.connections_list = self.line_list` after every reassignment of `line_list`:
```python
self.line_list = self.model.line_list
self.connections_list = self.line_list  # Keep alias in sync
```

**Lesson**: When using Python aliases (where two names point to the same object), reassigning one name breaks the alias. Either:
1. Update both names together after every reassignment
2. Use a property that always returns the current value
3. Avoid aliases entirely - just use one consistent name

For DSim, option 1 was chosen since `connections_list` is used by clipboard operations while `line_list` is the canonical name used elsewhere.

---

### Compiled Solver Execution Order Bug (February 2026)

**Problem**: Loading `c01_tank_feedback.diablos` (Step→Sum→Gain→TranFn→Scope feedback loop) and simulating produced all-zero output for the TranFn block. The "tank level" signal stayed at 0 the entire simulation.

**Root Cause**: `initialize_execution()` assigns `hierarchy=0` to memory blocks (TranFn, Integrator) because they can output without inputs. When `compile_system()` sorts blocks by hierarchy, the TranFn (hierarchy=0) ends up **before** its input chain (Sum at hierarchy=1, Gain at hierarchy=1). The execution sequence was:

```
0: step0    (source, hier=0)
1: tranfn3  (state block, hier=0)  ← reads gain2 which hasn't run yet!
2: sum1     (hier=1)
3: gain2    (hier=1)
4: scope4   (hier=1)
```

The pre-population step correctly sets `signals['tranfn3'] = C*x` for feedback, but the TranFn's **derivative** computation `dx = Ax + Bu` reads `signals['gain2']` which is 0 (not yet computed). With zero input and zero initial state, `dx = 0` forever.

**Fix**: In `compile_system()`, split blocks into three groups instead of two:
1. **Sources** (Step, Sine, etc.) — run first
2. **Algebraic** (Sum, Gain, Scope, etc.) — run second, using pre-populated state outputs
3. **State blocks** (TranFn, StateSpace, Integrator, PDE, etc.) — run last, computing derivatives with correct inputs

```python
sorted_order = sources + algebraic + state_blocks
```

This mirrors the mathematical structure: sources provide inputs, algebraic blocks compute the combined signal (using pre-populated state outputs for feedback), then state blocks compute correct derivatives.

**Lesson**: In compiled solvers with feedback loops, execution order must respect the dependency chain even when hierarchy values are set for the interpreter mode. State blocks need their outputs pre-populated (for feedback) but their derivative computation must happen AFTER all algebraic inputs are resolved.

---

### LaTeX Character Escaping: Single-Pass Required (February 2026)

**Problem**: `_escape_latex()` in `tikz_exporter.py` replaced characters sequentially. Backslash was replaced first → `\textbackslash{}`. Then `{` and `}` replacements corrupted the braces from step 1, producing `\textbackslash\{\}`.

**Fix**: Single-pass `re.sub` with a mapping dict:
```python
_ESCAPE_MAP = { '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', ... }
_ESCAPE_RE = re.compile(r'[\\{}_&%#~^]')

def _escape_latex(text):
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group()], text)
```

**Lesson**: When replacing characters in text where replacements contain characters that are also being replaced, ALWAYS use a single-pass approach (regex with mapping dict, or `str.translate()`). Sequential replacement is inherently broken for this pattern.

---

### TikZ Anchors Rotate With Shapes (February 2026)

**Problem**: Gain block port anchors were reversed when the block was flipped (`shape border rotate=180`). Code had a special branch swapping `left side` ↔ `apex` for flipped blocks.

**Root Cause**: In TikZ, when `shape border rotate=180` is applied, the anchors rotate WITH the shape. So `left side` is still the input side and `apex` is still the output side — no swap needed.

**Fix**: Removed the flipped branch entirely.

**Lesson**: TikZ shape anchors are shape-relative, not page-relative. When using `shape border rotate`, anchor names stay consistent with the shape orientation.

---

### MCP Server pip Dependencies vs System Tools (February 2026)

**Problem**: TexMCP's `requirements.txt` included `uv>=0.20.0`. `pip install` failed because `uv` is a Rust-based standalone tool, not a Python package.

**Fix**: Installed only runtime dependencies (`fastmcp`, `jinja2`, `python-dotenv`), skipping `uv` and optional deps (`openai`, `pytest`).

**Lesson**: When installing MCP servers, review `requirements.txt` before blindly running `pip install -r`. Some packages list build/dev tools (uv, meson, cmake) that are system-level tools, not pip packages. Install only runtime dependencies.

---

### Batch Simulation Doesn't Trigger Timer-Based State Detection (February 2026)

**Problem**: Live parameter tuning panel appeared but sliders had no effect — plots never updated when dragging sliders.

**Root Cause**: The `TuningController.store_sim_params()` was only called inside `safe_update()`, a QTimer-based method that detects `was_running → not is_running` transitions. Batch simulation runs **synchronously** (blocking the main thread), so the timer can never fire during the simulation. The timer never sees the running→stopped transition, so `store_sim_params()` is never called, leaving the controller in an inactive state (`_active = False`). All slider changes are silently ignored.

**Fix**: Call `store_sim_params()` directly after `canvas.start_simulation()` returns in `main_window.start_simulation()`:
```python
self.canvas.start_simulation()

# Arm tuning controller after batch simulation completes
if not self.canvas.is_simulation_running():
    sim_time = getattr(self.dsim, 'sim_time', None)
    sim_dt = getattr(self.dsim, 'sim_dt', None)
    if sim_time and sim_dt:
        self.tuning_controller.store_sim_params(sim_time, sim_dt)
```

**Lesson**: When using QTimer-based state detection, synchronous operations that block the main thread will prevent the timer from firing. Always provide an explicit notification path for synchronous operations alongside any timer-based detection.

---

### Interpreter data_recieved Guard Blocks Memory Block Updates (February 2026)

**Problem**: PRBS → TranFn → Scope produced constant 0 output.

**Root Cause**: `propagate_outputs()` in `simulation_engine.py` had a guard:
```python
if tuple_child['dstport'] not in mblock.input_queue:
    mblock.data_recieved += 1
```
Memory blocks preserve their `input_queue` across time steps (for feedback), so the port was always present after step 1. `data_recieved` was never incremented, causing memory blocks to fail the `has_enough_inputs` check in `execution_loop`. They never executed with full state update — outputting zeros forever.

A compounding issue: the initialization path (`initialize_execution` Loop 2) used a **different** readiness mechanism (`input_queue` presence) that always worked, masking the bug on step 1.

**Fixes**:
1. Removed the guard — always increment `data_recieved` in `propagate_outputs()`
2. Unified readiness checks: changed init Loop 2 to use `data_recieved >= required_ports` like the runtime loop
3. Removed dead `execute_and_propagate()` helper that had an inconsistent propagation guard

**Lesson**: Never use two different mechanisms for the same semantic check (readiness). The init path used `input_queue` presence while runtime used `data_recieved` counter — this divergence hid the bug for months. When state is preserved across cycles (`input_queue` for memory blocks), any guard that checks "is this new?" will break on the second cycle.

---

### Block State Stored on self Instead of params (February 2026)

**Problem**: Derivative block stored `t_old`, `i_old`, `didt_old` on `self` (the instance), unlike every other stateful block which uses `params['_xyz_']`.

**Root Cause**: Historical pattern from early development. The `_init_start_` flag was correctly in `params`, but the actual state was on the instance.

**Fix**: Moved state to `params['_t_old_']`, `params['_i_old_']`, `params['_didt_old_']`.

**Lesson**: All block state that persists across time steps must be stored in `params` (or `exec_params`), never on `self`. The simulation engine's `reset_memblocks()` only resets `_init_start_` in params/exec_params — instance attributes survive resets invisibly.

---

### LFSR Tap Convention: Left-Shift Fibonacci (February 2026)

**Problem**: Initial investigation of the PRBS block's tap table incorrectly assumed taps mapped directly to polynomial coefficients. 19 of 23 entries appeared broken.

**Root Cause**: The code uses a **left-shifting** Fibonacci LFSR where `feedback = XOR(bits at tap positions)` and the new bit is injected at the LSB. For this convention, taps `[t0, t1, ...]` with `max(t) = n-1` produce characteristic polynomial `p(x) = x^n + sum x^{n-1-t_j}`, NOT `p(x) = x^n + sum x^{t_j}`.

**Lesson**: Always verify LFSR conventions empirically (simulate a few steps, check period) before rewriting a tap table. The mapping from taps to polynomial depends on shift direction (left vs right) and feedback injection point (LSB vs MSB). The tap must include position `n-1` for the polynomial to have degree `n`.

