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

