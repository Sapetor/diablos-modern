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

