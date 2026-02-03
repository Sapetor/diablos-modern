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
