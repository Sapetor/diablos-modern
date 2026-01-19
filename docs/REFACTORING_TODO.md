# Refactoring TODO

This file tracks remaining refactoring opportunities for the DiaBloS codebase.

## Priority 1: Delegate More Helper Methods to Engine (Low Risk)

✅ **COMPLETED** - All Priority 1 methods have been delegated to SimulationEngine.

## Priority 2: Extract Plotting to Separate Module (Medium Risk)

✅ **COMPLETED** - Plotting logic extracted to `lib/plotting/scope_plotter.py`.

## Priority 3: Extract Run History to Separate Module (Low Risk)

✅ **COMPLETED** - Run history logic extracted to `lib/services/run_history_service.py`.

## Priority 4: Extract execution_init Loops (Medium-High Risk)

✅ **COMPLETED** - `execution_init` logic extracted to `engine.initialize_execution()`.


## Priority 5: Extract Canvas Tools and Rendering

✅ **COMPLETED** - All rendering and analysis logic extracted from `modern_canvas.py`, `block.py`, and `connection.py`.

## Priority 6: Simplify Modern Canvas (Logic Extraction)

✅ **COMPLETED** - Logic extracted to `InteractionManager`, `HistoryManager`, `SelectionManager`, `MenuManager`.

## Priority 7: Simplify Main Window

✅ **COMPLETED** - Logic extracted to `MenuBuilder`, `ProjectManager`.

## Completed ✅

- [x] Delegate `set_block_type()` to engine
- [x] Delegate `identify_memory_blocks()` to engine
- [x] Delegate `count_rk45_ints()` to engine
- [x] Fix and delegate `reset_execution_data()` to engine
- [x] Refactor `execution_loop` to use `engine.execute_block()`
- [x] Refactor `execution_loop` to use `engine.propagate_outputs()`
- [x] Code cleanup (indentation, duplicate logic)
- [x] Delegate `update_global_list()` to engine
- [x] Delegate `count_computed_global_list()` to engine
- [x] Delegate `get_max_hierarchy()` to engine
- [x] Delegate `detect_algebraic_loops()` to engine (40 lines → 3 lines)
- [x] Delegate `children_recognition()` to engine
- [x] Delegate `reset_memblocks()` to engine
- [x] Extract plotting logic to `lib/plotting/scope_plotter.py`
- [x] Extract run history logic to `lib/services/run_history_service.py`
- [x] Extract execution init logic to `engine.initialize_execution()`
- [x] Improvements: Dynamic height scaling for blocks based on port count (fixes SumBlock size)
- [x] Refactor ModernCanvas: Logic extracted to `InteractionManager` (mouse events, state machine)
- [x] Refactor ModernCanvas: Logic extracted to `HistoryManager`, `SelectionManager`, `MenuManager`.
- [x] Refactor MainWindow: Logic extracted to `MenuBuilder`, `ProjectManager`.
- [x] Fix: Implemented missing `Undo` and `Redo` actions
- [x] Fix: Implemented `Recent Files` loading logic
- [x] Testing: Added unit tests for `InteractionManager`, `SelectionManager`.
- [x] Refactor: Optimized `SystemCompiler` with closure-based executors (Fast Solver Speedup).
