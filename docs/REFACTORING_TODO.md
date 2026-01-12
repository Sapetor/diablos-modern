# Refactoring TODO

This file tracks remaining refactoring opportunities for the DiaBloS codebase.

## Priority 1: Delegate More Helper Methods to Engine (Low Risk)

✅ **COMPLETED** - All Priority 1 methods have been delegated to SimulationEngine.

## Priority 2: Extract Plotting to Separate Module (Medium Risk)

~250 lines of plotting logic could be extracted to `lib/plotting/scope_plotter.py`:

- [x] `plot_again()` 
- [x] `_plot_xygraph()`
- [x] `_plot_fft()`
- [x] `pyqtPlotScope()`
- [x] `dynamic_pyqtPlotScope()`
- [x] `get_scope_traces()`
- [x] `_scope_step_modes()`
- [x] `_is_discrete_upstream()`

✅ **COMPLETED** - Plotting logic extracted to `lib/plotting/scope_plotter.py`.

## Priority 3: Extract Run History to Separate Module (Low Risk)

~100 lines could be extracted to `lib/services/run_history_service.py`:

- [x] `_record_run_history()`
- [x] `_load_run_history()`
- [x] `save_run_history()`
- [x] `set_run_history_persist()`

✅ **COMPLETED** - Run history logic extracted to `lib/services/run_history_service.py`.

## Priority 4: Extract execution_init Loops (Medium-High Risk)

The main `execution_init()` (~270 lines) could be further simplified:

- [x] Source/memory block initial execution loop 
- [x] Main while loop that establishes hierarchy
- [x] Consider creating `engine.initialize_execution()`

✅ **COMPLETED** - `execution_init` logic extracted to `engine.initialize_execution()`.


## Priority 5: Extract Canvas Tools and Rendering

- [x] Extract Analysis tools (Bode, Root Locus) to `lib/analysis/control_system_analyzer.py`
- [x] Extract Block Rendering to `modern_ui/renderers/block_renderer.py`
- [x] Extract Connection Rendering to `modern_ui/renderers/connection_renderer.py`
- [x] Extract Canvas Rendering to `modern_ui/renderers/canvas_renderer.py` (Grid, Selection, HUDs)

✅ **COMPLETED** - All rendering and analysis logic extracted from `modern_canvas.py`, `block.py`, and `connection.py`.

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


