# Refactoring TODO

This file tracks remaining refactoring opportunities for the DiaBloS codebase.

## Priority 1: Delegate More Helper Methods to Engine (Low Risk)

These DSim methods can be delegated to SimulationEngine:

- [ ] `update_global_list()` (~15 lines) - engine has similar method
- [ ] `count_computed_global_list()` - simple count, engine can do it
- [ ] `get_max_hierarchy()` - engine already has this method
- [ ] `detect_algebraic_loops()` (~40 lines) - pure algorithm, move to engine
- [ ] `children_recognition()` - engine has `_children_recognition()`
- [ ] `reset_memblocks()` (~13 lines) - engine has this method

## Priority 2: Extract Plotting to Separate Module (Medium Risk)

~250 lines of plotting logic could be extracted to `lib/plotting/scope_plotter.py`:

- [ ] `plot_again()` 
- [ ] `_plot_xygraph()`
- [ ] `_plot_fft()`
- [ ] `pyqtPlotScope()`
- [ ] `dynamic_pyqtPlotScope()`
- [ ] `get_scope_traces()`
- [ ] `_scope_step_modes()`
- [ ] `_is_discrete_upstream()`

## Priority 3: Extract Run History to Separate Module (Low Risk)

~100 lines could be extracted to `lib/services/run_history_service.py`:

- [ ] `_record_run_history()`
- [ ] `_load_run_history()`
- [ ] `save_run_history()`
- [ ] `set_run_history_persist()`

## Priority 4: Extract execution_init Loops (Medium-High Risk)

The main `execution_init()` (~270 lines) could be further simplified:

- [ ] Source/memory block initial execution loop 
- [ ] Main while loop that establishes hierarchy
- [ ] Consider creating `engine.initialize_execution()`

## Completed ✅

- [x] Delegate `set_block_type()` to engine
- [x] Delegate `identify_memory_blocks()` to engine
- [x] Delegate `count_rk45_ints()` to engine
- [x] Fix and delegate `reset_execution_data()` to engine
- [x] Refactor `execution_loop` to use `engine.execute_block()`
- [x] Refactor `execution_loop` to use `engine.propagate_outputs()`
- [x] Code cleanup (indentation, duplicate logic)
