"""
SimulationEngine - Execution and analysis logic for DiaBloS.
Handles simulation initialization, execution loops, and diagram analysis.
"""

import logging
import time as time_module
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.workspace import WorkspaceManager
from lib.engine.system_compiler import SystemCompiler
from lib.engine.flattener import Flattener
from lib.engine.solver_diagnostics import build_diagnostics, format_diagnostics_for_log
from lib.engine.compile_cache import source_params_fingerprint, compiled_system_fingerprint
from lib.engine.block_params import push_down_internal_params
from lib.engine import graph_analysis

# The compiled fast-solver run (ODE solve + post-solve replay) lives in
# lib/engine/compiled_runner.py.  Its module-level names are re-exported here
# because they were part of this module's public surface before the split.
from lib.engine import compiled_runner
from lib.engine.compiled_runner import (  # noqa: F401  (re-export)
    FIXED_STEP_METHODS,
    SCIPY_SOLVER_METHODS,
    _KERNEL_REPLAY_FNS,
    integrate_fixed_step,
)

logger = logging.getLogger(__name__)

# Compiled-system cache fingerprinting (key-set constants + the normalize/
# fingerprint functions) lives in lib/engine/compile_cache.py.


class SimulationEngine:
    """
    Simulation engine that manages execution logic.
    Analyzes diagrams, detects algebraic loops, and executes simulations.

    Attributes:
        model: SimulationModel containing diagram data
        execution_initialized: Whether simulation has been initialized
        execution_pause: Whether simulation is paused
        execution_stop: Whether simulation has been stopped
        error_msg: Last error message from simulation
        sim_time: Total simulation time
        sim_dt: Simulation time step
        real_time: Whether to run simulation in real-time
        global_computed_list: Tracking list for block computation
        timeline: Time values for each simulation step
        outs: Output values from simulation
    """

    # Cached propagation adjacency (see _propagation_targets). Declared at
    # class level as well as in __init__ so an engine built with __new__
    # (test stubs that call propagate_outputs directly) still works.
    _prop_adj = None
    _prop_adj_source = None

    def __init__(self, model: Any) -> None:
        """
        Initialize simulation engine.

        Args:
            model: SimulationModel instance containing blocks and lines
        """
        self.model = model
        # Execution state
        self.execution_initialized: bool = False
        self.execution_pause: bool = False
        self.execution_stop: bool = False
        self.error_msg: str = ""

        # Simulation parameters
        self.sim_time: float = 1.0
        self.sim_dt: float = 0.01
        # Compiled-solver selection (see run_compiled_simulation).
        self.solver_method: str = "RK45"
        self.rtol: float = 1e-9
        self.atol: float = 1e-12
        self.real_time: bool = True
        self.execution_time: float = 1.0
        self.time_step: float = 0.0

        # Execution tracking
        self.global_computed_list: List[Dict[str, Any]] = []
        # Lazy name->entry index over global_computed_list (see _global_list_index)
        self._gcl_index: Dict[str, Dict[str, Any]] = {}
        self._gcl_index_source: Optional[List[Dict[str, Any]]] = None
        # Lazy srcname -> [(dst_block, srcport, dstport)] adjacency used by
        # propagate_outputs (see _propagation_targets).
        self._prop_adj: Optional[Dict[str, List[Tuple[DBlock, int, int]]]] = None
        self._prop_adj_source: Optional[Tuple[Any, Any, Tuple[int, int]]] = None
        self.timeline: np.ndarray = np.array([0.0])
        self.outs: List[Any] = []
        self.memory_blocks: set = set()
        self.max_hier: int = 0
        self.rk45_len: bool = False
        self.rk_counter: int = 0
        self.execution_time_start: float = 0.0

        # System Compiler
        self.compiler = SystemCompiler()
        self.flattener = Flattener()

        # Compiled-system cache and diagnostics. The cache is intentionally
        # one-entry: the common hot path is rerunning the current topology after
        # scalar tweaks, and a single slot avoids unbounded closure retention.
        self._compiled_system_cache_key = None
        self._compiled_system_cache_value = None
        self.compile_cache_hits: int = 0
        self.compile_cache_misses: int = 0
        self.last_solver_diagnostics: Dict[str, Any] = {}

        # Active execution lists (may differ from model if flattened)
        self.active_blocks_list = []
        self.active_line_list = []

    def initialize_execution(
        self, blocks_list: List[DBlock], lines_list: Optional[List[DLine]] = None
    ) -> bool:
        """
        Initialize the execution sequence for the simulation.

        Args:
            blocks_list: List of blocks (Top Level)
            lines_list: List of lines (Top Level). Required for flattening.

        Returns:
            bool: True if block initialization successful
        """
        try:
            import time as _time

            _te0 = _time.time()
            logger.debug("Engine: Initializing execution...")

            # The active block/line lists are about to be (re)selected; drop any
            # adjacency built for the previous ones.
            self.invalidate_propagation_cache()

            # 1. Flatten Hierarchy if lines provided
            # If line_list is None, we fallback to model list, but flattening requires consistent lists.
            # DSim calls this. We assume DSim passes lines.

            if lines_list is None:
                lines_list = self.model.line_list

            # Check if flattening needed (if any Subsystem block exists)
            has_subsystems = any(getattr(b, "block_type", "") == "Subsystem" for b in blocks_list)

            if has_subsystems:
                logger.info("Flattening hierarchical system...")
                self.active_blocks_list, self.active_line_list = self.flattener.flatten(
                    blocks_list, lines_list
                )
                logger.info(
                    f"Flattening complete. Blocks: {len(self.active_blocks_list)}, Lines: {len(self.active_line_list)}"
                )
            else:
                self.active_blocks_list = blocks_list
                self.active_line_list = lines_list
            logger.debug(f"[ENGINE TIMING] flattening check: {_time.time() - _te0:.3f}s")

            # Integrity Check on the Active (Flattened) System
            _te1 = _time.time()
            if not self.check_diagram_integrity():
                self.error_msg = "Diagram integrity check failed (connections)."
                logger.error(self.error_msg)
                return False
            logger.debug(f"[ENGINE TIMING] integrity check: {_time.time() - _te1:.3f}s")

            # Reset temporary lists using ACTIVE list
            self.global_computed_list = [
                {"name": x.name, "computed_data": x.computed_data, "hierarchy": x.hierarchy}
                for x in self.active_blocks_list
            ]
            self.reset_execution_data()
            self.execution_time_start = time_module.time()

            # Check for algebraic loops (part 1)
            check_loop = self.count_computed_global_list()

            # Resolve workspace-variable params BEFORE identify_memory_blocks so
            # TranFn/StateSpace classification reads resolved arrays (not raw
            # workspace-variable strings).
            #
            # Skip blocks whose exec_params are already populated with the
            # current sim_dt (DSim.execution_init resolves root blocks before
            # calling this method).  Avoids a redundant pass over every block
            # in the normal execution path while keeping the resolve as a
            # fallback for direct callers (tests, alternate init paths).
            workspace_manager = WorkspaceManager()
            current_dt = self.sim_dt
            for block in self.active_blocks_list:
                self._resolve_block_params(block, current_dt, workspace_manager)

            # Identify memory blocks (on ACTIVE list, after param resolution)
            self.identify_memory_blocks()

            # Propagate sample times for multi-rate simulation
            self.propagate_sample_times()

            # Count RK45 integrators
            self.rk45_len = self.count_rk45_integrators()
            self.rk_counter = 0
            logger.debug(f"[ENGINE TIMING] pre-loop setup: {_time.time() - _te0:.3f}s")

            # Loop 1: Execute Source Blocks (b_type=0) and Initialize Memory Blocks
            # Iterate active_blocks_list instead of blocks_list input
            blocks_to_exec = self.active_blocks_list
            logger.info(
                f"Engine: Initializing execution for {len(blocks_to_exec)} blocks (flattened)"
            )
            _te2 = _time.time()

            if not self._init_execute_sources(blocks_to_exec):
                return False

            # Note: Memory blocks stay computed - they executed in Loop 1 and will receive
            # feedback via propagation. They don't need to re-execute in Loop 2.
            # Their input_queue is preserved between time steps so feedback is applied
            # at the START of the NEXT time step in Loop 1.
            logger.debug(f"[ENGINE TIMING] Loop 1 (source blocks): {_time.time() - _te2:.3f}s")

            # Loop 2: Hierarchy Resolution Matrix
            _te3 = _time.time()
            if not self._init_resolve_hierarchy(blocks_to_exec, check_loop):
                return False
            logger.debug(f"[ENGINE TIMING] Loop 2 (hierarchy): {_time.time() - _te3:.3f}s")

            # Loop 3: Advance memory block state by one step using the inputs
            # resolved by Loop 2.  Loop 1 ran memory blocks with output_only=True
            # so y[0] = C @ x[0] could feed downstream consumers, but the state
            # was never updated.  Without this pass, the simulation loop's first
            # iteration would re-read x[0] and produce a duplicate sample at
            # t=dt — a one-step lag visible on every memory-block trace.  Proper
            # blocks (b_type=2) already get their state advanced in Loop 2's
            # full execute; this loop brings memory blocks in line with that.
            # Output is NOT propagated: scope already has y[0] from Loop 1.
            if not self._init_advance_memory_state(blocks_to_exec):
                return False

            # Sync hierarchies back to blocks
            self.reset_execution_data()

            # Calculate max hierarchy
            self.max_hier = self.get_max_hierarchy()

            logger.debug(f"Engine: Execution initialized. Max hierarchy: {self.max_hier}")
            logger.debug(f"[ENGINE TIMING] initialize_execution TOTAL: {_time.time() - _te0:.3f}s")
            self.execution_initialized = True
            return True

        except Exception as e:
            import traceback

            logger.error(f"Engine: Error during execution init: {e}")
            logger.error(traceback.format_exc())
            self.error_msg = str(e)
            return False

    def _init_execute_sources(self, blocks_to_exec):
        """Loop 1 of execution init: execute the source blocks (those whose
        every input port is optional or absent) and run the memory blocks
        output-only, pinning each to hierarchy 0 and propagating its
        outputs. Returns False with self.error_msg set on a block error.
        """
        import time as _time

        for block in blocks_to_exec:
            logger.debug(
                f"Engine: Initial processing of block: {block.name}, b_type: {block.b_type}"
            )
            out_value = {}

            # Determine whether this block can run with no upstream data.
            # b_type==0 by itself does NOT mean "source": Sum, MatrixGain,
            # Demux, etc. are all b_type==0 instantaneous blocks but they
            # do require inputs. Force-executing them here pins them to
            # hierarchy=0 with stale input values, which freezes feedback
            # loops in the interpreter path. Mirror the readiness check
            # used in Loop 2 below: a block is a source only if every
            # input port is optional (or there are none).
            optional_inputs = set()
            if hasattr(block, "block_instance") and block.block_instance:
                if hasattr(block.block_instance, "optional_inputs"):
                    optional_inputs = set(block.block_instance.optional_inputs)
            required_ports = block.in_ports - len(optional_inputs)
            is_source = required_ports == 0

            if block.b_type == 0 and is_source:
                # Execute source block
                _tblk = _time.time()
                out_value = self.execute_block(block)
                logger.debug(
                    f"[ENGINE TIMING] execute_block({block.name}): {_time.time() - _tblk:.3f}s"
                )
                if out_value is False:  # execute_block handles errors and returns None/False/Dict
                    return False

                block.computed_data = True
                block.hierarchy = 0
                self.update_global_list(block.name, h_value=0, h_assign=True)

            elif block.name in self.memory_blocks:
                # Execute memory block (output_only=True)
                _tblk = _time.time()
                out_value = self.execute_block(block, output_only=True)
                logger.debug(
                    f"[ENGINE TIMING] execute_block({block.name}, memory): {_time.time() - _tblk:.3f}s"
                )
                if out_value is False:
                    return False

                block.computed_data = True
                self.update_global_list(block.name, h_value=0, h_assign=True)

            # Check for errors in output
            if out_value and isinstance(out_value, dict) and "E" in out_value and out_value["E"]:
                self.error_msg = out_value.get("error", "Unknown error")
                logger.error(self.error_msg)
                return False

            # Propagate outputs to children
            if out_value:
                if (
                    block.b_type not in [1, 3]
                ):  # Only propagate if valid type logic applies (memory blocks propagate manually here)
                    # Note: The original logic had custom propagation here.
                    pass

                # We can reuse propagate_outputs but need to be careful about the specific logic used in init
                # Original logic manually iterated children. Let's replicate or delegate.
                _tprop = _time.time()
                self.propagate_outputs(block, out_value)
                _prop_time = _time.time() - _tprop
                if _prop_time > 0.01:
                    logger.debug(
                        f"[ENGINE TIMING] propagate_outputs({block.name}): {_prop_time:.3f}s"
                    )
        return True

    def _init_resolve_hierarchy(self, blocks_to_exec, check_loop):
        """Loop 2 of execution init: the hierarchy-resolution fixpoint --
        repeatedly execute every block whose inputs are ready, assigning
        ascending hierarchy levels, until all blocks are computed. Detects
        algebraic loops and stalls. Returns False with self.error_msg set
        on an algebraic loop, a stall, or a block error.
        """
        h_count = 1
        while not self.check_global_list():
            for block in blocks_to_exec:
                # Check execution readiness - account for optional inputs
                optional_inputs = set()
                if hasattr(block, "block_instance") and block.block_instance:
                    if hasattr(block.block_instance, "optional_inputs"):
                        optional_inputs = set(block.block_instance.optional_inputs)

                required_ports = block.in_ports - len(optional_inputs)
                can_execute = block.data_received >= required_ports or block.in_ports == 0

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "LOOP %s: %s (computed=%s) Ready=%s (Recv=%s/Ports=%s, Req=%s)",
                        h_count,
                        block.name,
                        block.computed_data,
                        can_execute,
                        block.data_received,
                        block.in_ports,
                        required_ports,
                    )

                if can_execute and not block.computed_data:
                    # OUT_VALUE execute_block...
                    out_value = self.execute_block(block)
                    if out_value is False:
                        return False
                    # Check for error dict from block
                    if isinstance(out_value, dict) and (
                        out_value.get("E") or out_value.get("error")
                    ):
                        self.error_msg = out_value.get("error", "Block returned error")
                        logger.error(f"Block {block.name} error: {self.error_msg}")
                        return False

                    # Memory block special output update
                    if block.name in self.memory_blocks:
                        self.sync_integrator_output(block)

                    self.update_global_list(block.name, h_value=h_count, h_assign=True)
                    block.computed_data = True

                    # Sync the DBlock-level sample schedule for discrete
                    # blocks that just performed their t=0 sample.  Without
                    # this, the simulation loop's first should_execute()
                    # would still see _next_execution_time=0 and run the
                    # block at t=dt instead of waiting for t=Ts.
                    if block.effective_sample_time > 0 and isinstance(out_value, dict):
                        self.stamp_held_outputs(block, out_value)
                        block.schedule_next_execution(self.time_step)

                    if block.name not in self.memory_blocks and block.b_type != 3:
                        self.propagate_outputs(block, out_value)

                    logger.debug(f"EXECUTED in LOOP {h_count}: {block.name}")

            # Algebraic Loop Detection
            computed_count = self.count_computed_global_list()
            if computed_count == check_loop:
                uncomputed_blocks = [b for b in blocks_to_exec if not b.computed_data]
                if not uncomputed_blocks:
                    break

                is_algebraic, cycle_nodes = self.detect_algebraic_loops(uncomputed_blocks)
                if is_algebraic:
                    self.error_msg = f"Algebraic loop detected involving blocks: {cycle_nodes}"
                    logger.error(self.error_msg)
                    return False
                else:
                    # The only legitimate stall is when the remaining
                    # uncomputed blocks are all memory blocks (they execute
                    # in Loop 3). Any uncomputed non-memory block would
                    # silently never run, so surface that as a hard error.
                    stalled_non_memory = [
                        b.name for b in uncomputed_blocks if b.name not in self.memory_blocks
                    ]
                    if stalled_non_memory:
                        self.error_msg = (
                            f"Hierarchy resolution stalled with uncomputed "
                            f"non-memory blocks: {stalled_non_memory}"
                        )
                        logger.error(self.error_msg)
                        return False
                    break
            else:
                check_loop = computed_count

            h_count += 1
        return True

    def _init_advance_memory_state(self, blocks_to_exec):
        """Loop 3 of execution init: advance each memory block state by one
        step using the inputs resolved in Loop 2 (Loop 1 ran them
        output-only so y[0] could feed downstream, but never advanced the
        state). Output is not propagated. Returns False with self.error_msg
        set on error.
        """
        for block in blocks_to_exec:
            if block.name in self.memory_blocks:
                # Capture the output belonging to t=0 before the state is
                # advanced: for a discrete-rate block this is the value held
                # across the first sample interval [0, Ts).
                pre_update_value = None
                if block.effective_sample_time > 0:
                    pre = self.execute_block(block, output_only=True)
                    if isinstance(pre, dict) and not pre.get("E"):
                        pre_update_value = pre

                out_value = self.execute_block(block)
                if out_value is False:
                    return False
                if isinstance(out_value, dict) and out_value.get("E"):
                    self.error_msg = out_value.get("error", "State advance failed")
                    logger.error(f"Loop 3 state advance failed for {block.name}: {self.error_msg}")
                    return False
                self.sync_integrator_output(block)
                # For discrete blocks, sync the DBlock-level sample schedule
                # with the block-internal sample state so the simulation
                # loop's should_execute() agrees with the block's own
                # bookkeeping.  Without this, the loop would think the
                # block's "next sample" is still at t=0 and would call
                # execute every dt instead of every Ts.
                if block.effective_sample_time > 0 and isinstance(out_value, dict):
                    self.stamp_held_outputs(block, out_value, pre_update_value)
                    block.schedule_next_execution(self.time_step)
        return True

    def _global_list_index(self) -> Dict[str, Dict[str, Any]]:
        """Name -> entry index over global_computed_list, rebuilt whenever the
        list object is replaced (init paths assign a fresh list; per-step code
        only mutates entries in place, so identity tracking is sufficient)."""
        if self._gcl_index_source is not self.global_computed_list:
            self._gcl_index = {g["name"]: g for g in self.global_computed_list}
            self._gcl_index_source = self.global_computed_list
        return self._gcl_index

    def update_global_list(
        self,
        block_name: str,
        h_value: int = 0,
        h_assign: bool = False,
        reset_computed: bool = False,
    ) -> None:
        """Update global computed list."""
        g_block = self._global_list_index().get(block_name)
        if g_block is not None:
            g_block["computed_data"] = not reset_computed
            if h_assign:
                g_block["hierarchy"] = h_value

    def stamp_held_outputs(
        self,
        block: DBlock,
        out_value: Any,
        pre_update_value: Optional[Dict[int, Any]] = None,
    ) -> None:
        """Store the outputs a discrete-rate block emits until its next sample.

        Between sample instants the loop propagates these held values, so they
        must be the outputs belonging to the instant that just fired.  Most
        stateful blocks return exactly that from execute() (they compute
        y = Cx + Du before advancing x), but a block declaring
        ``output_is_post_update`` returns the advanced state instead; holding
        that would make its staircase lead the true sampled response by a full
        sample period.  For those, ``pre_update_value`` — the output_only
        result captured *before* the state advance — is held instead.
        """
        source = out_value
        instance = getattr(block, "block_instance", None)
        if instance is not None and getattr(instance, "output_is_post_update", False):
            if pre_update_value is None:
                # No pre-update output available: keep whatever is already
                # held rather than stamping the advanced state.
                return
            source = pre_update_value
        if not isinstance(source, dict):
            return
        for port, value in source.items():
            if isinstance(port, int):
                block.set_held_output(port, value)

    @staticmethod
    def sync_integrator_output(block) -> None:
        """Publish an Integrator's freshly advanced state as its reported output.

        The copy is the whole point.  Every in-place integration method
        (FWD_EULER / BWD_EULER / TUSTIN / RK45 all do ``params["mem"] += ...``)
        mutates the state array in place, so binding "output" to the same object
        made the reported output track the state instead of lagging it by one
        step -- shifting every sample of those methods one step early (the
        integral of a unit step read t + dtime).  SOLVE_IVP rebinds "mem" to a
        fresh array on each call and so never exhibited it.
        """
        if getattr(block, "block_fn", None) != "Integrator":
            return
        exec_params = getattr(block, "exec_params", None)
        if not exec_params or "mem" not in exec_params:
            return
        exec_params["output"] = np.array(exec_params["mem"], copy=True)

    def _resolve_block_params(
        self, block: DBlock, dt: float, workspace_manager: Optional[WorkspaceManager] = None
    ) -> None:
        """Resolve a block's exec_params for the given simulation step.

        Skips the (potentially expensive) workspace resolution when exec_params
        is already populated for this dt, then resolves params, copies internal
        ('_'-prefixed) parameters, and stamps the current dtime. Shared by
        initialize_execution and run_compiled_simulation so the cache-skip logic
        stays consistent.
        """
        # Blocks gated to a discrete rate integrate over their own sample
        # period, not the base step (see DBlock.execution_step).  Before
        # propagate_sample_times() has run this is still dt, so the stamp is
        # refreshed there once the effective rates are known.
        step = block.execution_step(dt)
        cached = getattr(block, "exec_params", None)
        params_fp = source_params_fingerprint(block.params)
        cached_fp = cached.get("_source_params_fingerprint") if cached else None
        if cached and cached.get("dtime") == step and cached_fp == params_fp:
            # The cache hit skips the '_'-prefixed copy below, and '_' keys are
            # excluded from the fingerprint, so a mid-run write to params would
            # otherwise never reach the block.  Refresh the narrow set that the
            # engine/UI legitimately push down (see block_params.PUSH_DOWN_KEYS);
            # a blanket copy here would clobber the state the block owns.
            push_down_internal_params(block)
            return
        if workspace_manager is None:
            workspace_manager = WorkspaceManager()
        block.exec_params = workspace_manager.resolve_params(block.params)
        # Copy internal parameters (those starting with '_')
        block.exec_params.update({k: v for k, v in block.params.items() if k.startswith("_")})
        block.exec_params["dtime"] = step
        block.exec_params["_source_params_fingerprint"] = params_fp

    def execute_block(
        self, block: DBlock, output_only: bool = False
    ) -> Union[Dict[int, Any], bool]:
        """
        Execute a single block.
        Returns output value (dict) or False on failure.
        """
        try:
            # Lazy %s args: this runs for every block on every step, so the
            # message must not be formatted when INFO is disabled.
            logger.info("ENGINE EXECUTE: %s (b_type=%s)", block.name, block.b_type)
            kwargs = {
                "time": self.time_step,
                "inputs": block.input_queue,
                "params": block.exec_params,
            }

            if output_only:
                kwargs["output_only"] = True
                if block.block_fn == "Integrator":
                    kwargs["next_add_in_memory"] = False
                    kwargs["dtime"] = block.execution_step(self.sim_dt)

            if block.external:
                # The External block is an unimplemented stub (see blocks/external.py).
                # Do NOT dynamically dispatch into a file-supplied function here: the
                # `external`/`fn_name` flags are persisted in project files, so honoring
                # them would turn opening a malicious .diablos file into arbitrary code
                # execution the moment external loading is ever implemented. Treat as a
                # hard, explicit "not supported" error instead.
                logger.error(
                    f"Block {block.name}: external function execution is not supported "
                    f"and is disabled for security reasons."
                )
                return False
            else:
                if block.block_instance is None:
                    # Logic for blocks without instance (e.g. Subsystem if not flattened correctly)
                    # If it's a Subsystem, we shouldn't be here unless flattening failed.
                    b_type_logs = getattr(block, "block_type", "Unknown")
                    logger.error(
                        f"Block {block.name} (type={b_type_logs}) has no block_instance. Skipping execution."
                    )
                    return False

                out_value = block.block_instance.execute(**kwargs)

            if out_value is None:
                logger.error(f"Block {block.name} returned None")
                return False

            if isinstance(out_value, dict) and "E" in out_value and out_value["E"]:
                return out_value  # Caller checks for error

            return out_value

        except Exception as e:
            logger.error(f"Error executing block {block.name}: {e}", exc_info=True)
            self.error_msg = f"Block '{block.name}' failed: {e}"
            return False

    def _active_line_source(self):
        """Lines to analyze: the active (post-init) list once execution is set
        up, else the model's edit-time list — mirrors the pre-extraction
        ``use_active`` selection the graph queries shared."""
        use_active = len(self.active_blocks_list) > 0
        return self.active_line_list if use_active else self.model.line_list

    def check_diagram_integrity(self):
        """Verify that all block ports are properly connected.

        Returns:
            bool: True if diagram is valid, False otherwise
        """
        blocks_to_check = (
            self.active_blocks_list if self.active_blocks_list else self.model.blocks_list
        )
        return graph_analysis.check_diagram_integrity(blocks_to_check, self._active_line_source())

    def get_neighbors(self, block_name):
        """Get all input and output connections for a block.

        Returns:
            tuple: (inputs, outputs) where each is a list of connection dicts
        """
        return graph_analysis.get_neighbors(block_name, self._active_line_source())

    def get_outputs(self, block_name):
        """Get all output connections for a block.

        Returns:
            list: Output connections
        """
        return graph_analysis.get_outputs(block_name, self._active_line_source())

    def get_max_hierarchy(self):
        """Find the maximum hierarchy level in the diagram.

        Returns:
            int: Maximum hierarchy value
        """
        return graph_analysis.get_max_hierarchy(self.active_blocks_list)

    def reset_execution_data(self) -> None:
        """Reset execution state for all blocks.

        IMPORTANT: Must update global_computed_list AND restore hierarchy from it.
        Memory blocks preserve their input_queue so feedback from previous step can be used.
        """
        # Safety check - if global_computed_list isn't populated yet, use simple reset
        if not self.global_computed_list or len(self.global_computed_list) != len(
            self.active_blocks_list
        ):
            for block in self.active_blocks_list:
                block.computed_data = False
                block.data_received = 0
                block.data_sent = 0
                block.hierarchy = -1
                # Preserve input_queue for memory blocks (they need feedback from previous step)
                if block.name not in self.memory_blocks:
                    block.input_queue = {}
            return

        for i in range(len(self.active_blocks_list)):
            block = self.active_blocks_list[i]
            self.global_computed_list[i]["computed_data"] = False
            block.computed_data = False
            block.data_received = 0
            block.data_sent = 0
            # Preserve input_queue for memory blocks (they need feedback from previous step)
            if block.name not in self.memory_blocks:
                block.input_queue = {}
            block.hierarchy = self.global_computed_list[i]["hierarchy"]

    @staticmethod
    def _integrator_method(params) -> str:
        """Canonical integration-strategy name stored in ``params['method']``.

        Delegates to ``blocks.integrator.resolve_method`` so the legacy "RK45"
        spelling of the fixed-step 4-stage scheme resolves to the same strategy
        ("RK4") the block itself will run.  Falls back to the raw string if the
        block module cannot be imported (frozen/partial installs).
        """
        raw = (params or {}).get("method")
        try:
            from blocks.integrator import resolve_method
        except ImportError:  # pragma: no cover - defensive
            return str(raw)
        return resolve_method(raw)

    def count_rk45_integrators(self):
        """
        Check if any integrators use the RK4 (4-sub-step) method.

        The interpreter runs four sub-steps per simulation step when this is
        true, so it must agree with the block's own method resolution: a
        diagram saved with the legacy "RK45" spelling would otherwise get one
        sub-step per step and a trace stretched by 4x.

        Returns:
            bool: True if fixed-step RK4 integrators exist
        """
        for block in self.active_blocks_list:
            if block.block_fn in ("Integrator", "External"):
                if self._integrator_method(block.params) == "RK4":
                    return True
        return False

    def reset_memblocks(self) -> None:
        """Reset memory blocks (integrators, transfer functions, etc.).

        Resets _init_start_ in both params and exec_params, and clears stale
        per-run state accumulators (_prev, mem, output). Memory blocks are still
        expected to fully re-initialize their state when _init_start_ is True
        (block contract); clearing these keys defensively prevents a partial
        re-init from silently reusing the previous run's final values.
        """
        for block in self.active_blocks_list:
            if "_init_start_" in block.params:
                block.params["_init_start_"] = True
            # Also reset in exec_params if it exists (used during execution)
            if hasattr(block, "exec_params") and block.exec_params:
                if "_init_start_" in block.exec_params:
                    block.exec_params["_init_start_"] = True
                # Clear stored per-run state accumulators so a stale value from
                # a previous run cannot leak into the next one.
                for stale_key in ("_prev", "mem", "output"):
                    if stale_key in block.exec_params:
                        del block.exec_params[stale_key]

    def detect_algebraic_loops(self, uncomputed_blocks):
        """Detect algebraic loops among uncomputed blocks (Kahn's algorithm).

        Args:
            uncomputed_blocks: List of blocks that haven't been computed

        Returns:
            tuple: (is_algebraic: bool, cycle_nodes: list) - True if loop detected,
                   with list of block names involved in the cycle
        """
        return graph_analysis.detect_algebraic_loops(
            uncomputed_blocks, self._active_line_source(), self.memory_blocks
        )

    def children_recognition(self, block_name, children_list):
        """Recursively find all children (downstream blocks) of a block.

        Args:
            block_name: Name of the parent block
            children_list: List to accumulate children

        Returns:
            list: Updated children list
        """
        return graph_analysis.children_recognition(
            block_name, children_list, self._active_line_source()
        )

    def update_sim_params(
        self,
        sim_time: float,
        sim_dt: float,
        solver_method: str = None,
        rtol: float = None,
        atol: float = None,
    ) -> None:
        """
        Update simulation parameters.

        Args:
            sim_time: Total simulation time
            sim_dt: Time step
            solver_method: Compiled-solver method (e.g. 'RK45', 'RK4', 'Euler',
                'LSODA', 'BDF', 'Radau', 'RK23', 'DOP853'). Unchanged if None.
            rtol: Relative tolerance for adaptive scipy solvers. Unchanged if None.
            atol: Absolute tolerance for adaptive scipy solvers. Unchanged if None.
        """
        self.sim_time = sim_time
        self.sim_dt = sim_dt
        if solver_method is not None:
            self.solver_method = solver_method
        if rtol is not None:
            self.rtol = rtol
        if atol is not None:
            self.atol = atol

    def get_execution_status(self):
        """
        Get current execution status.

        Returns:
            dict: Status information
        """
        return {
            "initialized": self.execution_initialized,
            "paused": self.execution_pause,
            "stopped": self.execution_stop,
            "error": self.error_msg if self.error_msg else None,
            "sim_time": self.sim_time,
            "sim_dt": self.sim_dt,
        }

    # =========================================================================
    # Core Execution Methods - Migrated from DSim
    # =========================================================================

    def prepare_execution(self, execution_time: float) -> bool:
        """
        Prepare the simulation for execution by resolving parameters and
        identifying memory blocks.

        Args:
            execution_time: Total simulation time in seconds

        Returns:
            bool: True if preparation successful, False otherwise
        """
        logger.debug("*****INIT NEW EXECUTION*****")

        self.execution_stop = False
        self.error_msg = ""
        self.time_step = 0
        self.timeline = np.array([self.time_step])
        self.execution_time = execution_time
        self.invalidate_propagation_cache()

        workspace_manager = WorkspaceManager()

        for block in self.model.blocks_list:
            # Resolve parameters using WorkspaceManager
            logger.debug(f"Block {block.name}: params before resolve = {block.params}")
            block.exec_params = workspace_manager.resolve_params(block.params)
            logger.debug(f"Block {block.name}: exec_params after resolve = {block.exec_params}")

            # Copy internal parameters that start with '_'
            block.exec_params.update({k: v for k, v in block.params.items() if k.startswith("_")})

            # Dynamically set b_type for Transfer Functions
            self.set_block_type(block)

            block.exec_params["dtime"] = self.sim_dt
            try:
                missing_file_flag = block.reload_external_data()
                if missing_file_flag == 1:
                    logger.error(f"Missing external file for block: {block.name}")
                    return False
            except Exception as e:
                logger.error(f"Error reloading external data for block {block.name}: {str(e)}")
                return False

        if not self.check_diagram_integrity():
            logger.error("Diagram integrity check failed")
            return False

        # Initialize global computed list
        self.global_computed_list = [
            {"name": x.name, "computed_data": x.computed_data, "hierarchy": x.hierarchy}
            for x in self.model.blocks_list
        ]
        self.reset_execution_data()
        self.execution_time_start = time_module.time()

        # Identify memory blocks
        self.identify_memory_blocks()

        # Check for RK45 integrators
        self.rk45_len = self.count_rk45_integrators()
        self.rk_counter = 0

        # Auto-connect Goto/From tags
        try:
            self.model.link_goto_from()
        except Exception as e:
            logger.warning(f"Goto/From linking failed: {e}")

        logger.debug("Execution preparation complete")
        return True

    def set_block_type(self, block: DBlock) -> None:
        """Set block type based on transfer function properness."""
        if block.block_fn == "TranFn":
            num = block.exec_params.get("numerator", [])
            den = block.exec_params.get("denominator", [])
            block.b_type = 1 if len(den) > len(num) else 2
        elif block.block_fn == "DiscreteTranFn":
            num = block.exec_params.get("numerator", [])
            den = block.exec_params.get("denominator", [])
            block.b_type = 1 if len(den) > len(num) else 2
        elif block.block_fn == "DiscreteStateSpace":
            # Coerce to float so a ragged/malformed D becomes a clear error
            # instead of an object array where elementwise `== 0` misbehaves.
            try:
                D = np.asarray(block.exec_params.get("D", [[0.0]]), dtype=float)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Block '{block.name}': invalid D matrix for DiscreteStateSpace: {e}"
                )
            block.b_type = 1 if np.all(D == 0) else 2

    def identify_memory_blocks(self) -> None:
        """Identify blocks with memory (integrators, strictly proper TFs, state variables).

        Uses the shared taxonomy in lib/engine/memory_blocks.py — see that
        module for the unconditional set (OUTPUT_ONLY_SAFE_BLOCK_FNS) and
        conditional helpers (is_strictly_proper_tf, is_zero_D_statespace).
        """
        from lib.engine.memory_blocks import is_memory_block

        self.memory_blocks = set()
        for block in self.active_blocks_list:
            # requires_inputs=False blocks (sources) are always safe to call
            # with output_only=True; treat them as memory blocks too so the
            # init loop runs them once.
            block_class = getattr(block, "block_class", None)
            if block_class:
                # requires_inputs is a class-level attribute (block contract), so
                # read it off the class directly — no need to instantiate, which
                # would incur constructor cost/side effects on every init.
                if not getattr(block_class, "requires_inputs", True):
                    self.memory_blocks.add(block.name)
                    continue

            if is_memory_block(block):
                self.memory_blocks.add(block.name)
        logger.debug(f"MEMORY BLOCKS IDENTIFIED: {self.memory_blocks}")

    def propagate_sample_times(self) -> None:
        """
        Propagate sample times through the diagram.

        Resolves effective sample times for all blocks based on:
        - Explicit sample_time parameter (>0 = fixed discrete rate)
        - Inherited rate (0 = inherit from fastest connected input)
        - Continuous (-1 = execute every timestep, default)

        Must be called after identify_memory_blocks() during initialization.
        """
        logger.debug("Propagating sample times...")

        # Build connection map for efficient lookup
        # Maps block_name -> list of source block names
        input_sources: Dict[str, List[str]] = {b.name: [] for b in self.active_blocks_list}
        for line in self.active_line_list:
            if line.dstblock in input_sources:
                input_sources[line.dstblock].append(line.srcblock)

        # Create block lookup
        block_map = {b.name: b for b in self.active_blocks_list}

        # Phase 1: Resolve explicit sample times from parameters
        for block in self.active_blocks_list:
            declared_rate = block.resolve_sample_time()
            block.effective_sample_time = declared_rate
            # Reset execution state for new simulation
            block.reset_sample_time_state()

        # Phase 2: Propagate inherited rates (sample_time = 0)
        # Use iterative propagation until no changes occur
        max_iterations = len(self.active_blocks_list) + 1
        for iteration in range(max_iterations):
            changed = False

            for block in self.active_blocks_list:
                # Only process blocks that inherit (sample_time = 0)
                if block.effective_sample_time != 0.0:
                    continue

                # Find fastest (smallest positive) sample time from inputs
                fastest_rate = -1.0  # Default to continuous if no discrete inputs
                for src_name in input_sources.get(block.name, []):
                    src_block = block_map.get(src_name)
                    if src_block and src_block.effective_sample_time > 0:
                        if fastest_rate < 0 or src_block.effective_sample_time < fastest_rate:
                            fastest_rate = src_block.effective_sample_time

                # Apply inherited rate
                if fastest_rate != block.effective_sample_time:
                    block.effective_sample_time = fastest_rate
                    changed = True

            if not changed:
                break

        # Phase 3: Mark connections as discrete based on source block sample time
        for line in self.active_line_list:
            src_block = block_map.get(line.srcblock)
            if src_block and src_block.effective_sample_time > 0:
                line.discrete_signal = True
            else:
                line.discrete_signal = False

        # Phase 3b: Warn about blocks that need a rate but did not get one.
        # These are pure sample-index recursions; with nothing to inherit they
        # fall back to one sample per solver step, which means the same diagram
        # gives a different physical response when sim_dt changes.  That used
        # to happen silently — it is the app's one remaining place where a
        # solver setting alters the modelled system rather than its accuracy.
        for block in self.active_blocks_list:
            instance = getattr(block, "block_instance", None)
            if instance is None or not getattr(instance, "requires_sample_time", False):
                continue
            if block.effective_sample_time <= 0:
                logger.warning(
                    f"{block.name} ({block.block_fn}) has no resolved sample time: "
                    f"no upstream discrete rate to inherit and none set. It will "
                    f"advance one sample per solver step ({self.sim_dt}s), so its "
                    f"response depends on the simulation step size. Set its "
                    f"'sampling_time' to the intended period in seconds."
                )

        # Phase 4: Stamp each block's own execution step into exec_params.
        # Blocks gated to a discrete rate are only executed every Ts seconds,
        # so the dtime they integrate with must be Ts, not the base solver
        # step — otherwise a continuous block given sampling_time=Ts advances
        # dt per sample and runs Ts/dt times too slowly.  This runs after the
        # rates are resolved (including inheritance), and before any block is
        # executed, so init-time discretisations (TranFn/StateSpace
        # cont2discrete) see the correct step.
        for block in self.active_blocks_list:
            if getattr(block, "exec_params", None):
                block.exec_params["dtime"] = block.execution_step(self.sim_dt)

        # Log resolved sample times
        discrete_blocks = [
            (b.name, b.effective_sample_time)
            for b in self.active_blocks_list
            if b.effective_sample_time > 0
        ]
        if discrete_blocks:
            logger.info(f"DISCRETE BLOCKS: {discrete_blocks}")
        discrete_lines = sum(
            1 for line in self.active_line_list if getattr(line, "discrete_signal", False)
        )
        if discrete_lines:
            logger.info(f"DISCRETE CONNECTIONS: {discrete_lines}")
        logger.debug("Sample time propagation complete")

    def propagate_outputs(
        self, block: DBlock, out_value: Dict[int, Any], count: bool = True
    ) -> None:
        """
        Propagate block outputs to connected downstream blocks.

        Args:
            block: Source block
            out_value: Output values from the block
            count: Whether this delivery counts as a new input arrival. Pass
                False to overwrite a value already delivered this step without
                re-counting it: a feedthrough memory block seeds its consumers
                with a stale held value early in the step and then refreshes
                them once it has actually sampled. Counting twice would inflate
                data_received and let a multi-input consumer fire before all of
                its real inputs had arrived.
        """
        targets = self._propagation_targets(block.name)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ENGINE PROPAGATE: %s -> %s", block.name, [t[0].name for t in targets])

        for mblock, srcport, dstport in targets:
            # A block may legitimately omit a port from its output dict
            # (sparse/partial output). Guard so a missing-but-wired port
            # logs a clear diagnostic instead of raising a raw KeyError
            # that the broad init except surfaces as an opaque failure.
            if srcport not in out_value:
                logger.warning(
                    f"Block '{block.name}' produced no value for output port "
                    f"{srcport} wired to '{mblock.name}' port "
                    f"{dstport}; skipping propagation."
                )
                continue
            mblock.input_queue[dstport] = out_value[srcport]
            if count:
                mblock.data_received += 1
                block.data_sent += 1

    def _propagation_targets(self, block_name: str) -> List[Tuple[DBlock, int, int]]:
        """Downstream ``(dst_block, srcport, dstport)`` deliveries for a source.

        The adjacency is built once for the current (blocks, lines) pair and
        cached.  Before this, every propagate_outputs call rescanned every line
        (``get_outputs``) and then every block (``is_child_of``), making the
        interpreter's per-step propagation O(blocks x (blocks + lines)).
        """
        # Use active blocks if execution initialized (flattened copies),
        # otherwise the model list (fallback for pre-init callers).
        blocks = (
            self.active_blocks_list if len(self.active_blocks_list) > 0 else self.model.blocks_list
        )
        lines = self._active_line_source()

        # Rebuild whenever either list is a different object or changed length.
        # Both are replaced wholesale by initialize_execution / the editor, and
        # invalidate_propagation_cache() covers an in-place rewrite.
        source = self._prop_adj_source
        if (
            self._prop_adj is None
            or source is None
            or source[0] is not blocks
            or source[1] is not lines
            or source[2] != (len(blocks), len(lines))
        ):
            self._prop_adj = self._build_propagation_adjacency(blocks, lines)
            self._prop_adj_source = (blocks, lines, (len(blocks), len(lines)))

        return self._prop_adj.get(block_name, ())

    @staticmethod
    def _build_propagation_adjacency(blocks, lines) -> Dict[str, List[Tuple[DBlock, int, int]]]:
        """srcblock name -> [(dst_block, srcport, dstport), ...].

        Lines pointing at a block that is not in ``blocks`` are dropped, which
        is what the old block-scan did implicitly.
        """
        block_by_name = {b.name: b for b in blocks}
        adjacency: Dict[str, List[Tuple[DBlock, int, int]]] = {}
        for line in lines:
            dst_block = block_by_name.get(line.dstblock)
            if dst_block is None:
                continue
            adjacency.setdefault(line.srcblock, []).append((dst_block, line.srcport, line.dstport))
        return adjacency

    def invalidate_propagation_cache(self) -> None:
        """Drop the cached propagation adjacency (call after editing the
        diagram in place, i.e. without replacing the block/line lists)."""
        self._prop_adj = None
        self._prop_adj_source = None

    def _children_recognition(
        self, block_name: str, children_list: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """Check if block_name is in the children list.

        Returns:
            Tuple of (is_child, matching_connections)
        """
        return graph_analysis.is_child_of(block_name, children_list)

    def check_global_list(self) -> bool:
        """Check if all blocks have been computed."""
        return all(elem["computed_data"] for elem in self.global_computed_list)

    def count_computed_global_list(self) -> int:
        """Count the number of computed blocks."""
        return sum(1 for x in self.global_computed_list if x["computed_data"])

    def execution_failed(self, msg: str = "") -> None:
        """
        Handle execution failure.

        Args:
            msg: Error message
        """
        self.execution_initialized = False
        self.reset_memblocks()
        self.error_msg = msg
        logger.error("*****EXECUTION STOPPED*****")

    def check_compilability(self, blocks: List[DBlock]) -> bool:
        """Check if the system can be compiled."""
        return self.compiler.check_compilability(blocks)

    def clear_compile_cache(self) -> None:
        """Drop the cached compiled RHS, state map, and replay executors."""
        self._compiled_system_cache_key = None
        self._compiled_system_cache_value = None

    def _compile_system_cached(self, blocks, sorted_blocks, lines, dt):
        """Compile the current diagram, reusing the previous compile on a hit."""
        key = compiled_system_fingerprint(blocks, sorted_blocks, lines, dt)
        if key == self._compiled_system_cache_key and self._compiled_system_cache_value:
            self.compile_cache_hits += 1
            model_func, y0, state_map, block_matrices, block_executors = (
                self._compiled_system_cache_value
            )
            self.compiler.block_executors = block_executors
            return model_func, y0.copy(), state_map, block_matrices, True

        self.compile_cache_misses += 1
        model_func, y0, state_map, block_matrices = self.compiler.compile_system(
            blocks, sorted_blocks, lines
        )
        block_executors = dict(getattr(self.compiler, "block_executors", {}))
        self._compiled_system_cache_key = key
        self._compiled_system_cache_value = (
            model_func,
            y0.copy(),
            state_map,
            block_matrices,
            block_executors,
        )
        return model_func, y0.copy(), state_map, block_matrices, False

    def get_solver_diagnostics(self) -> Dict[str, Any]:
        """Return a copy of the most recent compiled-solver diagnostics."""
        return dict(self.last_solver_diagnostics)

    def format_last_solver_diagnostics(self) -> str:
        """One-line summary of the most recent compiled run, or '' when none
        was recorded (e.g. the interpreter path ran)."""
        if not self.last_solver_diagnostics:
            return ""
        return format_diagnostics_for_log(self.last_solver_diagnostics)

    def _record_solver_diagnostics(self, **kwargs) -> None:
        """Store a compact, UI/log-friendly summary of the compiled run.

        Thin wrapper over :func:`lib.engine.solver_diagnostics.build_diagnostics`
        that injects the engine's running compile-cache counters.
        """
        self.last_solver_diagnostics = build_diagnostics(
            compile_cache_hits_total=self.compile_cache_hits,
            compile_cache_misses_total=self.compile_cache_misses,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Compiled (fast-solver) path.  The implementation lives in
    # lib/engine/compiled_runner.py; these stay as the public API.
    # ------------------------------------------------------------------

    @staticmethod
    def _replay_has_feedthrough(block, block_matrices) -> bool:
        """See :func:`lib.engine.compiled_runner.replay_has_feedthrough`."""
        return compiled_runner.replay_has_feedthrough(block, block_matrices)

    def _replay_compiled_signals(
        self, sol, current_blocks, current_lines, state_map, block_matrices
    ):
        """See :func:`lib.engine.compiled_runner.replay_compiled_signals`."""
        return compiled_runner.replay_compiled_signals(
            self, sol, current_blocks, current_lines, state_map, block_matrices
        )

    def run_compiled_simulation(
        self, blocks: List[DBlock], lines: List[Any], t_span: Tuple[float, float], dt: float
    ) -> bool:
        """Run the simulation using the compiled fast solver.

        See :func:`lib.engine.compiled_runner.run_compiled_simulation`.
        """
        return compiled_runner.run_compiled_simulation(self, blocks, lines, t_span, dt)
