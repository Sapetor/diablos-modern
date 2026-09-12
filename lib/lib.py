"lib.py - Contains all the core functions and classes for the simulation and execution of the graphs."

import numpy as np
import os
import time
import sys
from typing import Dict, Any, Optional
from tqdm import tqdm
from PyQt6.QtWidgets import QDialog
from lib.workspace import WorkspaceManager
from lib.dialogs import SimulationDialog
from lib.i18n import tr
from lib.sim_prefs import ask_before_run as _stored_ask_before_run, set_ask_before_run
import logging


# Import block size configuration

# Import extracted classes
from lib.plotting.scope_plotter import ScopePlotter
from lib.managers.subsystem_manager import SubsystemManager

logger = logging.getLogger(__name__)


class DSim:
    """
    Class that manages the simulation interface and main functions.

    :param SCREEN_WIDTH: The width of the window
    :param SCREEN_HEIGHT: The height of the window
    :param colors: List of predefined colors for elements that show in the canvas.
    :param fps: Base frames per seconds for pygame's loop.
    :param filename: Name of the file that was recently loaded. By default is 'data.dat'.
    :param sim_time: Simulation time for graph execution.
    :param sim_dt: Simulation sampling time for graph execution.
    :param plot_trange: Width in number of elements that must be shown when a graph is getting executed with dynamic plot enabled.
    :type SCREEN_WIDTH: int
    :type SCREEN_HEIGHT: int
    :type colors: dict
    :type fps: int
    :type filename: str
    :type sim_time: float
    :type sim_dt: float
    :type plot_trange: int

    """

    def __init__(self):
        logger.debug("Initializing DSim with MVC architecture")

        # Initialize MVC components
        from lib.models.simulation_model import SimulationModel
        from lib.engine.simulation_engine import SimulationEngine
        from lib.services.file_service import FileService
        from lib.services.run_history_service import RunHistoryService
        from lib.diagram_validator import DiagramValidator

        self.model = SimulationModel()
        self.engine = SimulationEngine(self.model)
        self.file_service = FileService(self.model)
        self.diagram_validator = DiagramValidator(self.model)

        # Screen/UI parameters
        self.SCREEN_WIDTH = 1280
        self.SCREEN_HEIGHT = 720 + 50
        self.FPS = 60

        # Delegate commonly used properties to model for backward compatibility
        self.colors = self.model.colors
        self.menu_blocks = self.model.menu_blocks
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list
        self.connections_list = self.line_list  # Alias for backward/forward compatibility

        # UI state
        self.ss_count = 0

        # Delegate simulation parameters to engine
        self.sim_time = self.engine.sim_time
        self.sim_dt = self.engine.sim_dt
        self.solver_method = self.engine.solver_method
        self.rtol = self.engine.rtol
        self.atol = self.engine.atol
        self.zero_crossing = self.engine.zero_crossing
        self.plot_trange = 100

        # Execution state (properties delegate to engine)
        # execution_initialized, execution_stop, error_msg are now properties
        self.execution_pause = self.engine.execution_pause
        self.real_time = self.engine.real_time
        self.dynamic_plot = False
        # Application preference (QSettings, not diagram data): when True,
        # Play re-opens the Simulation-settings dialog before every run. Read
        # once here; the settings dialog keeps the two in step.
        self.ask_before_run = _stored_ask_before_run()
        # Set while execution_batch runs off the GUI thread: suppresses every
        # Qt call made from the interpreter loop (see execution_batch).
        self._defer_gui_plots = False
        # True when the diagram was edited after the last run, i.e. the data
        # held in the Scope blocks no longer describes this diagram. This used
        # to piggy-back on ``dirty`` (which execution_init cleared), conflating
        # "unsaved" with "stale plot data"; the two are now separate.
        self.diagram_changed_since_run = True

        # Delegate filename to file service
        self.filename = self.file_service.filename

        # Execution tracking (timeline, global_computed_list are now properties)
        self.scope_plotter = ScopePlotter(self)
        self.outs = []
        self.plotty = None

        # Run history service
        self.run_history_service = RunHistoryService()
        self.run_history_service.load_history()

        # Subsystem manager
        self.subsystem_manager = SubsystemManager(self.model, self)

    # ``dirty`` is a live view of the model's flag, not a copy.
    #
    # FileService clears ``model.dirty`` on save while the whole GUI
    # (status bar, property controller, clipboard/connection managers) reads
    # and writes ``dsim.dirty``.  While these were two independent booleans a
    # save never cleared the flag the GUI showed, and ``execution_init`` used
    # to clear ``dsim.dirty`` -- so pressing Run marked unsaved edits as saved
    # and closing the window lost them silently.
    @property
    def dirty(self) -> bool:
        """True when the diagram has unsaved changes (delegates to the model)."""
        return self.model.dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
        value = bool(value)
        if value:
            # Every caller that marks the diagram dirty has just edited it, so
            # any plotted run is now stale (see diagram_changed_since_run).
            self.diagram_changed_since_run = True
        self.model.dirty = value

    # Properties for backward compatibility with subsystem navigation
    @property
    def navigation_stack(self):
        """Navigation stack - delegated to subsystem_manager."""
        return self.subsystem_manager.navigation_stack

    @property
    def current_subsystem(self):
        """Current subsystem name - delegated to subsystem_manager."""
        return self.subsystem_manager.current_subsystem

    def enter_subsystem(self, subsystem_block):
        """
        Enter a subsystem block to edit its contents.
        Pushes the current context to the stack and enters the subsystem.
        """
        return self.subsystem_manager.enter_subsystem(subsystem_block)

    def exit_subsystem(self):
        """
        Exit the current subsystem and return to the parent scope.
        Syncs external ports with internal Inport/Outport blocks.
        """
        return self.subsystem_manager.exit_subsystem()

    def get_current_path(self):
        """
        Return the current navigation path as a list of strings.
        Example: ['Top Level', 'Subsystem1', 'Nested2']
        """
        return self.subsystem_manager.get_current_path()

    def get_root_context(self):
        """
        Get the root context (blocks_list, line_list) of the simulation model.
        Used for execution to ensure we always simulate the full system.
        """
        return self.subsystem_manager.get_root_context()

    # Properties for state shared with SimulationEngine
    @property
    def timeline(self):
        """Timeline array - shared with engine."""
        return self.engine.timeline

    @timeline.setter
    def timeline(self, value):
        self.engine.timeline = value

    @property
    def time_step(self):
        """Current time step - shared with engine."""
        return self.engine.time_step

    @time_step.setter
    def time_step(self, value):
        self.engine.time_step = value

    @property
    def global_computed_list(self):
        """Block computation tracking - shared with engine."""
        return self.engine.global_computed_list

    @global_computed_list.setter
    def global_computed_list(self, value):
        self.engine.global_computed_list = value

    @property
    def execution_initialized(self):
        """Whether simulation is initialized - shared with engine."""
        return self.engine.execution_initialized

    @execution_initialized.setter
    def execution_initialized(self, value):
        self.engine.execution_initialized = value

    @property
    def execution_stop(self):
        """Whether simulation is stopped - shared with engine."""
        return self.engine.execution_stop

    @execution_stop.setter
    def execution_stop(self, value):
        self.engine.execution_stop = value

    @property
    def error_msg(self):
        """Error message - shared with engine."""
        return self.engine.error_msg

    @error_msg.setter
    def error_msg(self, value):
        self.engine.error_msg = value

    @property
    def last_solver_diagnostics(self):
        """Most recent compiled-solver diagnostics - shared with engine."""
        return self.engine.get_solver_diagnostics()

    @property
    def last_solver_diagnostics_summary(self):
        """One-line summary of the most recent compiled run, or '' when the last
        run used the interpreter path (which records no compiled diagnostics)."""
        return self.engine.format_last_solver_diagnostics()

    @property
    def execution_time_start(self):
        """Execution start time - shared with engine."""
        return self.engine.execution_time_start

    @execution_time_start.setter
    def execution_time_start(self, value):
        self.engine.execution_time_start = value

    @property
    def memory_blocks(self):
        """Memory blocks set - shared with engine."""
        return self.engine.memory_blocks

    @memory_blocks.setter
    def memory_blocks(self, value):
        self.engine.memory_blocks = value

    @property
    def max_hier(self):
        """Maximum block hierarchy level - shared with engine."""
        return self.engine.max_hier

    @max_hier.setter
    def max_hier(self, value):
        self.engine.max_hier = value

    @property
    def rk45_len(self):
        """Whether any integrator uses RK45 sub-stepping - shared with engine."""
        return self.engine.rk45_len

    @rk45_len.setter
    def rk45_len(self, value):
        self.engine.rk45_len = value

    @property
    def rk_counter(self):
        """RK45 sub-step counter - shared with engine."""
        return self.engine.rk_counter

    @rk_counter.setter
    def rk_counter(self, value):
        self.engine.rk_counter = value

    ##### ADD OR REMOVE BLOCKS AND LINES #####

    def add_block(self, block, m_pos):
        """Add a block to the diagram. Delegates to model."""
        new_block = self.model.add_block(block, m_pos)
        self.diagram_changed_since_run = True
        return new_block

    def add_line(self, srcData, dstData):
        """Add a connection line between two blocks. Delegates to model."""
        new_line = self.model.add_line(srcData, dstData)
        self.diagram_changed_since_run = True
        return new_line

    def remove_block_and_lines(self, block):
        """Remove a block and its associated lines. Delegates to model."""
        self.model.remove_block(block)
        self.line_list = self.model.line_list  # Sync line_list after removal
        self.connections_list = self.line_list  # Keep alias in sync
        self.diagram_changed_since_run = True

    # NOTE: display_lines, display_blocks, display_ports, update_lines moved to ModernCanvas
    # NOTE: Block loading moved to SimulationModel.load_all_blocks()

    ##### LOADING AND SAVING #####

    def save(
        self,
        autosave: bool = False,
        modern_ui_data: Optional[Dict] = None,
        filepath: Optional[str] = None,
    ) -> int:
        """
        Save diagram to file. Delegates to FileService.

        Args:
            autosave: If True, save to autosave location without dialog
            modern_ui_data: Additional UI state data to save
            filepath: Explicit destination. When given, no file dialog is shown
                and the file is written there (the autosave path takes this
                route -- see ``ModernDiaBloSWindow._auto_save``).

        Returns:
            0 on success, 1 if user cancelled
        """
        # Sync parameters to file_service
        sim_params = {
            "sim_time": self.sim_time,
            "sim_dt": self.sim_dt,
            "plot_trange": self.plot_trange,
            "solver_method": self.solver_method,
            "rtol": self.rtol,
            "atol": self.atol,
            "zero_crossing": self.zero_crossing,
        }
        self.file_service.SCREEN_WIDTH = self.SCREEN_WIDTH
        self.file_service.SCREEN_HEIGHT = self.SCREEN_HEIGHT

        # An autosave is a crash-recovery snapshot, not the user saving their
        # file: it must not clear the unsaved-changes flag. FileService clears
        # model.dirty on every successful write, so restore it afterwards --
        # otherwise pressing Run (execution_init autosaves) or the 2-minute
        # timer would silently mark unsaved edits as saved.
        was_dirty = self.dirty

        result = self.file_service.save(
            autosave=autosave,
            modern_ui_data=modern_ui_data,
            sim_params=sim_params,
            filepath=filepath,
        )

        if autosave:
            self.dirty = was_dirty
        elif result == 0:
            # Sync filename back for backward compatibility
            self.filename = self.file_service.filename

        return result

    def serialize(self, modern_ui_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Serialize current diagram state to dict.
        Used by DiagramService.
        """
        sim_params = {
            "sim_time": self.sim_time,
            "sim_dt": self.sim_dt,
            "plot_trange": self.plot_trange,
            "solver_method": self.solver_method,
            "rtol": self.rtol,
            "atol": self.atol,
            "zero_crossing": self.zero_crossing,
        }
        return self.file_service.serialize(modern_ui_data, sim_params)

    def deserialize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize diagram state from dict.
        Used by DiagramService.
        """
        # A different diagram is replacing the current one: drop the scope
        # data held for the "Previous run" overlay (see clear_all).
        self.scope_plotter.reset_held_runs()
        sim_params = self.file_service.apply_loaded_data(data)

        # Sync simulation parameters back to DSim
        self.sim_time = sim_params.get("sim_time", 1.0)
        self.sim_dt = sim_params.get("sim_dt", 0.01)
        self.plot_trange = sim_params.get("plot_trange", 100)
        self.solver_method = sim_params.get("solver_method", "RK45")
        self.rtol = sim_params.get("rtol", 1e-9)
        self.atol = sim_params.get("atol", 1e-12)
        self.zero_crossing = bool(sim_params.get("zero_crossing", True))
        self.ss_count = 0
        self.filename = self.file_service.filename
        return sim_params

    def clone_for_analysis(self) -> "DSim":
        """Return an independent ``DSim`` holding a copy of this diagram.

        Experiment runners (Monte-Carlo, parameter sweep) rewrite
        ``blocks_list``/``line_list``/``timeline``/``execution_initialized``
        and re-enter the interpreter loop.  Doing that to the live ``DSim`` from
        a worker thread races the GUI's 60 FPS repaint timer (which iterates the
        same lists and may itself call ``execution_loop``).  Running the
        experiment against a private copy removes the shared state entirely.

        The copy goes through ``serialize``/``deserialize``, i.e. exactly the
        save/reopen path, so block names (which the runners use to address
        parameters) and simulation settings are preserved.
        """
        clone = DSim()
        clone.deserialize(self.serialize())
        clone.use_fast_solver = getattr(self, "use_fast_solver", True)
        # Never let a cloned run touch the GUI: no dynamic plotting, no scope
        # windows -- the caller harvests block data directly.
        clone.dynamic_plot = False
        return clone

    def open(self) -> Optional[Dict]:
        """
        Load diagram from file. Delegates to FileService.

        Returns:
            modern_ui_data dict if present in file, None otherwise
        """
        data = self.file_service.load()

        if data is None:
            return None

        version = data.get("version", "1.0")
        if version != "2.0":
            logger.warning(f"Loading file version {version}, current is 2.0")

        # Apply loaded data using internal deserialize which handles syncing
        self.deserialize(data)

        return data.get("modern_ui_data")

    def clear_all(self):
        """Clear all blocks and lines from the diagram. Delegates to model."""
        self.model.clear_all()
        # Drop the scope data held for the "Previous run" overlay: it belongs
        # to the diagram being discarded, and overlaying it onto the next
        # diagram would silently mix unrelated signals.
        self.scope_plotter.reset_held_runs()
        # Update references
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list
        self.connections_list = self.line_list  # Keep alias in sync
        self.diagram_changed_since_run = True

        # Reset UI state
        self.ss_count = 0
        self.filename = "data.dat"
        self.sim_time = 1.0
        self.sim_dt = 0.01
        self.solver_method = "RK45"
        self.rtol = 1e-9
        self.atol = 1e-12
        self.zero_crossing = True
        self.plot_trange = 100
        self.dynamic_plot = False

        # The compiled RHS cached for the old diagram no longer applies.
        self.engine.clear_compile_cache()

    def new_diagram(self):
        """Start a fresh, untitled diagram (the File > New reset).

        ``DiagramService.new_diagram`` (and therefore ``ProjectManager`` and the
        File > New action) has always called this; ``DSim`` never defined it, so
        File > New raised ``AttributeError``. It resets the diagram via
        ``clear_all`` and drops the previous file's name, so the next save asks
        where to put the new diagram instead of silently overwriting the old
        file.
        """
        self.clear_all()
        # Untitled: use the canonical default extension on both sides (clear_all
        # still leaves the legacy .dat name behind).
        self.filename = "data.diablos"
        self.file_service.filename = "data.diablos"
        # An empty diagram has nothing unsaved.
        self.dirty = False

    ##### DIAGRAM EXECUTION #####

    # Settings the SimulationDialog edits that are also written to the .diablos
    # file (FileService stores plot_trange under the "sim_trange" key). Changing
    # one of these dirties the diagram; ``real_time``/``dynamic_plot`` are
    # session-only run modes and do not.
    PERSISTED_SIM_SETTINGS = (
        "sim_time",
        "sim_dt",
        "plot_trange",
        "solver_method",
        "rtol",
        "atol",
        "zero_crossing",
    )

    # Every setting the dialog round-trips onto this DSim. The dialog's result
    # keys are deliberately the attribute names, so applying it is a loop.
    _SIM_SETTINGS = PERSISTED_SIM_SETTINGS + ("real_time", "dynamic_plot")

    def open_simulation_dialog(self, parent=None, accept_label=None):
        """Show the Simulation-settings dialog pre-filled with the live values.

        Returns the dialog's value dict, or ``None`` if the user cancelled.
        Applying the result is the caller's job (``apply_sim_settings``), so
        the same dialog serves both the Simulation > Simulation Settings...
        action and the opt-in "ask before every run" path.
        """
        dialog = SimulationDialog(
            self.sim_time,
            self.sim_dt,
            self.plot_trange,
            parent=parent,
            solver_method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
            zero_crossing=self.zero_crossing,
            real_time=self.real_time,
            dynamic_plot=self.dynamic_plot,
            ask_before_run=self.ask_before_run,
            accept_label=accept_label,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        try:
            return dialog.get_values()
        except (ValueError, TypeError):
            # The dialog validates its numeric fields before accepting, so this
            # is belt-and-braces: keep the current settings rather than raise.
            logger.warning("Invalid simulation settings. Keeping the current values.")
            return None

    def apply_sim_settings(self, values) -> bool:
        """Apply a ``SimulationDialog`` result dict onto this DSim.

        Returns True when a setting that is stored in the ``.diablos`` file
        changed, i.e. when the caller should mark the diagram dirty. The
        "ask before every run" flag is an application preference, so it is
        written straight to QSettings and never dirties the diagram.
        """
        if not values:
            return False

        diagram_changed = False
        for key in self._SIM_SETTINGS:
            if key not in values:
                continue
            new = values[key]
            if getattr(self, key) != new:
                setattr(self, key, new)
                if key in self.PERSISTED_SIM_SETTINGS:
                    diagram_changed = True

        if "ask_before_run" in values:
            ask = bool(values["ask_before_run"])
            if ask != self.ask_before_run:
                self.ask_before_run = ask
                set_ask_before_run(ask)

        return diagram_changed

    def execution_init_time(self):
        """
        :purpose: Creates a pop-up window to ask for graph simulation setup values.
        :description: Opens the Simulation-settings dialog and applies what the user accepted, returning the simulation duration (or -1 if they cancelled). This is the *ask* path: since 2026-09 Play runs straight away with the stored settings, and this is reached only from Simulation > Simulation Settings... or when the "Ask before every run" preference is on.
        """
        values = self.open_simulation_dialog(accept_label=tr("Simulate"))
        if values is None:
            return -1
        self.apply_sim_settings(values)
        return self.sim_time

    def _resolve_block_params(self, blocks, workspace_manager, sim_dt, mask_scope=None) -> bool:
        """
        Recursively resolve execution parameters for a block hierarchy.

        Shared by execution_init (interactive) and run_tuning_simulation
        (headless) so both paths handle Transfer-Function typing, External
        data reload, and Subsystem recursion identically.

        ``mask_scope`` carries the enclosing masked subsystem's resolved
        parameters (see ``lib/masks.py``).  Mask expressions are folded into
        ``exec_params`` only -- the stored ``params`` keep the user's
        expression strings, so they survive save/load and every re-run.  The
        flattener repeats the same resolution on its execution clones; doing
        it here as well means ``set_block_type`` below sees resolved
        numerator/denominator arrays rather than raw mask variable names.

        Returns True on success, False if an external file is missing or a
        reload raised (self.error_msg is set in those cases).
        """
        from lib.masks import MaskError, child_scope_for, resolve_params_in_scope

        for block in blocks:
            # Resolve parameters using WorkspaceManager, with the enclosing
            # mask's parameters layered over the diagram workspace.
            block.exec_params = workspace_manager.resolve_params(
                resolve_params_in_scope(block.params, mask_scope)
            )
            # Copy internal parameters that start with '_'
            block.exec_params.update({k: v for k, v in block.params.items() if k.startswith("_")})

            # Dynamically set b_type for Transfer Functions (delegated to engine)
            self.engine.set_block_type(block)

            block.exec_params["dtime"] = sim_dt

            # Reload external data
            try:
                if (
                    block.block_fn == "External"
                ):  # Check explicitly or reload_external_data handles it
                    missing_file_flag = block.reload_external_data()
                    if missing_file_flag == 1:
                        msg = f"Missing external file for block: {block.name}"
                        logger.error(msg)
                        self.error_msg = msg
                        return False
            except Exception as e:
                msg = f"Error reloading external data for block {block.name}: {str(e)}"
                logger.error(msg)
                self.error_msg = msg
                return False

            # Recurse if subsystem
            if getattr(block, "block_type", "") == "Subsystem":
                try:
                    child_scope = child_scope_for(block, mask_scope)
                except MaskError as e:
                    logger.error(str(e))
                    self.error_msg = str(e)
                    return False
                if (
                    self._resolve_block_params(
                        block.sub_blocks, workspace_manager, sim_dt, child_scope
                    )
                    is False
                ):
                    return False
        return True

    def execution_init(self, ask: Optional[bool] = None) -> bool:
        """
        :purpose: Initializes the graph execution.
        :description: This is the first stage of the graph simulation, where variables and vectors are initialized, as well as testing to verify that everything is working properly. A previous autosave is done, as well as a block connection check and possible algebraic loops. If everything goes well, we continue with the loop stage.

        ``ask`` decides whether the Simulation-settings dialog is shown first.
        ``None`` (the default) resolves it from ``self.ask_before_run``, which
        mirrors the ``simulation/ask_before_run`` QSettings preference and is
        off by default -- so Play runs immediately with the settings stored in
        the diagram. Pass True/False to force either path (the headless and
        analysis runners never want the dialog).
        """
        try:
            logger.debug("Starting execution initialization...")
            # The class containing the functions for the execution is called

            self.execution_stop = False  # Prevent execution from stopping before executing in error
            self.error_msg = ""  # Clear any previous error message
            self.time_step = (
                0  # First iteration of the time which will be incrementing self.sim_dt seconds
            )
            self._timeline_list = [
                self.time_step
            ]  # Accumulate as list, convert to np.array when done
            self.timeline = np.array([self.time_step])  # Also keep np version for compatibility

            # Some parameters are initialized including the maximum simulation
            # time. Only the opt-in "ask before every run" path pops the dialog;
            # otherwise the stored duration is used as-is.
            if ask is None:
                ask = bool(getattr(self, "ask_before_run", False))
            self.execution_time = self.execution_init_time() if ask else self.sim_time

            # To cancel the simulation before running it (having pressed X in the pop up)
            if self.execution_time == -1 or len(self.blocks_list) == 0:
                self.execution_initialized = False
                return False

            # Force save before executing (so as not to lose the diagram)
            if self.save(True) == 1:
                return False

            logger.debug("*****INIT NEW EXECUTION*****")
            _t0 = time.time()

            workspace_manager = WorkspaceManager()
            logger.debug(f"[TIMING] WorkspaceManager created: {time.time() - _t0:.3f}s")

            # Get Root Context for execution
            root_blocks, root_lines = self.get_root_context()

            # Resolve params for ALL blocks in hierarchy
            logger.debug("Resolving parameters for hierarchy...")
            _t1 = time.time()
            if not self._resolve_block_params(root_blocks, workspace_manager, self.sim_dt):
                return False
            logger.debug(f"[TIMING] _resolve_block_params: {time.time() - _t1:.3f}s")

            logger.debug("Initializing execution...")

            # Sync simulation parameters to engine before initialization
            self.engine.update_sim_params(
                self.sim_time,
                self.sim_dt,
                solver_method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                zero_crossing=self.zero_crossing,
            )

            # Initialize engine with ROOT context (will trigger flattening)
            # Pass lines explicitly!
            _t2 = time.time()
            if not self.engine.initialize_execution(root_blocks, root_lines):
                self.execution_failed(self.engine.error_msg)
                return False
            logger.debug(f"[TIMING] engine.initialize_execution: {time.time() - _t2:.3f}s")

            # self.global_computed_list is synced from the engine via property;
            # the engine owns the per-block computation checklist.

            self.engine.reset_execution_data()
            self.execution_time_start = time.time()
            logger.debug("Execution initialization complete")
        except Exception as e:
            logger.exception("Error during execution initialization")
            self.error_msg = f"Error during execution initialization: {e}"
            return False

        logger.debug("*****EXECUTION START*****")
        _t3 = time.time()

        # Initialization of the progress bar
        self.pbar = tqdm(
            desc="SIMULATION PROGRESS", total=int(self.execution_time / self.sim_dt), unit="itr"
        )
        # The Scope data about to be produced belongs to *this* diagram, so
        # plot_again may use it until the next edit. (This is what the old
        # ``self.dirty = False`` here was really for -- it also wiped the
        # unsaved-changes flag, which is why pressing Run lost work.)
        self.diagram_changed_since_run = False

        # Identify memory blocks to correctly solve algebraic loops (delegated to engine)
        self.engine.identify_memory_blocks()

        # Check for integrators using Runge-Kutta 45 and initialize counter
        self.rk45_len = self.engine.count_rk45_integrators()
        self.rk_counter = 0

        # Auto-connect Goto/From tags before execution starts
        try:
            self.model.link_goto_from()
            # refresh references
            self.blocks_list = self.model.blocks_list
            self.line_list = self.model.line_list
            self.connections_list = self.line_list  # Keep alias in sync
        except Exception as e:
            logger.warning(f"Goto/From linking failed: {e}")

        # Validate signal dimensions between connected blocks
        self._validate_signal_dimensions()
        logger.debug(f"[TIMING] post-init checks: {time.time() - _t3:.3f}s")

        # NOTE: The earlier engine.initialize_execution() call in this method already
        # initialized the engine with root_blocks/root_lines; a second call here would
        # only double the initialization overhead, so it is intentionally omitted.

        # max_hier / rk45_len / rk_counter / execution_time_start /
        # execution_initialized are property bridges onto the engine, which
        # owns them — no re-copy needed here.
        self.rk_counter += 1

        # The dynamic plot function is initialized, if the Boolean is active
        _t4 = time.time()
        self.dynamic_pyqtPlotScope(step=0)
        logger.debug(f"[TIMING] dynamic_pyqtPlotScope: {time.time() - _t4:.3f}s")
        logger.debug(f"[TIMING] execution_init TOTAL: {time.time() - _t0:.3f}s")

        return True

    def execution_batch(self, progress_cb=None, cancel_cb=None, defer_plots=False) -> None:
        """Run the entire simulation as fast as possible.

        Args:
            progress_cb: optional callable(t_now, t_end), called once per
                interpreter step (the compiled solver has no step hook).
            cancel_cb: optional callable() -> bool, polled once per interpreter
                step; when it returns True the run stops early and the blocks
                are re-armed exactly as a completed run leaves them.
            defer_plots: when True, the interpreter path skips every Qt call
                (dynamic scope updates and the end-of-run ``pyqtPlotScope``) so
                the batch can be driven from a worker thread; the caller is then
                responsible for plotting on the GUI thread (``plot_again``).
        """
        _tb0 = time.time()
        # Drop any diagnostics from a previous run so the interpreter path (which
        # records none) never surfaces a stale compiled-solver summary.
        self.engine.last_solver_diagnostics = {}
        # FAST SOLVER CHECK
        # Check if fast solver is enabled (default True if attr missing)
        use_fast = getattr(self, "use_fast_solver", True)

        _tb1 = time.time()
        compilable = self.engine.check_compilability(self.blocks_list) if use_fast else False
        logger.debug(f"[TIMING] check_compilability: {time.time() - _tb1:.3f}s")

        if use_fast and compilable:
            logger.info("System is compilable. Using Fast Solver.")
            self.last_solver_type = "Fast (Compiled)"
            t_span = (0.0, self.execution_time)
            _tb2 = time.time()
            success = self.engine.run_compiled_simulation(
                self.blocks_list, self.line_list, t_span, self.sim_dt
            )
            logger.debug(f"[TIMING] run_compiled_simulation: {time.time() - _tb2:.3f}s")
            if success:
                logger.info("Fast simulation successful.")

                # Sync timeline from engine (Required for plotting)
                self.timeline = self.engine.timeline

                # Finalize execution state
                self.execution_initialized = False

                # Update progress bar to 100%
                if hasattr(self, "pbar") and self.pbar:
                    self.pbar.n = self.pbar.total
                    self.pbar.last_print_n = self.pbar.total
                    self.pbar.refresh()
                    self.pbar.close()

                # Perform post-simulation tasks normally handled by loop
                self.export_data()
                try:
                    self._record_run_history()
                except Exception as e:
                    logger.warning(f"Failed to record run history: {e}")

                return

        logger.info("System not fully compilable. Using Interpreter Mode.")
        self.last_solver_type = "Standard (Interpreter)"
        self._defer_gui_plots = bool(defer_plots)
        try:
            while self.execution_initialized:
                self.execution_loop()
                if progress_cb is not None:
                    progress_cb(self.time_step, self.execution_time)
                if cancel_cb is not None and cancel_cb():
                    logger.info("Batch simulation cancelled at t=%.6g", self.time_step)
                    self.timeline = np.array(self._timeline_list)
                    self.execution_initialized = False
                    if getattr(self, "pbar", None) is not None:
                        self.pbar.close()
                    self.engine.reset_memblocks()
                    break
        finally:
            self._defer_gui_plots = False

    def run_tuning_simulation(self, sim_time, sim_dt, cancel_cb=None, cancel_every=200):
        """
        Headless re-simulation for live parameter tuning.

        Runs a full simulation without UI dialogs, progress bars, or plot calls.
        The caller reads scope data directly from blocks after this returns.

        Args:
            sim_time: Total simulation time in seconds
            sim_dt: Simulation time step in seconds
            cancel_cb: optional callable() -> bool.  Polled every
                ``cancel_every`` interpreter steps (and once before the
                compiled solver, which cannot be interrupted mid-solve); when it
                returns True the run aborts and ``(False, "cancelled")`` is
                returned.  Without this a cancelled ensemble/sweep still had to
                wait out a whole run.
            cancel_every: how many interpreter steps between cancel polls.

        Returns:
            (success: bool, error_msg: str)
        """
        try:
            # Reset execution state
            self.execution_stop = False
            self.error_msg = ""
            self.time_step = 0
            self._timeline_list = [self.time_step]
            self.timeline = np.array([self.time_step])
            self.sim_time = sim_time
            self.sim_dt = sim_dt
            # Mirror execution_init: keep a single duration attribute (execution_time)
            # used consistently by both the interactive and headless loops.
            self.execution_time = sim_time

            # Resolve parameters for all blocks (handles External reload and
            # Subsystem recursion identically to execution_init)
            workspace_manager = WorkspaceManager()
            root_blocks, root_lines = self.get_root_context()

            if not self._resolve_block_params(root_blocks, workspace_manager, sim_dt):
                return (False, self.error_msg or "Parameter resolution failed")

            # Sync sim params to the engine before init. Without this the engine
            # keeps its default sim_dt (0.01) and re-stamps every block's
            # exec_params['dtime'] with it during initialize_execution, so
            # interpreter state blocks (TransferFunction, PID, Integrator)
            # discretize/integrate at 0.01 regardless of the requested sim_dt.
            self.engine.update_sim_params(
                sim_time,
                sim_dt,
                solver_method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
                zero_crossing=self.zero_crossing,
            )

            # Initialize engine
            if not self.engine.initialize_execution(root_blocks, root_lines):
                return (False, self.engine.error_msg or "Engine init failed")

            self.engine.reset_execution_data()
            self.engine.identify_memory_blocks()
            self.rk45_len = self.engine.count_rk45_integrators()
            self.rk_counter = 0

            # Auto-connect Goto/From tags
            try:
                self.model.link_goto_from()
                self.blocks_list = self.model.blocks_list
                self.line_list = self.model.line_list
                self.connections_list = self.line_list
            except Exception:
                logger.debug("Auto-connecting Goto/From tags failed", exc_info=True)

            self.execution_initialized = True
            self.diagram_changed_since_run = False
            self.rk_counter += 1

            # Run batch (compiled or interpreter)
            use_fast = getattr(self, "use_fast_solver", True)
            compilable = self.engine.check_compilability(self.blocks_list) if use_fast else False

            if cancel_cb is not None and cancel_cb():
                self.execution_initialized = False
                self.engine.reset_memblocks()
                return (False, "cancelled")

            if use_fast and compilable:
                t_span = (0.0, sim_time)
                success = self.engine.run_compiled_simulation(
                    self.blocks_list, self.line_list, t_span, sim_dt
                )
                if success:
                    self.timeline = self.engine.timeline
                    self.execution_initialized = False
                    return (True, "")
                # Fall through to interpreter if compiled fails

            # Interpreter mode
            steps = 0
            while self.execution_initialized:
                self.execution_loop_headless()
                steps += 1
                if cancel_cb is not None and steps % max(1, int(cancel_every)) == 0 and cancel_cb():
                    self.execution_initialized = False
                    self.engine.reset_memblocks()
                    return (False, "cancelled")

            if self.error_msg:
                return (False, self.error_msg)
            return (True, "")

        except Exception as e:
            logger.error(f"Tuning simulation error: {e}")
            self.execution_initialized = False
            return (False, str(e))

    def execution_loop_headless(self):
        """Stripped-down execution loop for tuning (no progress bar, no plots)."""
        self._interpreter_step(interactive=False)

    def _interpreter_step(self, interactive):
        """
        Advance the interpreter simulation by exactly one time step.

        Shared core of execution_loop (interactive GUI path) and
        execution_loop_headless (tuning / CLI / ensemble path). The numerical
        stepping is identical on both paths; `interactive` only gates the UI
        side effects: progress bar updates, live plotting, and the end-of-run
        export / run-history / scope-plot sequence.

        The step runs as a fixed sequence of phases, each a method below:
        ``_advance_clock`` (time / RK4 sub-step bookkeeping),
        ``_publish_memory_outputs`` (memory blocks report their pre-update
        outputs), ``_run_hierarchy_passes`` (every block executes once, in
        hierarchy order, via ``_execute_ready_block``), then
        ``_is_end_of_run`` / ``_finish_run``.  A block failure stops the run
        through ``execution_failed`` and returns before ``rk_counter`` advances.
        """
        try:
            if self.execution_pause:
                return

            # The run ends by clearing execution_initialized (see _is_end_of_run),
            # at which point reset_memblocks() has already re-armed every block.
            # Re-executing would append samples past the horizon computed from
            # re-initialised state, so stop doing work — but keep advancing the
            # clock, so a caller driving the loop on a time comparison rather
            # than on execution_initialized still terminates instead of
            # spinning forever.
            if not self.execution_initialized:
                self.time_step += self.sim_dt
                return

            self.engine.reset_execution_data()
            sample_recorded = self._advance_clock(interactive)

            # Use the active list from engine (flattened if needed); fall back
            # to the local list if the engine is not ready (though it should be).
            current_blocks = self.engine.active_blocks_list or self.blocks_list

            pre_update_outputs = self._publish_memory_outputs(current_blocks)
            if pre_update_outputs is None:
                return
            if not self._run_hierarchy_passes(current_blocks, pre_update_outputs):
                return

            if interactive and not self._defer_gui_plots:
                # The dynamic plot function is called to save the new data, if active
                self.dynamic_pyqtPlotScope(step=1)

            if self._is_end_of_run(sample_recorded):
                self._finish_run(interactive)

            self.rk_counter += 1

        except Exception as e:
            import traceback

            logger.error(f"Error during execution loop: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.execution_failed(f"Error during execution loop: {e}")

    def _advance_clock(self, interactive) -> bool:
        """Advance ``time_step`` for this call.

        Returns True when this call advanced a whole step and recorded a
        timeline sample — the only point at which ``_is_end_of_run`` may decide
        the run is complete.

        RK4 evaluates its four stages at t, t+h/2, t+h/2, t+h and completes the
        state update on the last one, so a whole cycle must advance the clock by
        exactly one sim_dt: +0, +h/2, +0, +h/2.  Sub-step 0 is the *start* of a
        step and must not advance time — it publishes the state the previous
        cycle finished computing, which is the sample for this grid point.  It
        used to add a full sim_dt here, so each four-call cycle advanced
        2*sim_dt while the integrator advanced one h, and every RK45 trace came
        out stretched 2x in time (the integral of a unit step reached only 0.5
        at t=1.0).
        """
        if self.rk45_len:
            self.rk_counter %= 4
            if self.rk_counter in [1, 3]:
                self.time_step += self.sim_dt / 2
                return False
            if self.rk_counter != 0:
                return False
        else:
            self.time_step += self.sim_dt
        if interactive:
            self.pbar.update(1)
        self._timeline_list.append(self.time_step)
        return True

    def _block_failed(self, out_value) -> bool:
        """Stop the run and return True when a block reported an error.

        Anything that is not a dict (``None``, or the ``False`` the engine
        returns when a block cannot be run at all) is a failure too.
        """
        if not isinstance(out_value, dict):
            self.execution_failed(
                "Block returned None" if out_value is None else f"Block returned {out_value!r}"
            )
            return True
        if out_value.get("E"):
            self.execution_failed(out_value.get("error", "Unknown error"))
            return True
        return False

    def _propagate_held_outputs(self, block):
        """Deliver a discrete block's held (last-sampled) outputs to its consumers."""
        held_outputs = {p: block.get_held_output(p) for p in range(block.out_ports)}
        self.engine.propagate_outputs(block, held_outputs)

    def _publish_memory_outputs(self, current_blocks):
        """First pass: memory blocks report their outputs *before* this step's state update.

        Returns the outputs of discrete-rate memory blocks keyed by block name
        (needed by ``_run_hierarchy_passes`` to hold the right value for blocks
        whose execute() returns the advanced state, see
        SimulationEngine.stamp_held_outputs), or None when a block failed and
        the run has been stopped.
        """
        pre_update_outputs = {}
        for block in current_blocks:
            try:
                if self.rk45_len:
                    # Sinks (Scope / Export) must record one sample per RK4
                    # cycle, not one per stage evaluation.  The flag has to be
                    # written to exec_params, which is the dict execute()
                    # actually receives; block.params is kept in sync because
                    # a cache-miss re-resolve re-copies '_'-prefixed keys from
                    # it (SimulationEngine._resolve_block_params).  Writing it
                    # to params alone left it invisible to the blocks —
                    # exec_params is normally served from cache — so every
                    # sub-step was recorded: 21 scope samples against a
                    # 6-entry timeline.
                    skip = self.rk_counter != 0
                    block.params["_skip_"] = skip
                    if getattr(block, "exec_params", None) is not None:
                        block.exec_params["_skip_"] = skip

                if block.name not in self.memory_blocks:
                    continue

                # Multi-rate: a discrete block off its sample instant only
                # propagates its held outputs.
                if not block.should_execute(self.time_step):
                    if block.b_type != 3:
                        self._propagate_held_outputs(block)
                    continue

                out_value = self.engine.execute_block(block, output_only=True)
                if self._block_failed(out_value):
                    return None

                if block.effective_sample_time > 0:
                    pre_update_outputs[block.name] = out_value

                # Propagate outputs to children. set_held_output and
                # schedule_next_execution are NOT called here — the hierarchy
                # pass runs the state-updating execute and handles them.
                # Calling them here would advance the next-sample time before
                # the state ever updates, leaving downstream consumers stuck
                # on the initial state.
                self.engine.propagate_outputs(block, out_value)

            except Exception as e:
                logger.error(f"Error executing block {block.name}: {str(e)}")
                self.execution_failed(f"Error executing block {block.name}: {e}")
                return None
        return pre_update_outputs

    @staticmethod
    def _has_enough_inputs(block) -> bool:
        """True when every required (non-optional) input port has received data."""
        optional_inputs = set()
        instance = getattr(block, "block_instance", None)
        if instance and hasattr(instance, "optional_inputs"):
            optional_inputs = set(instance.optional_inputs)
        required_ports = block.in_ports - len(optional_inputs)
        return block.data_received >= required_ports or block.in_ports == 0

    def _run_hierarchy_passes(self, current_blocks, pre_update_outputs) -> bool:
        """Execute every block once, in the hierarchy order fixed at init.

        Two layers of re-iteration are required:

         - Inner (within-level): two blocks at the same hierarchy can be ordered
           such that A's output is B's input but B precedes A in current_blocks.
           A single pass would skip B (no inputs yet) then never revisit it
           within this level.
         - Outer (cross-level): memory blocks (Integrator, StateSpace, strictly
           proper TF…) are force-pinned to hierarchy=0 by init Loop 1's memory
           branch, but their state-update execute consumes inputs produced at
           hierarchy>0. Without an outer re-pass, the for-hier loop completes
           hier=0 with the memory block still uncomputed, fires its producer at
           hier=N, then never returns to hier=0 — so the state-update never runs
           and the state freezes. Mirroring init Loop 2's
           `while not check_global_list` pattern keeps the loop bounded (each
           block fires at most once per timestep).

        Returns False when a block failed and the run has been stopped.
        """
        while True:
            outer_progressed = False
            for hier in range(self.max_hier + 1):
                while True:
                    progressed = False
                    for block in current_blocks:
                        if (
                            block.hierarchy != hier
                            or block.computed_data
                            or not self._has_enough_inputs(block)
                        ):
                            continue
                        progressed = True
                        outer_progressed = True
                        if not self._execute_ready_block(block, pre_update_outputs):
                            return False
                    if not progressed:
                        break
            if not outer_progressed:
                return True

    def _execute_ready_block(self, block, pre_update_outputs) -> bool:
        """Run one block whose inputs are complete and propagate its outputs.

        Returns False when the block failed and the run has been stopped.
        """
        is_memory = block.name in self.memory_blocks

        # Multi-rate: a discrete block off its sample instant is marked
        # computed and only propagates its held outputs.
        if not block.should_execute(self.time_step):
            self.engine.update_global_list(block.name, h_value=0)
            block.computed_data = True
            if not is_memory and block.b_type != 3:
                self._propagate_held_outputs(block)
            return True

        # Execute using engine (handles external vs internal, kwargs building)
        out_value = self.engine.execute_block(block)

        # After execution, for memory blocks, update the 'output' state for the next step
        if is_memory:
            self.engine.sync_integrator_output(block)

        if self._block_failed(out_value):
            return False

        # Multi-rate: Store outputs and schedule next execution for discrete blocks
        if block.effective_sample_time > 0:
            self.engine.stamp_held_outputs(block, out_value, pre_update_outputs.get(block.name))
            block.schedule_next_execution(self.time_step)

        # A memory block with direct feedthrough — a ZOH at a sample instant
        # outputs the value it just sampled — had only its *stale* held value
        # delivered, by the first pass's output_only call; the propagation
        # below skips memory blocks entirely, so the fresh sample did not reach
        # consumers until the next step and the staircase edges lagged by one
        # solver step.  Refresh the already-counted input queues in place,
        # before consumers at later hierarchy levels run.  Strictly-proper
        # memory blocks (b_type 1) are untouched: their output really does
        # depend only on past inputs, so the first pass's value is the correct
        # one.
        if is_memory and block.b_type == 2:
            self.engine.propagate_outputs(block, out_value, count=False)

        # The computed_data booleans are updated in the global list as well as in the block itself
        self.engine.update_global_list(block.name, h_value=0)
        block.computed_data = True

        # Propagate outputs to children (skip memory blocks and sinks)
        if not is_memory and block.b_type != 3:
            self.engine.propagate_outputs(block, out_value)
        return True

    def _is_end_of_run(self, sample_recorded) -> bool:
        """True once the last sample that fits inside the horizon has been produced.

        Stop then, rather than after computing one step past it: the old
        `time_step > execution_time` test could only fire on a step already
        beyond the end, so a 3.0 s run at dt=0.01 emitted 302 samples ending at
        t=3.01.  This matches the compiled solver's grid (arange(0, T+dt, dt)
        clipped to <= T), so both solvers return the same timeline.  The
        tolerance absorbs the drift of accumulating sim_dt.

        Only a call that recorded a sample may end the run: RK45 sub-steps land
        between grid points and accumulate the drift of adding sim_dt/2, so the
        last one of a cycle can test as just past the horizon and terminate
        before the cycle's sample is recorded, dropping the final grid point.
        What remains of the old test is a runaway guard, with a full sim_dt of
        margin so it cannot fire mid-cycle.
        """
        if sample_recorded:
            next_step = self.time_step + self.sim_dt
            return next_step > self.execution_time + self.sim_dt * 1e-6
        return self.time_step > self.execution_time + self.sim_dt

    def _finish_run(self, interactive):
        """End-of-run sequence: freeze the timeline, then (interactive only)
        export, record run history and plot; finally re-arm the blocks."""
        self.timeline = np.array(self._timeline_list)  # Convert list to numpy array
        self.execution_initialized = False  # The execution loop is terminated
        if interactive:
            self.pbar.close()  # The progress bar ends

            # Export
            self.export_data()

            # Record run history for inspector
            try:
                self._record_run_history()
            except Exception as e:
                logger.warning(f"Could not record run history: {e}")

            # Scope. Skipped when the batch is being driven from a worker
            # thread: Qt windows may only be built on the GUI thread, so the
            # caller plots via plot_again() instead.
            if not self.dynamic_plot and not self._defer_gui_plots:
                logger.debug("Calling pyqtPlotScope...")
                self.pyqtPlotScope()
                logger.debug("pyqtPlotScope call finished.")

        # Resets the initialization of the blocks with special initial executions
        self.engine.reset_memblocks()
        if interactive:
            logger.debug("*****EXECUTION DONE*****")

    def single_step(self) -> bool:
        """
        Execute exactly one timestep of the simulation.
        Used for step-by-step debugging when simulation is paused.

        If simulation is not initialized, it will be initialized first
        (starting from t=0) in paused state.

        Returns:
            bool: True if step was executed, False on error
        """
        try:
            # If not initialized, initialize first (allows stepping from start)
            if not self.execution_initialized:
                logger.info("Initializing simulation for step-by-step mode...")
                success = self.execution_init()
                if not success:
                    logger.error("Failed to initialize simulation for stepping")
                    return False
                # Start paused
                self.execution_pause = True
                logger.info("Simulation initialized at t=0, ready to step")

            # Temporarily unpause
            self.execution_pause = False

            # Execute one step
            self.execution_loop()

            # Re-pause (single-step always pauses after)
            self.execution_pause = True

            logger.debug(f"Single step executed: t={self.time_step:.4f}s")
            return True

        except Exception as e:
            logger.error(f"Error during single step: {str(e)}")
            self.execution_pause = True
            return False

    def execution_loop(self):
        """
        :purpose: Continues with the execution sequence in loop until time runs out or an special event stops it.
        :description: This is the second stage of the network simulation. Here the reading of the complete graph will be done cyclically until the time is up, the user indicates that it is finished (by pressing Stop) or simply until one of the blocks gives error. At the end, the data saved in blocks like 'Scope' and 'External_data', will be exported to other libraries to perform their functions.
        """
        self._interpreter_step(interactive=True)

    def execution_failed(self, msg=""):
        """
        :purpose: If an error is found while executing the graph, this function stops all the processes and resets values to the state before execution.
        """
        self.execution_initialized = False  # Finishes the simulation execution
        self.engine.reset_memblocks()  # Restores the initialization of the integrators (in case the error was due to vectors of different dimensions).
        if hasattr(self, "pbar"):
            self.pbar.close()  # Finishes the progress bar
        self.error_msg = msg
        logger.error("*****EXECUTION STOPPED*****")

    def check_diagram_integrity(self):
        """Check diagram integrity. Delegates to SimulationEngine."""
        return self.engine.check_diagram_integrity()

    def _validate_signal_dimensions(self):
        """
        Validate signal dimensions between connected blocks.
        Logs warnings for potential dimension mismatches (non-fatal).
        """
        warnings = []

        for line in self.line_list:
            if line.hidden:
                continue

            # Find source and destination blocks
            src_block = next((b for b in self.blocks_list if b.name == line.srcblock), None)
            dst_block = next((b for b in self.blocks_list if b.name == line.dstblock), None)

            if not src_block or not dst_block:
                continue

            # Get expected output width from source block
            src_width = None
            if hasattr(src_block, "block_instance") and src_block.block_instance:
                outputs = getattr(src_block.block_instance, "outputs", [])
                if line.srcport < len(outputs):
                    port_def = outputs[line.srcport]
                    if isinstance(port_def, dict):
                        src_width = port_def.get("width", None)

            # Get expected input width from destination block
            dst_width = None
            if hasattr(dst_block, "block_instance") and dst_block.block_instance:
                try:
                    inputs = (
                        dst_block.block_instance.get_inputs(dst_block.params)
                        if hasattr(dst_block.block_instance, "get_inputs")
                        else getattr(dst_block.block_instance, "inputs", [])
                    )
                    if line.dstport < len(inputs):
                        port_def = inputs[line.dstport]
                        if isinstance(port_def, dict):
                            dst_width = port_def.get("width", None)
                except Exception:
                    logger.debug(
                        "Could not determine destination port width for dimension check",
                        exc_info=True,
                    )

            # Check for dimension mismatch (only if both specify a width)
            if src_width is not None and dst_width is not None:
                if src_width != dst_width and src_width != -1 and dst_width != -1:
                    warnings.append(
                        f"Dimension mismatch: {src_block.name}[{line.srcport}] → "
                        f"{dst_block.name}[{line.dstport}] (width {src_width} → {dst_width})"
                    )

        for warning in warnings:
            logger.warning(f"Signal dimension: {warning}")

        if warnings:
            logger.info(
                f"Signal dimension validation: {len(warnings)} potential mismatch(es) detected"
            )
        else:
            logger.debug("Signal dimension validation: No mismatches detected")

    def create_subsystem_from_selection(self, selected_blocks=None):
        """
        Create a subsystem from selected blocks.
        Moves selected blocks into a new Subsystem block and maintains connections.
        """
        if selected_blocks is None:
            selected_blocks = [b for b in self.blocks_list if b.selected]
        return self.subsystem_manager.create_subsystem_from_selection(selected_blocks)

    def export_data(self):
        """
        :purpose: Exports the data saved in Export blocks to the selected format.
        :description: This function is executed after the simulation has finished or stopped. It looks for export blocks, which have some vectors saved with signal outputs from previous blocks. Then it merges all vectors in one big matrix, which is exported with the time vector. The output format is selected by the Export block's ``format`` parameter:

            - ``npz`` (default): ``np.savez`` archive with one array per labelled signal plus ``t``.
            - ``csv``: ``np.savetxt`` with a header of ``t`` followed by the signal labels (one column per signal).
            - ``mat``: ``scipy.io.savemat`` with a dict of ``t`` plus one entry per labelled signal.
        """
        # Preserve column order so csv/mat columns mirror the order signals were
        # encountered. vec_dict maps label -> 1-D array of length len(timeline).
        vec_dict = {}
        export_toggle = False
        out_format = "npz"
        for block in self.blocks_list:
            if block.block_fn == "Export":
                export_toggle = True
                # Last Export block's format wins (diagrams typically have one).
                out_format = str(block.params.get("format", "npz")).lower()
                labels = block.params["vec_labels"]
                vector = block.params["vector"]
                if block.params["vec_dim"] == 1:
                    vec_dict[labels] = vector
                elif block.params["vec_dim"] > 1:
                    for i in range(block.params["vec_dim"]):
                        vec_dict[labels[i]] = vector[:, i]
        if not export_toggle:
            return

        if out_format not in ("npz", "csv", "mat"):
            logger.warning("Unknown export format '%s'; falling back to 'npz'.", out_format)
            out_format = "npz"

        # Derive basename without assuming a fixed-length extension
        basename = os.path.splitext(self.filename)[0]
        export_path = os.path.join("saves", basename)
        # In frozen mode, redirect saves/ to a writable location (mirrors FileService)
        if getattr(sys, "frozen", False) and not os.path.isabs(export_path):
            from lib.app_paths import get_user_data_dir

            export_path = os.path.join(get_user_data_dir(), export_path)
        # Ensure the target directory exists before writing
        os.makedirs(os.path.dirname(export_path) or ".", exist_ok=True)

        timeline = np.asarray(self.timeline)

        if out_format == "npz":
            np.savez(export_path, t=timeline, **vec_dict)
            out_file = export_path + ".npz"
        elif out_format == "csv":
            labels = list(vec_dict.keys())
            # Build (N, 1 + n_signals) matrix: time column first, then signals.
            columns = [timeline.reshape(-1)]
            for label in labels:
                columns.append(np.asarray(vec_dict[label]).reshape(-1))
            data = np.column_stack(columns)
            header = ",".join(["t"] + [str(label) for label in labels])
            out_file = export_path + ".csv"
            np.savetxt(out_file, data, delimiter=",", header=header, comments="")
        else:  # 'mat'
            from scipy.io import savemat

            mat_dict = {"t": timeline.reshape(-1)}
            for label, value in vec_dict.items():
                mat_dict[str(label)] = np.asarray(value).reshape(-1)
            out_file = export_path + ".mat"
            savemat(out_file, mat_dict)

        logger.info("DATA EXPORTED TO " + out_file)

    def run_optimization(self, callback=None):
        """
        Run parameter optimization on the current diagram.

        This method looks for Parameter, CostFunction, Constraint, and Optimizer
        blocks in the diagram and uses scipy.optimize to find optimal parameter
        values that minimize the cost function(s).

        Workflow:
        1. Find all optimization-related blocks in the diagram
        2. Extract tunable parameters from Parameter blocks
        3. Create objective function that runs simulation and returns cost
        4. Call scipy.optimize with the configured method
        5. Write optimal parameters back to Parameter blocks

        Args:
            callback: Optional callback function called after each evaluation
                      with signature callback(n_eval, cost, params_dict)

        Returns:
            Dict with optimization results:
                - success: bool, whether optimization converged
                - optimal_cost: float, final cost value
                - optimal_params: dict, parameter name -> optimal value
                - n_evaluations: int, number of function evaluations
                - history: list of dicts with evaluation history
                - message: str, optimizer message

        Example:
            # Place Parameter, CostFunction, and Optimizer blocks in diagram
            result = dsim.run_optimization()
            if result['success']:
                print(f"Optimal cost: {result['optimal_cost']}")
                print(f"Optimal parameters: {result['optimal_params']}")
        """
        from lib.engine.optimization_engine import OptimizationEngine

        logger.info("Starting optimization...")

        # Create optimization engine
        opt_engine = OptimizationEngine(dsim=self)

        # Get root context for optimization
        root_blocks, root_lines = self.get_root_context()

        # Run optimization
        result = opt_engine.run_optimization(blocks=root_blocks)

        if result.get("success"):
            logger.info("Optimization completed successfully!")
            logger.info(f"Optimal cost: {result.get('optimal_cost')}")
            logger.info(f"Optimal parameters: {result.get('optimal_params')}")
        else:
            logger.warning(f"Optimization did not converge: {result.get('message')}")

        return result

    def get_symbolic_equations(self, input_blocks=None, output_blocks=None):
        """
        Extract symbolic equations from the block diagram.

        Uses the SymbolicEngine to trace signal flow through the diagram
        and compose symbolic expressions (using SymPy).

        Args:
            input_blocks: List of block names to treat as inputs (auto-detected if None)
            output_blocks: List of block names to get equations for (all if None)

        Returns:
            Dict with:
                - equations: dict of block_name -> symbolic expression
                - transfer_functions: dict of (from, to) -> G(s) if computed
                - latex: dict of block_name -> LaTeX string

        Example:
            result = dsim.get_symbolic_equations()
            for name, eq in result['equations'].items():
                print(f"{name}: {eq}")
        """
        try:
            from lib.engine.symbolic_engine import SymbolicEngine
        except ImportError:
            logger.error("SymPy is required for symbolic features. Install with: pip install sympy")
            return None

        logger.info("Extracting symbolic equations...")

        # Create symbolic engine
        sym_engine = SymbolicEngine(dsim=self)

        # Get root context
        root_blocks, root_lines = self.get_root_context()

        # Build graph
        sym_engine.build_graph(blocks=root_blocks, lines=root_lines)

        # Create input symbols
        input_symbols = sym_engine.create_input_symbols(input_blocks)

        # Get all equations
        equations = sym_engine.get_all_equations(input_symbols)

        # Convert to LaTeX
        latex_eqs = {}
        for name, expr in equations.items():
            if expr is not None:
                try:
                    latex_eqs[name] = sym_engine.to_latex(expr)
                except Exception:
                    latex_eqs[name] = str(expr)

        return {
            "equations": equations,
            "latex": latex_eqs,
            "input_symbols": input_symbols,
        }

    def extract_transfer_function(self, from_block, to_block):
        """
        Extract transfer function G(s) between two signals.

        Args:
            from_block: Input block name
            to_block: Output block name

        Returns:
            SymPy expression for G(s) = Y(s)/U(s), or None on error
        """
        try:
            from lib.engine.symbolic_engine import SymbolicEngine
        except ImportError:
            logger.error("SymPy is required for symbolic features.")
            return None

        # Create symbolic engine
        sym_engine = SymbolicEngine(dsim=self)

        # Get root context
        root_blocks, root_lines = self.get_root_context()

        # Build graph
        sym_engine.build_graph(blocks=root_blocks, lines=root_lines)

        # Extract transfer function
        G = sym_engine.extract_transfer_function(from_block, to_block)

        return G

    def linearize(self, operating_point=None, input_blocks=None, output_blocks=None):
        """
        Linearize the system at an operating point.

        Computes state-space matrices (A, B, C, D) using numerical Jacobians.

        Args:
            operating_point: Dict of block_name -> value (uses current state if None)
            input_blocks: List of input block names
            output_blocks: List of output block names

        Returns:
            Dict with:
                - A, B, C, D: State-space matrices
                - n_states: Number of states
                - state_names: List of state variable names
                - eigenvalues: System eigenvalues
                - is_stable: Whether system is stable
                - is_controllable: Whether system is controllable
                - is_observable: Whether system is observable
        """
        from lib.analysis.linearizer import Linearizer

        logger.info("Linearizing system...")

        # Create linearizer
        linearizer = Linearizer(dsim=self)

        # Get root context
        root_blocks, root_lines = self.get_root_context()

        # Linearize
        result = linearizer.linearize_at_point(
            operating_point=operating_point, input_blocks=input_blocks, output_blocks=output_blocks
        )

        if result is not None:
            # Add controllability/observability
            A, B, C = result["A"], result["B"], result["C"]
            result["is_controllable"] = linearizer.is_controllable(A, B)
            result["is_observable"] = linearizer.is_observable(A, C)

            logger.info(f"System linearized: {result['n_states']} states")
            logger.info(f"Stable: {result['is_stable']}")
            logger.info(f"Controllable: {result['is_controllable']}")
            logger.info(f"Observable: {result['is_observable']}")

        return result

    def export_equations_latex(self, filename=None):
        """
        Export block diagram equations to LaTeX document.

        Args:
            filename: Output file path (returns string if None)

        Returns:
            LaTeX document string
        """
        try:
            from lib.engine.symbolic_engine import SymbolicEngine
        except ImportError:
            logger.error("SymPy is required for symbolic features.")
            return None

        # Get equations first
        result = self.get_symbolic_equations()
        if result is None:
            return None

        # Create symbolic engine for export
        sym_engine = SymbolicEngine(dsim=self)
        root_blocks, root_lines = self.get_root_context()
        sym_engine.build_graph(blocks=root_blocks, lines=root_lines)

        # Export to LaTeX
        latex_doc = sym_engine.export_equations_latex(
            equations=result["equations"], filename=filename
        )

        if filename:
            logger.info(f"Equations exported to {filename}")

        return latex_doc

    def _record_run_history(self):
        """Record the current run data for future inspection."""
        timeline, traces = self.scope_plotter.get_scope_traces()
        if timeline is None or len(timeline) == 0:
            return

        self.run_history_service.record_run(
            timeline=timeline, traces=traces, sim_dt=self.sim_dt, sim_time=self.sim_time
        )

    @property
    def run_history(self):
        """Delegated property for backward compatibility."""
        return self.run_history_service.history

    @run_history.setter
    def run_history(self, value):
        self.run_history_service.history = value

    def save_run_history(self):
        """Persist run history to disk if enabled."""
        self.run_history_service.save_history()

    def set_run_history_persist(self, enabled: bool):
        """Toggle persistence of waveform run history."""
        self.run_history_service.set_persist(enabled)

    # Pyqtgraph functions
    def pyqtPlotScope(self):
        """
        :purpose: Plots the data saved in Scope blocks using pyqtgraph.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.pyqtPlotScope()

    def plot_again(self):
        """
        :purpose: Re-plots the scope data and handling other plots.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.plot_again()

    def dynamic_pyqtPlotScope(self, step):
        """
        :purpose: Plots the data saved in Scope blocks dynamically with pyqtgraph.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.dynamic_pyqtPlotScope(step)
