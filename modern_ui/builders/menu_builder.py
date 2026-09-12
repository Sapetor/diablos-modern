import os
import logging

from PyQt6.QtGui import QAction, QActionGroup

from lib.i18n import tr

logger = logging.getLogger(__name__)


class MenuBuilder:
    """Builder for MainWindow menus."""

    def __init__(self, main_window):
        self.window = main_window

    def setup_menubar(self):
        """Setup the entire menu bar."""
        menubar = self.window.menuBar()
        menubar.clear()

        self._create_file_menu(menubar)
        self._create_edit_menu(menubar)
        self._create_simulation_menu(menubar)
        self._create_analysis_menu(menubar)
        self._create_view_menu(menubar)
        self._create_help_menu(menubar)

    def _create_analysis_menu(self, menubar):
        """Create Analysis menu (linearization-based system analysis)."""
        analysis_menu = menubar.addMenu(tr("&Analysis"))
        # "&&" renders a literal "&": label shows "Linearize & Analyze...".
        analysis_menu.addAction(tr("&Linearize && Analyze..."), self.window.linearize_and_analyze)
        analysis_menu.addAction(
            tr("&Find Operating Point (Trim)..."), self.window.find_operating_point
        )
        analysis_menu.addAction(tr("&Parameter Sweep..."), self.window.run_parameter_sweep)
        analysis_menu.addAction(tr("&Monte Carlo..."), self.window.run_monte_carlo)

    def _create_file_menu(self, menubar):
        """Create File menu."""
        file_menu = menubar.addMenu(tr("&File"))

        # Standard actions
        file_menu.addAction(tr("&New") + "\tCtrl+N", self.window.new_diagram)
        file_menu.addAction(tr("&Open") + "\tCtrl+O", self.window.open_diagram)
        file_menu.addAction(tr("&Save") + "\tCtrl+S", self.window.save_diagram)
        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu(tr("E&xport"))
        export_menu.addAction(tr("Export as &Image..."), self.window.export_image)
        export_menu.addAction(tr("Export as Ti&kZ..."), self.window.export_tikz)
        export_menu.addAction(tr("Export as &Python Script..."), self.window.export_python_script)

        file_menu.addSeparator()

        # Recent Files
        self.window.recent_files_menu = file_menu.addMenu(tr("Recent Files"))
        if hasattr(self.window, "_update_recent_files_menu"):
            self.window._update_recent_files_menu()

        # Examples
        examples_menu = file_menu.addMenu(tr("Examples"))
        self._populate_examples_menu(examples_menu)

        file_menu.addSeparator()
        exit_action = file_menu.addAction(tr("E&xit") + "\tAlt+F4", self.window.close)
        # Danger-color via dynamic property; QSS picks it up via [role="danger"]
        exit_action.setProperty("role", "danger")

    def _populate_examples_menu(self, menu):
        """Populate examples submenu. Filenames shown without extension."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        examples_dir = os.path.join(base_dir, "examples")

        if os.path.exists(examples_dir):
            try:
                files = sorted(
                    f for f in os.listdir(examples_dir) if f.endswith((".json", ".dat", ".diablos"))
                )
            except OSError as exc:
                logger.warning("Could not read examples directory %s: %s", examples_dir, exc)
                menu.addAction(tr("Examples directory not readable")).setEnabled(False)
                return
            if not files:
                menu.addAction(tr("No examples found")).setEnabled(False)
                return
            for f in files:
                display = os.path.splitext(f)[0].replace("_", " ")
                action = menu.addAction(display)
                action.triggered.connect(
                    lambda checked, fname=f: self.window.open_example(
                        os.path.join(examples_dir, fname)
                    )
                )
        else:
            menu.addAction(tr("Examples directory not found")).setEnabled(False)

    def _create_edit_menu(self, menubar):
        """Create Edit menu."""
        edit_menu = menubar.addMenu(tr("&Edit"))
        if hasattr(self.window, "undo_action"):
            edit_menu.addAction(tr("&Undo") + "\tCtrl+Z", self.window.undo_action)
        if hasattr(self.window, "redo_action"):
            edit_menu.addAction(tr("&Redo") + "\tCtrl+Y", self.window.redo_action)

        edit_menu.addSeparator()

        # Check if select_all is implemented, otherwise define it or skip
        if hasattr(self.window, "select_all"):
            edit_menu.addAction(tr("Select &All") + "\tCtrl+A", self.window.select_all)
        elif hasattr(self.window, "canvas") and hasattr(self.window.canvas, "_select_all_blocks"):
            # Fallback if method missing in window
            edit_menu.addAction(
                tr("Select &All") + "\tCtrl+A", self.window.canvas._select_all_blocks
            )

        edit_menu.addAction(tr("Copy Diagram as &Image"), self.window.copy_diagram_image)

        edit_menu.addSeparator()

        # Create Subsystem
        if hasattr(self.window, "create_subsystem"):
            action = edit_menu.addAction(
                tr("Create &Subsystem") + "\tCtrl+G", self.window.create_subsystem
            )
            action.setShortcut("Ctrl+G")
        elif hasattr(self.window, "canvas") and hasattr(
            self.window.canvas, "_create_subsystem_trigger"
        ):
            action = edit_menu.addAction(
                tr("Create &Subsystem") + "\tCtrl+G", self.window.canvas._create_subsystem_trigger
            )
            action.setShortcut("Ctrl+G")

        # Masks & user library (see modern_ui/managers/mask_library_manager.py)
        if hasattr(self.window, "edit_block_mask"):
            edit_menu.addAction(tr("Edit &Mask..."), self.window.edit_block_mask)
            edit_menu.addAction(tr("&Look Under Mask"), self.window.look_under_mask)
            edit_menu.addAction(tr("Save as &Library Block..."), self.window.save_as_library_block)
            edit_menu.addAction(tr("Reload from Li&brary"), self.window.reload_from_library)
            edit_menu.addAction(tr("Refresh Block Librar&y"), self.window.refresh_block_library)

        # Custom block modules (see lib/user_blocks.py, docs/BLOCK_API.md)
        if hasattr(self.window, "reload_user_blocks"):
            # Lambda, not the bound method: QAction.triggered(bool) would bind
            # the checked flag to reload_user_blocks' `quiet` argument.
            edit_menu.addAction(tr("Reload &User Blocks"), lambda: self.window.reload_user_blocks())
            edit_menu.addAction(
                tr("Open User Blocks &Folder..."), self.window.open_user_blocks_folder
            )

        edit_menu.addSeparator()

        if hasattr(self.window, "show_command_palette"):
            edit_menu.addAction(
                tr("Command &Palette") + "\tCtrl+K", self.window.show_command_palette
            )

    def _create_simulation_menu(self, menubar):
        """Create Simulation menu."""
        sim_menu = menubar.addMenu(tr("&Simulation"))
        sim_menu.addAction(tr("&Run") + "\tF5", self.window.start_simulation)
        sim_menu.addAction(tr("&Pause") + "\tF6", self.window.pause_simulation)
        sim_menu.addAction(tr("&Stop") + "\tF7", self.window.stop_simulation)
        sim_menu.addSeparator()

        # Run no longer pops the settings dialog; this is the way in.
        if hasattr(self.window, "open_simulation_settings"):
            action = sim_menu.addAction(
                tr("Simulation Settin&gs...") + "\tCtrl+E", self.window.open_simulation_settings
            )
            action.setShortcut("Ctrl+E")
            self.window.simulation_settings_action = action

        sim_menu.addSeparator()

        # Fast Solver Toggle
        fast_solver = sim_menu.addAction(tr("Enable Fast Solver (Experimental)"))
        fast_solver.setCheckable(True)
        # Default to True, but check DSim state if possible (MainWindow usually holds this state)
        # We'll assume MainWindow has 'use_fast_solver' attribute initialized to True
        is_fast = getattr(self.window, "use_fast_solver", True)
        fast_solver.setChecked(is_fast)
        fast_solver.triggered.connect(self.window.toggle_fast_solver)
        self.window.fast_solver_action = fast_solver

        sim_menu.addSeparator()
        sim_menu.addAction(tr("Show &Plots"), self.window.show_plots)

    def _create_view_menu(self, menubar):
        """Create View menu."""
        view_menu = menubar.addMenu(tr("&View"))

        # Zoom controls
        # Delegate to window methods if they exist, or lambdas
        if hasattr(self.window, "zoom_in"):
            view_menu.addAction(tr("&Zoom In") + "\tCtrl++", self.window.zoom_in)
        else:
            view_menu.addAction(
                tr("&Zoom In") + "\tCtrl++",
                lambda: self.window.set_zoom(self.window.zoom_level * 1.2),
            )

        if hasattr(self.window, "zoom_out"):
            view_menu.addAction(tr("Zoom &Out") + "\tCtrl+-", self.window.zoom_out)
        else:
            view_menu.addAction(
                tr("Zoom &Out") + "\tCtrl+-",
                lambda: self.window.set_zoom(self.window.zoom_level / 1.2),
            )

        if hasattr(self.window, "fit_to_window"):
            view_menu.addAction(tr("&Fit to Window") + "\tCtrl+0", self.window.fit_to_window)

        view_menu.addSeparator()

        # Grid toggle
        if hasattr(self.window, "toggle_grid"):
            action = view_menu.addAction(
                tr("Show &Grid") + "\tCtrl+Shift+G", self.window.toggle_grid
            )
            action.setCheckable(True)
            action.setChecked(getattr(self.window, "show_grid", True))  # default True
            action.setShortcut("Ctrl+Shift+G")
            self.window.grid_toggle_action = action

        view_menu.addSeparator()

        # Live overlay submenu (Section 4 of UX phase 2)
        live_menu = view_menu.addMenu(tr("Live overlay"))
        # V1 — port-value chips (default ON)
        chips_action = QAction(tr("Output value chips"), self.window, checkable=True)
        chips_action.setChecked(True)

        def _toggle_chips(checked):
            if hasattr(self.window, "canvas"):
                self.window.canvas.show_live_chips = bool(checked)
                self.window.canvas.update()

        chips_action.triggered.connect(_toggle_chips)
        live_menu.addAction(chips_action)
        self.window.live_chips_action = chips_action

        view_menu.addSeparator()
        view_menu.addAction(tr("Toggle &Theme") + "\tCtrl+T", self.window.toggle_theme)

        # Block palette submenu
        from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES, theme_manager

        palette_menu = view_menu.addMenu(tr("Block &Palette"))
        palette_group = QActionGroup(self.window)
        palette_group.setExclusive(True)
        for key, display in PALETTE_DISPLAY_NAMES.items():
            action = QAction(display, self.window, checkable=True)
            action.triggered.connect(lambda checked, k=key: self.window._set_palette(k))
            palette_group.addAction(action)
            palette_menu.addAction(action)
            if key == theme_manager.current_palette:
                action.setChecked(True)
        self.window.palette_actions = palette_group

        # Solid block fills toggle
        solid_fills_action = QAction(tr("Solid Block Fills"), self.window, checkable=True)
        solid_fills_action.setChecked(theme_manager.solid_fills)
        solid_fills_action.triggered.connect(self.window._toggle_solid_fills)
        view_menu.addAction(solid_fills_action)
        self.window.solid_fills_action = solid_fills_action

        view_menu.addSeparator()

        # Variable Editor toggle
        if hasattr(self.window, "toggle_variable_editor"):
            action = view_menu.addAction(
                tr("Show/Hide Variable &Editor") + "\tCtrl+Shift+V",
                self.window.toggle_variable_editor,
            )
            action.setCheckable(True)
            action.setChecked(False)
            action.setShortcut("Ctrl+Shift+V")
            self.window.variable_editor_action = action

        # Workspace Editor toggle
        if hasattr(self.window, "toggle_workspace_editor"):
            action = view_menu.addAction(
                tr("Workspace &Variables") + "\tCtrl+Shift+W",
                self.window.toggle_workspace_editor,
            )
            action.setCheckable(True)
            action.setChecked(False)
            action.setShortcut("Ctrl+Shift+W")
            self.window.workspace_editor_action = action

        # Minimap toggle
        if hasattr(self.window, "toggle_minimap"):
            action = view_menu.addAction(
                tr("&Minimap") + "\tCtrl+Shift+M", self.window.toggle_minimap
            )
            action.setCheckable(True)
            action.setChecked(False)
            action.setShortcut("Ctrl+Shift+M")
            self.window.minimap_action = action

        # Parameter Tuning Panel toggle
        if hasattr(self.window, "toggle_tuning_panel"):
            action = view_menu.addAction(
                tr("Parameter &Tuning Panel") + "\tCtrl+Shift+T",
                self.window.toggle_tuning_panel,
            )
            action.setCheckable(True)
            action.setChecked(False)
            action.setShortcut("Ctrl+Shift+T")
            self.window.tuning_panel_action = action

        view_menu.addSeparator()

        # UI Scale
        scaling_menu = view_menu.addMenu(tr("UI Scale"))
        scaling_menu.addAction("100%").triggered.connect(lambda: self.window._set_scaling(1.0))
        scaling_menu.addAction("125%").triggered.connect(lambda: self.window._set_scaling(1.25))
        scaling_menu.addAction("150%").triggered.connect(lambda: self.window._set_scaling(1.5))

        view_menu.addSeparator()

        # Routing Menu
        routing_menu = view_menu.addMenu(tr("Default Connection Routing"))

        bezier = routing_menu.addAction(tr("Bezier (Curved)"))
        bezier.setCheckable(True)
        bezier.setChecked(True)  # Assuming default
        bezier.triggered.connect(lambda: self.window._set_default_routing_mode("bezier"))

        ortho = routing_menu.addAction(tr("Orthogonal (Manhattan)"))
        ortho.setCheckable(True)
        ortho.triggered.connect(lambda: self.window._set_default_routing_mode("orthogonal"))

        # Store actions in window for exclusive checking logic if needed
        self.window.bezier_routing_action = bezier
        self.window.orthogonal_routing_action = ortho

        view_menu.addSeparator()
        self._create_language_menu(view_menu)

    def _create_language_menu(self, view_menu):
        """Build View ▸ Language from the catalogs in ``locales/``.

        Entries are discovered at runtime, so dropping a new ``locales/xx.json``
        in (with a ``_meta.name``) is all it takes to offer another language --
        no code change. "System" follows the host locale.
        """
        from lib.i18n import (
            SYSTEM_LANGUAGE,
            available_languages,
            stored_language_setting,
        )

        language_menu = view_menu.addMenu(tr("&Language"))
        group = QActionGroup(self.window)
        group.setExclusive(True)
        current = stored_language_setting()

        system_action = QAction(tr("System default"), self.window, checkable=True)
        system_action.setChecked(current == SYSTEM_LANGUAGE)
        system_action.triggered.connect(lambda _checked: self.window.set_language(SYSTEM_LANGUAGE))
        group.addAction(system_action)
        language_menu.addAction(system_action)
        language_menu.addSeparator()

        for entry in available_languages():
            code = entry["code"]
            # The native name is deliberately NOT translated: a language is
            # listed in its own language so a user who cannot read the current
            # UI language can still find theirs.
            action = QAction(entry["name"], self.window, checkable=True)
            action.setChecked(current == code)
            action.triggered.connect(lambda _checked, c=code: self.window.set_language(c))
            group.addAction(action)
            language_menu.addAction(action)

        self.window.language_menu = language_menu
        self.window.language_actions = group

    def _create_help_menu(self, menubar):
        """Create Help menu."""
        help_menu = menubar.addMenu(tr("&Help"))

        help_menu.addAction(tr("&Keyboard Shortcuts") + "\tF1", self._show_shortcuts)

        # Reuse the existing Command Palette action when the window exposes it.
        if hasattr(self.window, "show_command_palette"):
            help_menu.addAction(tr("Command &Palette"), self.window.show_command_palette)

        help_menu.addAction(tr("Open &Examples"), self._open_examples_folder)
        help_menu.addAction(tr("User &Manual"), self._open_user_manual)
        help_menu.addSeparator()
        help_menu.addAction(tr("&About"), self._show_about)

        # F1 opens the shortcuts dialog from anywhere in the window. Held on the
        # window so the QShortcut isn't garbage-collected with this builder.
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QKeySequence, QShortcut

        self.window._shortcuts_help_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F1), self.window)
        self.window._shortcuts_help_shortcut.activated.connect(self._show_shortcuts)

    def _show_shortcuts(self):
        """Open the read-only keyboard-shortcuts reference dialog."""
        from modern_ui.widgets.shortcuts_dialog import KeyboardShortcutsDialog

        dialog = KeyboardShortcutsDialog(self.window)
        dialog.exec()

    def _open_resource_in_os(self, rel_path: str, *, is_dir: bool) -> None:
        """Open a bundled resource (folder or file) with the OS default handler.

        Resolves via ``lib.app_paths.resource_path`` (the canonical bundled-asset
        resolver) so it works in dev and under PyInstaller frozen builds alike —
        the hand-rolled ``__file__`` walk it replaces broke in frozen mode.
        """
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from lib.app_paths import resource_path

        path = resource_path(rel_path)
        exists = os.path.isdir(path) if is_dir else os.path.isfile(path)
        if not exists:
            logger.warning("Resource not found: %s", path)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _open_examples_folder(self):
        """Open the examples folder in the OS file browser."""
        self._open_resource_in_os("examples", is_dir=True)

    def _open_user_manual(self):
        """Open docs/USER_MANUAL.md with the OS default handler."""
        self._open_resource_in_os(os.path.join("docs", "USER_MANUAL.md"), is_dir=False)

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.about(
            self.window,
            tr("About Modern DiaBloS"),
            tr(
                "Modern DiaBloS - Diagram Block System\n\n"
                "Phase 2 Refactoring\n"
                "A modern control system simulation environment."
            ),
        )
