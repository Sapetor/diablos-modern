import os
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence

from lib.i18n import tr

logger = logging.getLogger(__name__)

SK = QKeySequence.StandardKey


def _sequences(*keys):
    """Flatten StandardKey / "Ctrl+X" / QKeySequence specs into unique QKeySequences.

    A ``StandardKey`` expands to every platform binding (``Redo`` is Ctrl+Shift+Z
    on macOS/Linux and Ctrl+Y on Windows), so Qt renders and handles the native
    one while the extras keep the app's historical keys alive.
    """
    out = []
    for key in keys:
        if isinstance(key, SK):
            candidates = QKeySequence.keyBindings(key)
        elif isinstance(key, QKeySequence):
            candidates = [key]
        else:
            candidates = [QKeySequence(key)]
        for seq in candidates:
            if not seq.isEmpty() and seq not in out:
                out.append(seq)
    return out


class MenuBuilder:
    """Builder for MainWindow menus.

    Shortcuts are bound as real ``QAction`` shortcuts (never ``"\\tCtrl+X"``
    label suffixes) so Qt renders them natively and every key has exactly one
    owner. Ownership rules:

    * window-wide actions (undo/redo, zoom, simulation, panels, ...) use the
      default ``WindowShortcut`` context;
    * canvas editing actions (cut/copy/paste/select-all/flip/align) are also added
      to the canvas and use ``WidgetWithChildrenShortcut`` so they never steal
      keys from other widgets (a table's own copy / select-all, say);
    * ``Delete``/``Backspace``/``Esc`` stay in ``ModernCanvas.keyPressEvent``
      (context dependent, and Backspace must keep working in text fields).
    """

    def __init__(self, main_window):
        self.window = main_window

    # -- helpers ------------------------------------------------------------

    def _action(
        self,
        menu,
        text,
        slot=None,
        keys=None,
        *,
        canvas_scope=False,
        app_scope=False,
        checkable=False,
        tooltip=None,
    ):
        """Create a QAction owned by the window, add it to ``menu``, bind keys."""
        action = QAction(text, self.window)
        if checkable:
            action.setCheckable(True)
        if slot is not None:
            action.triggered.connect(slot)
        if keys:
            keys = keys if isinstance(keys, (list, tuple)) else [keys]
            action.setShortcuts(_sequences(*keys))
            if canvas_scope and hasattr(self.window, "canvas"):
                action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                self.window.canvas.addAction(action)
                self.window._canvas_scoped_actions.append(action)
            elif app_scope:
                action.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        menu.addAction(action)
        return action

    def _scoped(self, menu, text, slot, keys, **kw):
        """Canvas-scoped action (see the class docstring)."""
        return self._action(menu, text, slot, keys, canvas_scope=True, **kw)

    @staticmethod
    def _dispose_menu(menu):
        """Unbind every shortcut under ``menu`` so a rebuild can't be ambiguous."""
        for action in menu.actions():
            if action.menu() is not None:
                MenuBuilder._dispose_menu(action.menu())
            action.setShortcuts([])
        menu.setParent(None)
        menu.deleteLater()

    def setup_menubar(self):
        """Setup the entire menu bar."""
        menubar = self.window.menuBar()
        # Rebuilt on every language switch: drop the old menus (and their key
        # bindings) first, or the old and new actions would collide.
        for old in menubar.actions():
            if old.menu() is not None:
                self._dispose_menu(old.menu())
        menubar.clear()
        for stale in getattr(self.window, "_canvas_scoped_actions", []):
            stale.setShortcuts([])
            if hasattr(self.window, "canvas"):
                self.window.canvas.removeAction(stale)
        self.window._canvas_scoped_actions = []

        self._create_file_menu(menubar)
        self._create_edit_menu(menubar)
        self._create_library_menu(menubar)
        self._create_simulation_menu(menubar)
        self._create_analysis_menu(menubar)
        self._create_view_menu(menubar)
        self._create_help_menu(menubar)

    # -- File ---------------------------------------------------------------

    def _create_file_menu(self, menubar):
        """Create File menu."""
        file_menu = menubar.addMenu(tr("&File"))

        self._action(file_menu, tr("&New"), self.window.new_diagram, SK.New)
        self._action(file_menu, tr("&Open"), self.window.open_diagram, SK.Open)
        self._action(file_menu, tr("&Save"), self.window.save_diagram, SK.Save)
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
        if hasattr(self.window, "show_preferences"):
            prefs = self._action(
                file_menu,
                tr("&Preferences..."),
                self.window.show_preferences,
                [SK.Preferences, "Ctrl+,"],
            )
            # macOS moves this into the application menu on its own.
            prefs.setMenuRole(QAction.MenuRole.PreferencesRole)
            self.window.preferences_action = prefs
            file_menu.addSeparator()

        # Quit is Ctrl+Q on macOS/Linux; Windows has no standard key, so keep the
        # Alt+F4 chord the label used to advertise.
        exit_keys = QKeySequence.keyBindings(SK.Quit) or [QKeySequence("Alt+F4")]
        exit_action = self._action(file_menu, tr("E&xit"), self.window.close, exit_keys)
        exit_action.setMenuRole(QAction.MenuRole.QuitRole)
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

    # -- Edit ---------------------------------------------------------------

    def _create_edit_menu(self, menubar):
        """Create Edit menu: undo/redo, clipboard, selection and layout editing."""
        win = self.window
        edit_menu = menubar.addMenu(tr("&Edit"))
        win.edit_menu = edit_menu

        # Undo/redo: window-wide. Text widgets accept the ShortcutOverride for
        # these keys themselves, so a focused line edit keeps its own undo.
        if hasattr(win, "undo_action"):
            self._action(edit_menu, tr("&Undo"), win.undo_action, SK.Undo)
        if hasattr(win, "redo_action"):
            self._action(
                edit_menu, tr("&Redo"), win.redo_action, [SK.Redo, "Ctrl+Shift+Z", "Ctrl+Y"]
            )

        edit_menu.addSeparator()

        canvas = getattr(win, "canvas", None)
        if canvas is not None and hasattr(canvas, "copy_selected_blocks"):
            self._scoped(edit_menu, tr("Cu&t"), canvas._cut_selected_blocks, SK.Cut)
            self._scoped(edit_menu, tr("&Copy"), canvas.copy_selected_blocks, SK.Copy)
            self._scoped(edit_menu, tr("&Paste"), canvas.paste_blocks, SK.Paste)

        if hasattr(win, "select_all"):
            self._scoped(edit_menu, tr("Select &All"), win.select_all, SK.SelectAll)
        elif canvas is not None and hasattr(canvas, "_select_all_blocks"):
            self._scoped(edit_menu, tr("Select &All"), canvas._select_all_blocks, SK.SelectAll)

        edit_menu.addAction(tr("Copy Diagram as &Image"), win.copy_diagram_image)

        edit_menu.addSeparator()

        # Create Subsystem (window-wide, as it always was)
        if hasattr(win, "create_subsystem"):
            self._action(edit_menu, tr("Create &Subsystem"), win.create_subsystem, "Ctrl+G")
        elif canvas is not None and hasattr(canvas, "_create_subsystem_trigger"):
            self._action(
                edit_menu, tr("Create &Subsystem"), canvas._create_subsystem_trigger, "Ctrl+G"
            )

        if canvas is not None and hasattr(canvas, "flip_selected_blocks"):
            self._scoped(edit_menu, tr("&Flip Block"), canvas.flip_selected_blocks, "Ctrl+F")

            align_menu = edit_menu.addMenu(tr("A&lign"))
            # Align Top uses U ("up"): Ctrl+Shift+T belongs to the tuning panel.
            for label, method, key in (
                (tr("Align &Left"), "align_left", "Ctrl+Shift+L"),
                (tr("Align &Right"), "align_right", "Ctrl+Shift+R"),
                (tr("Align &Center Horizontally"), "align_center_horizontal", "Ctrl+Shift+H"),
                (tr("Align &Top"), "align_top", "Ctrl+Shift+U"),
                (tr("Align &Bottom"), "align_bottom", "Ctrl+Shift+B"),
                (tr("Align Center &Vertically"), "align_center_vertical", None),
            ):
                if hasattr(canvas, method):
                    self._scoped(align_menu, label, getattr(canvas, method), key)

        edit_menu.addSeparator()

        if hasattr(win, "show_command_palette"):
            palette = self._action(
                edit_menu,
                tr("Command &Palette"),
                win.show_command_palette,
                "Ctrl+K",
                app_scope=True,
            )
            win.command_palette_action = palette

    # -- Library ------------------------------------------------------------

    def _create_library_menu(self, menubar):
        """Create Library menu: masks, the user block library and custom blocks."""
        win = self.window
        lib_menu = menubar.addMenu(tr("&Library"))
        win.library_menu = lib_menu

        # Masks & user library (see modern_ui/managers/mask_library_manager.py).
        # Lambdas, not the bound methods: QAction.triggered(bool) would otherwise
        # bind the checked flag to their optional `block` argument.
        if hasattr(win, "edit_block_mask"):
            lib_menu.addAction(tr("Edit &Mask..."), lambda: win.edit_block_mask())
            lib_menu.addAction(tr("&Look Under Mask"), lambda: win.look_under_mask())
            lib_menu.addAction(tr("Save as &Library Block..."), lambda: win.save_as_library_block())
            lib_menu.addAction(tr("Reload from Li&brary"), lambda: win.reload_from_library())
            lib_menu.addAction(tr("Refresh Block Librar&y"), lambda: win.refresh_block_library())

        # Custom block modules (see lib/user_blocks.py, docs/BLOCK_API.md)
        if hasattr(win, "reload_user_blocks"):
            lib_menu.addSeparator()
            # Lambda, not the bound method: QAction.triggered(bool) would bind
            # the checked flag to reload_user_blocks' `quiet` argument.
            lib_menu.addAction(tr("Reload &User Blocks"), lambda: win.reload_user_blocks())
            lib_menu.addAction(tr("Open User Blocks &Folder..."), win.open_user_blocks_folder)

    # -- Simulation ---------------------------------------------------------

    def _create_simulation_menu(self, menubar):
        """Create Simulation menu."""
        win = self.window
        sim_menu = menubar.addMenu(tr("&Simulation"))
        sim_menu.setToolTipsVisible(True)
        self._action(sim_menu, tr("&Run"), win.start_simulation, "F5")
        self._action(sim_menu, tr("&Pause"), win.pause_simulation, "F6")
        # Shift+F5 (stop) is the historical canvas chord; kept as an alias.
        self._action(sim_menu, tr("&Stop"), win.stop_simulation, ["F7", "Shift+F5"])
        if hasattr(win, "step_simulation"):
            self._action(sim_menu, tr("S&tep"), win.step_simulation, "F8")
        sim_menu.addSeparator()

        # Run no longer pops the settings dialog; this is the way in.
        if hasattr(win, "open_simulation_settings"):
            action = self._action(
                sim_menu, tr("Simulation Settin&gs..."), win.open_simulation_settings, "Ctrl+E"
            )
            win.simulation_settings_action = action

        sim_menu.addSeparator()

        # Compiled-solver toggle (the stored setting is still ``use_fast_solver``).
        fast_solver = self._action(
            sim_menu,
            tr("Use Compiled Solver"),
            win.toggle_fast_solver,
            checkable=True,
            tooltip=tr(
                "Run diagrams through the fast compiled ODE solver. "
                "Diagrams with blocks it cannot compile fall back to the interpreter."
            ),
        )
        # MainWindow holds this state and initialises it to True.
        fast_solver.setChecked(getattr(win, "use_fast_solver", True))
        win.fast_solver_action = fast_solver

        sim_menu.addSeparator()
        self._action(sim_menu, tr("Show &Plots"), win.show_plots, "Ctrl+Shift+P")

    # -- Analysis -----------------------------------------------------------

    def _create_analysis_menu(self, menubar):
        """Create Analysis menu (linearization-based system analysis)."""
        win = self.window
        analysis_menu = menubar.addMenu(tr("&Analysis"))
        # "&&" renders a literal "&": label shows "Linearize & Analyze...".
        self._action(
            analysis_menu, tr("&Linearize && Analyze..."), win.linearize_and_analyze, "Ctrl+Alt+L"
        )
        self._action(
            analysis_menu,
            tr("&Find Operating Point (Trim)..."),
            win.find_operating_point,
            "Ctrl+Alt+T",
        )
        self._action(
            analysis_menu, tr("&Parameter Sweep..."), win.run_parameter_sweep, "Ctrl+Alt+S"
        )
        self._action(analysis_menu, tr("&Monte Carlo..."), win.run_monte_carlo, "Ctrl+Alt+M")

    # -- View ---------------------------------------------------------------

    def _create_view_menu(self, menubar):
        """Create View menu: quick view toggles only (persistent settings live in
        File > Preferences)."""
        win = self.window
        view_menu = menubar.addMenu(tr("&View"))

        if hasattr(win, "zoom_in"):
            zoom_in = win.zoom_in
        else:

            def zoom_in():
                win.set_zoom(win.zoom_level * 1.2)

        if hasattr(win, "zoom_out"):
            zoom_out = win.zoom_out
        else:

            def zoom_out():
                win.set_zoom(win.zoom_level / 1.2)

        # Ctrl+= is the unshifted spelling of Ctrl++ on most keyboards.
        self._action(view_menu, tr("&Zoom In"), zoom_in, [SK.ZoomIn, "Ctrl+="])
        self._action(view_menu, tr("Zoom &Out"), zoom_out, SK.ZoomOut)
        if hasattr(win, "fit_to_window"):
            self._action(view_menu, tr("&Fit to Window"), win.fit_to_window, "Ctrl+0")

        view_menu.addSeparator()

        # Grid toggle
        if hasattr(win, "toggle_grid"):
            action = self._action(
                view_menu, tr("Show &Grid"), win.toggle_grid, "Ctrl+Shift+G", checkable=True
            )
            action.setChecked(getattr(win, "show_grid", True))  # default True
            win.grid_toggle_action = action

        view_menu.addSeparator()

        # Live overlay submenu (Section 4 of UX phase 2)
        live_menu = view_menu.addMenu(tr("Live overlay"))
        # V1 — port-value chips (default ON)
        chips_action = QAction(tr("Output value chips"), win, checkable=True)
        chips_action.setChecked(True)

        def _toggle_chips(checked):
            if hasattr(win, "canvas"):
                win.canvas.show_live_chips = bool(checked)
                win.canvas.update()

        chips_action.triggered.connect(_toggle_chips)
        live_menu.addAction(chips_action)
        win.live_chips_action = chips_action

        view_menu.addSeparator()
        self._action(view_menu, tr("Toggle &Theme"), win.toggle_theme, "Ctrl+T")

        view_menu.addSeparator()

        # Dock / panel toggles
        for attr, label, slot, key in (
            (
                "variable_editor_action",
                tr("Show/Hide Variable &Editor"),
                "toggle_variable_editor",
                "Ctrl+Shift+V",
            ),
            (
                "workspace_editor_action",
                tr("Workspace &Variables"),
                "toggle_workspace_editor",
                "Ctrl+Shift+W",
            ),
            ("minimap_action", tr("&Minimap"), "toggle_minimap", "Ctrl+Shift+M"),
            (
                "tuning_panel_action",
                tr("Parameter &Tuning Panel"),
                "toggle_tuning_panel",
                "Ctrl+Shift+T",
            ),
        ):
            if hasattr(win, slot):
                action = self._action(view_menu, label, getattr(win, slot), key, checkable=True)
                action.setChecked(False)
                setattr(win, attr, action)

    def _create_help_menu(self, menubar):
        """Create Help menu."""
        help_menu = menubar.addMenu(tr("&Help"))

        # F1 opens the shortcuts dialog from anywhere in the window.
        self._action(help_menu, tr("&Keyboard Shortcuts"), self._show_shortcuts, "F1")

        # Reuse the existing Command Palette action when the window exposes it.
        if hasattr(self.window, "show_command_palette"):
            help_menu.addAction(tr("Command &Palette"), self.window.show_command_palette)

        help_menu.addAction(tr("Open &Examples"), self._open_examples_folder)
        help_menu.addAction(tr("User &Manual"), self._open_user_manual)
        help_menu.addSeparator()
        help_menu.addAction(tr("&About"), self._show_about)

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

        from modern_ui import __version__

        QMessageBox.about(
            self.window,
            tr("About DiaBloS Modern"),
            # The licence lines are not decoration: MIT requires the copyright
            # notice to accompany every copy, and LGPL v3 s4(c) requires the Qt
            # notice to appear among the notices the program displays at run
            # time. THIRD_PARTY_LICENSES.md and licenses/ ship in the bundle.
            tr(
                "DiaBloS Modern {version}\n"
                "A block-diagram simulation environment for dynamics and control.\n\n"
                "This program's own source code is free software under the MIT "
                "licence.\n"
                "It bundles Qt 6 under the LGPL v3 and the PyQt6 bindings under "
                "the GPL v3;\n"
                "see THIRD_PARTY_LICENSES.md and licenses/ next to the "
                "application for the\n"
                "notices and the corresponding source.",
                version=__version__,
            ),
        )
