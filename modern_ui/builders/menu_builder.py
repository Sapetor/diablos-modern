
import os
import logging
from PyQt5.QtWidgets import QMenu, QAction

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
        self._create_view_menu(menubar)
        self._create_help_menu(menubar)

    def _create_file_menu(self, menubar):
        """Create File menu."""
        file_menu = menubar.addMenu("&File")
        
        # Standard actions
        file_menu.addAction("&New\tCtrl+N", self.window.new_diagram)
        file_menu.addAction("&Open\tCtrl+O", self.window.open_diagram)
        file_menu.addAction("&Save\tCtrl+S", self.window.save_diagram)
        file_menu.addSeparator()

        # Recent Files
        self.window.recent_files_menu = file_menu.addMenu("Recent Files")
        if hasattr(self.window, '_update_recent_files_menu'):
            self.window._update_recent_files_menu()

        # Examples
        examples_menu = file_menu.addMenu("Examples")
        self._populate_examples_menu(examples_menu)

        file_menu.addSeparator()
        file_menu.addAction("E&xit\tAlt+F4", self.window.close)

    def _populate_examples_menu(self, menu):
        """Populate examples submenu."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        examples_dir = os.path.join(base_dir, 'examples')
        
        if os.path.exists(examples_dir):
            found = False
            for f in os.listdir(examples_dir):
                if f.endswith('.json') or f.endswith('.dat'):
                    action = menu.addAction(f)
                    # Use default argument capture
                    action.triggered.connect(lambda checked, fname=f: self.window.open_example(os.path.join(examples_dir, fname)))
                    found = True
            
            if not found:
                 menu.addAction("No examples found").setEnabled(False)
        else:
            menu.addAction("Examples directory not found").setEnabled(False)

    def _create_edit_menu(self, menubar):
        """Create Edit menu."""
        edit_menu = menubar.addMenu("&Edit")
        if hasattr(self.window, 'undo_action'):
            edit_menu.addAction("&Undo\tCtrl+Z", self.window.undo_action)
        if hasattr(self.window, 'redo_action'):
            edit_menu.addAction("&Redo\tCtrl+Y", self.window.redo_action)
            
        edit_menu.addSeparator()
        
        # Check if select_all is implemented, otherwise define it or skip
        if hasattr(self.window, 'select_all'):
            edit_menu.addAction("Select &All\tCtrl+A", self.window.select_all)
        elif hasattr(self.window, 'canvas') and hasattr(self.window.canvas, '_select_all_blocks'):
            # Fallback if method missing in window
            edit_menu.addAction("Select &All\tCtrl+A", self.window.canvas._select_all_blocks)
            
        edit_menu.addSeparator()
        
        if hasattr(self.window, 'show_command_palette'):
            edit_menu.addAction("Command &Palette\tCtrl+P", self.window.show_command_palette)

    def _create_simulation_menu(self, menubar):
        """Create Simulation menu."""
        sim_menu = menubar.addMenu("&Simulation")
        sim_menu.addAction("&Run\tF5", self.window.start_simulation)
        sim_menu.addAction("&Pause\tF6", self.window.pause_simulation)
        sim_menu.addAction("&Stop\tF7", self.window.stop_simulation)
        sim_menu.addSeparator()
        
        # Fast Solver Toggle
        fast_solver = sim_menu.addAction("Enable Fast Solver (Experimental)")
        fast_solver.setCheckable(True)
        # Default to True, but check DSim state if possible (MainWindow usually holds this state)
        # We'll assume MainWindow has 'use_fast_solver' attribute initialized to True
        is_fast = getattr(self.window, 'use_fast_solver', True)
        fast_solver.setChecked(is_fast)
        fast_solver.triggered.connect(self.window.toggle_fast_solver)
        self.window.fast_solver_action = fast_solver
        
        sim_menu.addSeparator()
        sim_menu.addAction("Show &Plots", self.window.show_plots)

    def _create_view_menu(self, menubar):
        """Create View menu."""
        view_menu = menubar.addMenu("&View")
        
        # Zoom controls
        # Delegate to window methods if they exist, or lambdas
        if hasattr(self.window, 'zoom_in'):
             view_menu.addAction("&Zoom In\tCtrl++", self.window.zoom_in)
        else:
             view_menu.addAction("&Zoom In\tCtrl++", lambda: self.window.set_zoom(self.window.zoom_level * 1.2))

        if hasattr(self.window, 'zoom_out'):
             view_menu.addAction("Zoom &Out\tCtrl+-", self.window.zoom_out)
        else:
             view_menu.addAction("Zoom &Out\tCtrl+-", lambda: self.window.set_zoom(self.window.zoom_level / 1.2))
             
        if hasattr(self.window, 'fit_to_window'):
            view_menu.addAction("&Fit to Window\tCtrl+0", self.window.fit_to_window)
        
        view_menu.addSeparator()

        # Grid toggle
        if hasattr(self.window, 'toggle_grid'):
            action = view_menu.addAction("Show &Grid\tCtrl+G", self.window.toggle_grid)
            action.setCheckable(True)
            action.setChecked(getattr(self.window, 'show_grid', True)) # default True
            self.window.grid_toggle_action = action

        view_menu.addSeparator()
        view_menu.addAction("Toggle &Theme\tCtrl+T", self.window.toggle_theme)
        view_menu.addSeparator()
        
        # Variable Editor toggle
        if hasattr(self.window, 'toggle_variable_editor'):
             action = view_menu.addAction("Show/Hide Variable &Editor\tCmd+Shift+V", self.window.toggle_variable_editor)
             action.setCheckable(True)
             action.setChecked(False)
             self.window.variable_editor_action = action
             
             # Shortcut handling might remain in MainWindow or move here?
             # MainWindow had: self.variable_editor_shortcut = QShortcut(...)
             # We will assume MainWindow handles shortcut separately if it's complex, 
             # or we can move it here if we import QShortcut. 
             # For now, let's leave shortcut creation keying off the method existance, 
             # but the MENU action is handled here.

        view_menu.addSeparator()
        
        # UI Scale
        scaling_menu = view_menu.addMenu("UI Scale")
        scaling_menu.addAction("100%").triggered.connect(lambda: self.window._set_scaling(1.0))
        scaling_menu.addAction("125%").triggered.connect(lambda: self.window._set_scaling(1.25))
        scaling_menu.addAction("150%").triggered.connect(lambda: self.window._set_scaling(1.5))

        view_menu.addSeparator()
        
        # Routing Menu
        routing_menu = view_menu.addMenu("Default Connection Routing")
        
        bezier = routing_menu.addAction("Bezier (Curved)")
        bezier.setCheckable(True)
        bezier.setChecked(True) # Assuming default
        bezier.triggered.connect(lambda: self.window._set_default_routing_mode("bezier"))
        
        ortho = routing_menu.addAction("Orthogonal (Manhattan)")
        ortho.setCheckable(True)
        ortho.triggered.connect(lambda: self.window._set_default_routing_mode("orthogonal"))
        
        # Store actions in window for exclusive checking logic if needed
        self.window.bezier_routing_action = bezier
        self.window.orthogonal_routing_action = ortho

    def _create_help_menu(self, menubar):
        """Create Help menu."""
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About", self._show_about)

    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(self.window, "About Modern DiaBloS",
                          "Modern DiaBloS - Diagram Block System\n\n"
                          "Phase 2 Refactoring\n"
                          "A modern control system simulation environment.")
