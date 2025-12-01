#!/usr/bin/env python3
"""
Script to integrate VariableEditor into MainWindow
Safer than trying to use replace_file_content which was causing corruption
"""

def integrate_variable_editor():
    import re
    
    filepath = 'modern_ui/main_window.py'
    
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add import for VariableEditor (after CommandPalette import)
    import_pattern = r'(from modern_ui\.widgets\.command_palette import CommandPalette)'
    import_replacement = r'\1\nfrom modern_ui.widgets.variable_editor import VariableEditor'
    
    if 'from modern_ui.widgets.variable_editor import VariableEditor' not in content:
        content = re.sub(import_pattern, import_replacement, content)
        print("✓ Added VariableEditor import")
    else:
        print("✓ VariableEditor import already exists")
    
    # 2. Add VariableEditor initialization in __init__ (after command palette setup)
    init_pattern = r'(        self\.command_palette\.command_selected\.connect\(self\._on_command_executed\)\n        self\._setup_command_palette\(\))'
    init_code = '''        self.command_palette.command_selected.connect(self._on_command_executed)
        self._setup_command_palette()
        
        # Initialize Variable Editor (Dockable)
        from PyQt5.QtWidgets import QDockWidget
        self.variable_editor = VariableEditor(self)
        self.variable_editor_dock = QDockWidget("Variable Editor", self)
        self.variable_editor_dock.setWidget(self.variable_editor)
        self.variable_editor_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.variable_editor_dock)
        self.variable_editor_dock.hide()  # Hidden by default
        
        # Connect variable editor signals
        self.variable_editor.variables_updated.connect(self._on_variables_updated)'''
    
    if 'self.variable_editor = VariableEditor' not in content:
        content = re.sub(init_pattern, init_code, content)
        print("✓ Added VariableEditor initialization")
    else:
        print("✓ VariableEditor initialization already exists")
    
    # 3. Add menu action in View menu (after Toggle Theme)
    menu_pattern = r'(        view_menu\.addAction\("Toggle &Theme\\tCtrl\+T", self\.toggle_theme\))'
    menu_replacement = r'''\1
        view_menu.addSeparator()
        
        # Variable Editor toggle
        self.variable_editor_action = view_menu.addAction("Show/Hide Variable &Editor\\tCtrl+Shift+V", self.toggle_variable_editor)
        self.variable_editor_action.setCheckable(True)
        self.variable_editor_action.setChecked(False)'''
    
    if 'Show/Hide Variable' not in content:
        content = re.sub(menu_pattern, menu_replacement, content)
        print("✓ Added Variable Editor menu action")
    else:
        print("✓ Variable Editor menu action already exists")
    
    # 4. Add toggle method and signal handler (before closeEvent)
    methods_code = '''
    def toggle_variable_editor(self):
        """Toggle Variable Editor visibility."""
        if self.variable_editor_dock.isVisible():
            self.variable_editor_dock.hide()
            self.variable_editor_action.setChecked(False)
            self.toast.show_message("Variable Editor hidden")
        else:
            self.variable_editor_dock.show()
            self.variable_editor_action.setChecked(True)
            self.toast.show_message("Variable Editor shown")
    
    def _on_variables_updated(self):
        """Handle variable updates from the Variable Editor."""
        try:
            var_count = len(WorkspaceManager().variables)
            self.toast.show_message(f"✓ Workspace updated ({var_count} variables)", duration=2000)
            self.status_message.setText(f"Workspace updated with {var_count} variable(s)")
            logger.info(f"Workspace updated from Variable Editor: {var_count} variables")
        except Exception as e:
            logger.error(f"Error handling variable update: {str(e)}")
            self.toast.show_message(f"Error updating workspace: {str(e)}", duration=3000, is_error=True)
    
    def closeEvent(self, event):'''
    
    if 'def toggle_variable_editor' not in content:
        # Find closeEvent and insert before it
        close_event_pattern = r'(    def closeEvent\(self, event\):)'
        content = re.sub(close_event_pattern, methods_code, content)
        print("✓ Added toggle_variable_editor and _on_variables_updated methods")
    else:
        print("✓ Methods already exist")
    
    # Write the updated content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✅ Integration complete!")
    print("\nNext steps:")
    print("1. Run the application")
    print("2. Press Ctrl+Shift+V to show the Variable Editor")
    print("3. Define variables like: num = [1, 2]")
    print("4. Click 'Update Workspace'")
    print("5. In block parameters, type 'num' instead of [1, 2]")

if __name__ == '__main__':
    integrate_variable_editor()
