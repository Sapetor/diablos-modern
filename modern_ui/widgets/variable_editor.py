
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                             QLabel, QToolBar, QAction,
                             QDockWidget, QFrame)
from PyQt5.QtCore import pyqtSignal, QRegExp
from PyQt5.QtGui import QFont, QColor, QSyntaxHighlighter, QTextCharFormat
import logging
from lib.workspace import WorkspaceManager
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code."""
    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []

        # Keywords
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True',
            'try', 'while', 'with', 'yield'
        ]
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#CC7832"))  # Orange-ish
        keyword_format.setFontWeight(QFont.Bold)
        for word in keywords:
            pattern = QRegExp(r'\b' + word + r'\b')
            self.highlighting_rules.append((pattern, keyword_format))

        # Strings (single and double quotes)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#6A8759"))  # Green-ish
        self.highlighting_rules.append((QRegExp(r'".*"'), string_format))
        self.highlighting_rules.append((QRegExp(r"'.*'"), string_format))

        # Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))  # Grey
        self.highlighting_rules.append((QRegExp(r'#[^\n]*'), comment_format))

        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#6897BB"))  # Blue-ish
        self.highlighting_rules.append((QRegExp(r'\b[0-9]+\b'), number_format))
        self.highlighting_rules.append((QRegExp(r'\b[0-9]*\.[0-9]+\b'), number_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = QRegExp(pattern)
            index = expression.indexIn(text)
            while index >= 0:
                length = expression.matchedLength()
                self.setFormat(index, length, format)
                index = expression.indexIn(text, index + length)

class VariableEditor(QWidget):
    """
    A widget that provides a text editor for defining workspace variables.
    Allows users to write Python code to define variables (e.g., K=1, A=[1,2]).
    """
    variables_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.workspace_manager = WorkspaceManager()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(theme_manager.get_icon_size())
        # Style applied in _apply_theme
        layout.addWidget(self.toolbar)

        # Actions
        self.action_load = QAction("Load Script", self)
        self.action_load.setToolTip("Load Python script from file")
        self.action_load.triggered.connect(self.load_script)
        self.action_load.setText("📂 Load")
        self.toolbar.addAction(self.action_load)

        self.action_run = QAction("Update Workspace", self)
        self.action_run.setShortcut("Ctrl+Enter")
        self.action_run.setToolTip("Run code and update workspace (Ctrl+Enter)")
        self.action_run.triggered.connect(self.update_workspace)
        # Simple text icon for now if no image assets
        self.action_run.setText("▶ Run") 
        self.toolbar.addAction(self.action_run)

        self.action_clear = QAction("Clear", self)
        self.action_clear.setToolTip("Clear editor")
        self.action_clear.triggered.connect(self.clear_editor)
        self.action_clear.setText("🗑 Clear")
        self.toolbar.addAction(self.action_clear)

        self.toolbar.addSeparator()

        self.action_float = QAction("Float", self)
        self.action_float.setToolTip("Detach/Attach window")
        self.action_float.setCheckable(True)
        self.action_float.triggered.connect(self.toggle_float)
        self.action_float.setText("⧉ Float")
        self.toolbar.addAction(self.action_float)

        # Text Editor
        self.editor = QTextEdit()
        self.editor.setFont(QFont("Monospace", 11))
        self.editor.setPlaceholderText("# Define variables here\nK = 10\namplitude = 5\n\n# You can use math\nimport math\nomega = 2 * math.pi * 50")
        self.editor.setStyleSheet("border: none;")
        
        # Apply syntax highlighting
        self.highlighter = PythonHighlighter(self.editor.document())
        
        layout.addWidget(self.editor)

        # Status Bar (Inline Feedback)
        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(30)
        # Style applied in _apply_theme
        status_layout = QHBoxLayout(self.status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)
        
        self.status_label = QLabel("Ready")
        # Style applied in _apply_theme
        status_layout.addWidget(self.status_label)
        
        # Connect to theme changes
        theme_manager.theme_changed.connect(self._apply_theme)
        
        # Apply initial theme
        self._apply_theme()
        
        layout.addWidget(self.status_bar)

    def _apply_theme(self):
        """Apply the current theme to the widget."""
        bg_color = theme_manager.get_color("background_secondary").name()
        border_color = theme_manager.get_color("border_primary").name()
        text_color = theme_manager.get_color("text_primary").name()
        status_bg = theme_manager.get_color("surface_variant").name()
        
        # Toolbar styling
        self.toolbar.setStyleSheet(f"""
            QToolBar {{ 
                border-bottom: 1px solid {border_color}; 
                background: {bg_color}; 
                spacing: 5px; 
                color: {text_color};
            }}
            QToolButton {{
                color: {text_color};
                background: transparent;
                padding: 5px;
                border-radius: 4px;
            }}
            QToolButton:hover {{
                background: {theme_manager.get_color("surface_elevated").name()};
            }}
        """)
        
        # Status bar styling
        self.status_bar.setStyleSheet(f"background: {status_bg}; border-top: 1px solid {border_color};")
        self.status_label.setStyleSheet(f"color: {text_color};")
        
        # Editor styling (optional, but good for consistency)
        editor_bg = theme_manager.get_color("surface").name()
        self.editor.setStyleSheet(f"border: none; background-color: {editor_bg}; color: {text_color};")

    def clear_editor(self):
        self.editor.clear()
        self.status_label.setText("Cleared")
        self.status_label.setStyleSheet("color: #666;")

    def load_script(self):
        """Load a Python script from a file."""
        from PyQt5.QtWidgets import QFileDialog
        import os
        
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Python Script", 
            "", 
            "Python Files (*.py);;Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                self.editor.setPlainText(content)
                self.status_label.setText(f"Loaded {os.path.basename(filename)}")
                self.status_label.setStyleSheet("color: #666;")
                logger.info(f"Loaded script from {filename}")
            except Exception as e:
                logger.error(f"Error loading script: {e}")
                self.status_label.setText(f"Error loading file: {str(e)}")
                self.status_label.setStyleSheet("color: red;")

    def toggle_float(self, checked):
        """Toggle floating state of the parent dock widget."""
        # Find the parent QDockWidget
        parent = self.parent()
        while parent and not isinstance(parent, QDockWidget):
            parent = parent.parent()
        
        if parent and isinstance(parent, QDockWidget):
            parent.setFloating(checked)
            if checked:
                self.action_float.setText("⧉ Dock")
            else:
                self.action_float.setText("⧉ Float")
        else:
            logger.warning("Could not find parent QDockWidget to float/dock")

    def update_workspace(self):
        """Execute the code in the editor and update the workspace."""
        code = self.editor.toPlainText()
        if not code.strip():
            self.status_label.setText("No code to execute")
            return

        try:
            # Create a dictionary to serve as the local namespace
            local_vars = {}
            
            # Execute the code
            exec(code, {}, local_vars)
            
            # Filter out modules and internal variables
            import types
            new_vars = {k: v for k, v in local_vars.items() 
                       if not isinstance(v, types.ModuleType) and not k.startswith('_')}
            
            # Update the workspace manager
            self.workspace_manager.variables.update(new_vars)
            
            # Log success
            var_list = list(new_vars.keys())
            logger.info(f"Workspace updated with {len(new_vars)} variables: {var_list}")
            
            # Emit signal
            self.variables_updated.emit()
            
            # Update status inline
            self.status_label.setText(f"✓ Updated {len(new_vars)} variables: {', '.join(var_list[:3])}{'...' if len(var_list)>3 else ''}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
        except Exception as e:
            logger.error(f"Error updating workspace: {e}")
            self.status_label.setText(f"⚠ Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def set_text(self, text):
        self.editor.setPlainText(text)

    def get_text(self):
        return self.editor.toPlainText()
