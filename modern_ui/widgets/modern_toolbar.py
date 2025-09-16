"""
Modern Toolbar Widget for DiaBloS
Features enhanced styling, modern icons, and improved usability.
"""

from PyQt5.QtWidgets import (QToolBar, QToolButton, QAction, QWidget, 
                             QHBoxLayout, QLabel, QSlider, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from modern_ui.themes.theme_manager import theme_manager


class ModernToolBar(QToolBar):
    """Modern styled toolbar with enhanced functionality."""
    
    # Signals for toolbar actions
    new_diagram = pyqtSignal()
    open_diagram = pyqtSignal()
    save_diagram = pyqtSignal()
    play_simulation = pyqtSignal()
    pause_simulation = pyqtSignal()
    stop_simulation = pyqtSignal()
    plot_results = pyqtSignal()
    capture_screen = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    theme_toggled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setObjectName("ModernToolBar")
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        self._setup_actions()
        self._setup_toolbar()
        
        # Connect to theme changes
        theme_manager.theme_changed.connect(self._update_theme)
    
    def _create_icon(self, symbol: str, color: str = None) -> QIcon:
        """Create a simple text-based icon."""
        if color is None:
            color = theme_manager.get_color('text_primary').name()
        
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(color))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, symbol)
        painter.end()
        
        return QIcon(pixmap)
    
    def _setup_actions(self):
        """Create toolbar actions."""
        # File actions
        self.new_action = QAction(self._create_icon("📄"), "New", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setToolTip("Create new diagram (Ctrl+N)")
        self.new_action.triggered.connect(self.new_diagram.emit)
        
        self.open_action = QAction(self._create_icon("📁"), "Open", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setToolTip("Open diagram (Ctrl+O)")
        self.open_action.triggered.connect(self.open_diagram.emit)
        
        self.save_action = QAction(self._create_icon("💾"), "Save", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setToolTip("Save diagram (Ctrl+S)")
        self.save_action.triggered.connect(self.save_diagram.emit)
        
        # Simulation actions
        self.play_action = QAction(self._create_icon("▶️"), "Play", self)
        self.play_action.setShortcut("F5")
        self.play_action.setToolTip("Start simulation (F5)")
        self.play_action.triggered.connect(self.play_simulation.emit)
        
        self.pause_action = QAction(self._create_icon("⏸️"), "Pause", self)
        self.pause_action.setShortcut("F6")
        self.pause_action.setToolTip("Pause simulation (F6)")
        self.pause_action.triggered.connect(self.pause_simulation.emit)
        
        self.stop_action = QAction(self._create_icon("⏹️"), "Stop", self)
        self.stop_action.setShortcut("F7")
        self.stop_action.setToolTip("Stop simulation (F7)")
        self.stop_action.triggered.connect(self.stop_simulation.emit)
        
        # View actions
        self.plot_action = QAction(self._create_icon("📊"), "Plot", self)
        self.plot_action.setToolTip("Show plots")
        self.plot_action.triggered.connect(self.plot_results.emit)
        
        self.capture_action = QAction(self._create_icon("📷"), "Capture", self)
        self.capture_action.setToolTip("Take screenshot")
        self.capture_action.triggered.connect(self.capture_screen.emit)
        
        # Theme toggle action
        self.theme_action = QAction(self._create_icon("🌙"), "Theme", self)
        self.theme_action.setToolTip("Toggle dark/light theme")
        self.theme_action.triggered.connect(self._toggle_theme)
    
    def _setup_toolbar(self):
        """Setup toolbar layout and widgets."""
        # File group
        self.addAction(self.new_action)
        self.addAction(self.open_action)
        self.addAction(self.save_action)
        self.addSeparator()
        
        # Simulation group
        self.addAction(self.play_action)
        self.addAction(self.pause_action)
        self.addAction(self.stop_action)
        self.addSeparator()
        
        # View group
        self.addAction(self.plot_action)
        self.addAction(self.capture_action)
        self.addSeparator()
        
        # Zoom control
        zoom_widget = self._create_zoom_widget()
        self.addWidget(zoom_widget)
        self.addSeparator()
        
        # Theme toggle
        self.addAction(self.theme_action)
        
        # Add spacer to push remaining items to the right
        spacer = QWidget()
        spacer.setSizePolicy(spacer.sizePolicy().Expanding, spacer.sizePolicy().Preferred)
        self.addWidget(spacer)
        
        # Status indicators
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        self.addWidget(self.status_label)
    
    def _create_zoom_widget(self) -> QWidget:
        """Create zoom control widget."""
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(4, 0, 4, 0)
        zoom_layout.setSpacing(8)
        
        # Zoom label
        zoom_label = QLabel("Zoom:")
        zoom_layout.addWidget(zoom_label)
        
        # Zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(25)  # 25%
        self.zoom_slider.setMaximum(200)  # 200%
        self.zoom_slider.setValue(100)   # 100%
        self.zoom_slider.setFixedWidth(100)
        self.zoom_slider.setToolTip("Zoom level")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        zoom_layout.addWidget(self.zoom_slider)
        
        # Zoom percentage label
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(40)
        zoom_layout.addWidget(self.zoom_label)
        
        return zoom_widget
    
    def _on_zoom_changed(self, value: int):
        """Handle zoom slider changes."""
        zoom_factor = value / 100.0
        self.zoom_label.setText(f"{value}%")
        self.zoom_changed.emit(zoom_factor)
    
    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        theme_manager.toggle_theme()
        self.theme_toggled.emit()
    
    def _update_theme(self):
        """Update icons and styling when theme changes."""
        # Recreate icons with new theme colors
        self.new_action.setIcon(self._create_icon("📄"))
        self.open_action.setIcon(self._create_icon("📁"))
        self.save_action.setIcon(self._create_icon("💾"))
        self.play_action.setIcon(self._create_icon("▶️"))
        self.pause_action.setIcon(self._create_icon("⏸️"))
        self.stop_action.setIcon(self._create_icon("⏹️"))
        self.plot_action.setIcon(self._create_icon("📊"))
        self.capture_action.setIcon(self._create_icon("📷"))
        
        # Update theme icon based on current theme
        if theme_manager.current_theme.value == "dark":
            self.theme_action.setIcon(self._create_icon("☀️"))
            self.theme_action.setToolTip("Switch to light theme")
        else:
            self.theme_action.setIcon(self._create_icon("🌙"))
            self.theme_action.setToolTip("Switch to dark theme")
    
    def set_status(self, message: str):
        """Update status message."""
        self.status_label.setText(message)
    
    def set_simulation_state(self, running: bool, paused: bool = False):
        """Update toolbar based on simulation state."""
        self.play_action.setEnabled(not running or paused)
        self.pause_action.setEnabled(running and not paused)
        self.stop_action.setEnabled(running)
        
        if running and not paused:
            self.set_status("Simulation running...")
        elif running and paused:
            self.set_status("Simulation paused")
        else:
            self.set_status("Ready")
    
    def get_zoom_factor(self) -> float:
        """Get current zoom factor."""
        return self.zoom_slider.value() / 100.0
    
    def set_zoom_factor(self, factor: float):
        """Set zoom factor."""
        value = int(factor * 100)
        self.zoom_slider.setValue(value)