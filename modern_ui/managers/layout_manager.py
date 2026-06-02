"""
LayoutManager -- builds the main window's central layout: the nested splitters
and the left (block palette), centre (canvas + breadcrumb + error panel) and
right (property editor) panels, plus the width-based splitter sizing.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

Like ``StatusBarManager``, the build methods assign the constructed widgets back
onto the window as attributes (``canvas``, ``property_editor``, ``main_splitter``,
``center_splitter``, ``block_palette``, ``canvas_area``, ...) because those names
are referenced widely -- inside the window and externally (e.g. ``diagram_service``
reads ``main_window.main_splitter`` / ``center_splitter``).

The property scroll-area event filter stays installed on the *window* (a
QObject whose ``eventFilter`` override handles it); this manager just installs it
and stores ``window._prop_scroll_viewport``.
"""

import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
                             QLabel, QFrame, QScrollArea)

from modern_ui.widgets.modern_palette import ModernBlockPalette
from modern_ui.widgets.modern_canvas import ModernCanvas
from modern_ui.widgets.property_editor import PropertyEditor
from modern_ui.widgets.error_panel import ErrorPanel
from modern_ui.widgets.breadcrumb_bar import BreadcrumbBar
from modern_ui.platform_config import get_platform_config

logger = logging.getLogger(__name__)


class LayoutManager:
    """Builds the central widget, splitters, and panels of the main window."""

    def __init__(self, main_window):
        self.window = main_window

    def setup_layout(self):
        """Setup modern layout with splitters."""
        window = self.window
        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setObjectName("MainSplitter")
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_layout.addWidget(main_splitter)

        # Left panel (Block Palette)
        window.left_panel = self.create_left_panel()
        main_splitter.addWidget(window.left_panel)

        # Center area (Canvas + Property Panel on right)
        center_splitter = QSplitter(Qt.Horizontal)
        center_splitter.setObjectName("CenterSplitter")
        center_splitter.setChildrenCollapsible(False)
        center_splitter.setHandleWidth(5)

        # Canvas area (will contain the drawing canvas)
        window.canvas_area = self.create_canvas_area()
        center_splitter.addWidget(window.canvas_area)

        # Property panel (right side, vertical)
        window.property_panel = self.create_property_panel()
        center_splitter.addWidget(window.property_panel)

        # Set stretch factors: canvas gets priority when resizing
        center_splitter.setStretchFactor(0, 1)  # Canvas stretches
        center_splitter.setStretchFactor(1, 0)  # Property panel stays fixed size

        # Don't set splitter sizes here - will be set after window is shown
        # This ensures calculations use actual window width, not screen width

        main_splitter.addWidget(center_splitter)

        # Set stretch factors: center (canvas+properties) gets priority when resizing
        main_splitter.setStretchFactor(0, 0)  # Left panel stays fixed size
        main_splitter.setStretchFactor(1, 1)  # Center area stretches

        # Store reference to center splitter for initial sizing
        window._center_splitter_for_init = center_splitter

        # Store splitters for theme updates
        window.main_splitter = main_splitter
        window.center_splitter = center_splitter

    def initialize_splitter_sizes(self):
        """Initialize splitter sizes based on actual window dimensions.
        Called after window is shown to ensure accurate sizing."""
        window = self.window
        # Get platform configuration and actual window width
        config = get_platform_config()
        actual_width = window.width()

        logger.info(f"Initializing splitters with actual window width: {actual_width}px")

        # Main splitter: left panel gets fixed width, rest goes to center
        left_width = config.splitter_left_width
        center_width = actual_width - left_width
        window.main_splitter.setSizes([left_width, center_width])

        # Center splitter: canvas gets most space, property panel gets configured percentage
        property_width = int(center_width * config.splitter_property_percent)
        canvas_width = center_width - property_width

        # Ensure property panel is at least minimum width
        min_property_width = config.splitter_property_min_width
        if property_width < min_property_width:
            property_width = min_property_width
            canvas_width = center_width - property_width

        window.center_splitter.setSizes([canvas_width, property_width])

        logger.info(f"Splitter sizes: left={left_width}, canvas={canvas_width}, properties={property_width}")

    def create_left_panel(self) -> QWidget:
        """Create modern left panel for block palette."""
        window = self.window
        panel = QFrame()
        panel.setObjectName("ModernPanel")

        # Get platform configuration
        config = get_platform_config()

        panel.setMinimumWidth(config.left_panel_min_width)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Panel title
        title = QLabel("Block Palette")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 4px;")
        layout.addWidget(title)

        # Modern block palette widget (Phase 2)
        window.block_palette = ModernBlockPalette(window.dsim)
        layout.addWidget(window.block_palette)
        return panel

    def create_canvas_area(self) -> QWidget:
        """Create modern canvas area with responsive sizing and error panel."""
        window = self.window
        # Create container widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        # Create the modern canvas widget
        window.canvas = ModernCanvas(window.dsim)

        # Set default routing mode for new connections
        window.canvas.default_routing_mode = window.default_routing_mode

        # Get platform configuration
        config = get_platform_config()

        window.canvas.setMinimumSize(config.canvas_min_width, config.canvas_min_height)

        logger.info(f"Canvas minimum size: {config.canvas_min_width}×{config.canvas_min_height}")

        # Connect canvas signals
        window.canvas.block_selected.connect(window._on_block_selected)
        window.canvas.connection_created.connect(window._on_connection_created)
        window.canvas.simulation_status_changed.connect(window._on_simulation_status_changed)
        window.canvas.command_palette_requested.connect(window.show_command_palette)
        window.toolbar.auto_route_wires.connect(window.canvas.auto_route_lines)

        # Create breadcrumb bar
        window.breadcrumb_bar = BreadcrumbBar()
        window.breadcrumb_bar.path_clicked.connect(window.canvas.navigate_scope)
        window.canvas.scope_changed.connect(window.breadcrumb_bar.set_path)

        # Add widgets to container
        container_layout.addWidget(window.breadcrumb_bar)
        container_layout.addWidget(window.canvas, 1)  # Canvas gets stretch priority

        # Create error panel
        window.error_panel = ErrorPanel()
        window.error_panel.error_clicked.connect(window._on_error_clicked)
        window.error_panel.setMaximumHeight(200)  # Limit height
        container_layout.addWidget(window.error_panel, 0)  # Error panel doesn't stretch

        return container

    def create_property_panel(self) -> QWidget:
        """Create modern property panel on right side with size constraints."""
        window = self.window
        panel = QFrame()
        panel.setObjectName("ModernPanel")
        panel.setFrameStyle(QFrame.StyledPanel)

        # Get platform configuration
        config = get_platform_config()

        panel.setMinimumWidth(config.property_panel_min_width)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Panel title
        title = QLabel("Properties")
        title.setObjectName("PanelTitle")
        title.setStyleSheet("font-weight: bold; font-size: 12pt; padding: 4px;")
        layout.addWidget(title)

        # Scroll area for properties
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameStyle(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(scroll_area)

        # Property editor
        window.property_editor = PropertyEditor()
        scroll_area.setWidget(window.property_editor)

        # PyQt5 5.15: QScrollArea viewport absorbs mouse clicks and doesn't
        # transfer focus to child widgets. Install event filter to fix this.
        window._prop_scroll_viewport = scroll_area.viewport()
        window._prop_scroll_viewport.installEventFilter(window)

        return panel
