"""Modern Block Palette Widget for DiaBloS Phase 2
Interactive palette with draggable blocks organized by categories.
"""

import logging
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QFrame, QPushButton, QButtonGroup, QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QPoint, QByteArray, QRect
from PyQt5.QtGui import QDrag, QPainter, QPixmap, QFont, QColor
from PyQt5.QtSvg import QSvgWidget

# Import DSim modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.lib import DSim
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class DraggableBlockWidget(QFrame):
    """A draggable widget representing a block in the palette."""
    
    block_drag_started = pyqtSignal(object, object)  # menu_block, position
    
    def __init__(self, menu_block, category_name, colors, parent=None):
        super().__init__(parent)
        self.menu_block = menu_block
        self.category_name = category_name
        self.colors = colors
        
        # Setup widget
        self._setup_widget()
        self._apply_styling()
    
    def _setup_widget(self):
        """Setup the widget layout and content."""
        # Get platform configuration for consistent sizing
        from modern_ui.platform_config import get_platform_config
        config = get_platform_config()

        scaled_size = config.palette_block_size

        logger.debug(f"Block widget sizing: size={scaled_size}px")

        self.setFixedSize(scaled_size, scaled_size)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setCursor(Qt.OpenHandCursor)
        # Tooltip with doc, ports, params
        try:
            doc_lines = []
            block_cls = getattr(self.menu_block, 'block_class', None)
            inst = block_cls() if block_cls else None
            if inst and hasattr(inst, 'doc'):
                doc_lines.append(str(inst.doc))
            # Inputs / outputs
            def _names(seq):
                try:
                    return ", ".join([i.get('name', '') for i in seq if isinstance(i, dict)]) or "—"
                except Exception:
                    return "—"
            if inst:
                doc_lines.append(f"Inputs: {_names(getattr(inst, 'inputs', []))}")
                doc_lines.append(f"Outputs: {_names(getattr(inst, 'outputs', []))}")
                params = getattr(self.menu_block, 'param_meta', getattr(inst, 'params', {}))
                if isinstance(params, dict) and params:
                    doc_lines.append("Params: " + ", ".join(list(params.keys())[:6]) + ("…" if len(params) > 6 else ""))
            tooltip = "\n".join([line for line in doc_lines if line])
            if tooltip:
                self.setToolTip(tooltip)
        except Exception:
            pass

    def paintEvent(self, event):
        painter = QPainter(self)
        self._perform_drawing(painter)
        painter.end()

    def _perform_drawing(self, painter):
        painter.setRenderHint(QPainter.Antialiasing)

        from lib.simulation.block import DBlock
        from modern_ui.themes.theme_manager import ThemeType
        menu_block = self.menu_block

        # Adjust block rect to leave more margin for ports (10px on sides for port circles)
        block_rect = self.rect().adjusted(12, 12, -12, -30)

        temp_dblock = DBlock(
            block_fn=menu_block.block_fn,
            sid=0,
            coords=block_rect,
            color=menu_block.b_color,
            in_ports=menu_block.ins,
            out_ports=menu_block.outs,
            b_type=menu_block.b_type,
            io_edit=menu_block.io_edit,
            fn_name=menu_block.fn_name,
            params=menu_block.params,
            external=menu_block.external,
            colors=self.colors,
            block_class=getattr(menu_block, 'block_class', None)
        )

        temp_dblock.update_Block()

        # In dark mode, temporarily override text_primary to use dark color for block icons
        original_theme = None
        if theme_manager.current_theme == ThemeType.DARK:
            original_theme = theme_manager.themes[ThemeType.DARK]['text_primary']
            theme_manager.themes[ThemeType.DARK]['text_primary'] = '#1F2937'

        # Draw block with ports for better visualization
        temp_dblock.draw_Block(painter, draw_name=False, draw_ports=True)

        # Restore original theme color if we changed it
        if original_theme:
            theme_manager.themes[ThemeType.DARK]['text_primary'] = original_theme

        # Draw block name with background for visibility
        name_rect = QRect(4, self.height() - 22, self.width() - 8, 20)
        
        # Draw semi-transparent background behind text for contrast
        from modern_ui.themes.theme_manager import ThemeType
        if theme_manager.current_theme == ThemeType.DARK:
            # Dark mode: dark background, white text
            bg_color = QColor('#1A1F26')
            bg_color.setAlpha(220)
            text_color = QColor('#FFFFFF')
        else:
            # Light mode: light background, dark text  
            bg_color = QColor('#F3F4F6')
            bg_color.setAlpha(220)
            text_color = QColor('#1F2937')
        
        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(name_rect, 4, 4)
        
        # Draw text
        painter.setPen(text_color)
        font = QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(name_rect, Qt.AlignCenter, menu_block.fn_name)

    def _apply_styling(self):
        """Apply theme-aware styling with elevated surfaces."""
        # Use palette-specific theme colors
        item_bg = theme_manager.get_color('palette_item_bg')
        item_hover = theme_manager.get_color('palette_item_hover')
        item_border = theme_manager.get_color('palette_item_border')
        item_border_hover = theme_manager.get_color('palette_item_border_hover')

        self.setStyleSheet(f"""
            DraggableBlockWidget {{
                background-color: {item_bg.name()};
                border: 1px solid {item_border.name()};
                border-radius: 10px;
                margin: 2px;
            }}
            DraggableBlockWidget:hover {{
                background-color: {item_hover.name()};
                border: 2px solid {item_border_hover.name()};
            }}
        """)
    
    def mousePressEvent(self, event):
        """Handle mouse press for drag initiation."""
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.drag_start_position = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move to start drag operation."""
        if not (event.buttons() & Qt.LeftButton):
            return
        
        if not hasattr(self, 'drag_start_position'):
            return
        
        if ((event.pos() - self.drag_start_position).manhattanLength() < 
            10):  # Minimum drag distance
            return
        
        self._start_drag(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.setCursor(Qt.OpenHandCursor)
    
    def _start_drag(self, event):
        """Start the drag operation."""
        try:
            logger.debug(f"Starting drag for block: {getattr(self.menu_block, 'fn_name', 'Unknown')}")
            
            drag = QDrag(self)
            
            mime_data = QMimeData()
            mime_data.setText(f"diablo_block:{getattr(self.menu_block, 'fn_name', 'Unknown')}")
            
            drag.menu_block = self.menu_block
            
            pixmap = self._create_drag_pixmap()
            drag.setPixmap(pixmap)
            drag.setHotSpot(QPoint(pixmap.width()//2, pixmap.height()//2))
            
            drag.setMimeData(mime_data)
            
            drop_action = drag.exec_(Qt.CopyAction | Qt.MoveAction, Qt.CopyAction)
            
            logger.debug(f"Drag completed with action: {drop_action}")
            
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")
    
    def _create_drag_pixmap(self):
        """Create a pixmap for the drag operation."""
        try:
            pixmap = QPixmap(self.size())
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setOpacity(0.8)
            self._perform_drawing(painter)
            painter.end()
            
            return pixmap
            
        except Exception as e:
            logger.error(f"Error creating drag pixmap: {str(e)}")
            pixmap = QPixmap(100, 30)
            pixmap.fill(theme_manager.get_color('accent_primary'))
            return pixmap


class BlockCategoryWidget(QFrame):
    """Widget for a category of blocks."""
    
    def __init__(self, category_name, blocks, colors, parent=None):
        super().__init__(parent)
        self.category_name = category_name
        self.blocks = blocks
        self.colors = colors
        
        self._setup_widget()
        self._apply_styling()
    
    def _setup_widget(self):
        """Setup the category widget."""
        from modern_ui.platform_config import get_platform_config
        config = get_platform_config()

        self.setFrameStyle(QFrame.StyledPanel)

        layout = QVBoxLayout(self)
        padding = config.palette_container_padding
        layout.setContentsMargins(padding, padding, padding, padding)
        layout.setSpacing(config.palette_grid_spacing)

        # Category header
        header = QLabel(self.category_name)
        header.setFont(QFont("Segoe UI", 9, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        # Add blocks in a grid (2 columns)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(config.palette_grid_spacing)

        row = 0
        col = 0
        for block in self.blocks:
            block_widget = DraggableBlockWidget(block, self.category_name, colors=self.colors)
            grid_layout.addWidget(block_widget, row, col)
            col += 1
            if col > 1:  # 2 blocks per row
                col = 0
                row += 1

        layout.addLayout(grid_layout)
    
    def _apply_styling(self):
        """Apply theme-aware styling."""
        bg_color = theme_manager.get_color('surface_secondary')
        border_color = theme_manager.get_color('border_primary')
        text_color = theme_manager.get_color('text_secondary')
        
        self.setStyleSheet(f"""
            BlockCategoryWidget {{
                background-color: transparent;
                border: none;
                border-radius: 6px;
                margin: 4px;
            }}
            QLabel {{
                color: {text_color.name()};
                padding: 4px;
                border-bottom: 1px solid {border_color.name()};
                margin-bottom: 4px;
                font-size: 9pt;
                font-weight: bold;
            }}
        """)


class ModernBlockPalette(QWidget):
    """Modern block palette with organized categories and drag-and-drop support."""
    
    block_drag_started = pyqtSignal(object)  # Emitted when a block drag starts
    
    def __init__(self, dsim, parent=None):
        super().__init__(parent)
        
        self.dsim = dsim
        
        # Setup widget
        self._setup_widget()
        self._load_blocks()
        self._apply_styling()
        
        # Connect to theme changes
        theme_manager.theme_changed.connect(self._apply_styling)
        
        logger.info("Modern block palette initialized")

    def _setup_widget(self):
        """Setup the main widget layout."""
        from modern_ui.platform_config import get_platform_config
        config = get_platform_config()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(config.palette_grid_spacing)

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search blocks...")
        self.search_bar.textChanged.connect(self._filter_blocks)
        layout.addWidget(self.search_bar)

        # Create scroll area for blocks
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container for block categories
        self.blocks_container = QWidget()
        self.blocks_layout = QVBoxLayout(self.blocks_container)
        padding = config.palette_container_padding
        self.blocks_layout.setContentsMargins(padding, padding, padding, padding)
        self.blocks_layout.setSpacing(config.palette_grid_spacing * 2)  # Double spacing between categories
        
        scroll_area.setWidget(self.blocks_container)
        layout.addWidget(scroll_area)

    def _filter_blocks(self, text):
        """Filter blocks based on search text."""
        text = text.lower()

        for i in range(self.blocks_layout.count()):
            category_widget = self.blocks_layout.itemAt(i).widget()
            if isinstance(category_widget, BlockCategoryWidget):
                category_visible = False

                # Get the grid layout (it's the second item in the category's layout)
                category_layout = category_widget.layout()
                if category_layout and category_layout.count() >= 2:
                    grid_item = category_layout.itemAt(1)  # 0=header, 1=grid
                    if grid_item:
                        grid_layout = grid_item.layout()
                        if grid_layout:
                            # Iterate through grid layout items
                            for j in range(grid_layout.count()):
                                block_widget = grid_layout.itemAt(j).widget()
                                if isinstance(block_widget, DraggableBlockWidget):
                                    block_name = getattr(block_widget.menu_block, 'fn_name', '').lower()
                                    # Show block if search text matches or if search is empty
                                    if not text or text in block_name:
                                        block_widget.show()
                                        category_visible = True
                                    else:
                                        block_widget.hide()

                # Show/hide entire category based on whether any blocks are visible
                category_widget.setVisible(category_visible)

    def _load_blocks(self):
        """Load blocks from DSim and organize them into categories."""
        try:
            # Get menu blocks from DSim
            menu_blocks = getattr(self.dsim, 'menu_blocks', [])
            
            if not menu_blocks:
                logger.warning("No menu blocks found in DSim")
                self._add_placeholder()
                return
            
            # Organize blocks by category
            categories = self._categorize_blocks(menu_blocks)
            
            # Create category widgets
            for category_name, blocks in categories.items():
                category_widget = BlockCategoryWidget(category_name, blocks, colors=self.dsim.colors)
                self.blocks_layout.addWidget(category_widget)
            
            # Add stretch to push categories to top
            self.blocks_layout.addStretch()
            
            logger.info(f"Loaded {len(menu_blocks)} blocks in {len(categories)} categories")
            
        except Exception as e:
            logger.error(f"Error loading blocks: {str(e)}")
            self._add_placeholder()
    
    def _categorize_blocks(self, menu_blocks):
        """Organize blocks into logical categories."""
        categories = {
            "Sources": [],
            "Math": [],
            "Control": [],
            "Filters": [],
            "Sinks": [],
            "Routing": [],
            "Other": []
        }

        # Simple categorization based on block names (fallback)
        source_keywords = ['step', 'ramp', 'sine', 'square', 'constant', 'source']
        math_keywords = ['sum', 'gain', 'multiply', 'add', 'subtract', 'divide', 'abs', 'sqrt', 'product']
        control_keywords = ['integrator', 'derivative', 'pid', 'controller', 'delay', 'transfer']
        filter_keywords = ['filter', 'lowpass', 'highpass', 'bandpass']
        sink_keywords = ['scope', 'display', 'sink', 'plot', 'output', 'export', 'term']
        routing_keywords = ['mux', 'demux', 'switch', 'selector']

        for block in menu_blocks:
            # Check if block has a block_class with category property
            category = None
            if hasattr(block, 'block_class') and block.block_class:
                try:
                    temp_instance = block.block_class()
                    if hasattr(temp_instance, 'category'):
                        category = temp_instance.category
                except Exception:
                    pass

            # If no category property, use keyword-based categorization
            if not category:
                block_name = getattr(block, 'fn_name', '').lower()

                if any(keyword in block_name for keyword in source_keywords):
                    category = "Sources"
                elif any(keyword in block_name for keyword in math_keywords):
                    category = "Math"
                elif any(keyword in block_name for keyword in control_keywords):
                    category = "Control"
                elif any(keyword in block_name for keyword in filter_keywords):
                    category = "Filters"
                elif any(keyword in block_name for keyword in sink_keywords):
                    category = "Sinks"
                elif any(keyword in block_name for keyword in routing_keywords):
                    category = "Routing"
                else:
                    category = "Other"

            # Add block to the appropriate category
            if category in categories:
                categories[category].append(block)
            else:
                categories["Other"].append(block)
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        return categories
    
    def _add_placeholder(self):
        """Add placeholder content when no blocks are available."""
        placeholder = QLabel("No blocks available.\nCheck DSim initialization.")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet(f"""
            color: {theme_manager.get_color('text_secondary').name()};
            font-style: italic;
            padding: 20px;
        """)
        self.blocks_layout.addWidget(placeholder)
    
    def _apply_styling(self):
        """Apply theme-aware styling."""
        from modern_ui.themes.theme_manager import ThemeType

        bg_color = theme_manager.get_color('surface_primary')
        text_color = theme_manager.get_color('text_primary')
        border_color = theme_manager.get_color('border_primary')
        search_bg_color = theme_manager.get_color('surface_secondary')

        # Use theme-appropriate colors for search box
        if theme_manager.current_theme == ThemeType.DARK:
            search_input_bg = search_bg_color.name()
            search_input_text = text_color.name()
        else:
            # Light gray background with dark text for light mode
            search_input_bg = '#E0E0E0'
            search_input_text = '#111827'

        self.setStyleSheet(f"""
            ModernBlockPalette {{
                background-color: {bg_color.name()};
                border: 1px solid {border_color.name()};
                border-radius: 6px;
            }}
            QLineEdit {{
                background-color: {search_input_bg};
                color: {search_input_text};
                border: 1px solid {border_color.name()};
                border-radius: 4px;
                padding: 5px;
            }}
            QLabel {{
                color: {text_color.name()};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {theme_manager.get_color('surface_secondary').name()};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme_manager.get_color('accent_primary').name()};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme_manager.get_color('accent_secondary').name()};
            }}
        """)
    
    def refresh_blocks(self):
        """Refresh the block palette by reloading blocks."""
        try:
            # Clear existing blocks
            for i in reversed(range(self.blocks_layout.count())):
                child = self.blocks_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # Reload blocks
            self._load_blocks()
            
            logger.info("Block palette refreshed")
            
        except Exception as e:
            logger.error(f"Error refreshing blocks: {str(e)}")
    
    def get_available_blocks(self):
        """Get list of all available blocks."""
        return getattr(self.dsim, 'menu_blocks', [])
