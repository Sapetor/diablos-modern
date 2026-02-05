
import logging
from PyQt5.QtWidgets import QMenu
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class MenuManager:
    """
    Manages context menus for the ModernCanvas.
    Handles Block, Connection, and Canvas context menus.
    """
    def __init__(self, canvas):
        self.canvas = canvas

    def handle_context_menu(self, pos):
        """Handle right-click context menu event."""
        # Check if clicked on a block
        clicked_block = self.canvas._get_clicked_block(pos)
        if clicked_block:
            self.show_block_context_menu(clicked_block, pos)
            return

        # Check if clicked on a line (returns tuple: line, collision_type)
        clicked_line, _ = self.canvas._get_clicked_line(pos)
        if clicked_line:
            self.show_connection_context_menu(clicked_line, pos)
            return

        # Otherwise show canvas menu
        self.show_canvas_context_menu(pos)

    def _get_menu_stylesheet(self):
        """Return the standard stylesheet for menus."""
        return f"""
            QMenu {{
                background-color: {theme_manager.get_color('surface_secondary').name()};
                color: {theme_manager.get_color('text_primary').name()};
                border: 1px solid {theme_manager.get_color('border_primary').name()};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 12px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {theme_manager.get_color('accent_primary').name()};
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {theme_manager.get_color('border_secondary').name()};
                margin: 4px 8px;
            }}
        """

    def show_block_context_menu(self, block, pos):
        """Show context menu for a block."""
        menu = QMenu(self.canvas)
        menu.setStyleSheet(self._get_menu_stylesheet())

        # Ensure block is selected
        if not block.selected:
            self.canvas._clear_selections()
            block.selected = True
            self.canvas.update()

        # Block actions
        delete_action = menu.addAction("Delete")
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(self.canvas.remove_selected_items)

        duplicate_action = menu.addAction("Duplicate")
        duplicate_action.setShortcut("Ctrl+D")
        duplicate_action.triggered.connect(lambda: self.canvas._duplicate_block(block))

        menu.addSeparator()

        copy_action = menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.canvas._copy_selected_blocks)

        cut_action = menu.addAction("Cut")
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self.canvas._cut_selected_blocks)

        # Create Subsystem (if multiple blocks selected)
        selected_blocks = [b for b in self.canvas.dsim.blocks_list if b.selected]
        if len(selected_blocks) > 1:
            menu.addSeparator()
            create_subsys_action = menu.addAction("Create Subsystem")
            create_subsys_action.setShortcut("Ctrl+G")
            create_subsys_action.triggered.connect(lambda: self.canvas.dsim.create_subsystem_from_selection(selected_blocks))

            # Alignment submenu (only when 2+ blocks selected)
            align_menu = menu.addMenu("Align")

            # Horizontal alignment
            align_left = align_menu.addAction("Align Left")
            align_left.setShortcut("Ctrl+Shift+L")
            align_left.triggered.connect(self.canvas.align_left)

            align_right = align_menu.addAction("Align Right")
            align_right.setShortcut("Ctrl+Shift+R")
            align_right.triggered.connect(self.canvas.align_right)

            align_center_h = align_menu.addAction("Align Center (Horizontal)")
            align_center_h.setShortcut("Ctrl+Shift+H")
            align_center_h.triggered.connect(self.canvas.align_center_horizontal)

            align_menu.addSeparator()

            # Vertical alignment
            align_top = align_menu.addAction("Align Top")
            align_top.setShortcut("Ctrl+Shift+T")
            align_top.triggered.connect(self.canvas.align_top)

            align_bottom = align_menu.addAction("Align Bottom")
            align_bottom.setShortcut("Ctrl+Shift+B")
            align_bottom.triggered.connect(self.canvas.align_bottom)

            align_center_v = align_menu.addAction("Align Center (Vertical)")
            align_center_v.setShortcut("Ctrl+Shift+V")
            align_center_v.triggered.connect(self.canvas.align_center_vertical)

            # Distribution (only when 3+ blocks selected)
            if len(selected_blocks) >= 3:
                align_menu.addSeparator()

                dist_h = align_menu.addAction("Distribute Horizontally")
                dist_h.triggered.connect(self.canvas.distribute_horizontal)

                dist_v = align_menu.addAction("Distribute Vertically")
                dist_v.triggered.connect(self.canvas.distribute_vertical)

        menu.addSeparator()

        properties_action = menu.addAction("Properties...")
        properties_action.triggered.connect(lambda: self.canvas._show_block_properties(block))

        # Add special actions for analysis blocks
        if block.block_fn == "BodeMag":
            menu.addSeparator()
            bode_action = menu.addAction("Generate Bode Plot")
            bode_action.triggered.connect(lambda: self.canvas.generate_bode_plot(block))
        elif block.block_fn == "RootLocus":
            menu.addSeparator()
            rlocus_action = menu.addAction("Generate Root Locus Plot")
            rlocus_action.triggered.connect(lambda: self.canvas.generate_root_locus(block))

        # Show menu at cursor position
        screen_pos = self.canvas.mapToGlobal(pos)
        menu.exec_(screen_pos)

    def show_connection_context_menu(self, line, pos):
        """Show context menu for a connection line."""
        menu = QMenu(self.canvas)
        menu.setStyleSheet(self._get_menu_stylesheet())

        # Ensure line is selected
        if not line.selected:
            self.canvas._clear_line_selections()
            line.selected = True
            self.canvas.update()

        # Edit Label action
        edit_label_action = menu.addAction("Edit Label...")
        edit_label_action.triggered.connect(lambda: self.canvas._edit_connection_label(line))

        # Routing mode submenu
        routing_menu = menu.addMenu("Routing Mode")

        # Bezier mode
        bezier_action = routing_menu.addAction("Bezier (Curved)")
        bezier_action.setCheckable(True)
        bezier_action.setChecked(getattr(line, 'routing_mode', 'bezier') == "bezier")
        bezier_action.triggered.connect(lambda: self.canvas._set_connection_routing_mode(line, "bezier"))

        # Orthogonal mode
        orthogonal_action = routing_menu.addAction("Orthogonal (Manhattan)")
        orthogonal_action.setCheckable(True)
        orthogonal_action.setChecked(getattr(line, 'routing_mode', 'bezier') == "orthogonal")
        orthogonal_action.triggered.connect(lambda: self.canvas._set_connection_routing_mode(line, "orthogonal"))

        menu.addSeparator()

        delete_action = menu.addAction("Delete Connection")
        delete_action.setShortcut("Del")
        delete_action.triggered.connect(lambda: self.canvas._delete_line(line))

        highlight_action = menu.addAction("Highlight Path")
        highlight_action.triggered.connect(lambda: self.canvas._highlight_connection_path(line))

        # Show menu at cursor position
        screen_pos = self.canvas.mapToGlobal(pos)
        menu.exec_(screen_pos)

    def show_canvas_context_menu(self, pos):
        """Show context menu for empty canvas area."""
        menu = QMenu(self.canvas)
        menu.setStyleSheet(self._get_menu_stylesheet())

        # Paste (only enabled if clipboard has blocks)
        # Assuming clipboard_blocks is on canvas for now, or we can move it later
        clipboard_blocks = getattr(self.canvas, 'clipboard_blocks', [])
        
        paste_action = menu.addAction("Paste")
        paste_action.setShortcut("Ctrl+V")
        paste_action.setEnabled(len(clipboard_blocks) > 0)
        paste_action.triggered.connect(lambda: self.canvas._paste_blocks(pos))

        menu.addSeparator()

        select_all_action = menu.addAction("Select All")
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self.canvas._select_all_blocks)

        clear_selection_action = menu.addAction("Clear Selection")
        clear_selection_action.setShortcut("Esc")
        clear_selection_action.triggered.connect(self.canvas._clear_selections)

        menu.addSeparator()

        # Zoom submenu
        zoom_menu = menu.addMenu("Zoom")
        zoom_in_action = zoom_menu.addAction("Zoom In")
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.canvas.zoom_in)

        zoom_out_action = zoom_menu.addAction("Zoom Out")
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.canvas.zoom_out)

        zoom_fit_action = zoom_menu.addAction("Fit All")
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.canvas.zoom_to_fit)

        # Show menu at cursor position
        screen_pos = self.canvas.mapToGlobal(pos)
        menu.exec_(screen_pos)
