"""Context-menu manager for ModernCanvas.

Implements three scoped right-click menus per the V1 handoff:
  • Block menu      — edit / tune / copy / wrap / scope / delete
  • Canvas menu     — inline search QWidgetAction + quick-add + view toggles
  • Wire menu       — probe / insert / routing (radio group) / delete

Styling uses theme tokens; QMenu styling is also covered by the global QSS,
so this is additive. Inline kbd hints are rendered as a right-aligned monospace
QLabel inside a QWidgetAction (Qt QSS can't right-align portions of QMenu::item
text directly).
"""

import logging
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QMenu, QAction, QActionGroup, QWidgetAction, QWidget, QHBoxLayout,
    QLabel, QLineEdit, QFrame
)

from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_font(size: int = 9) -> QFont:
    f = QFont("Menlo")
    f.setStyleHint(QFont.Monospace)
    if hasattr(f, 'setFamilies'):
        f.setFamilies(["Menlo", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"])
    f.setPointSize(size)
    return f


def _menu_stylesheet() -> str:
    """Style matching the cmdk / handoff dark theme. Reads from theme_manager."""
    bg = theme_manager.get_color('surface_elevated').name()
    text = theme_manager.get_color('text_primary').name()
    text_dim = theme_manager.get_color('text_secondary').name()
    accent = theme_manager.get_color('accent_primary').name()
    border = theme_manager.get_color('border_primary').name()
    error = theme_manager.get_color('error').name()
    # 12% accent overlay — approximated via a fixed alpha-blended hex
    accent_qc = theme_manager.get_color('accent_primary')
    accent_bg = accent_qc.name()  # selection highlight uses solid accent on hover

    return f"""
        QMenu {{
            background-color: {bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 6px 0px;
        }}
        QMenu::item {{
            padding: 6px 16px 6px 16px;
            border-radius: 5px;
            margin: 1px 4px;
        }}
        QMenu::item:selected {{
            background-color: {accent_bg};
            color: white;
        }}
        QMenu::item:disabled {{
            color: {text_dim};
        }}
        QMenu::separator {{
            height: 1px;
            background-color: {border};
            margin: 4px 8px;
        }}
        QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}
        QMenu::icon {{ padding-left: 6px; }}
    """


def _build_header_action(menu: QMenu, text: str) -> QAction:
    """Create a disabled, eyebrow-style header action ('CAPS', dim color)."""
    dim = theme_manager.get_color('text_secondary').name()
    label = QLabel(text.upper())
    f = label.font()
    f.setPointSize(8)
    f.setBold(True)
    f.setLetterSpacing(QFont.PercentageSpacing, 110)
    label.setFont(f)
    label.setStyleSheet(f"color: {dim}; padding: 6px 14px 2px 14px;")
    wa = QWidgetAction(menu)
    wa.setDefaultWidget(label)
    wa.setEnabled(False)
    menu.addAction(wa)
    return wa


def _build_kbd_row(menu: QMenu, label_text: str, kbd_text: str = "", *,
                   danger: bool = False, dim: bool = False,
                   on_trigger=None, enabled: bool = True) -> QAction:
    """Build a menu row with right-aligned monospace kbd hint.

    QSS can't style portions of text within QMenu::item; we use a QWidgetAction
    with a custom widget. Hover/selection highlight is approximated via stylesheet
    on the host widget.
    """
    text_primary = theme_manager.get_color('text_primary').name()
    text_dim = theme_manager.get_color('text_secondary').name()
    text_disabled = theme_manager.get_color('text_disabled').name()
    error = theme_manager.get_color('error').name()
    surface = theme_manager.get_color('surface_elevated').name()
    accent = theme_manager.get_color('accent_primary').name()
    border = theme_manager.get_color('border_primary').name()

    fg = error if danger else (text_disabled if dim else text_primary)

    container = QWidget()
    container.setObjectName("CtxRow")
    lay = QHBoxLayout(container)
    lay.setContentsMargins(14, 4, 14, 4)
    lay.setSpacing(10)

    lbl = QLabel(label_text)
    lbl.setStyleSheet(f"color: {fg}; background: transparent;")
    lay.addWidget(lbl, 1)

    if kbd_text:
        kbd = QLabel(kbd_text)
        kbd.setFont(_mono_font(8))
        kbd.setStyleSheet(
            f"color: {text_dim}; background: rgba(255,255,255,0.04);"
            f" border: 1px solid {border}; border-radius: 3px;"
            f" padding: 1px 5px;"
        )
        lay.addWidget(kbd, 0)

    # Container hover effect via stylesheet on the host widget
    container.setStyleSheet(f"""
        QWidget#CtxRow {{ background: transparent; border-radius: 5px; }}
        QWidget#CtxRow:hover {{ background-color: {accent}; }}
        QWidget#CtxRow:hover QLabel {{ color: white; }}
    """)

    wa = QWidgetAction(menu)
    wa.setDefaultWidget(container)
    wa.setEnabled(enabled)
    if on_trigger is not None:
        wa.triggered.connect(on_trigger)
        # also fire on mouse click of the widget
        container.mousePressEvent = lambda ev, fn=on_trigger, m=menu: (m.close(), fn())
    menu.addAction(wa)
    return wa


# ---------------------------------------------------------------------------
# Canvas search QWidgetAction
# ---------------------------------------------------------------------------

class _CanvasSearchWidget(QWidget):
    """Inline search input for the empty-canvas context menu.

    Lives inside a QWidgetAction at the top of the canvas menu. Typing filters
    the visible quick-add rows below it; Enter adds the top match at the cursor.
    """

    def __init__(self, on_add, all_blocks, parent=None):
        super().__init__(parent)
        self._on_add = on_add  # callable(menu_block) -> add at cursor
        self._all_blocks = all_blocks  # list of menu_block objects
        self._top_match = None

        bg = theme_manager.get_color('surface_secondary').name()
        text = theme_manager.get_color('text_primary').name()
        border = theme_manager.get_color('border_primary').name()
        dim = theme_manager.get_color('text_disabled').name()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(0)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Type to add a block…  ('s', 'sum', 'pid'…)")
        self.edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 4px 9px 4px 9px;
                color: {text};
                font-size: 11pt;
                selection-background-color: {theme_manager.get_color('accent_primary').name()};
            }}
            QLineEdit:focus {{ border-color: {theme_manager.get_color('accent_primary').name()}; }}
        """)
        self.edit.textChanged.connect(self._on_text_changed)
        self.edit.returnPressed.connect(self._on_return)
        layout.addWidget(self.edit, 1)

    def _on_text_changed(self, text: str):
        # Update top match — naive substring on fn_name / block_fn
        text = (text or "").strip().lower()
        if not text:
            self._top_match = None
            return
        for mb in self._all_blocks:
            fn = (getattr(mb, 'fn_name', '') or '').lower()
            bf = (getattr(mb, 'block_fn', '') or '').lower()
            if text in fn or text in bf:
                self._top_match = mb
                return
        self._top_match = None

    def _on_return(self):
        if self._top_match is not None and self._on_add:
            mb = self._top_match
            # Find the parent QMenu and close it before adding
            p = self.parent()
            while p is not None and not isinstance(p, QMenu):
                p = p.parent()
            if isinstance(p, QMenu):
                p.close()
            self._on_add(mb)


# ---------------------------------------------------------------------------
# Main MenuManager
# ---------------------------------------------------------------------------

class MenuManager:
    """Builds and shows right-click context menus for the canvas.

    The dispatcher (`handle_context_menu`) routes to one of three menus based
    on what's under the cursor (block / wire / empty canvas).
    """

    def __init__(self, canvas):
        self.canvas = canvas

    # -- Dispatcher --------------------------------------------------------

    def handle_context_menu(self, pos):
        """Route the right-click to block / wire / canvas menu."""
        clicked_block = self.canvas._get_clicked_block(pos)
        if clicked_block:
            self.show_block_context_menu(clicked_block, pos)
            return

        clicked_line, _ = self.canvas._get_clicked_line(pos)
        if clicked_line:
            self.show_connection_context_menu(clicked_line, pos)
            return

        self.show_canvas_context_menu(pos)

    # -- V1: Block menu ---------------------------------------------------

    def show_block_context_menu(self, block, pos):
        menu = QMenu(self.canvas)
        menu.setStyleSheet(_menu_stylesheet())

        # Promote click target into selection if it wasn't selected
        if not block.selected:
            self.canvas._clear_selections()
            block.selected = True
            self.canvas.update()
        selected = [b for b in self.canvas.dsim.blocks_list if b.selected]
        multi = len(selected) > 1

        # Header — block name (disabled)
        block_label = getattr(block, 'username', None) or getattr(block, 'name', 'block')
        if multi:
            _build_header_action(menu, f"{len(selected)} blocks selected")
        else:
            _build_header_action(menu, str(block_label))

        # Edit / Tune
        _build_kbd_row(menu, "Edit parameters…", "Enter",
                       on_trigger=lambda: self.canvas._show_block_properties(block))
        # "Tune live" — pin first tunable param to tuning panel if available
        first_tunable = self._first_tunable(block)
        if first_tunable is not None:
            _build_kbd_row(menu, "Tune live", "T",
                           on_trigger=lambda: self._tune_live(block, first_tunable))

        menu.addSeparator()

        # Copy / Duplicate / Flip / Rename
        _build_kbd_row(menu, "Copy", "Ctrl+C", on_trigger=self.canvas._copy_selected_blocks)
        _build_kbd_row(menu, "Duplicate", "Ctrl+D",
                       on_trigger=lambda: self.canvas._duplicate_block(block))
        _build_kbd_row(menu, "Cut", "Ctrl+X", on_trigger=self.canvas._cut_selected_blocks)
        if hasattr(self.canvas, 'flip_selected_blocks'):
            _build_kbd_row(menu, "Flip horizontal", "F",
                           on_trigger=self.canvas.flip_selected_blocks)
        _build_kbd_row(menu, "Rename…", "F2",
                       on_trigger=lambda: self._rename_block(block))

        menu.addSeparator()

        # Detach wires
        if not multi:
            _build_kbd_row(menu, "Detach all wires", "",
                           on_trigger=lambda: self._detach_wires(block))

        # Wrap-in-subsystem (when >=1 selected)
        if hasattr(self.canvas, '_create_subsystem_trigger'):
            menu.addSeparator()
            _build_kbd_row(menu, "Wrap in subsystem", "Ctrl+G",
                           on_trigger=self.canvas._create_subsystem_trigger)

        # Port editing (variable-port blocks only)
        if getattr(block, 'io_edit', 'none') not in ('none', False, None):
            menu.addSeparator()
            _build_kbd_row(menu, "Edit ports…", "",
                           on_trigger=lambda: self._edit_ports(block))

        # Multi-select alignment submenu
        if multi:
            menu.addSeparator()
            align_menu = menu.addMenu("Align && Distribute")
            align_menu.setStyleSheet(_menu_stylesheet())
            for label, cb_name in [
                ("Align Left",    'align_left'),
                ("Align Right",   'align_right'),
                ("Align Center (H)", 'align_center_horizontal'),
                ("Align Top",     'align_top'),
                ("Align Bottom",  'align_bottom'),
                ("Align Center (V)", 'align_center_vertical'),
            ]:
                if hasattr(self.canvas, cb_name):
                    a = align_menu.addAction(label)
                    a.triggered.connect(getattr(self.canvas, cb_name))
            if len(selected) >= 3:
                align_menu.addSeparator()
                if hasattr(self.canvas, 'distribute_horizontal'):
                    align_menu.addAction("Distribute Horizontally").triggered.connect(self.canvas.distribute_horizontal)
                if hasattr(self.canvas, 'distribute_vertical'):
                    align_menu.addAction("Distribute Vertically").triggered.connect(self.canvas.distribute_vertical)

        # Tunable params submenu (existing UX kept under "Add to Tuning")
        tunables = self._tunable_params(block)
        if tunables:
            menu.addSeparator()
            tune_menu = menu.addMenu("Add to Tuning")
            tune_menu.setStyleSheet(_menu_stylesheet())
            main_win = self._find_main_window()
            for pname, pval in tunables:
                a = tune_menu.addAction(f"{pname} = {pval}")
                if main_win is not None:
                    a.triggered.connect(
                        lambda checked=False, b=block, p=pname: main_win._add_to_tuning(b, p)
                    )

        # Block-fn specific analysis actions
        if hasattr(block, 'block_fn'):
            if block.block_fn == 'BodeMag':
                menu.addSeparator()
                _build_kbd_row(menu, "Generate Bode magnitude plot", "",
                               on_trigger=lambda: self.canvas.generate_bode_plot(block))
            elif block.block_fn == 'BodePhase':
                menu.addSeparator()
                _build_kbd_row(menu, "Generate Bode phase plot", "",
                               on_trigger=lambda: self.canvas.generate_bode_phase_plot(block))
            elif block.block_fn == 'RootLocus':
                menu.addSeparator()
                _build_kbd_row(menu, "Generate root-locus plot", "",
                               on_trigger=lambda: self.canvas.generate_root_locus(block))
            elif block.block_fn == 'Nyquist':
                menu.addSeparator()
                _build_kbd_row(menu, "Generate Nyquist plot", "",
                               on_trigger=lambda: self.canvas.generate_nyquist_plot(block))
            elif block.block_fn == 'LQR' and hasattr(self.canvas, 'compute_lqr'):
                menu.addSeparator()
                _build_kbd_row(menu, "Compute LQR gain", "",
                               on_trigger=lambda: self.canvas.compute_lqr(block))

        # Delete (danger)
        menu.addSeparator()
        _build_kbd_row(menu, "Delete", "Del", danger=True,
                       on_trigger=self.canvas.remove_selected_items)

        menu.exec_(QCursor.pos())

    # -- V2: Canvas (empty) menu ------------------------------------------

    def show_canvas_context_menu(self, pos):
        menu = QMenu(self.canvas)
        menu.setStyleSheet(_menu_stylesheet())

        # Inline search at the top
        all_blocks = list(getattr(self.canvas.dsim, 'menu_blocks', []))
        if all_blocks:
            search = _CanvasSearchWidget(
                on_add=lambda mb, p=pos: self._quick_add(mb, p),
                all_blocks=all_blocks,
            )
            wa = QWidgetAction(menu)
            wa.setDefaultWidget(search)
            menu.addAction(wa)
            # Focus the search input as soon as the menu shows. Qt swallows focus
            # for QWidgetAction widgets by default; defer with a 0-ms timer.
            QTimer.singleShot(0, search.edit.setFocus)
            menu.addSeparator()

        # Paste / paste-in-place
        has_clipboard = bool(getattr(self.canvas, 'clipboard_blocks', []))
        _build_kbd_row(menu, "Paste", "Ctrl+V",
                       enabled=has_clipboard,
                       on_trigger=lambda: self.canvas._paste_blocks(pos))

        menu.addSeparator()

        # Quick-add: 5 commonly-used blocks (best-effort match by fn_name)
        quick_names = ('sine', 'step', 'sum', 'gain', 'scope')
        # Build a map fn_name(lower) -> menu_block
        by_fn = {(getattr(mb, 'fn_name', '') or '').lower(): mb for mb in all_blocks}
        added_any = False
        if any(name in by_fn for name in quick_names):
            _build_header_action(menu, "Add commonly used")
        for name in quick_names:
            mb = by_fn.get(name)
            if mb is None:
                continue
            added_any = True
            _build_kbd_row(
                menu,
                getattr(mb, 'block_fn', name).lower(),
                "add here",
                on_trigger=lambda b=mb, p=pos: self._quick_add(b, p),
            )
        if added_any:
            menu.addSeparator()

        # View toggles
        _build_kbd_row(menu, "Toggle grid", "",
                       on_trigger=lambda: self._toggle_grid())
        if hasattr(self.canvas, 'zoom_to_fit'):
            _build_kbd_row(menu, "Fit view", "Ctrl+0",
                           on_trigger=self.canvas.zoom_to_fit)

        menu.addSeparator()

        # Select-all / clear / auto-route (preserve old features)
        _build_kbd_row(menu, "Select all", "Ctrl+A",
                       on_trigger=self.canvas._select_all_blocks)
        _build_kbd_row(menu, "Clear selection", "Esc",
                       on_trigger=self.canvas._clear_selections)
        has_lines = bool(getattr(self.canvas.dsim, 'line_list', []))
        if has_lines and hasattr(self.canvas, 'auto_route_lines'):
            _build_kbd_row(menu, "Auto-route wires", "",
                           enabled=has_lines,
                           on_trigger=self.canvas.auto_route_lines)

        menu.exec_(QCursor.pos())

    # -- V3: Wire menu -----------------------------------------------------

    def show_connection_context_menu(self, line, pos):
        menu = QMenu(self.canvas)
        menu.setStyleSheet(_menu_stylesheet())

        if not line.selected:
            self.canvas._clear_line_selections() if hasattr(self.canvas, '_clear_line_selections') else None
            line.selected = True
            self.canvas.update()

        # Header label — `src.out[i] → dst.in[j]` if possible
        header = self._wire_label(line)
        _build_header_action(menu, header)

        _build_kbd_row(menu, "Edit label…", "",
                       on_trigger=lambda: self.canvas._edit_connection_label(line))

        menu.addSeparator()

        # Routing radio group
        cur_mode = getattr(line, 'routing_mode', 'bezier')
        route_menu = menu.addMenu("Routing")
        route_menu.setStyleSheet(_menu_stylesheet())
        rg = QActionGroup(route_menu)
        rg.setExclusive(True)
        for mode_key, label in [
            ('bezier',     'Bezier (curved)'),
            ('orthogonal', 'Orthogonal (Manhattan)'),
        ]:
            a = route_menu.addAction(label)
            a.setCheckable(True)
            a.setChecked(cur_mode == mode_key)
            a.triggered.connect(
                lambda checked=False, m=mode_key, l=line: self.canvas._set_connection_routing_mode(l, m)
            )
            rg.addAction(a)

        if hasattr(self.canvas, '_highlight_connection_path'):
            menu.addSeparator()
            _build_kbd_row(menu, "Highlight path", "",
                           on_trigger=lambda: self.canvas._highlight_connection_path(line))

        menu.addSeparator()
        _build_kbd_row(menu, "Delete wire", "Del", danger=True,
                       on_trigger=lambda: self.canvas._delete_line(line))

        menu.exec_(QCursor.pos())

    # -- Internals --------------------------------------------------------

    def _quick_add(self, menu_block, pos):
        try:
            self.canvas.add_block_from_palette(menu_block, pos)
        except Exception as e:
            logger.error(f"Quick-add failed: {e}")

    def _detach_wires(self, block):
        """Remove all wires connected to this block."""
        dsim = self.canvas.dsim
        to_remove = []
        for line in list(getattr(dsim, 'line_list', [])):
            src = getattr(line, 'src_block', None) or getattr(line, 'srcblock', None)
            dst = getattr(line, 'dst_block', None) or getattr(line, 'dstblock', None)
            # Some impls use names rather than block refs
            if src is block or dst is block:
                to_remove.append(line)
            else:
                src_name = getattr(line, 'srcblock', None)
                dst_name = getattr(line, 'dstblock', None)
                if src_name == getattr(block, 'name', None) or dst_name == getattr(block, 'name', None):
                    to_remove.append(line)
        for line in to_remove:
            try:
                self.canvas._delete_line(line)
            except Exception:
                pass
        self.canvas.update()

    def _rename_block(self, block):
        from PyQt5.QtWidgets import QInputDialog
        current = getattr(block, 'username', None) or getattr(block, 'name', '')
        new_name, ok = QInputDialog.getText(self.canvas, "Rename block", "New name:", text=current)
        if ok and new_name.strip():
            block.username = new_name.strip()
            main_win = self._find_main_window()
            if main_win is not None:
                try:
                    main_win._on_property_changed(block.name, '_username_', block.username)
                except Exception:
                    pass
            self.canvas.dsim.dirty = True
            self.canvas.update()

    def _edit_ports(self, block):
        self.canvas._push_undo("Edit Ports")
        block.change_port_numbers()
        self.canvas.update()
        self.canvas.dsim.dirty = True

    def _toggle_grid(self):
        main_win = self._find_main_window()
        if main_win is not None and hasattr(main_win, 'toggle_grid'):
            main_win.toggle_grid()
        elif hasattr(self.canvas, 'toggle_grid'):
            self.canvas.toggle_grid()

    def _first_tunable(self, block):
        params = self._tunable_params(block)
        return params[0][0] if params else None

    def _tunable_params(self, block):
        """Return list of (param_name, value) for scalar / list-of-scalar params."""
        out = []
        params = getattr(block, 'params', {}) or {}
        for k, v in params.items():
            if k.startswith('_'):
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out.append((k, v))
            elif isinstance(v, (list, tuple)):
                for i, elem in enumerate(v):
                    if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                        out.append((f"{k}[{i}]", elem))
        return out

    def _tune_live(self, block, param_name):
        main_win = self._find_main_window()
        if main_win is not None and hasattr(main_win, '_add_to_tuning'):
            try:
                main_win._add_to_tuning(block, param_name)
                if hasattr(main_win, 'tuning_panel'):
                    main_win.tuning_panel.show()
            except Exception as e:
                logger.warning(f"Tune-live failed: {e}")

    def _wire_label(self, line):
        src = getattr(line, 'srcblock', '') or getattr(getattr(line, 'src_block', None), 'name', '')
        dst = getattr(line, 'dstblock', '') or getattr(getattr(line, 'dst_block', None), 'name', '')
        src_port = getattr(line, 'srcport', getattr(line, 'src_port', 0))
        dst_port = getattr(line, 'dstport', getattr(line, 'dst_port', 0))
        return f"wire: {src}.out[{src_port}] → {dst}.in[{dst_port}]"

    def _find_main_window(self):
        w = self.canvas.parent()
        while w is not None and not hasattr(w, '_add_to_tuning'):
            w = w.parent()
        return w
