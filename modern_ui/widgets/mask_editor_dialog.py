"""Mask editor -- turn a Subsystem into a reusable, parameterised block.

Three tabs, matching the three things a mask carries (see ``lib/masks.py``):

  * **Parameters** -- the ordered table of exposed parameters
    (name / type / default / options / doc) with add, remove and reorder.
    Inner blocks reference these names in their own values, e.g. an inner
    Gain with ``gain = "K"``.
  * **Appearance** -- display name, palette category, icon text and outline
    shape used when the masked block is drawn and exported.
  * **Documentation** -- the description shown in the property panel and in
    the palette tooltip.

``get_mask()`` returns a normalized mask dict (or raises
:class:`lib.masks.MaskError` for an invalid table); the caller decides what to
do with it.  ``accept()`` validates first and keeps the dialog open with an
inline message when the mask does not normalize, so an invalid mask can never
be committed to a block.
"""

import logging

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lib.i18n import tr, tr_noop
from lib.masks import (
    MASK_PARAM_TYPES,
    MASK_SHAPES,
    MaskError,
    default_mask,
    get_mask,
    mask_parameters,
    normalize_mask,
)

logger = logging.getLogger(__name__)

_COL_NAME = 0
_COL_TYPE = 1
_COL_DEFAULT = 2
_COL_OPTIONS = 3
_COL_DOC = 4
# English source keys (tr_noop marks them for extraction); translated at
# display time in _build_parameters_tab() via ``tr(c)`` so the column
# count/order stays a stable module-level constant.
_COLUMNS = (
    tr_noop("Name"),
    tr_noop("Type"),
    tr_noop("Default"),
    tr_noop("Options"),
    tr_noop("Description"),
)


class MaskEditorDialog(QDialog):
    """Create or edit the mask of a Subsystem block."""

    def __init__(self, block=None, mask=None, parent=None):
        """
        Args:
            block: The Subsystem being masked. Used for the initial mask and
                the window title; may be None when editing a bare mask dict.
            mask: Optional mask dict to start from (defaults to the block's
                current mask, or a blank one).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.block = block
        self.setWindowTitle(tr("Edit Mask"))
        self.setModal(True)
        self.setMinimumWidth(620)

        if mask is None and block is not None:
            mask = get_mask(block)
        if mask is None:
            suggested = ""
            if block is not None:
                suggested = getattr(block, "username", "") or getattr(block, "name", "")
            mask = default_mask(suggested or tr("Masked Subsystem"))

        self._build_ui()
        self._load(mask)

    # ------------------------------------------------------------------ UI ---

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_parameters_tab(), tr("Parameters"))
        self.tabs.addTab(self._build_appearance_tab(), tr("Icon && Appearance"))
        self.tabs.addTab(self._build_documentation_tab(), tr("Documentation"))
        layout.addWidget(self.tabs, 1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #d9534f;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _build_parameters_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)

        hint = QLabel(
            tr(
                "Parameters exposed by this block. Inside the subsystem, reference "
                "a parameter by its name in a block value (for example a Gain with "
                "gain = K). Values are resolved against these parameters first, "
                "then the diagram's workspace variables."
            )
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([tr(c) for c in _COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_COL_DOC, QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.add_btn = QPushButton(tr("Add"))
        self.remove_btn = QPushButton(tr("Remove"))
        self.up_btn = QPushButton(tr("Move up"))
        self.down_btn = QPushButton(tr("Move down"))
        self.add_btn.clicked.connect(self.add_parameter)
        self.remove_btn.clicked.connect(self.remove_parameter)
        self.up_btn.clicked.connect(lambda: self.move_parameter(-1))
        self.down_btn.clicked.connect(lambda: self.move_parameter(1))
        for btn in (self.add_btn, self.remove_btn, self.up_btn, self.down_btn):
            row.addWidget(btn)
        row.addStretch(1)
        v.addLayout(row)
        return page

    def _build_appearance_tab(self):
        page = QWidget()
        form = QFormLayout(page)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Vehicle")
        form.addRow(tr("Display name:"), self.name_edit)

        self.category_edit = QLineEdit()
        self.category_edit.setPlaceholderText(tr("User Library"))
        self.category_edit.setToolTip(tr("Palette section this block is listed under"))
        form.addRow(tr("Palette category:"), self.category_edit)

        self.icon_edit = QLineEdit()
        self.icon_edit.setPlaceholderText(tr("A short label or emoji, e.g. 1/(ms+b)"))
        self.icon_edit.setToolTip(
            tr("Drawn inside the block. Leave empty to draw the display name instead.")
        )
        form.addRow(tr("Icon text:"), self.icon_edit)

        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(MASK_SHAPES))
        form.addRow(tr("Outline shape:"), self.shape_combo)
        return page

    def _build_documentation_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel(tr("Shown in the property panel and the palette tooltip:")))
        self.description_edit = QPlainTextEdit()
        v.addWidget(self.description_edit, 1)
        return page

    # ---------------------------------------------------------------- data ---

    def _load(self, mask):
        self.name_edit.setText(str(mask.get("name", "")))
        self.category_edit.setText(str(mask.get("category", "") or "User Library"))
        self.icon_edit.setText(str(mask.get("icon", "")))
        shape = str(mask.get("shape", "rect"))
        idx = self.shape_combo.findText(shape)
        self.shape_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.description_edit.setPlainText(str(mask.get("description", "")))

        self.table.setRowCount(0)
        for spec in mask_parameters(mask):
            self._append_row(spec)

    def _append_row(self, spec=None):
        spec = spec or {}
        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, _COL_NAME, QTableWidgetItem(str(spec.get("name", ""))))

        type_combo = QComboBox()
        type_combo.addItems(list(MASK_PARAM_TYPES))
        idx = type_combo.findText(str(spec.get("type", "float")))
        type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.table.setCellWidget(row, _COL_TYPE, type_combo)

        default = spec.get("default", "")
        self.table.setItem(row, _COL_DEFAULT, QTableWidgetItem(_to_text(default)))

        options = spec.get("options") or []
        self.table.setItem(row, _COL_OPTIONS, QTableWidgetItem(", ".join(str(o) for o in options)))
        self.table.item(row, _COL_OPTIONS).setToolTip(
            tr("Comma-separated allowed values (required for a 'choice' parameter)")
        )

        self.table.setItem(row, _COL_DOC, QTableWidgetItem(str(spec.get("doc", ""))))
        return row

    # -------------------------------------------------------------- actions ---

    def add_parameter(self, spec=None):
        """Append a parameter row and select it."""
        row = self._append_row(spec if isinstance(spec, dict) else None)
        self.table.selectRow(row)
        return row

    def remove_parameter(self):
        """Remove the selected parameter row."""
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)

    def move_parameter(self, delta):
        """Move the selected row by ``delta`` (-1 up, +1 down)."""
        row = self.table.currentRow()
        target = row + delta
        if row < 0 or target < 0 or target >= self.table.rowCount():
            return
        spec_a = self._row_spec(row)
        spec_b = self._row_spec(target)
        self._write_row(row, spec_b)
        self._write_row(target, spec_a)
        self.table.selectRow(target)

    def _write_row(self, row, spec):
        self.table.item(row, _COL_NAME).setText(str(spec.get("name", "")))
        combo = self.table.cellWidget(row, _COL_TYPE)
        idx = combo.findText(str(spec.get("type", "float")))
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.table.item(row, _COL_DEFAULT).setText(_to_text(spec.get("default", "")))
        self.table.item(row, _COL_OPTIONS).setText(
            ", ".join(str(o) for o in (spec.get("options") or []))
        )
        self.table.item(row, _COL_DOC).setText(str(spec.get("doc", "")))

    def _row_spec(self, row):
        def cell(col):
            item = self.table.item(row, col)
            return item.text() if item is not None else ""

        combo = self.table.cellWidget(row, _COL_TYPE)
        ptype = combo.currentText() if combo is not None else "float"
        options = [o.strip() for o in cell(_COL_OPTIONS).split(",") if o.strip()]
        return {
            "name": cell(_COL_NAME).strip(),
            "type": ptype,
            "default": _from_text(cell(_COL_DEFAULT), ptype, options),
            "options": options,
            "doc": cell(_COL_DOC).strip(),
        }

    # --------------------------------------------------------------- result ---

    def get_mask(self):
        """Return the edited mask as a normalized dict (raises MaskError)."""
        mask = {
            "name": self.name_edit.text().strip(),
            "description": self.description_edit.toPlainText(),
            "icon": self.icon_edit.text(),
            "shape": self.shape_combo.currentText(),
            "category": self.category_edit.text().strip() or "User Library",
            "parameters": [self._row_spec(r) for r in range(self.table.rowCount())],
        }
        return normalize_mask(mask)

    def accept(self):
        """Validate before closing; an invalid mask keeps the dialog open."""
        try:
            self.get_mask()
        except MaskError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            logger.info("Mask rejected: %s", exc)
            return
        self.error_label.hide()
        super().accept()


# ---------------------------------------------------------------------------
# Value <-> text helpers
# ---------------------------------------------------------------------------


def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_to_text(v) for v in value) + "]"
    return str(value)


def _from_text(text, ptype, options):
    """Parse a default-value cell for the declared type.

    Text that does not parse is kept as-is: a default may legitimately be an
    expression over the diagram's workspace variables.
    """
    from lib.safe_eval import SafeEvalError, safe_expr

    text = (text or "").strip()
    if ptype in ("string", "choice"):
        if not text and ptype == "choice" and options:
            return options[0]
        return text
    if not text:
        return {"float": 0.0, "int": 0, "bool": False, "list": []}.get(ptype, "")
    if ptype == "bool":
        return text.strip().lower() in ("1", "true", "yes", "on")
    try:
        value = safe_expr(text, variables={}, allow_numpy=False)
    except (SafeEvalError, ValueError, SyntaxError, TypeError):
        return text
    if ptype == "int":
        try:
            return int(value)
        except (TypeError, ValueError):
            return text
    if ptype == "float":
        try:
            return float(value)
        except (TypeError, ValueError):
            return text
    if ptype == "list":
        if isinstance(value, (list, tuple)):
            return list(value)
        return text
    return value


def edit_block_mask(block, parent=None):
    """Open the editor for ``block`` and return the new mask, or None.

    The dialog is validated on accept, so a returned mask is always
    normalized and safe to hand to :func:`lib.masks.set_mask`.
    """
    dialog = MaskEditorDialog(block=block, parent=parent)
    if dialog.exec_() != QDialog.Accepted:
        return None
    try:
        return dialog.get_mask()
    except MaskError as exc:  # pragma: no cover - accept() already validated
        logger.error("Mask validation failed after accept: %s", exc)
        return None


__all__ = ["MaskEditorDialog", "edit_block_mask"]
