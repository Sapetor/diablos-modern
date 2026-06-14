"""Results window for the "Find Operating Point (Trim)" feature.

``OperatingPointWindow`` is a pure *view*: it CONSUMES the headless trim
result-dict produced by :meth:`AnalysisController.find_trim` and renders the
equilibrium state values in a table, with a copy-to-clipboard action. It
performs no solving of its own.

Result-dict contract::

    result = {
      "ok": bool, "error": str,
      "success": bool, "message": str,
      "residual_norm": float|None,
      "states": [{"name": str, "value": float}],
      "operating_point": {block_name: value},
      "summary": str,
    }
"""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QApplication,
    QHeaderView,
)
from PyQt5.QtCore import Qt

from modern_ui.themes.theme_manager import theme_manager, TYPE


class OperatingPointWindow(QWidget):
    """Window presenting an operating-point (trim) result."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result or {}

        self.setWindowTitle("Operating Point (Trim)")
        self.resize(480, 460)

        layout = QVBoxLayout()
        self.setLayout(layout)

        if not self.result.get("ok", False):
            self._build_error_view(layout)
            return

        # Summary header (convergence, residual, state count).
        summary = self.result.get("summary") or ""
        header = QLabel(summary)
        header.setWordWrap(True)
        ok = self.result.get("success")
        color = theme_manager.get_color('success' if ok else 'warning').name()
        header.setStyleSheet(f"color: {color}; font-size: {TYPE['body']}pt; padding: 4px;")
        layout.addWidget(header)

        # State table.
        states = self.result.get("states") or []
        table = QTableWidget(len(states), 2)
        table.setHorizontalHeaderLabels(["State", "Value"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row, st in enumerate(states):
            name_item = QTableWidgetItem(str(st.get("name", f"x{row}")))
            val_item = QTableWidgetItem(f"{float(st.get('value', 0.0)):.6g}")
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, 0, name_item)
            table.setItem(row, 1, val_item)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)

        if not states:
            layout.addWidget(QLabel("No continuous states in this diagram."))

        # Copy button.
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        copy_btn = QPushButton("Copy (Python dict)")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        btn_row.addWidget(copy_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ error
    def _build_error_view(self, layout):
        msg = self.result.get("error") or "Operating-point search failed."
        label = QLabel(str(msg))
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"color: {theme_manager.get_color('error').name()}; "
            f"font-size: {TYPE['body_strong']}pt; padding: 24px;"
        )
        layout.addWidget(label)

    # ----------------------------------------------------------------- helpers
    def _copy_to_clipboard(self):
        """Copy the operating point as a Python dict literal {name: value}."""
        op = self.result.get("operating_point") or {}
        if not op:
            op = {st.get("name"): st.get("value")
                  for st in (self.result.get("states") or [])}
        items = []
        for k, v in op.items():
            if isinstance(v, (list, tuple)):
                v_str = "[" + ", ".join(f"{float(x):.6g}" for x in v) + "]"
            else:
                v_str = f"{float(v):.6g}"
            items.append(f"{k!r}: {v_str}")
        text = "{" + ", ".join(items) + "}"
        QApplication.clipboard().setText(text)
