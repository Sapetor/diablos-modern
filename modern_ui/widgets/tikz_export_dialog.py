"""
TikZ Export Dialog - Configure and preview TikZ diagram export.

Provides format selection (standalone vs snippet), export options,
live preview, clipboard copy, and file export.

Layout: options panel (left) | live preview (right) using QSplitter.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QCheckBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QTextEdit, QMessageBox, QApplication, QSplitter,
    QDialogButtonBox, QWidget, QFrame, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class TikZExportDialog(QDialog):
    """Dialog for configuring and exporting TikZ diagram code."""

    def __init__(self, blocks_list, line_list, parent=None):
        super().__init__(parent)
        self.blocks_list = blocks_list
        self.line_list = line_list

        self.setWindowTitle("Export as TikZ")
        self.setMinimumSize(820, 520)
        self.resize(900, 580)
        self.setModal(True)

        self._setup_ui()
        self._update_preview()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # --- Compact info header ---
        from lib.export.tikz_exporter import TikZExporter
        exporter = TikZExporter(self.blocks_list, self.line_list)
        info = exporter.get_info()
        info_text = (
            f"{info['block_count']} blocks  \u00b7  "
            f"{info['connection_count']} connections  \u00b7  "
            f"{', '.join(info['block_types'])}"
        )
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: gray; padding: 2px 0;")
        root.addWidget(info_label)

        # --- Splitter: options (left) | preview (right) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -- Left panel: scrollable options --
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 6, 0)
        left.setSpacing(10)

        # Format
        fmt_group = QGroupBox("Format")
        fmt_layout = QHBoxLayout(fmt_group)
        fmt_layout.setContentsMargins(8, 4, 8, 4)

        self.format_btn_group = QButtonGroup(self)
        self.standalone_radio = QRadioButton("Standalone .tex")
        self.snippet_radio = QRadioButton("Snippet only")
        self.standalone_radio.setChecked(True)
        self.format_btn_group.addButton(self.standalone_radio, 0)
        self.format_btn_group.addButton(self.snippet_radio, 1)

        fmt_layout.addWidget(self.standalone_radio)
        fmt_layout.addWidget(self.snippet_radio)
        fmt_layout.addStretch()
        left.addWidget(fmt_group)

        # Diagram Style
        style_group = QGroupBox("Diagram Style")
        style_layout = QVBoxLayout(style_group)
        style_layout.setContentsMargins(8, 4, 8, 4)
        style_layout.setSpacing(3)

        self.source_as_arrow_cb = QCheckBox("Sources as input arrows")
        self.source_as_arrow_cb.setChecked(True)

        self.sink_as_arrow_cb = QCheckBox("Sinks as output arrows")
        self.sink_as_arrow_cb.setChecked(True)

        self.show_signal_labels_cb = QCheckBox("Signal labels on connections")
        self.show_signal_labels_cb.setChecked(True)

        self.fill_blocks_cb = QCheckBox("Category fill colors")
        self.fill_blocks_cb.setChecked(True)

        for cb in (self.source_as_arrow_cb, self.sink_as_arrow_cb,
                   self.show_signal_labels_cb, self.fill_blocks_cb):
            style_layout.addWidget(cb)

        left.addWidget(style_group)

        # Content
        content_group = QGroupBox("Content")
        content_layout = QVBoxLayout(content_group)
        content_layout.setContentsMargins(8, 4, 8, 4)
        content_layout.setSpacing(3)

        self.show_usernames_cb = QCheckBox("Block labels")
        self.show_usernames_cb.setChecked(True)

        self.show_values_cb = QCheckBox("Parameter values")
        self.show_values_cb.setChecked(True)

        self.include_sinks_cb = QCheckBox("Include sink blocks")
        self.include_sinks_cb.setChecked(True)

        for cb in (self.show_usernames_cb, self.show_values_cb,
                   self.include_sinks_cb):
            content_layout.addWidget(cb)

        left.addWidget(content_group)

        # LaTeX
        latex_group = QGroupBox("LaTeX")
        latex_layout = QFormLayout(latex_group)
        latex_layout.setContentsMargins(8, 4, 8, 4)
        latex_layout.setSpacing(4)

        self.use_resizebox_cb = QCheckBox(r"\resizebox{\textwidth}{!}{...}")
        self.use_resizebox_cb.setChecked(False)

        self.page_width_spin = QDoubleSpinBox()
        self.page_width_spin.setRange(4.0, 30.0)
        self.page_width_spin.setValue(14.0)
        self.page_width_spin.setSuffix(" cm")
        self.page_width_spin.setSingleStep(1.0)

        latex_layout.addRow(self.use_resizebox_cb)
        latex_layout.addRow("Page width:", self.page_width_spin)
        left.addWidget(latex_group)

        # Output path
        out_group = QGroupBox("Output")
        out_layout = QHBoxLayout(out_group)
        out_layout.setContentsMargins(8, 4, 8, 4)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output file...")
        default_path = os.path.join(os.path.expanduser("~"), "diagram.tex")
        self.path_edit.setText(default_path)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setFixedWidth(72)
        self.browse_btn.clicked.connect(self._browse_file)

        out_layout.addWidget(self.path_edit, 1)
        out_layout.addWidget(self.browse_btn)
        left.addWidget(out_group)

        left.addStretch()
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # -- Right panel: preview --
        preview_frame = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(4, 4, 4, 4)

        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        mono = QFont("Menlo", 10)
        mono.setStyleHint(QFont.Monospace)
        self.preview_edit.setFont(mono)
        self.preview_edit.setLineWrapMode(QTextEdit.NoWrap)

        preview_layout.addWidget(self.preview_edit)
        splitter.addWidget(preview_frame)

        # Splitter proportions: ~35% options, ~65% preview
        splitter.setSizes([310, 580])
        root.addWidget(splitter, 1)

        # --- Button bar ---
        btn_bar = QHBoxLayout()

        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._copy_to_clipboard)
        btn_bar.addWidget(self.copy_btn)
        btn_bar.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.Cancel | QDialogButtonBox.Save
        )
        button_box.button(QDialogButtonBox.Save).setText("Export")
        button_box.rejected.connect(self.reject)
        button_box.accepted.connect(self._export)
        btn_bar.addWidget(button_box)

        root.addLayout(btn_bar)

        # --- Wire option changes to preview ---
        self.format_btn_group.buttonClicked.connect(self._update_preview)
        for cb in (self.include_sinks_cb, self.sink_as_arrow_cb,
                   self.source_as_arrow_cb, self.show_usernames_cb,
                   self.show_values_cb, self.show_signal_labels_cb,
                   self.fill_blocks_cb, self.use_resizebox_cb):
            cb.stateChanged.connect(self._update_preview)
        self.page_width_spin.valueChanged.connect(self._update_preview)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_options(self):
        return {
            'include_sinks': self.include_sinks_cb.isChecked(),
            'sink_as_arrow': self.sink_as_arrow_cb.isChecked(),
            'source_as_arrow': self.source_as_arrow_cb.isChecked(),
            'show_usernames': self.show_usernames_cb.isChecked(),
            'show_values': self.show_values_cb.isChecked(),
            'show_signal_labels': self.show_signal_labels_cb.isChecked(),
            'fill_blocks': self.fill_blocks_cb.isChecked(),
            'use_resizebox': self.use_resizebox_cb.isChecked(),
            'page_width_cm': self.page_width_spin.value(),
        }

    def _generate_tikz(self):
        from lib.export.tikz_exporter import TikZExporter
        exporter = TikZExporter(self.blocks_list, self.line_list)
        options = self._get_options()
        if self.standalone_radio.isChecked():
            return exporter.export_document(options)
        return exporter.export_snippet(options)

    def _update_preview(self, *_args):
        try:
            tikz_code = self._generate_tikz()
            self.preview_edit.setPlainText(tikz_code)
        except Exception as e:
            logger.error(f"TikZ preview error: {e}")
            self.preview_edit.setPlainText(f"% Error generating preview:\n% {e}")

    def _browse_file(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save TikZ File", self.path_edit.text(),
            "TeX files (*.tex);;All files (*)"
        )
        if filepath:
            if not filepath.lower().endswith('.tex'):
                filepath += '.tex'
            self.path_edit.setText(filepath)

    def _copy_to_clipboard(self):
        tikz_code = self._generate_tikz()
        clipboard = QApplication.clipboard()
        clipboard.setText(tikz_code)
        QMessageBox.information(self, "Copied", "TikZ code copied to clipboard.")

    def _export(self):
        filepath = self.path_edit.text().strip()
        if not filepath:
            QMessageBox.warning(self, "Error", "Please specify an output file path.")
            return

        try:
            tikz_code = self._generate_tikz()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(tikz_code)
            QMessageBox.information(
                self, "Export Complete",
                f"TikZ diagram exported to:\n{filepath}"
            )
            self.accept()
        except Exception as e:
            logger.error(f"TikZ export error: {e}")
            QMessageBox.critical(self, "Export Failed", str(e))
