"""
Preferences dialog (File > Preferences...).

Gathers the settings that used to be scattered through the View menu (theme,
block palette, solid fills, UI scale, language, default connection routing) plus
the persistent simulation preference, in one place.

The dialog stores nothing itself. Every control calls the *same* main-window
method the old menu action called, so persistence is unchanged:

========================  =====================================================
Setting                   Applied through
========================  =====================================================
Theme                     ``window.toggle_theme()`` (user_preferences.json)
Block palette             ``window._set_palette(key)`` (user_preferences.json)
Solid block fills         ``window._toggle_solid_fills(bool)`` (same file)
UI scale                  ``window._set_scaling(factor)`` (config + restart note)
Language                  ``window.set_language(code)`` (QSettings ``ui/language``)
Default routing           ``window._set_default_routing_mode(mode)``
Ask before every run      ``lib.sim_prefs`` (QSettings ``simulation/ask_before_run``)
========================  =====================================================

Changes apply immediately, as the menu entries did; the dialog just has Close.
"""

import json
import logging

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lib.i18n import SYSTEM_LANGUAGE, available_languages, stored_language_setting, tr, tr_noop
from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES, ThemeType, theme_manager

logger = logging.getLogger(__name__)

SCALE_CHOICES = (1.0, 1.25, 1.5)
ROUTING_CHOICES = (
    ("bezier", tr_noop("Bezier (Curved)")),
    ("orthogonal", tr_noop("Orthogonal (Manhattan)")),
)


def _stored_scaling() -> float:
    """The scaling factor the app will use on next start (user config, else 1.0)."""
    from lib.app_paths import user_data_path

    try:
        with open(user_data_path("config/default_config.json"), "r") as f:
            return float(json.load(f).get("display", {}).get("scaling_factor", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setEnabled(False)  # dimmed secondary text
    return label


class PreferencesDialog(QDialog):
    """Sectioned preferences; each control writes through the window's own setter."""

    def __init__(self, window):
        super().__init__(window)
        self.window_ref = window
        self.setWindowTitle(tr("Preferences"))
        self.setMinimumSize(560, 360)
        self.setModal(True)

        root = QVBoxLayout(self)
        body = QHBoxLayout()
        self.sections = QListWidget()
        self.sections.setFixedWidth(150)
        self.pages = QStackedWidget()
        body.addWidget(self.sections)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)

        for title, builder in (
            (tr("Appearance"), self._build_appearance),
            (tr("Language"), self._build_language),
            (tr("Editing"), self._build_editing),
            (tr("Simulation"), self._build_simulation),
        ):
            self.sections.addItem(title)
            self.pages.addWidget(builder())
        self.sections.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.sections.setCurrentRow(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    # -- pages ---------------------------------------------------------------

    def _page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        outer.addLayout(form)
        outer.addStretch(1)
        return page, form, outer

    def _build_appearance(self):
        page, form, outer = self._page()
        win = self.window_ref

        self.theme_combo = QComboBox()
        self.theme_combo.addItem(tr("Light"), ThemeType.LIGHT)
        self.theme_combo.addItem(tr("Dark"), ThemeType.DARK)
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(theme_manager.current_theme))
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        form.addRow(tr("Theme"), self.theme_combo)

        self.palette_combo = QComboBox()
        for key, display in PALETTE_DISPLAY_NAMES.items():
            self.palette_combo.addItem(display, key)
        self.palette_combo.setCurrentIndex(
            self.palette_combo.findData(theme_manager.current_palette)
        )
        self.palette_combo.currentIndexChanged.connect(
            lambda _i: win._set_palette(self.palette_combo.currentData())
        )
        form.addRow(tr("Block palette"), self.palette_combo)

        self.solid_fills_check = QCheckBox(tr("Solid Block Fills"))
        self.solid_fills_check.setChecked(theme_manager.solid_fills)
        self.solid_fills_check.toggled.connect(win._toggle_solid_fills)
        form.addRow("", self.solid_fills_check)

        self.scale_combo = QComboBox()
        for factor in SCALE_CHOICES:
            self.scale_combo.addItem(f"{int(round(factor * 100))}%", factor)
        stored = _stored_scaling()
        idx = self.scale_combo.findData(stored)
        self.scale_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.scale_combo.currentIndexChanged.connect(
            lambda _i: win._set_scaling(self.scale_combo.currentData())
        )
        form.addRow(tr("UI Scale"), self.scale_combo)
        outer.insertWidget(
            1, _note(tr("The UI scale takes effect after restarting the application."))
        )
        return page

    def _build_language(self):
        page, form, outer = self._page()
        win = self.window_ref

        self.language_combo = QComboBox()
        self.language_combo.addItem(tr("System default"), SYSTEM_LANGUAGE)
        for entry in available_languages():
            # Native names are deliberately not translated: a language is listed
            # in its own language so users can always find theirs.
            self.language_combo.addItem(entry["name"], entry["code"])
        idx = self.language_combo.findData(stored_language_setting())
        self.language_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.language_combo.currentIndexChanged.connect(
            lambda _i: win.set_language(self.language_combo.currentData())
        )
        form.addRow(tr("Language"), self.language_combo)
        outer.insertWidget(
            1, _note(tr("Open windows keep the previous language until they are reopened."))
        )
        return page

    def _build_editing(self):
        page, form, _outer = self._page()
        win = self.window_ref

        self.routing_combo = QComboBox()
        for mode, label in ROUTING_CHOICES:
            self.routing_combo.addItem(tr(label), mode)
        current = getattr(win, "default_routing_mode", "bezier")
        self.routing_combo.setCurrentIndex(max(0, self.routing_combo.findData(current)))
        self.routing_combo.currentIndexChanged.connect(
            lambda _i: win._set_default_routing_mode(self.routing_combo.currentData())
        )
        form.addRow(tr("Default Connection Routing"), self.routing_combo)
        return page

    def _build_simulation(self):
        page, form, _outer = self._page()
        from lib import sim_prefs

        self.ask_before_run_check = QCheckBox(tr("Ask for simulation settings before every run"))
        self.ask_before_run_check.setChecked(sim_prefs.ask_before_run())
        self.ask_before_run_check.toggled.connect(self._on_ask_before_run)
        form.addRow("", self.ask_before_run_check)
        return page

    # -- handlers ------------------------------------------------------------

    def _on_theme(self, _index):
        wanted = self.theme_combo.currentData()
        if wanted != theme_manager.current_theme:
            self.window_ref.toggle_theme()

    def _on_ask_before_run(self, checked):
        # Same two writes lib.lib.DSim.apply_settings performs.
        from lib import sim_prefs

        dsim = getattr(self.window_ref, "dsim", None)
        if dsim is not None:
            dsim.ask_before_run = bool(checked)
        sim_prefs.set_ask_before_run(bool(checked))
