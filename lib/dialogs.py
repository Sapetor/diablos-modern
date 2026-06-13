from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QMessageBox
from PyQt5.QtCore import Qt
import logging

logger = logging.getLogger(__name__)

class ParamDialog(QDialog):
    def __init__(self, name, params, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)

        logger.debug(f"Initializing ParamDialog for {name}")
        self.params = params
        self.setWindowTitle(f"{name} Parameters")
        self.layout = QVBoxLayout()
        self.entries = {}

        for key, value in params.items():
            self.layout.addWidget(QLabel(key))
            if key in ['numerator', 'denominator']:
                entry = QLineEdit(', '.join(map(str, value)))
            else:
                entry = QLineEdit(str(value))
            self.layout.addWidget(entry)
            self.entries[key] = entry

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        self.layout.addWidget(ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        self.layout.addWidget(cancel_button)

        self.setLayout(self.layout)

    def get_values(self):
        values = {}
        for key, entry in self.entries.items():
            if key in ['numerator', 'denominator']:
                try:
                    values[key] = list(map(float, entry.text().split(',')))
                except ValueError:
                    pass # Ignore errors while typing
            else:
                values[key] = entry.text()
        return values

class PortDialog(QDialog):
    def __init__(self, name, params, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{name} Port Configuration")
        self.params = params
        self.layout = QVBoxLayout()
        self.entries = {}

        for key, value in params.items():
            self.layout.addWidget(QLabel(key))
            entry = QLineEdit(str(value))
            self.layout.addWidget(entry)
            self.entries[key] = entry

        button = QPushButton("OK")
        button.clicked.connect(self.accept)
        self.layout.addWidget(button)

        self.setLayout(self.layout)

    def get_values(self):
        return {key: entry.text() for key, entry in self.entries.items()}

class SimulationDialog(QDialog):
    # Methods offered in the solver dropdown. Fixed-step methods (Euler, RK4)
    # use the base step size; the rest are adaptive scipy.integrate solvers.
    SOLVER_METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "RK4", "Euler"]

    def __init__(self, sim_time, sim_dt, plot_trange, parent=None,
                 solver_method="RK45", rtol=1e-9, atol=1e-12):
        super().__init__(parent)
        from PyQt5.QtWidgets import QGroupBox  # Local import to avoid circular dep issues if any, or just convenience

        self.setWindowTitle("Simulation Configuration")
        self.resize(480, 420)
        self.layout = QVBoxLayout()

        # --- Solver Configuration Group ---
        solver_group = QGroupBox("Solver Configuration")
        solver_layout = QVBoxLayout()

        # Solver method
        solver_layout.addWidget(QLabel("Solver Method"))
        self.solver_method_combo = QComboBox()
        self.solver_method_combo.addItems(self.SOLVER_METHODS)
        idx = self.solver_method_combo.findText(str(solver_method))
        self.solver_method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        solver_layout.addWidget(self.solver_method_combo)

        method_hint = QLabel(
            "Adaptive: RK45 (default), RK23, DOP853; stiff: Radau, BDF, LSODA. "
            "Fixed-step (use the step size below): RK4, Euler."
        )
        method_hint.setObjectName("HintLabel")
        method_hint.setWordWrap(True)
        solver_layout.addWidget(method_hint)

        solver_layout.addSpacing(10)

        # Base Step Size
        solver_layout.addWidget(QLabel("Base Step Size (dt) [s]"))
        self.sampling_time_input = QLineEdit(str(sim_dt))
        solver_layout.addWidget(self.sampling_time_input)

        # Explanation Hint — themed via QSS, not inline styles
        hint_label = QLabel("Global solver step. Discrete blocks execute at their independent 'sampling_time' or synchronized to this step.")
        hint_label.setObjectName("HintLabel")
        hint_label.setWordWrap(True)
        solver_layout.addWidget(hint_label)

        solver_layout.addSpacing(10)

        # Simulation Time
        solver_layout.addWidget(QLabel("Simulation Duration [s]"))
        self.sim_time_input = QLineEdit(str(sim_time))
        solver_layout.addWidget(self.sim_time_input)

        # Tolerances (adaptive solvers only)
        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Rel. tol"))
        self.rtol_input = QLineEdit(str(rtol))
        tol_row.addWidget(self.rtol_input)
        tol_row.addWidget(QLabel("Abs. tol"))
        self.atol_input = QLineEdit(str(atol))
        tol_row.addWidget(self.atol_input)
        solver_layout.addLayout(tol_row)

        # Real-time Checkbox
        self.real_time_checkbox = QCheckBox("Run in real-time")
        solver_layout.addWidget(self.real_time_checkbox)

        solver_group.setLayout(solver_layout)
        self.layout.addWidget(solver_group)

        # --- Visualization Group ---
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        viz_layout.addWidget(QLabel("Plot Window Range [samples]"))
        self.plot_range_input = QLineEdit(str(plot_trange))
        viz_layout.addWidget(self.plot_range_input)
        
        self.dynamic_plot_checkbox = QCheckBox("Enable Dynamic Plotting")
        viz_layout.addWidget(self.dynamic_plot_checkbox)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        accept_btn = QPushButton("Simulate")
        accept_btn.clicked.connect(self.accept)
        accept_btn.setDefault(True)
        button_layout.addWidget(accept_btn)
        
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    # Numeric fields validated before the dialog is accepted. Maps the user-facing
    # label to the QLineEdit so validation errors can name the offending field.
    def _numeric_fields(self):
        return [
            ("Simulation Duration", self.sim_time_input),
            ("Base Step Size (dt)", self.sampling_time_input),
            ("Plot Window Range", self.plot_range_input),
            ("Rel. tol", self.rtol_input),
            ("Abs. tol", self.atol_input),
        ]

    def accept(self):
        # Validate all numeric fields before closing so get_values() (which calls
        # float() unconditionally) cannot raise into the caller's accept flow.
        invalid = []
        for label, entry in self._numeric_fields():
            try:
                float(entry.text())
            except (ValueError, TypeError):
                invalid.append(label)

        if invalid:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid number for: " + ", ".join(invalid) + ".",
            )
            return  # Keep the dialog open for correction.

        super().accept()

    def get_values(self):
        return {
            'sim_time': float(self.sim_time_input.text()),
            'sim_dt': float(self.sampling_time_input.text()),
            'plot_trange': float(self.plot_range_input.text()),
            'dynamic_plot': self.dynamic_plot_checkbox.isChecked(),
            'real_time': self.real_time_checkbox.isChecked(),
            'solver_method': self.solver_method_combo.currentText(),
            'rtol': float(self.rtol_input.text()),
            'atol': float(self.atol_input.text())
        }