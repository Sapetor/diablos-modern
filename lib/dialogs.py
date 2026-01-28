from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt5.QtCore import Qt
import logging

logger = logging.getLogger(__name__)

class ParamDialog(QDialog):
    def __init__(self, name, params):
        super().__init__()
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
    def __init__(self, name, params):
        super().__init__()
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
    def __init__(self, sim_time, sim_dt, plot_trange):
        super().__init__()
        from PyQt5.QtWidgets import QGroupBox, QFormLayout  # Local import to avoid circular dep issues if any, or just convenience
        
        self.setWindowTitle("Simulation Configuration")
        self.resize(480, 350)
        self.layout = QVBoxLayout()

        # --- Solver Configuration Group ---
        solver_group = QGroupBox("Solver Configuration")
        solver_layout = QVBoxLayout()
        
        # Base Step Size
        solver_layout.addWidget(QLabel("Base Step Size (dt) [s]"))
        self.sampling_time_input = QLineEdit(str(sim_dt))
        solver_layout.addWidget(self.sampling_time_input)
        
        # Explanation Hint
        hint_label = QLabel("<i>Global solver step. Discrete blocks execute at their<br>independent 'sampling_time' or synchronized to this step.</i>")
        hint_label.setStyleSheet("color: gray; font-size: 11px;")
        hint_label.setWordWrap(True)
        solver_layout.addWidget(hint_label)
        
        solver_layout.addSpacing(10)
        
        # Simulation Time
        solver_layout.addWidget(QLabel("Simulation Duration [s]"))
        self.sim_time_input = QLineEdit(str(sim_time))
        solver_layout.addWidget(self.sim_time_input)
        
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
        
        accept_btn = QPushButton("Start Simulation")
        accept_btn.clicked.connect(self.accept)
        accept_btn.setDefault(True)
        button_layout.addWidget(accept_btn)
        
        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

    def get_values(self):
        return {
            'sim_time': float(self.sim_time_input.text()),
            'sim_dt': float(self.sampling_time_input.text()),
            'plot_trange': float(self.plot_range_input.text()),
            'dynamic_plot': self.dynamic_plot_checkbox.isChecked(),
            'real_time': self.real_time_checkbox.isChecked()
        }