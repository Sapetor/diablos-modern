from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox
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
        self.setWindowTitle("Simulate")
        self.layout = QVBoxLayout()

        self.sim_time_input = QLineEdit(str(sim_time))
        self.sampling_time_input = QLineEdit(str(sim_dt))
        self.plot_range_input = QLineEdit(str(plot_trange))
        self.dynamic_plot_checkbox = QCheckBox("Dynamic Plot")
        self.real_time_checkbox = QCheckBox("Run in real-time")

        self.layout.addWidget(QLabel("Simulation Time"))
        self.layout.addWidget(self.sim_time_input)
        self.layout.addWidget(QLabel("Sampling Time"))
        self.layout.addWidget(self.sampling_time_input)
        self.layout.addWidget(QLabel("Time range Plot"))
        self.layout.addWidget(self.plot_range_input)
        self.layout.addWidget(self.dynamic_plot_checkbox)
        self.layout.addWidget(self.real_time_checkbox)

        button = QPushButton("Accept")
        button.clicked.connect(self.accept)
        self.layout.addWidget(button)

        self.setLayout(self.layout)

    def get_values(self):
        return {
            'sim_time': float(self.sim_time_input.text()),
            'sim_dt': float(self.sampling_time_input.text()),
            'plot_trange': float(self.plot_range_input.text()),
            'dynamic_plot': self.dynamic_plot_checkbox.isChecked(),
            'real_time': self.real_time_checkbox.isChecked()
        }