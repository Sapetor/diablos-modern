"""
Control System Analysis Module
Extracts Bode Plot and Root Locus logic from the UI.
"""

import logging
import numpy as np
from scipy import signal
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

class ControlSystemAnalyzer:
    """
    Handles control system analysis (Bode, Root Locus) for DiaBloS Blocks.
    Separates scientific calculation and plotting from the connection canvas.
    """

    def __init__(self, dsim, parent=None):
        """
        Initialize the analyzer.

        Args:
            dsim: The simulation controller class (DSim/SimulationEngine interface)
            parent: Optional parent widget (usually the main window or canvas) for message boxes
        """
        self.dsim = dsim
        self.parent = parent
        self.plot_windows = []  # Keep references to windows to prevent GC

    def _find_connected_transfer_function(self, target_block):
        """
        Find the Transfer Function block connected to the input of the target block.

        Args:
            target_block: The block (BodeMag/RootLocus) checking for input.

        Returns:
            The source DBlock instance, or None if not found/invalid.
        """
        source_block = None
        # Logic matches original modern_canvas implementation
        for line in self.dsim.line_list:
            if line.dstblock == target_block.name:
                for block in self.dsim.blocks_list:
                    if block.name == line.srcblock:
                        if block.block_fn == 'TranFn':
                            source_block = block
                            break
                if source_block:
                    break
        return source_block

    def generate_bode_plot(self, bode_block):
        """
        Find the connected transfer function, calculate, and plot the Bode diagram.
        """
        source_block = self._find_connected_transfer_function(bode_block)
        
        if not source_block:
            if self.parent:
                QMessageBox.warning(self.parent, "Bode Plot Error", 
                                  "BodeMag block must be connected to the output of a Transfer Function block.")
            return

        # Get numerator and denominator
        num = source_block.params.get('numerator')
        den = source_block.params.get('denominator')

        if not num or not den:
            if self.parent:
                QMessageBox.warning(self.parent, "Bode Plot Error", 
                                  "Connected Transfer Function has invalid parameters.")
            return

        try:
            # Calculate Bode plot data
            w, mag, phase = signal.bode((num, den))

            # Display the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Bode Plot: {source_block.name}")
            layout = QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)

            plot_widget.setLogMode(x=True, y=False)
            plot_widget.setLabel('left', 'Magnitude', units='dB')
            plot_widget.setLabel('bottom', 'Frequency', units='rad/s')
            plot_widget.setTitle(f"Bode Magnitude Plot: {source_block.name}")
            plot_widget.plot(w, mag)
            plot_widget.showGrid(x=True, y=True)

            self.plot_windows.append(plot_window)
            plot_window.show()
            
        except Exception as e:
            error_msg = f"Error generating Bode plot: {str(e)}"
            logger.error(error_msg)
            if self.parent:
                QMessageBox.critical(self.parent, "Bode Plot Error", error_msg)

    def generate_root_locus(self, rootlocus_block):
        """
        Find the connected transfer function, calculate, and plot the root locus.
        """
        source_block = self._find_connected_transfer_function(rootlocus_block)

        if not source_block:
            if self.parent:
                QMessageBox.warning(self.parent, "Root Locus Error", 
                                  "RootLocus block must be connected to the output of a Transfer Function block.")
            return

        # Get numerator and denominator
        num = source_block.params.get('numerator')
        den = source_block.params.get('denominator')

        if not num or not den:
            if self.parent:
                QMessageBox.warning(self.parent, "Root Locus Error", 
                                  "Connected Transfer Function has invalid parameters.")
            return

        try:
            # Ensure num and den are numpy arrays
            num = np.atleast_1d(num)
            den = np.atleast_1d(den)

            # Pad numerator to same length as denominator
            if len(num) < len(den):
                num = np.pad(num, (len(den) - len(num), 0), 'constant')
            elif len(den) < len(num):
                den = np.pad(den, (len(num) - len(den), 0), 'constant')

            # Calculate root locus for gains
            k_values = np.concatenate([
                np.linspace(0, 1, 150),           # Fine detail near K=0
                np.linspace(1, 10, 150),          # Fine detail 1-10
                np.logspace(1, 4, 400)            # Logarithmic from 10 to 10000
            ])

            all_roots = []
            for k in k_values:
                # Characteristic equation: den(s) + k*num(s) = 0
                char_poly = den + k * num
                roots = np.roots(char_poly)
                all_roots.append(roots)

            all_roots = np.array(all_roots)

            # Track branches and sort roots
            num_poles = all_roots.shape[1]
            sorted_roots = [all_roots[0]]
            
            for i in range(1, len(all_roots)):
                current = all_roots[i]
                previous = sorted_roots[-1]
                matched = []
                available = list(range(len(current)))

                for prev_root in previous:
                    if not available:
                        break
                    distances = [abs(current[j] - prev_root) for j in available]
                    closest_idx = available[np.argmin(distances)]
                    matched.append(current[closest_idx])
                    available.remove(closest_idx)

                for idx in available:
                    matched.append(current[idx])

                sorted_roots.append(np.array(matched))

            sorted_roots = np.array(sorted_roots)

            # Display the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Root Locus: {source_block.name}")
            layout = QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)

            plot_widget.setLabel('left', 'Imaginary Axis', units='rad/s')
            plot_widget.setLabel('bottom', 'Real Axis', units='1/s')
            plot_widget.setTitle(f"Root Locus Plot: {source_block.name}")

            colors = [(100, 149, 237), (237, 100, 100), (100, 237, 149),
                      (237, 149, 100), (149, 100, 237), (237, 237, 100)]

            for pole_idx in range(num_poles):
                branch_real = sorted_roots[:, pole_idx].real
                branch_imag = sorted_roots[:, pole_idx].imag
                color = colors[pole_idx % len(colors)]
                plot_widget.plot(branch_real, branch_imag,
                               pen=pg.mkPen(color, width=2),
                               name=f'Branch {pole_idx+1}' if pole_idx == 0 else None)

            # Mark open-loop poles
            poles = np.roots(den)
            plot_widget.plot(poles.real, poles.imag, pen=None, symbol='x', symbolSize=12,
                           symbolPen=pg.mkPen('r', width=3), name='Poles')

            # Mark open-loop zeros
            if np.any(num):
                zeros = np.roots(num)
                if len(zeros) > 0:
                    plot_widget.plot(zeros.real, zeros.imag, pen=None, symbol='o', symbolSize=12,
                                   symbolPen=pg.mkPen('g', width=3), symbolBrush=None, name='Zeros')

            plot_widget.addLine(x=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
            plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()

            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            error_msg = f"Error calculating root locus: {str(e)}"
            logger.error(error_msg)
            if self.parent:
                QMessageBox.critical(self.parent, "Root Locus Error", error_msg)
