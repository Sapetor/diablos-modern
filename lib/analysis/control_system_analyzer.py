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
        Find the dynamic block connected to the input of the target block.
        Modified to support Transfer Functions, State Space, PID, etc.
        """
        source_block = None
        # Logic matches original modern_canvas implementation
        allowed_types = ['TranFn', 'DiscreteTranFn', 'StateSpace', 'DiscreteStateSpace', 'PID']
        
        logger.info(f"Looking for source block for {target_block.name}")
        logger.info(f"  Line list size: {len(self.dsim.line_list)}")
        
        for line in self.dsim.line_list:
            # logger.info(f"  Checking line: {line.srcblock} -> {line.dstblock}")
            if line.dstblock == target_block.name:
                logger.info(f"  Found incoming line from {line.srcblock}")
                for block in self.dsim.blocks_list:
                    if block.name == line.srcblock:
                        logger.info(f"  Source block fn: {block.block_fn}")
                        if block.block_fn in allowed_types:
                            source_block = block
                            logger.info(f"  Match found: {source_block.name}")
                            break
                        else:
                            logger.info(f"  Block type {block.block_fn} not in allowed types")
                if source_block:
                    break
        
        if not source_block:
            logger.warning(f"No valid source block found for {target_block.name}")
            
        return source_block

    def _position_window(self, window):
        """Position the plot window near the mouse cursor."""
        try:
            from PyQt5.QtGui import QCursor
            cursor_pos = QCursor.pos()
            logger.info(f"Positioning window at cursor: {cursor_pos.x()}, {cursor_pos.y()}")
            
            # Position slightly offset from cursor
            # Ensure we resize first so geometry is valid? (optional but good practice)
            window.resize(600, 400) # Ensure a default size
            
            new_x = cursor_pos.x() + 20
            new_y = cursor_pos.y() + 20
            
            window.move(new_x, new_y)
            logger.info(f"Window moved to: {new_x}, {new_y}")
            
        except Exception as e:
            logger.warning(f"Could not position window: {e}")
            if self.parent:
                try:
                    geo = self.parent.window().frameGeometry()
                    window.move(geo.center() - window.rect().center())
                except:
                    pass

    def generate_bode_plot(self, bode_block):
        """
        Find the connected block, extract system model, and plot the Bode diagram.
        """
        logger.info(f"Generating Bode Plot for {bode_block.name}")
        source_block = self._find_connected_transfer_function(bode_block)
        
        if not source_block:
            if self.parent:
                QMessageBox.warning(self.parent, "Bode Plot Error", 
                                  "BodeMag block must be connected to a supported dynamic block.")
            return

        try:
            # Extract system model based on block type
            system = None
            title = f"Bode Magnitude Plot: {source_block.name}"
            
            # 1. Continuous Transfer Function
            if source_block.block_fn == 'TranFn':
                num = source_block.params.get('numerator')
                den = source_block.params.get('denominator')
                if num and den:
                    system = signal.TransferFunction(num, den)

            # 2. Discrete Transfer Function
            elif source_block.block_fn == 'DiscreteTranFn':
                num = source_block.params.get('numerator')
                den = source_block.params.get('denominator')
                # Check for 'sampling_time' (standard) or 'dt' (legacy/alt)
                st = source_block.params.get('sampling_time', -1.0)
                if st <= 0:
                    st = source_block.params.get('dt', 1.0)
                
                # If still <= 0 (e.g. inherited -1), default to 1.0 for analysis
                # This matches standard MATLAB behavior for unspecified/inherited analysis
                if st <= 0:
                    st = 1.0
                    
                if num and den:
                     # Scipy's dlti supports (num, den, dt=dt)
                    system = signal.TransferFunction(num, den, dt=st)

            # 3. State Space (Continuous)
            elif source_block.block_fn == 'StateSpace':
                A = source_block.params.get('A')
                B = source_block.params.get('B')
                C = source_block.params.get('C')
                D = source_block.params.get('D')
                if all(v is not None for v in [A, B, C, D]):
                    system = signal.StateSpace(A, B, C, D)

            # 4. Discrete State Space
            elif source_block.block_fn == 'DiscreteStateSpace':
                A = source_block.params.get('A')
                B = source_block.params.get('B')
                C = source_block.params.get('C')
                D = source_block.params.get('D')
                
                st = source_block.params.get('sampling_time', -1.0)
                if st <= 0:
                     st = source_block.params.get('dt', 1.0)
                
                if st <= 0:
                     st = 1.0
                    
                if all(v is not None for v in [A, B, C, D]):
                    system = signal.StateSpace(A, B, C, D, dt=st)
            
            # 5. PID Controller
            elif source_block.block_fn == 'PID':
                kp = source_block.params.get('kp', 1.0)
                ki = source_block.params.get('ki', 0.0)
                kd = source_block.params.get('kd', 0.0)
                # PID = Kp + Ki/s + Kd*s = (Kd*s^2 + Kp*s + Ki) / s
                num = [kd, kp, ki]
                den = [1, 0]
                system = signal.TransferFunction(num, den)

            if system is None:
                raise ValueError(f"Could not extract valid system model from {source_block.name}")

            # Calculate Frequency Response
            # For discrete systems, generate log-spaced frequencies up to Nyquist
            # to ensure the plot looks good on a log-x scale.
            # Default scipy behavior is linear spacing for discrete, which looks bad on log plots.
            w_vector = None
            nyquist_freq = None
            
            if hasattr(system, 'dt') and system.dt is not None:
                # Discrete System
                dt = system.dt
                if dt > 0:
                    nyquist_freq = np.pi / dt
                    # 3 - 4 decades below Nyquist is usually sufficient
                    w_vector = np.logspace(np.log10(nyquist_freq) - 4, np.log10(nyquist_freq), num=500)
            
            # Calculate Bode
            # Note: signal.bode returns w, mag, phase
            w, mag, phase = system.bode(w=w_vector)


            # Filter out zero or negative frequencies for log plot
            valid_indices = w > 0
            w = w[valid_indices]
            mag = mag[valid_indices]
            phase = phase[valid_indices]

            if len(w) == 0:
                 raise ValueError("No valid positive frequencies to plot.")

            # Display the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Bode Plot: {source_block.name}")
            
            # Use specific layout for potentially adding Phase later, but for now just Mag
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w') # White background for scientific plots often looks cleaner/standard
            
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)

            # Style the plot
            plot_widget.setLogMode(x=True, y=False)
            
            # styling axes
            styles = {'color': 'k', 'font-size': '10pt'}
            plot_widget.setLabel('left', 'Magnitude', units='dB', **styles)
            # Disable auto SI units to prevent confusion (e.g. krad/s)
            plot_widget.setLabel('bottom', 'Frequency (rad/s)', **styles)
            plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
            
            plot_widget.setTitle(title, color='k', size='12pt')
            
            # Grid
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Curve
            # Use a slightly thicker blue line
            pen = pg.mkPen(color=(0, 114, 189), width=2.5) # Matlab-like blue
            plot_widget.plot(w, mag, pen=pen)
            
            logger.info(f"Bode Plot Generated: {source_block.name}")
            logger.info(f"  Num: {num}, Den: {den}, St: {st}")
            logger.info(f"  Nyquist: {nyquist_freq:.4f} rad/s" if nyquist_freq else "  Continuous")

            ax_left = plot_widget.getAxis('left')
            ax_bottom = plot_widget.getAxis('bottom')
            ax_bottom.setPen('k') 
            ax_left.setPen('k')
            ax_bottom.setTextPen('k')
            ax_bottom.setTextPen('k')
            ax_left.setTextPen('k')
            
            # self._position_window(plot_window) # Removed to match Nyquist/Phase behavior
            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            error_msg = f"Error generating Bode plot: {str(e)}"
            logger.error(error_msg)
            if self.parent:
                QMessageBox.critical(self.parent, "Error", error_msg)

    def generate_bode_phase_plot(self, block):
        """
        Generate Bode Phase plot.
        """
        logger.info(f"Generating Bode Phase Plot for {block.name}")
        source_block = self._find_connected_transfer_function(block)
        if not source_block:
             if self.parent:
                QMessageBox.warning(self.parent, "Bode Phase Error", "Block must be connected to a supported dynamic block.")
             return

        try:
            logger.info(f"Extracting system model from {source_block.name}")
            system = self._extract_system_model(source_block)
            
            # Use log-spaced w for discrete systems
            w_vector = self._get_frequency_vector(system)
            
            w, mag, phase = system.bode(w=w_vector)
            
            # Filter valid w
            valid = w > 0
            w = w[valid]
            phase = phase[valid]
            
            if len(w) == 0: raise ValueError("No valid frequencies.")

            # Display
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Bode Phase Plot: {source_block.name}")
            layout = QVBoxLayout()
            layout.setContentsMargins(0,0,0,0)
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)
            
            plot_widget.setLogMode(x=True, y=False)
            
            styles = {'color': 'k', 'font-size': '10pt'}
            plot_widget.setLabel('left', 'Phase', units='deg', **styles)
            plot_widget.setLabel('bottom', 'Frequency (rad/s)', **styles)
            plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
            plot_widget.setTitle(f"Bode Phase Plot: {source_block.name}", color='k', size='12pt')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            plot_widget.plot(w, phase, pen=pg.mkPen(color=(0, 114, 189), width=2.5))
            
            # Styling
            for ax_name in ['left', 'bottom']:
                ax = plot_widget.getAxis(ax_name)
                ax.setPen('k')
                ax.setTextPen('k')

            # Nyquist line for discrete
            if hasattr(system, 'dt') and system.dt and system.dt > 0:
                 nyq_freq = np.pi/system.dt
                 line = pg.InfiniteLine(pos=np.log10(nyq_freq), angle=90, pen=pg.mkPen('r', style=Qt.DashLine))
                 plot_widget.addItem(line)

            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            logger.error(f"Error generating Bode Phase: {e}")
            if self.parent: QMessageBox.critical(self.parent, "Error", str(e))

    def generate_nyquist_plot(self, block):
        """
        Generate Nyquist plot (Im vs Re).
        """
        logger.info(f"Generating Nyquist Plot for {block.name}")
        source_block = self._find_connected_transfer_function(block)
        if not source_block:
             if self.parent:
                QMessageBox.warning(self.parent, "Nyquist Error", "Block must be connected to a supported dynamic block.")
             return

        try:
            logger.info(f"Extracting system model from {source_block.name}")
            system = self._extract_system_model(source_block)
            
            # Frequency vector
            w_vector = self._get_frequency_vector(system, default_pts=1000)
            if w_vector is None:
                # Continuous default
                w_vector = np.logspace(-2, 3, 1000)

            # Calculate complex frequency response
            if hasattr(system, 'freqresp'):
                w, H = system.freqresp(w=w_vector)
            else:
                # Fallback for systems that might not implement freqresp directly (though scipy should)
                # w, mag, phase = system.bode(w=w_vector)
                # H = 10^(mag/20) * exp(j * phase * pi/180)
                # Better to trust scipy freqresp
                raise ValueError("System does not support frequency response calculation.")
            
            real = H.real
            imag = H.imag
            
            # Display
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Nyquist Plot: {source_block.name}")
            layout = QVBoxLayout()
            layout.setContentsMargins(0,0,0,0)
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('w')
            layout.addWidget(plot_widget)
            plot_window.setLayout(layout)
            
            # Nyquist is Linear-Linear usually, or Equal Aspect Ratio?
            # Equal Aspect Ratio is important for Nyquist to see -1 circle correctly
            plot_widget.setAspectLocked(True)
            
            styles = {'color': 'k', 'font-size': '10pt'}
            plot_widget.setLabel('left', 'Imaginary Axis', **styles)
            plot_widget.setLabel('bottom', 'Real Axis', **styles)
            plot_widget.setTitle(f"Nyquist Plot: {source_block.name}", color='k', size='12pt')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Plot Curve
            plot_widget.plot(real, imag, pen=pg.mkPen(color=(0, 114, 189), width=2.5))
            
            # Mirror for negative frequencies?
            # Usually Nyquist plots show -Inf to +Inf. Scipy gives 0 to Inf.
            # Mirroring: Real is even, Imag is odd.
            plot_widget.plot(real, -imag, pen=pg.mkPen(color=(0, 114, 189), width=2.5, style=Qt.DashLine))
            
            # Mark -1+0j point
            plot_widget.plot([-1], [0], symbol='+', symbolSize=12, symbolPen='r')

            # Arrows? (maybe later)

            # Styling
            for ax_name in ['left', 'bottom']:
                ax = plot_widget.getAxis(ax_name)
                ax.setPen('k')
                ax.setTextPen('k')

            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            logger.error(f"Error generating Nyquist: {e}")
            if self.parent: QMessageBox.critical(self.parent, "Error", str(e))

    def _extract_system_model(self, source_block):
        """Helper to extract Scipy system from block."""
        system = None
        if source_block.block_fn == 'TranFn':
            num = source_block.params.get('numerator')
            den = source_block.params.get('denominator')
            if num and den: system = signal.TransferFunction(num, den)
        elif source_block.block_fn == 'DiscreteTranFn':
            num = source_block.params.get('numerator')
            den = source_block.params.get('denominator')
            st = source_block.params.get('sampling_time', -1.0)
            if st <= 0: st = source_block.params.get('dt', 1.0)
            if st <= 0: st = 1.0
            if num and den: system = signal.TransferFunction(num, den, dt=st)
        elif source_block.block_fn == 'StateSpace':
             A = source_block.params.get('A'); B = source_block.params.get('B')
             C = source_block.params.get('C'); D = source_block.params.get('D')
             if all(v is not None for v in [A,B,C,D]): system = signal.StateSpace(A,B,C,D)
        elif source_block.block_fn == 'DiscreteStateSpace':
             A = source_block.params.get('A'); B = source_block.params.get('B')
             C = source_block.params.get('C'); D = source_block.params.get('D')
             st = source_block.params.get('sampling_time', -1.0)
             if st <= 0: st = source_block.params.get('dt', 1.0)
             if st <= 0: st = 1.0
             if all(v is not None for v in [A,B,C,D]): system = signal.StateSpace(A,B,C,D, dt=st)
        elif source_block.block_fn == 'PID':
             kp = source_block.params.get('kp', 1.0); ki = source_block.params.get('ki', 0.0); kd = source_block.params.get('kd', 0.0)
             num = [kd, kp, ki]; den = [1, 0]
             system = signal.TransferFunction(num, den)
        
        if system is None: raise ValueError("Invalid system parameters")
        return system

    def _get_frequency_vector(self, system, default_pts=500):
        """Helper for freq vector."""
        if hasattr(system, 'dt') and system.dt and system.dt > 0:
            nyq = np.pi / system.dt
            return np.logspace(np.log10(nyq)-4, np.log10(nyq), default_pts)
        return None


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

            self._position_window(plot_window)
            self.plot_windows.append(plot_window)
            plot_window.show()

        except Exception as e:
            error_msg = f"Error calculating root locus: {str(e)}"
            logger.error(error_msg)
            if self.parent:
                QMessageBox.critical(self.parent, "Root Locus Error", error_msg)
