import logging
import numpy as np
import scipy.signal as signal
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class NyquistAnalyzer(BaseAnalyzer):
    """Analyzer for generating Nyquist plots."""
    
    def analyze(self, source_block, canvas):
        """Generate Nyquist plot."""
        logger.debug(f"NyquistAnalyzer called for {source_block.name}")
        
        model, sys_block = self._find_connected_transfer_function(source_block, canvas)
        
        if not model:
            return None
            
        try:
            num, den, dt = model
            
            # Calculation
             # Create discrete or continuous system
            # Auto-range the frequency sweep from the system poles/zeros so
            # both fast and slow dynamics land on screen, instead of a fixed
            # np.logspace(-3, 3) window.
            w = self._auto_frequency_range(num, den, dt=dt, n_points=1000)
            if dt > 0:
                sys = signal.TransferFunction(num, den, dt=dt)
                # Discrete systems require dfreqresp; the module-level freqresp
                # only supports continuous-time systems and raises otherwise.
                w, H = signal.dfreqresp(sys, w=w)
            else:
                sys = signal.TransferFunction(num, den)
                w, H = signal.freqresp(sys, w=w)

            real = H.real
            imag = H.imag

            # Stability margins, computed from the same frequency response.
            mag_db = 20.0 * np.log10(np.abs(H))
            phase_deg = np.degrees(np.unwrap(np.angle(H)))
            margins = self._compute_stability_margins(w, mag_db, phase_deg)
            
            # Plotting
            from PyQt5.QtWidgets import QWidget
            plot_window = QWidget()
            t = f"Nyquist Plot: {sys_block.name}"
            plot_widget = self._setup_plot_widget(plot_window, t)
            
            plot_widget.setLabel('left', 'Imaginary', color='#000000')
            plot_widget.setLabel('bottom', 'Real', color='#000000')
            plot_widget.setAspectLocked(True)
            
            # Plot curve
            plot_widget.plot(real, imag, pen=pg.mkPen('b', width=2), name='System')
            # Mirror for full Nyquist
            plot_widget.plot(real, -imag, pen=pg.mkPen('b', width=2, style=Qt.DashLine))
            
            # Critical point -1+0j
            plot_widget.plot([-1], [0], symbol='+', symbolSize=12, symbolPen='r')

            # Annotate stability margins. The gain margin corresponds to where
            # the locus crosses the negative real axis (-180 deg); 1/GM gives
            # the |G| at that point, i.e. its distance from the origin.
            gm = margins.get('gain_margin_db')
            pm = margins.get('phase_margin_deg')
            ann_lines = []
            if gm is not None and np.isfinite(gm):
                ann_lines.append(f"GM = {gm:.1f} dB")
                gm_lin = 10.0 ** (-gm / 20.0)  # |G| at -180 deg crossing
                plot_widget.plot([-gm_lin], [0], symbol='o', symbolSize=10,
                                 symbolPen=pg.mkPen('r', width=2), symbolBrush=None)
            if pm is not None and np.isfinite(pm):
                ann_lines.append(f"PM = {pm:.1f} deg")
            if ann_lines:
                margin_text = pg.TextItem("\n".join(ann_lines), color='r')
                margin_text.setPos(-1, 0)
                plot_widget.addItem(margin_text)

            # Axes lines
            plot_widget.addLine(x=0, pen=pg.mkPen('k', width=1))
            plot_widget.addLine(y=0, pen=pg.mkPen('k', width=1))
            
            plot_widget.showGrid(x=True, y=True)
            self._position_window(plot_window)
            plot_window.show()
            return plot_window
            
        except Exception as e:
            logger.error(f"Error calculating Nyquist plot: {e}")
            return None
