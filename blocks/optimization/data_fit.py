"""
DataFit Block - Model calibration against experimental data

This block loads experimental data and computes the fit error
between simulation output and measured data. Used for parameter
identification and model calibration.
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class DataFitBlock(BaseBlock):
    """
    Data Fit Block for model calibration.

    Loads experimental data from a file and computes the error
    between simulation output and the measured data. The error
    is accumulated for use by the optimizer.

    Supported file formats:
    - CSV: Comma-separated values
    - NPZ: NumPy compressed archive
    - TXT: Space/tab separated text
    """

    @property
    def block_name(self):
        return "DataFit"

    @property
    def category(self):
        return "Optimization"

    @property
    def color(self):
        return "brown"

    @property
    def doc(self):
        return (
            "Data Fit - Model calibration against experimental data"
            "\n\nLoads measured data and computes fit error."
            "\nUsed for parameter identification."
            "\n\nSupported formats:"
            "\n- CSV files (comma-separated)"
            "\n- NPZ files (numpy archive)"
            "\n- TXT files (space/tab separated)"
            "\n\nParameters:"
            "\n- data_file: Path to data file"
            "\n- time_col: Column name/index for time"
            "\n- signal_col: Column name/index for signal"
            "\n- fit_type: Error metric (MSE, MAE, R2)"
            "\n\nInputs:"
            "\n- signal: Simulation output to compare"
            "\n\nOutputs:"
            "\n- error: Current fit error"
            "\n- measured: Interpolated measured value"
        )

    @property
    def params(self):
        return {
            "data_file": {
                "type": "string",
                "default": "",
                "doc": "Path to data file"
            },
            "time_col": {
                "type": "string",
                "default": "t",
                "doc": "Time column name or index"
            },
            "signal_col": {
                "type": "string",
                "default": "y",
                "doc": "Signal column name or index"
            },
            "fit_type": {
                "type": "string",
                "default": "MSE",
                "doc": "Fit type: MSE, MAE, RMSE, R2"
            },
            "weight": {
                "type": "float",
                "default": 1.0,
                "doc": "Weight in overall objective"
            },
            "interpolation": {
                "type": "string",
                "default": "linear",
                "doc": "Interpolation: linear, nearest, spline"
            },
            "_init_start_": {
                "type": "bool",
                "default": True,
                "doc": "Internal: initialization flag"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "signal", "type": "float", "doc": "Simulation output"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "error", "type": "float", "doc": "Fit error"},
            {"name": "measured", "type": "float", "doc": "Measured value at current time"},
        ]

    @property
    def requires_outputs(self):
        return False

    def draw_icon(self, block_rect):
        """Draw data fit icon - data points with fitted curve."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw data points
        path.addEllipse(0.15, 0.55, 0.08, 0.08)
        path.addEllipse(0.28, 0.35, 0.08, 0.08)
        path.addEllipse(0.45, 0.25, 0.08, 0.08)
        path.addEllipse(0.62, 0.35, 0.08, 0.08)
        path.addEllipse(0.78, 0.5, 0.08, 0.08)
        # Draw fitted curve
        path.moveTo(0.12, 0.6)
        path.cubicTo(0.3, 0.3, 0.6, 0.25, 0.88, 0.55)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Compare simulation output to measured data."""

        # Initialization
        if params.get('_init_start_', True):
            self._load_data(params)
            params['_accumulated_error_'] = 0.0
            params['_n_points_'] = 0
            params['_ss_tot_'] = 0.0  # For R² calculation
            params['_ss_res_'] = 0.0
            params['_init_start_'] = False

        dtime = float(params.get('dtime', 0.01))
        weight = float(params.get('weight', 1.0))
        fit_type = params.get('fit_type', 'MSE')

        # Get measured data at current time
        t_data = params.get('_time_data_', np.array([0.0]))
        y_data = params.get('_signal_data_', np.array([0.0]))

        if len(t_data) == 0 or len(y_data) == 0:
            return {0: 0.0, 1: 0.0, 'E': False}

        # Interpolate measured value
        interpolation = params.get('interpolation', 'linear')

        if time < t_data[0] or time > t_data[-1]:
            # Outside data range - extrapolate or return edge value
            if time < t_data[0]:
                measured = y_data[0]
            else:
                measured = y_data[-1]
        else:
            if interpolation == 'nearest':
                idx = np.argmin(np.abs(t_data - time))
                measured = y_data[idx]
            elif interpolation == 'spline':
                try:
                    from scipy.interpolate import UnivariateSpline
                    spline = UnivariateSpline(t_data, y_data, s=0)
                    measured = float(spline(time))
                except Exception:
                    measured = float(np.interp(time, t_data, y_data))
            else:
                # Linear interpolation
                measured = float(np.interp(time, t_data, y_data))

        # Get simulation signal
        signal = inputs.get(0, 0.0)
        if isinstance(signal, np.ndarray):
            signal = float(signal.flatten()[0])

        # Compute error
        error = signal - measured

        # Accumulate based on fit type
        if fit_type.upper() == 'MSE':
            params['_accumulated_error_'] = params.get('_accumulated_error_', 0.0) + error**2
        elif fit_type.upper() == 'MAE':
            params['_accumulated_error_'] = params.get('_accumulated_error_', 0.0) + abs(error)
        elif fit_type.upper() == 'RMSE':
            params['_accumulated_error_'] = params.get('_accumulated_error_', 0.0) + error**2
        elif fit_type.upper() == 'R2':
            # R² = 1 - SS_res / SS_tot
            mean_y = params.get('_mean_y_', np.mean(y_data))
            params['_ss_res_'] = params.get('_ss_res_', 0.0) + error**2
            params['_ss_tot_'] = params.get('_ss_tot_', 0.0) + (measured - mean_y)**2

        params['_n_points_'] = params.get('_n_points_', 0) + 1

        # Return current accumulated error
        accumulated = params.get('_accumulated_error_', 0.0)
        n_points = params.get('_n_points_', 1)

        if fit_type.upper() == 'MSE':
            current_error = accumulated / n_points if n_points > 0 else 0.0
        elif fit_type.upper() == 'MAE':
            current_error = accumulated / n_points if n_points > 0 else 0.0
        elif fit_type.upper() == 'RMSE':
            current_error = np.sqrt(accumulated / n_points) if n_points > 0 else 0.0
        elif fit_type.upper() == 'R2':
            ss_res = params.get('_ss_res_', 0.0)
            ss_tot = params.get('_ss_tot_', 1.0)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            current_error = 1 - r2  # Minimize 1 - R²
        else:
            current_error = accumulated / n_points if n_points > 0 else 0.0

        return {0: current_error * weight, 1: measured, 'E': False}

    def _load_data(self, params):
        """Load experimental data from file."""
        data_file = params.get('data_file', '')
        time_col = params.get('time_col', 't')
        signal_col = params.get('signal_col', 'y')

        if not data_file:
            params['_time_data_'] = np.array([0.0, 1.0])
            params['_signal_data_'] = np.array([0.0, 0.0])
            params['_mean_y_'] = 0.0
            return

        try:
            if data_file.endswith('.npz'):
                data = np.load(data_file)
                t_data = data.get(time_col, data.get('t', np.array([0.0, 1.0])))
                y_data = data.get(signal_col, data.get('y', np.array([0.0, 0.0])))
            elif data_file.endswith('.csv'):
                import csv
                with open(data_file, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                if time_col.isdigit():
                    time_idx = int(time_col)
                    signal_idx = int(signal_col) if signal_col.isdigit() else 1
                    t_data = np.array([float(list(row.values())[time_idx]) for row in rows])
                    y_data = np.array([float(list(row.values())[signal_idx]) for row in rows])
                else:
                    t_data = np.array([float(row.get(time_col, 0)) for row in rows])
                    y_data = np.array([float(row.get(signal_col, 0)) for row in rows])
            else:
                # Try numpy loadtxt
                data = np.loadtxt(data_file)
                if data.ndim == 1:
                    t_data = np.arange(len(data))
                    y_data = data
                else:
                    time_idx = int(time_col) if time_col.isdigit() else 0
                    signal_idx = int(signal_col) if signal_col.isdigit() else 1
                    t_data = data[:, time_idx]
                    y_data = data[:, signal_idx]

            params['_time_data_'] = np.atleast_1d(t_data).flatten()
            params['_signal_data_'] = np.atleast_1d(y_data).flatten()
            params['_mean_y_'] = np.mean(params['_signal_data_'])

            logger.info(f"DataFit: Loaded {len(t_data)} points from {data_file}")

        except Exception as e:
            logger.error(f"DataFit: Failed to load data from {data_file}: {e}")
            params['_time_data_'] = np.array([0.0, 1.0])
            params['_signal_data_'] = np.array([0.0, 0.0])
            params['_mean_y_'] = 0.0

    def get_final_error(self, params):
        """Get the final fit error (called by optimizer after simulation)."""
        fit_type = params.get('fit_type', 'MSE')
        weight = float(params.get('weight', 1.0))
        accumulated = params.get('_accumulated_error_', 0.0)
        n_points = params.get('_n_points_', 1)

        if fit_type.upper() == 'MSE':
            error = accumulated / n_points if n_points > 0 else 0.0
        elif fit_type.upper() == 'MAE':
            error = accumulated / n_points if n_points > 0 else 0.0
        elif fit_type.upper() == 'RMSE':
            error = np.sqrt(accumulated / n_points) if n_points > 0 else 0.0
        elif fit_type.upper() == 'R2':
            ss_res = params.get('_ss_res_', 0.0)
            ss_tot = params.get('_ss_tot_', 1.0)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            error = 1 - r2
        else:
            error = accumulated / n_points if n_points > 0 else 0.0

        return error * weight

    def reset(self, params):
        """Reset for a new optimization iteration."""
        params['_accumulated_error_'] = 0.0
        params['_n_points_'] = 0
        params['_ss_tot_'] = 0.0
        params['_ss_res_'] = 0.0
        params['_init_start_'] = True
