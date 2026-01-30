"""
Field Processing Blocks for PDE outputs

These blocks process array/field outputs from PDE blocks:
- FieldProbe: Extract value at a specific location
- FieldIntegral: Integrate field over domain
- FieldMax: Find maximum value and location
- FieldScope: Visualize spatiotemporal field data
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class FieldProbeBlock(BaseBlock):
    """
    Extract field value at a specific spatial location.

    Takes a field array and outputs the value at position x (or index i).
    Uses linear interpolation for positions between nodes.
    """

    @property
    def block_name(self):
        return "FieldProbe"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Probe: Extract value at a specific location."
            "\n\nInterpolates the field value at the specified position."
            "\n\nParameters:"
            "\n- position: Spatial position x [m] or fraction [0-1]"
            "\n- position_mode: 'absolute' or 'normalized'"
            "\n- L: Domain length (for absolute mode)"
            "\n\nInputs:"
            "\n- field: Field array from PDE block"
            "\n- position: Dynamic position input (optional)"
            "\n\nOutputs:"
            "\n- value: Field value at the probed location"
        )

    @property
    def params(self):
        return {
            "position": {
                "type": "float",
                "default": 0.5,
                "doc": "Probe position (absolute x or normalized 0-1)"
            },
            "position_mode": {
                "type": "string",
                "default": "normalized",
                "doc": "Position mode: 'absolute' or 'normalized'"
            },
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length for absolute mode"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "Field array to probe"},
            {"name": "position", "type": "float", "doc": "Dynamic position (optional)"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "value", "type": "float", "doc": "Field value at probe location"},
        ]

    @property
    def optional_inputs(self):
        """Input 1 (position) is optional - uses parameter if not connected."""
        return [1]

    def draw_icon(self, block_rect):
        """Draw probe icon - crosshair on a curve."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw a curve
        path.moveTo(0.1, 0.7)
        path.cubicTo(0.3, 0.3, 0.6, 0.5, 0.9, 0.4)
        # Draw crosshair at probe point
        path.moveTo(0.5, 0.3)
        path.lineTo(0.5, 0.7)
        path.moveTo(0.35, 0.5)
        path.lineTo(0.65, 0.5)
        # Small circle at probe point
        path.addEllipse(0.45, 0.45, 0.1, 0.1)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Execute the field probe."""
        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            return {0: 0.0, 'E': False}

        # Get position (use input if available, otherwise parameter)
        position = inputs.get(1, None)
        if position is None:
            position = float(params.get('position', 0.5))

        mode = params.get('position_mode', 'normalized')
        L = float(params.get('L', 1.0))
        N = len(field)

        # Convert to normalized position [0, 1]
        if mode == 'absolute':
            pos_norm = position / L
        else:
            pos_norm = position

        # Clamp to valid range
        pos_norm = np.clip(pos_norm, 0.0, 1.0)

        # Linear interpolation
        idx_float = pos_norm * (N - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, N - 1)
        frac = idx_float - idx_low

        value = field[idx_low] * (1 - frac) + field[idx_high] * frac

        return {0: float(value), 'E': False}


class FieldIntegralBlock(BaseBlock):
    """
    Integrate field over the spatial domain.

    Computes ∫ field(x) dx using trapezoidal rule.
    """

    @property
    def block_name(self):
        return "FieldIntegral"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Integral: Integrate field over domain."
            "\n\nComputes ∫ field(x) dx using trapezoidal rule."
            "\n\nParameters:"
            "\n- L: Domain length [m]"
            "\n- normalize: If True, divide by L to get average"
            "\n\nInputs:"
            "\n- field: Field array from PDE block"
            "\n\nOutputs:"
            "\n- integral: Integral value"
        )

    @property
    def params(self):
        return {
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
            "normalize": {
                "type": "bool",
                "default": False,
                "doc": "Normalize by domain length (gives average)"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "Field array to integrate"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "integral", "type": "float", "doc": "Integral value"},
        ]

    def draw_icon(self, block_rect):
        """Draw integral icon - integral symbol with curve."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw integral symbol
        path.moveTo(0.45, 0.15)
        path.cubicTo(0.55, 0.15, 0.55, 0.25, 0.5, 0.35)
        path.lineTo(0.5, 0.65)
        path.cubicTo(0.45, 0.75, 0.45, 0.85, 0.55, 0.85)
        # Shaded area under curve
        path.moveTo(0.6, 0.7)
        path.cubicTo(0.7, 0.5, 0.8, 0.4, 0.9, 0.35)
        path.lineTo(0.9, 0.7)
        path.lineTo(0.6, 0.7)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Execute the field integral."""
        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            return {0: 0.0, 'E': False}

        L = float(params.get('L', 1.0))
        N = len(field)
        dx = L / (N - 1) if N > 1 else L

        # Trapezoidal integration
        integral = np.trapz(field, dx=dx)

        if params.get('normalize', False):
            integral = integral / L

        return {0: float(integral), 'E': False}


class FieldMaxBlock(BaseBlock):
    """
    Find maximum (or minimum) value in a field and its location.
    """

    @property
    def block_name(self):
        return "FieldMax"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Max: Find maximum value and location."
            "\n\nFinds the maximum (or minimum) value in the field."
            "\n\nParameters:"
            "\n- mode: 'max' or 'min'"
            "\n- L: Domain length for location output"
            "\n\nInputs:"
            "\n- field: Field array from PDE block"
            "\n\nOutputs:"
            "\n- value: Maximum (or minimum) value"
            "\n- location: Spatial location of extremum"
            "\n- index: Array index of extremum"
        )

    @property
    def params(self):
        return {
            "mode": {
                "type": "string",
                "default": "max",
                "doc": "Mode: 'max' or 'min'"
            },
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length for location output"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "Field array to analyze"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "value", "type": "float", "doc": "Extreme value"},
            {"name": "location", "type": "float", "doc": "Spatial location"},
            {"name": "index", "type": "int", "doc": "Array index"},
        ]

    @property
    def optional_outputs(self):
        """Outputs 1 (location) and 2 (index) are auxiliary."""
        return [1, 2]

    def draw_icon(self, block_rect):
        """Draw max icon - curve with peak marker."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw a curve with a peak
        path.moveTo(0.1, 0.7)
        path.cubicTo(0.3, 0.7, 0.4, 0.25, 0.5, 0.25)
        path.cubicTo(0.6, 0.25, 0.7, 0.7, 0.9, 0.7)
        # Arrow pointing to max
        path.moveTo(0.5, 0.15)
        path.lineTo(0.5, 0.25)
        path.moveTo(0.45, 0.2)
        path.lineTo(0.5, 0.15)
        path.lineTo(0.55, 0.2)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Execute the field max/min finder."""
        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        if len(field) == 0:
            return {0: 0.0, 1: 0.0, 2: 0, 'E': False}

        mode = params.get('mode', 'max')
        L = float(params.get('L', 1.0))
        N = len(field)

        if mode == 'min':
            idx = int(np.argmin(field))
            value = field[idx]
        else:
            idx = int(np.argmax(field))
            value = field[idx]

        location = (idx / (N - 1)) * L if N > 1 else 0.0

        return {0: float(value), 1: float(location), 2: idx, 'E': False}


class FieldScopeBlock(BaseBlock):
    """
    Visualize spatiotemporal field data as a 2D heatmap.

    Stores field snapshots over time and displays them as a heatmap
    with space on one axis and time on the other.
    """

    @property
    def block_name(self):
        return "FieldScope"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return (
            "Field Scope: Visualize spatiotemporal field evolution."
            "\n\nDisplays field data as 2D heatmap (x vs t)."
            "\n\nParameters:"
            "\n- L: Domain length [m]"
            "\n- colormap: Colormap name (viridis, hot, coolwarm, etc.)"
            "\n- title: Plot title"
            "\n\nInputs:"
            "\n- field: Field array from PDE block"
            "\n\nOutputs:"
            "\n- None (visualization block)"
        )

    @property
    def params(self):
        return {
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
            "colormap": {
                "type": "string",
                "default": "viridis",
                "doc": "Matplotlib colormap name"
            },
            "title": {
                "type": "string",
                "default": "Field Evolution",
                "doc": "Plot title"
            },
            "vec_labels": {
                "type": "string",
                "default": "field",
                "doc": "Signal label for data export"
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
            {"name": "field", "type": "array", "doc": "Field array to visualize"},
        ]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        """FieldScope is a sink block."""
        return False

    def draw_icon(self, block_rect):
        """Draw field scope icon - 2D grid/heatmap."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw grid pattern representing heatmap
        # Outer rectangle
        path.addRect(0.15, 0.15, 0.7, 0.7)
        # Grid lines
        for i in range(1, 4):
            x = 0.15 + i * 0.175
            path.moveTo(x, 0.15)
            path.lineTo(x, 0.85)
        for i in range(1, 4):
            y = 0.15 + i * 0.175
            path.moveTo(0.15, y)
            path.lineTo(0.85, y)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Execute the field scope."""
        # Initialization
        if params.get('_init_start_', True):
            params['_field_history_'] = []
            params['_time_history_'] = []
            params['_init_start_'] = False

        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        # Store snapshot
        params['_field_history_'].append(field.copy())
        params['_time_history_'].append(time)

        # Also store in 'vector' format for compatibility with ScopePlotter
        history = np.array(params['_field_history_'])
        params['vector'] = history

        return {'E': False}

    def plot_field(self, params, timeline=None):
        """
        Generate the spatiotemporal heatmap plot.

        Called by the plotting system after simulation.
        """
        import matplotlib.pyplot as plt

        history = params.get('_field_history_', [])
        times = params.get('_time_history_', [])

        if not history or len(history) == 0:
            logger.warning("FieldScope: No data to plot")
            return

        # Convert to numpy array
        data = np.array(history)  # Shape: (n_times, n_points)

        L = float(params.get('L', 1.0))
        N = data.shape[1] if data.ndim > 1 else 1
        colormap = params.get('colormap', 'viridis')
        title = params.get('title', 'Field Evolution')

        # Create meshgrid
        x = np.linspace(0, L, N)
        t = np.array(times)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot heatmap
        im = ax.pcolormesh(x, t, data, cmap=colormap, shading='auto')
        ax.set_xlabel('Position x [m]')
        ax.set_ylabel('Time t [s]')
        ax.set_title(title)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field Value')

        plt.tight_layout()
        plt.show()

        return fig


class FieldGradientBlock(BaseBlock):
    """
    Compute spatial gradient of a field.

    Outputs ∂field/∂x using central differences.
    """

    @property
    def block_name(self):
        return "FieldGradient"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Gradient: Compute spatial derivative."
            "\n\nComputes ∂field/∂x using central differences."
            "\n\nParameters:"
            "\n- L: Domain length [m]"
            "\n\nInputs:"
            "\n- field: Field array"
            "\n\nOutputs:"
            "\n- gradient: Spatial gradient array"
        )

    @property
    def params(self):
        return {
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "Field array"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "gradient", "type": "array", "doc": "Spatial gradient"},
        ]

    def draw_icon(self, block_rect):
        """Draw gradient icon - slope with nabla symbol."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw nabla (inverted triangle)
        path.moveTo(0.3, 0.2)
        path.lineTo(0.5, 0.6)
        path.lineTo(0.7, 0.2)
        path.lineTo(0.3, 0.2)
        # Draw sloped line below
        path.moveTo(0.2, 0.85)
        path.lineTo(0.8, 0.65)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Compute the field gradient."""
        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        if len(field) < 2:
            return {0: np.array([0.0]), 'E': False}

        L = float(params.get('L', 1.0))
        N = len(field)
        dx = L / (N - 1)

        # Central differences for interior, one-sided at boundaries
        gradient = np.gradient(field, dx)

        return {0: gradient, 'E': False}


class FieldLaplacianBlock(BaseBlock):
    """
    Compute Laplacian (second derivative) of a field.

    Outputs ∇²field = ∂²field/∂x² using central differences.
    """

    @property
    def block_name(self):
        return "FieldLaplacian"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Laplacian: Compute second spatial derivative."
            "\n\nComputes ∇²field = ∂²field/∂x²."
            "\n\nParameters:"
            "\n- L: Domain length [m]"
            "\n\nInputs:"
            "\n- field: Field array"
            "\n\nOutputs:"
            "\n- laplacian: Laplacian array"
        )

    @property
    def params(self):
        return {
            "L": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length [m]"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "Field array"},
        ]

    @property
    def outputs(self):
        return [
            {"name": "laplacian", "type": "array", "doc": "Laplacian (∇²)"},
        ]

    def draw_icon(self, block_rect):
        """Draw Laplacian icon - nabla squared symbol."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # Draw nabla squared (two inverted triangles)
        # First nabla
        path.moveTo(0.2, 0.25)
        path.lineTo(0.35, 0.55)
        path.lineTo(0.5, 0.25)
        path.lineTo(0.2, 0.25)
        # Second nabla (superscript 2 position)
        path.moveTo(0.5, 0.2)
        path.lineTo(0.6, 0.4)
        path.lineTo(0.7, 0.2)
        path.lineTo(0.5, 0.2)
        # Curve below representing second derivative
        path.moveTo(0.15, 0.75)
        path.cubicTo(0.35, 0.9, 0.65, 0.6, 0.85, 0.75)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """Compute the field Laplacian."""
        field = inputs.get(0, np.array([0.0]))
        field = np.atleast_1d(field).flatten()

        if len(field) < 3:
            return {0: np.zeros(len(field)), 'E': False}

        L = float(params.get('L', 1.0))
        N = len(field)
        dx = L / (N - 1)
        dx_sq = dx * dx

        # Central difference for second derivative
        laplacian = np.zeros(N)
        for i in range(1, N-1):
            laplacian[i] = (field[i+1] - 2*field[i] + field[i-1]) / dx_sq

        # Boundary values (use one-sided differences)
        laplacian[0] = (field[2] - 2*field[1] + field[0]) / dx_sq
        laplacian[N-1] = (field[N-1] - 2*field[N-2] + field[N-3]) / dx_sq

        return {0: laplacian, 'E': False}
