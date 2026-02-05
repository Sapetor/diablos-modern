"""
2D Field Processing Blocks for PDE outputs

These blocks process 2D array/field outputs from 2D PDE blocks:
- FieldProbe2D: Extract value at a specific (x,y) location
- FieldScope2D: Visualize 2D field as animated heatmap
- FieldSlice: Extract 1D slice from 2D field
"""

import logging
import numpy as np
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class FieldProbe2DBlock(BaseBlock):
    """
    Extract field value at a specific (x,y) location from a 2D field.

    Uses bilinear interpolation for positions between nodes.
    """

    @property
    def block_name(self):
        return "FieldProbe2D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "2D Field Probe: Extract value at (x,y) location."
            "\n\nUses bilinear interpolation for positions between nodes."
            "\n\nParameters:"
            "\n- x_position, y_position: Probe coordinates"
            "\n- position_mode: 'absolute' or 'normalized' (0-1)"
            "\n- Lx, Ly: Domain dimensions (for absolute mode)"
            "\n\nInputs:"
            "\n- field: 2D field array from PDE block"
            "\n- x_pos: Dynamic x position (optional)"
            "\n- y_pos: Dynamic y position (optional)"
            "\n\nOutputs:"
            "\n- value: Field value at probed location"
        )

    @property
    def params(self):
        return {
            "x_position": {
                "type": "float",
                "default": 0.5,
                "doc": "X probe position"
            },
            "y_position": {
                "type": "float",
                "default": 0.5,
                "doc": "Y probe position"
            },
            "position_mode": {
                "type": "string",
                "default": "normalized",
                "doc": "Position mode: 'absolute' or 'normalized'"
            },
            "Lx": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in x"
            },
            "Ly": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in y"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "2D field array to probe"},
            {"name": "x_pos", "type": "float", "doc": "Dynamic x position (optional)"},
            {"name": "y_pos", "type": "float", "doc": "Dynamic y position (optional)"},
        ]

    @property
    def optional_inputs(self):
        return [1, 2]

    @property
    def outputs(self):
        return [
            {"name": "value", "type": "float", "doc": "Field value at (x,y)"},
        ]

    def draw_icon(self, block_rect):
        """Draw 2D probe icon - crosshairs on grid."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()

        # Grid
        path.addRect(0.2, 0.2, 0.6, 0.6)
        path.moveTo(0.2, 0.5)
        path.lineTo(0.8, 0.5)
        path.moveTo(0.5, 0.2)
        path.lineTo(0.5, 0.8)

        # Crosshair
        path.addEllipse(0.4, 0.4, 0.2, 0.2)

        return path

    def execute(self, time, inputs, params, **kwargs):
        """Extract value from 2D field at specified position."""
        field = inputs.get(0, None)
        if field is None:
            return {0: 0.0, 'E': False}

        field = np.atleast_2d(field)
        if field.ndim != 2:
            return {0: 0.0, 'E': False}

        Ny, Nx = field.shape

        # Get position
        x_pos = inputs.get(1, None)
        if x_pos is None:
            x_pos = float(params.get('x_position', 0.5))
        y_pos = inputs.get(2, None)
        if y_pos is None:
            y_pos = float(params.get('y_position', 0.5))

        position_mode = params.get('position_mode', 'normalized')
        Lx = float(params.get('Lx', 1.0))
        Ly = float(params.get('Ly', 1.0))

        # Convert to normalized coordinates
        if position_mode == 'absolute':
            x_norm = x_pos / Lx
            y_norm = y_pos / Ly
        else:
            x_norm = x_pos
            y_norm = y_pos

        # Clamp to valid range
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        # Convert to array indices (float)
        i_float = x_norm * (Nx - 1)
        j_float = y_norm * (Ny - 1)

        # Bilinear interpolation
        i0 = int(np.floor(i_float))
        i1 = min(i0 + 1, Nx - 1)
        j0 = int(np.floor(j_float))
        j1 = min(j0 + 1, Ny - 1)

        di = i_float - i0
        dj = j_float - j0

        # Interpolate
        val = (field[j0, i0] * (1 - di) * (1 - dj) +
               field[j0, i1] * di * (1 - dj) +
               field[j1, i0] * (1 - di) * dj +
               field[j1, i1] * di * dj)

        return {0: float(val), 'E': False}


class FieldScope2DBlock(BaseBlock):
    """
    Visualize 2D field as an animated heatmap.

    Shows the spatial distribution of the field at each time step.
    Can also show time evolution as a 3D surface or animation.
    """

    @property
    def block_name(self):
        return "FieldScope2D"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "tomato"

    @property
    def doc(self):
        return (
            "2D Field Scope: Visualize 2D field evolution."
            "\n\nDisplays animated heatmap of field distribution."
            "\n\nParameters:"
            "\n- Lx, Ly: Domain dimensions [m]"
            "\n- colormap: Color scheme (viridis, hot, coolwarm, etc.)"
            "\n- title: Plot title"
            "\n- clim_min/max: Color scale limits (auto if None)"
            "\n- sample_interval: Store every N timesteps"
            "\n\nInputs:"
            "\n- field: 2D field array (Ny × Nx)"
            "\n\nOutputs: None (visualization only)"
        )

    @property
    def params(self):
        return {
            "Lx": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in x [m]"
            },
            "Ly": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in y [m]"
            },
            "colormap": {
                "type": "string",
                "default": "viridis",
                "doc": "Colormap: viridis, hot, coolwarm, plasma, etc."
            },
            "title": {
                "type": "string",
                "default": "2D Field",
                "doc": "Plot title"
            },
            "clim_min": {
                "type": "float",
                "default": None,
                "doc": "Color scale minimum (None for auto)"
            },
            "clim_max": {
                "type": "float",
                "default": None,
                "doc": "Color scale maximum (None for auto)"
            },
            "sample_interval": {
                "type": "int",
                "default": 5,
                "doc": "Store every N timesteps (reduces memory)"
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
            {"name": "field", "type": "array", "doc": "2D field to visualize"},
        ]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        return False

    def draw_icon(self, block_rect):
        """Draw 2D scope icon - heatmap pattern."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()

        # Screen/frame
        path.addRect(0.15, 0.15, 0.7, 0.7)

        # Grid pattern suggesting heatmap
        for i in range(3):
            for j in range(3):
                x = 0.2 + i * 0.2
                y = 0.2 + j * 0.2
                size = 0.15
                path.addRect(x, y, size, size)

        return path

    def execute(self, time, inputs, params, **kwargs):
        """Store field snapshot for visualization."""
        # Initialize storage
        if params.get('_init_start_', True):
            params['_field_history_2d_'] = []
            params['_time_history_'] = []
            params['_sample_count_'] = 0
            params['_init_start_'] = False

        field = inputs.get(0, None)
        if field is None:
            return {'E': False}

        field = np.atleast_2d(field)

        # Sample at specified interval to reduce memory
        sample_interval = int(params.get('sample_interval', 5))
        params['_sample_count_'] = params.get('_sample_count_', 0) + 1

        if params['_sample_count_'] >= sample_interval:
            params['_field_history_2d_'].append(field.copy())
            params['_time_history_'].append(time)
            params['_sample_count_'] = 0

        return {'E': False}


class FieldSliceBlock(BaseBlock):
    """
    Extract a 1D slice from a 2D field.

    Can extract horizontal (constant y) or vertical (constant x) slices.
    Useful for comparing 2D results with 1D analysis.
    """

    @property
    def block_name(self):
        return "FieldSlice"

    @property
    def category(self):
        return "PDE"

    @property
    def color(self):
        return "purple"

    @property
    def doc(self):
        return (
            "Field Slice: Extract 1D slice from 2D field."
            "\n\nParameters:"
            "\n- slice_direction: 'x' (horizontal) or 'y' (vertical)"
            "\n- slice_position: Position of slice (normalized 0-1)"
            "\n- Lx, Ly: Domain dimensions"
            "\n\nInputs:"
            "\n- field: 2D field array"
            "\n- position: Dynamic slice position (optional)"
            "\n\nOutputs:"
            "\n- slice: 1D array along the slice"
        )

    @property
    def params(self):
        return {
            "slice_direction": {
                "type": "string",
                "default": "x",
                "doc": "Slice direction: 'x' (horizontal) or 'y' (vertical)"
            },
            "slice_position": {
                "type": "float",
                "default": 0.5,
                "doc": "Position of slice (normalized 0-1)"
            },
            "Lx": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in x"
            },
            "Ly": {
                "type": "float",
                "default": 1.0,
                "doc": "Domain length in y"
            },
        }

    @property
    def inputs(self):
        return [
            {"name": "field", "type": "array", "doc": "2D field array"},
            {"name": "position", "type": "float", "doc": "Dynamic slice position"},
        ]

    @property
    def optional_inputs(self):
        return [1]

    @property
    def outputs(self):
        return [
            {"name": "slice", "type": "array", "doc": "1D slice array"},
        ]

    def draw_icon(self, block_rect):
        """Draw slice icon - grid with line through it."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()

        # Grid
        path.addRect(0.2, 0.2, 0.6, 0.6)

        # Slice line
        path.moveTo(0.2, 0.5)
        path.lineTo(0.8, 0.5)

        # Arrow
        path.moveTo(0.7, 0.45)
        path.lineTo(0.8, 0.5)
        path.lineTo(0.7, 0.55)

        return path

    def execute(self, time, inputs, params, **kwargs):
        """Extract 1D slice from 2D field."""
        field = inputs.get(0, None)
        if field is None:
            return {0: np.array([0.0]), 'E': False}

        field = np.atleast_2d(field)
        Ny, Nx = field.shape

        # Get slice position
        position = inputs.get(1, None)
        if position is None:
            position = float(params.get('slice_position', 0.5))

        direction = params.get('slice_direction', 'x')

        if direction.lower() == 'x':
            # Horizontal slice (constant y)
            j = int(position * (Ny - 1))
            j = max(0, min(Ny - 1, j))
            slice_arr = field[j, :]
        else:
            # Vertical slice (constant x)
            i = int(position * (Nx - 1))
            i = max(0, min(Nx - 1, i))
            slice_arr = field[:, i]

        return {0: slice_arr, 'E': False}
