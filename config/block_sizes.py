"""Block Size Configuration
Defines default sizes for different block types in the DiaBloS Modern UI.
Sizes are specified as (width, height) tuples in pixels.
"""

from typing import Tuple

# Default size for blocks not specified
DEFAULT_BLOCK_SIZE = (100, 80)

# Block-specific sizes
# Format: 'BlockFunctionName': (width, height)
BLOCK_SIZES = {
    # Math operations - can be smaller
    'Sum': (60, 50),
    'Product': (60, 50),
    'Gain': (70, 50),
    'Abs': (60, 50),
    'Sqrt': (60, 50),

    # Sources - medium size
    'Step': (80, 60),
    'Ramp': (80, 60),
    'Sine': (80, 60),
    'Constant': (70, 50),

    # Control blocks - standard size
    'Integrator': (90, 70),
    'Derivative': (90, 70),
    'PID': (100, 80),
    'TranFn': (110, 80),
    'DiscreteTranFn': (110, 80),
    'StateSpace': (120, 90),
    'DiscreteStateSpace': (120, 90),
    'ZeroOrderHold': (90, 70),

    # Sinks - wider for labels/displays
    'Scope': (100, 80),
    'Display': (90, 70),
    'Export': (90, 70),
    'Term': (80, 60),

    # Routing - compact
    'Mux': (70, 60),
    'Demux': (70, 60),
    'Switch': (90, 90),
    'Goto': (70, 60),
    'From': (70, 60),
}

# Minimum block size to prevent blocks from becoming too small
MIN_BLOCK_WIDTH = 50
MIN_BLOCK_HEIGHT = 40

# Maximum block size to prevent blocks from becoming too large
MAX_BLOCK_WIDTH = 300
MAX_BLOCK_HEIGHT = 300

# Resize handle size (pixels)
RESIZE_HANDLE_SIZE = 8


def get_block_size(block_fn: str) -> Tuple[int, int]:
    """
    Get the default size for a block type.

    Args:
        block_fn: Block function name (e.g., 'Sum', 'Integrator')

    Returns:
        Tuple of (width, height) in pixels
    """
    return BLOCK_SIZES.get(block_fn, DEFAULT_BLOCK_SIZE)


def clamp_block_size(width: int, height: int) -> Tuple[int, int]:
    """
    Ensure block size is within allowed bounds.

    Args:
        width: Desired width
        height: Desired height

    Returns:
        Tuple of (clamped_width, clamped_height)
    """
    clamped_width = max(MIN_BLOCK_WIDTH, min(width, MAX_BLOCK_WIDTH))
    clamped_height = max(MIN_BLOCK_HEIGHT, min(height, MAX_BLOCK_HEIGHT))
    return (clamped_width, clamped_height)
