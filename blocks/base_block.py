
from abc import ABC, abstractmethod

class BaseBlock(ABC):
    """
    Abstract base class for all simulation blocks.
    """

    @property
    @abstractmethod
    def block_name(self):
        """The user-facing name of the block."""
        pass


    @property
    @abstractmethod
    def params(self):
        """A dictionary defining the block's parameters, their types, and default values."""
        pass

    @property
    @abstractmethod
    def inputs(self):
        """A list of input port definitions."""
        pass

    @property
    @abstractmethod
    def outputs(self):
        """A list of output port definitions."""
        pass

    @abstractmethod
    def execute(self, time, inputs, params):
        """
        The core simulation function for the block.

        :param time: The current simulation time.
        :param inputs: A dictionary of input values, keyed by port index.
        :param params: A dictionary of the block's current parameter values.
        :return: A dictionary of output values, keyed by port index.
        """
        pass

    @property
    def use_port_grid_snap(self):
        """
        Whether port positions should snap to grid.

        Some blocks (like triangular gain blocks) need precise port alignment
        without grid snapping for proper visual geometry.

        :return: True to snap ports to grid, False otherwise.
        """
        return True  # Default: use grid snapping

    @property
    def requires_inputs(self):
        """
        Whether this block requires all inputs to be connected.

        Source blocks typically don't require inputs, while most other blocks do.
        Override this in subclasses for custom behavior.

        :return: True if inputs must be connected, False otherwise.
        """
        # Default: blocks require inputs unless they're Sources
        return getattr(self, 'category', 'Other') not in ['Sources']

    @property
    def requires_outputs(self):
        """
        Whether this block requires outputs to be connected.

        Sink blocks and utility blocks typically don't require outputs to be connected.
        Override this in subclasses for custom behavior.

        :return: True if outputs must be connected, False otherwise.
        """
        # Default: blocks require outputs unless they're Sinks or Other
        return getattr(self, 'category', 'Other') not in ['Sinks', 'Other']

    def draw_icon(self, block_rect):
        """
        Return a QPainterPath for the block's icon in normalized coordinates.
        
        The path should use coordinates from 0.0 to 1.0 where (0,0) is top-left
        and (1,1) is bottom-right. The path will be automatically scaled and 
        positioned within the block's drawing area.
        
        Override this method in subclasses to provide custom block icons.
        If not overridden, the fallback switch-based drawing in DBlock is used.
        
        Args:
            block_rect: QRect of the block (for context, not for positioning)
            
        Returns:
            QPainterPath in 0-1 normalized coordinates, or None to use fallback
        """
        return None  # Default: use fallback switch-based drawing
