
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
