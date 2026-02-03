"""
Base block module providing the abstract base class for all simulation blocks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray

# Type aliases for block interfaces
BlockParams = Dict[str, Any]
BlockInputs = Dict[int, Union[float, int, NDArray[np.floating]]]
BlockOutput = Dict[int, Union[float, int, NDArray[np.floating]]]
BlockResult = Union[BlockOutput, Dict[str, Union[bool, str]]]
PortDefinition = Dict[str, str]


class BaseBlock(ABC):
    """
    Abstract base class for all simulation blocks.

    All custom blocks must inherit from this class and implement
    the required abstract properties and methods.

    Attributes:
        optional_inputs: Set of port indices that are optional (not required to be connected).
        optional_outputs: Set of port indices that are optional outputs.
    """

    # Optional ports (override in subclasses)
    optional_inputs: set = set()
    optional_outputs: set = set()

    @property
    @abstractmethod
    def block_name(self) -> str:
        """The user-facing name of the block."""
        pass

    @property
    @abstractmethod
    def params(self) -> BlockParams:
        """A dictionary defining the block's parameters, their types, and default values."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> List[PortDefinition]:
        """A list of input port definitions."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[PortDefinition]:
        """A list of output port definitions."""
        pass

    @abstractmethod
    def execute(self, time: float, inputs: BlockInputs, params: BlockParams, **kwargs: Any) -> BlockResult:
        """
        The core simulation function for the block.

        Args:
            time: The current simulation time.
            inputs: A dictionary of input values, keyed by port index.
            params: A dictionary of the block's current parameter values.
            **kwargs: Additional keyword arguments (e.g., output_only, dtime).

        Returns:
            A dictionary of output values keyed by port index,
            or an error dict {'E': True, 'error': 'message'} on failure.
        """
        pass

    @property
    def use_port_grid_snap(self) -> bool:
        """
        Whether port positions should snap to grid.

        Some blocks (like triangular gain blocks) need precise port alignment
        without grid snapping for proper visual geometry.

        Returns:
            True to snap ports to grid, False otherwise.
        """
        return True  # Default: use grid snapping

    @property
    def requires_inputs(self) -> bool:
        """
        Whether this block requires all inputs to be connected.

        Source blocks typically don't require inputs, while most other blocks do.
        Override this in subclasses for custom behavior.

        Returns:
            True if inputs must be connected, False otherwise.
        """
        # Default: blocks require inputs unless they're Sources
        return getattr(self, 'category', 'Other') not in ['Sources']

    @property
    def requires_outputs(self) -> bool:
        """
        Whether this block requires outputs to be connected.

        Sink blocks and utility blocks typically don't require outputs to be connected.
        Override this in subclasses for custom behavior.

        Returns:
            True if outputs must be connected, False otherwise.
        """
        # Default: blocks require outputs unless they're Sinks or Other
        return getattr(self, 'category', 'Other') not in ['Sinks', 'Other']

    def draw_icon(self, block_rect: Any) -> Optional[Any]:
        """
        Return a QPainterPath for the block's icon in normalized coordinates.

        The path should use coordinates from 0.0 to 1.0 where (0,0) is top-left
        and (1,1) is bottom-right. The path will be automatically scaled and
        positioned within the block's drawing area.

        Override this method in subclasses to provide custom block icons.
        If not overridden, the fallback switch-based drawing in DBlock is used.

        Args:
            block_rect: QRect of the block (for context, not for positioning).

        Returns:
            QPainterPath in 0-1 normalized coordinates, or None to use fallback.
        """
        return None  # Default: use fallback switch-based drawing

    def symbolic_execute(
        self, inputs: Dict[int, Any], params: BlockParams
    ) -> Optional[Dict[int, Any]]:
        """
        Symbolic execution for equation extraction and linearization.

        This method returns symbolic expressions (using SymPy) instead of
        numeric values. Used by the SymbolicEngine for:
        - Automatic equation extraction
        - Transfer function computation
        - Linearization at operating points
        - LaTeX/MathML export

        Override this method in subclasses to provide symbolic behavior.
        If not overridden, returns None indicating no symbolic support.

        Args:
            inputs: Dict of symbolic input expressions (port_idx -> sympy expr).
            params: Dict of block parameters (may contain symbolic params).

        Returns:
            Dict of symbolic output expressions {port_idx: sympy_expr}, or
            None if block doesn't support symbolic execution.

        Example for Gain block::

            def symbolic_execute(self, inputs, params):
                from sympy import Symbol
                K = params.get('gain', Symbol('K'))
                u = inputs.get(0, Symbol('u'))
                return {0: K * u}
        """
        return None  # Default: no symbolic support

    def get_symbolic_params(self, params: BlockParams) -> Dict[str, Any]:
        """
        Convert numeric parameters to symbolic for equation extraction.

        Override to specify which parameters should be symbolic
        and how they should be named.

        Args:
            params: Dict of block parameters.

        Returns:
            Dict of symbolic parameters (name -> sympy Symbol).
        """
        return {}  # Default: no symbolic parameters
