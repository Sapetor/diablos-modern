"""
External Block - Execute custom Python code from external files.

NOTE: This feature is currently not fully implemented. The external file
loading mechanism needs to be added to lib/simulation/block.py.
"""

import logging
from blocks.base_block import BaseBlock

logger = logging.getLogger(__name__)


class ExternalBlock(BaseBlock):
    """
    External Function Block.

    Executes custom Python code loaded from an external .py file.

    NOTE: This is a placeholder. The actual execution happens via
    block.file_function in the simulation engine, but the loading
    mechanism is not yet implemented.
    """

    def __init__(self):
        super().__init__()

    @property
    def doc(self):
        return (
            "External Function Block (NOT FULLY IMPLEMENTED)."
            "\n\nExecutes custom Python code loaded from an external file."
            "\n\nParameters:"
            "\n- filename: Path to the .py file."
            "\n- function: Name of the function to call."
            "\n\nThe function should have signature:"
            "\n  def my_function(time, inputs, params, **kwargs) -> dict"
            "\n\nReturns: {0: output_value, 'E': False}"
            "\n\nNOTE: External file loading is not yet implemented."
        )

    @property
    def block_name(self):
        return "External"

    @property
    def category(self):
        return "Other"

    @property
    def color(self):
        return "light_gray"

    @property
    def params(self):
        return {
            "filename": {
                "default": "",
                "type": "string",
                "doc": "Path to external Python file"
            },
            "function": {
                "default": "execute",
                "type": "string",
                "doc": "Function name to call"
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute external function.

        NOTE: When block.external is True, the simulation engine bypasses
        this method and calls block.file_function directly. This method
        is only called if the external file loading failed.
        """
        filename = params.get('filename', '')
        if not filename:
            logger.warning("External block: No filename specified")
            return {0: 0.0, 'E': True, 'error': 'No external file specified'}

        logger.warning(f"External block: file_function not loaded for {filename}")
        return {0: 0.0, 'E': True, 'error': f'External file not loaded: {filename}'}
