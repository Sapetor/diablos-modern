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

    NOTE: This is a placeholder. External execution is NOT implemented and
    is intentionally disabled in the simulation engine for security reasons:
    the `external`/`function` fields are persisted in project files, so
    dynamically loading and calling code from a file-supplied path would turn
    opening an untrusted .diablos file into arbitrary code execution. Any
    future implementation must require explicit user confirmation and must not
    auto-execute code referenced by a loaded project file.
    """

    # Keep this block out of the block palette. ``execute()`` only ever
    # returns an error dict and the engine hard-refuses External blocks, so
    # offering it for placement can only produce a diagram that fails to run.
    # Plain class attribute (not a BaseBlock property) so the palette can read
    # it with getattr(block_class, "hidden", False) without every other block
    # having to declare one.
    hidden = True

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
            "filename": {"default": "", "type": "string", "doc": "Path to external Python file"},
            "function": {"default": "execute", "type": "string", "doc": "Function name to call"},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a bracketed source-file glyph (external code)."""
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        path.moveTo(0.2, 0.2)
        path.lineTo(0.8, 0.2)
        path.moveTo(0.2, 0.5)
        path.lineTo(0.6, 0.5)
        path.moveTo(0.2, 0.8)
        path.lineTo(0.8, 0.8)
        path.moveTo(0.2, 0.2)
        path.lineTo(0.2, 0.8)
        return path

    def execute(self, time, inputs, params, **kwargs):
        """
        Execute external function.

        NOTE: External execution is disabled in the simulation engine for
        security reasons, so this stub is the only code path. It always
        returns an error.
        """
        filename = params.get("filename", "")
        if not filename:
            logger.warning("External block: No filename specified")
            return {0: 0.0, "E": True, "error": "No external file specified"}

        logger.warning(f"External block: file_function not loaded for {filename}")
        return {0: 0.0, "E": True, "error": f"External file not loaded: {filename}"}
