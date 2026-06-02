"""
PropertyController -- handles block parameter edits coming from the property
editor (and a few menu actions), plus pinning parameters to the tuning panel.

Extracted verbatim (behavior-preserving) from ``ModernDiaBloSWindow`` so the
main window keeps only thin facades. Follows the same manager pattern as the
other ``modern_ui/managers`` (constructed with the main window, held as
``self.window``).

Note: this is distinct from ``PropertyEditor._on_property_changed`` (a widget
method with signature ``(key, value)`` that emits the ``property_changed``
signal). This controller backs the window-side slot
``(block_name, prop_name, new_value)`` that signal connects to.
"""

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PropertyController:
    """Applies property-editor changes to blocks and the tuning panel."""

    def __init__(self, main_window):
        self.window = main_window

    def convert_param_value(self, new_value, target_type):
        """
        Convert a parameter value to the target type, handling variables.

        Args:
            new_value: The string input from the user.
            target_type: The expected type (int, float, bool, list).

        Returns:
            The converted value.

        Raises:
            ValueError, TypeError, SyntaxError: If conversion fails and it's not a valid variable name.
        """
        try:
            # Try to convert to the expected type
            if target_type == bool:
                if isinstance(new_value, str):
                    return new_value.lower() == 'true'
                return bool(new_value)
            elif target_type == list:
                converted = ast.literal_eval(new_value)
                if not isinstance(converted, list):
                    raise TypeError("Input must be a list (e.g., [1, 2, 3])")
                return converted
            elif target_type == int:
                return int(new_value)
            elif target_type == float:
                return float(new_value)
            else:
                return str(new_value)
        except (ValueError, TypeError, SyntaxError):
            # If conversion fails, treat as a string (potential variable name or expression)
            # We allow this so that expressions like '[K, K]' or '2*K' can be stored as strings
            # and resolved later by the WorkspaceManager.
            logger.debug(f"Could not convert '{new_value}' to {target_type}, keeping as string.")
            return str(new_value)

    def on_property_changed(self, block_name: str, prop_name: str, new_value: Any) -> None:
        """Handle property changes from the property editor."""
        canvas = self.window.canvas
        param_type = None
        try:
            for block in canvas.dsim.blocks_list:
                if block.name == block_name:
                    # Handle username change (special case - not in params)
                    if prop_name == '_username_':
                        canvas.dsim.dirty = True
                        canvas.update()
                        return

                    # Handle port count change from property editor
                    if prop_name in ('_inputs_', '_outputs_'):
                        canvas._push_undo("Edit Ports")
                        if prop_name == '_inputs_':
                            block.in_ports = int(new_value)
                        else:
                            block.out_ports = int(new_value)
                        block.update_Block()
                        block.params['_inputs_'] = block.in_ports
                        block.params['_outputs_'] = block.out_ports
                        canvas.dsim.dirty = True
                        canvas.update()
                        return

                    param_type = type(block.params.get(prop_name))

                    # If the value is already a list or numpy array, preserve it
                    # (property editor may have already converted it for accepts_array params)
                    if isinstance(new_value, (list, tuple)):
                        converted_value = list(new_value)
                    elif hasattr(new_value, 'tolist'):  # numpy array
                        converted_value = new_value
                    else:
                        converted_value = self.convert_param_value(new_value, param_type)

                    logger.debug(f"Updating {block_name}.{prop_name} to {converted_value} (type: {type(converted_value).__name__})")
                    block.update_params({prop_name: converted_value})
                    canvas.dsim.dirty = True
                    # For Goto/From blocks, refresh labels and virtual links immediately
                    if block.block_fn in ("Goto", "From") and prop_name in ("tag", "signal_name"):
                        try:
                            canvas.dsim.model.link_goto_from()
                        except Exception as e:
                            logger.warning(f"Could not relink Goto/From after property change: {e}")
                    # Refresh canvas to show updated block visuals (Sum signs, labels, etc.)
                    canvas.update()
                    break
        except (ValueError, TypeError, SyntaxError) as e:
            logger.error(f"Failed to convert property {prop_name} to type {param_type}: {e}")
            self.window.show_error(f"Invalid input for {prop_name}: {e}")
        except Exception as e:
            logger.error(f"Error updating property: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def add_to_tuning(self, block, param_name):
        """Add a block parameter to the tuning panel."""
        self.window.tuning_panel.add_parameter(block, param_name)
