import numpy as np
from blocks.base_block import BaseBlock


class AssertBlock(BaseBlock):
    """
    Stops simulation if the input violates a condition.
    Useful for detecting invalid states or debugging.
    """

    @property
    def block_name(self):
        return "Assert"

    @property
    def category(self):
        return "Sinks"

    @property
    def color(self):
        return "red"

    @property
    def doc(self):
        return "Stops simulation if input violates condition. Modes: >0, <0, >=0, <=0, ==0, !=0, finite."

    @property
    def params(self):
        return {
            "condition": {"type": "string", "default": ">0", "doc": "Condition: >0, <0, >=0, <=0, ==0, !=0, finite"},
            "message": {"type": "string", "default": "Assertion failed", "doc": "Error message on failure."},
            "enabled": {"type": "bool", "default": True, "doc": "Enable/disable assertion check."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return []

    @property
    def requires_outputs(self):
        """Sinks don't need output connections."""
        return False

    def execute(self, time, inputs, params):
        if not params.get("enabled", True):
            return {0: np.array([0.0])}
        
        condition = params.get("condition", ">0")
        message = params.get("message", "Assertion failed")
        
        input_value = np.atleast_1d(inputs.get(0, 0))
        
        # Check condition for all elements
        passed = True
        for val in input_value.flatten():
            val = float(val)
            
            if condition == ">0":
                passed = val > 0
            elif condition == "<0":
                passed = val < 0
            elif condition == ">=0":
                passed = val >= 0
            elif condition == "<=0":
                passed = val <= 0
            elif condition == "==0":
                passed = abs(val) < 1e-10
            elif condition == "!=0":
                passed = abs(val) >= 1e-10
            elif condition == "finite":
                passed = np.isfinite(val)
            else:
                # Unknown condition, pass
                passed = True
            
            if not passed:
                break
        
        if not passed:
            # Return error signal to stop simulation
            return {
                0: np.array([0.0]),
                'E': True,
                'error': f"{message} (value={val}, condition={condition}, time={time:.4f})"
            }
        
        return {0: np.array([0.0])}
