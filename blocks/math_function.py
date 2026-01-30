
import numpy as np
from blocks.base_block import BaseBlock

class MathFunctionBlock(BaseBlock):
    def __init__(self):
        super().__init__()

    @property
    def block_name(self):
        return "MathFunction"

    @property
    def category(self):
        return "Math"

    @property
    def color(self):
        return "green"

    @property
    def params(self):
        return {
            "function": {
                "default": "sin",
                "type": "choice",
                "options": [
                    "sin", "cos", "tan", 
                    "asin", "acos", "atan", 
                    "exp", "log", "log10", 
                    "sqrt", "square", "sign", 
                    "abs", "ceil", "floor",
                    "reciprocal"
                ]
            }
        }

    @property
    def doc(self):
        return """Apply a mathematical function to the input signal.

Supported Functions:
- Trigonometric: sin, cos, tan, asin, acos, atan
- Exponential: exp, log (ln), log10, sqrt, square, cube
- Operational: sign, abs, ceil, floor, reciprocal
- Python Syntax: You can enter any valid Python expression using 'u' (input) and 't' (time), e.g., 'u**2 + sin(t*10)'

Select the function via the block parameters."""

    @property
    def inputs(self):
        return [{"name": "u", "type": "float"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "float"}]

    def execute(self, time, inputs, params, **kwargs):
        # inputs are keyed by port index (0), but fallback to "u" just in case
        u = inputs.get(0, inputs.get("u", 0.0))
        
        # Ensure function name is lowercase for comparison
        func = str(params.get("function", "sin")).lower()
        
        try:
            if func == "sin":
                return {0: np.sin(u)}
            elif func == "cos":
                return {0: np.cos(u)}
            elif func == "tan":
                return {0: np.tan(u)}
            elif func == "asin":
                return {0: np.arcsin(u) if -1 <= u <= 1 else 0.0} # Safety or let it warn?
            elif func == "acos":
                return {0: np.arccos(u) if -1 <= u <= 1 else 0.0}
            elif func == "atan":
                return {0: np.arctan(u)}
            elif func == "exp":
                return {0: np.exp(u)}
            elif func == "log":
                return {0: np.log(u) if u > 0 else 0.0}
            elif func == "log10":
                return {0: np.log10(u) if u > 0 else 0.0}
            elif func == "sqrt":
                return {0: np.sqrt(u) if u >= 0 else 0.0}
            elif func == "square":
                return {0: u * u}
            elif func == "sign":
                return {0: np.sign(u)}
            elif func == "abs":
                return {0: np.abs(u)}
            elif func == "ceil":
                return {0: np.ceil(u)}
            elif func == "floor":
                return {0: np.floor(u)}
            elif func == "reciprocal":
                return {0: 1.0/u if u != 0 else 0.0}
            elif func == "cube":
                return {0: u * u * u}
            
            # Python Syntax Fallback
            # Try to evaluate as a Python expression
            # Context: u (input), t (time), np (numpy), and standard math functions
            context = {
                "u": u, "t": time,
                "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                "exp": np.exp, "log": np.log, "log10": np.log10,
                "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
                "ceil": np.ceil, "floor": np.floor,
                "pi": np.pi, "e": np.e,
                "np": np
            }
            # Use the raw function string (process original case) to allow correct python syntax if needed
            # but we already lowered it? 
            # Ideally we should use params.get('function') raw. 
            # But the 'func' variable above is lowered.
            # Let's use the raw param for eval to respect variable cases if user types 'np.sin(u)' (though we inject np).
            raw_func = str(params.get("function", "sin"))
            return {0: float(eval(raw_func, {"__builtins__": None}, context))}

        except Exception:
            # Fallback for domain errors or invalid syntax
            return {0: 0.0}

    def draw_icon(self, block_rect):
        """MathFunction uses f(·) text rendering - handled in DBlock switch."""
        return None
