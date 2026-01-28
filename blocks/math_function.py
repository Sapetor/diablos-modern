
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
- Exponential: exp, log (ln), log10, sqrt, square
- Operational: sign, abs, ceil, floor, reciprocal

Select the function via the block parameters."""

    @property
    def inputs(self):
        return [{"name": "u", "type": "float"}]

    @property
    def outputs(self):
        return [{"name": "y", "type": "float"}]

    def execute(self, time, inputs, params):
        u = inputs.get("u", 0.0)
        func = params.get("function", "sin")
        
        try:
            if func == "sin":
                return {"y": np.sin(u)}
            elif func == "cos":
                return {"y": np.cos(u)}
            elif func == "tan":
                return {"y": np.tan(u)}
            elif func == "asin":
                return {"y": np.arcsin(u) if -1 <= u <= 1 else 0.0} # Safety or let it warn?
            elif func == "acos":
                return {"y": np.arccos(u) if -1 <= u <= 1 else 0.0}
            elif func == "atan":
                return {"y": np.arctan(u)}
            elif func == "exp":
                return {"y": np.exp(u)}
            elif func == "log":
                return {"y": np.log(u) if u > 0 else 0.0}
            elif func == "log10":
                return {"y": np.log10(u) if u > 0 else 0.0}
            elif func == "sqrt":
                return {"y": np.sqrt(u) if u >= 0 else 0.0}
            elif func == "square":
                return {"y": u * u}
            elif func == "sign":
                return {"y": np.sign(u)}
            elif func == "abs":
                return {"y": np.abs(u)}
            elif func == "ceil":
                return {"y": np.ceil(u)}
            elif func == "floor":
                return {"y": np.floor(u)}
            elif func == "reciprocal":
                return {"y": 1.0/u if u != 0 else 0.0}
            else:
                return {"y": u}
        except Exception:
            # Fallback for domain errors like log(-1) if checks fail
            return {"y": 0.0}

    def draw_icon(self, block_rect):
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        
        # We can't easily draw generic text in path, 
        # so we rely on the labelrenderer in modern_canvas that handles 'name' or params?
        # Typically blocks draw a symbol. Here the symbol IS the function parameter.
        # But draw_icon returns a path, not text.
        # We'll draw a generic 'f(x)' looking curve or let the renderer handle text?
        # The block renderer usually draws the block name.
        # But for 'Math', showing structure is nice.
        
        # Generic function curve f(x)
        path.moveTo(0.2, 0.8)
        path.cubicTo(0.4, 0.8, 0.6, 0.2, 0.8, 0.2)
        
        return path
