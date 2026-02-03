# Blocks API

All simulation blocks inherit from the `BaseBlock` abstract class.

## BaseBlock

Abstract base class for all simulation blocks.

::: blocks.base_block.BaseBlock
    options:
      show_root_heading: true
      members:
        - block_name
        - params
        - inputs
        - outputs
        - execute
        - draw_icon
        - symbolic_execute

## Block Categories

### Sources

Blocks that generate signals:

- **Constant**: Outputs a constant value
- **Step**: Step function generator
- **Ramp**: Ramp signal generator
- **Sine**: Sinusoidal signal generator
- **Noise**: Random noise generator
- **WaveGenerator**: Multi-waveform generator

### Math

Mathematical operation blocks:

- **Gain**: Multiply by constant
- **Sum**: Add/subtract signals
- **Product**: Multiply signals
- **MathFunction**: Apply math functions (sin, cos, exp, etc.)
- **Abs**: Absolute value

### Control

Control system blocks:

- **Integrator**: Numeric integration
- **PID**: PID controller
- **TransferFcn**: Transfer function
- **StateSpace**: State-space system
- **Saturation**: Output limiting
- **RateLimiter**: Rate of change limiting

### PDE

Partial differential equation solvers:

- **HeatEquation1D**: 1D heat diffusion
- **WaveEquation1D**: 1D wave equation
- **AdvectionEquation1D**: 1D advection
- **HeatEquation2D**: 2D heat diffusion

### Sinks

Output and visualization blocks:

- **Scope**: Time-series plot
- **XYGraph**: X-Y plot
- **Display**: Numeric display
- **FieldScope**: 1D field visualization
- **FieldScope2D**: 2D field visualization

## Creating Custom Blocks

To create a custom block:

```python
from blocks.base_block import BaseBlock
import numpy as np

class MyCustomBlock(BaseBlock):
    @property
    def block_name(self):
        return "MyBlock"

    @property
    def category(self):
        return "Custom"

    @property
    def params(self):
        return {
            "gain": {"type": "float", "default": 1.0}
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        value = inputs.get(0, 0.0)
        gain = params.get("gain", 1.0)
        return {0: value * gain, 'E': False}
```
