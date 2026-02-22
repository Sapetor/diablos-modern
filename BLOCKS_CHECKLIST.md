# DiaBloS Block Development Checklist

## Before Creating a Block

- [ ] Determine block category (Sources, Sinks, Math, Control, Routing, PDE, Optimization, Other)
- [ ] Decide on file location:
  - [ ] `blocks/myblock.py` for standard blocks
  - [ ] `blocks/pde/myblock.py` for PDE blocks
  - [ ] `blocks/optimization_primitives/myblock.py` for optimization primitives
  - [ ] `blocks/optimization/myblock.py` for parameter optimization
- [ ] Review similar existing blocks for patterns
- [ ] Check if parameters can use templates from `blocks/param_templates.py`

## Block Class Implementation

### Required Properties (all blocks must have these)

- [ ] `block_name` - User-facing name (e.g., "MyGain")
- [ ] `category` - One of: Sources, Sinks, Math, Control, Routing, PDE, Other
- [ ] `color` - UI color name (yellow, magenta, green, orange, purple, etc.)
- [ ] `doc` - Documentation string with usage notes
- [ ] `params` - Dictionary of parameter definitions
- [ ] `inputs` - List of input port definitions
- [ ] `outputs` - List of output port definitions
- [ ] `execute()` - Core simulation function

### Parameter Definition

For each parameter in `params` dict:

- [ ] `type` field specified (float, int, string, list, bool)
- [ ] `default` value provided
- [ ] `doc` field with description (highly recommended)
- [ ] For enums: `"choices"` list provided
- [ ] For arrays: `"accepts_array": True` if scalar can promote to vector

Example:
```python
@property
def params(self):
    return {
        "gain": {
            "type": "float",
            "default": 1.0,
            "doc": "Scaling gain factor"
        },
        "method": {
            "type": "string",
            "default": "linear",
            "doc": "Interpolation method",
            "choices": ["linear", "cubic", "spline"]
        }
    }
```

### Port Definitions

For each input/output port:

- [ ] `name` field provided
- [ ] `type` field specified (any, float, vector, array, matrix)
- [ ] `doc` field with description

Example:
```python
@property
def inputs(self):
    return [
        {"name": "u", "type": "any", "doc": "Input signal"},
        {"name": "reference", "type": "float", "doc": "Reference value"}
    ]

@property
def outputs(self):
    return [
        {"name": "y", "type": "any", "doc": "Output signal"}
    ]
```

### Execute Method

- [ ] Signature: `execute(self, time, inputs, params, **kwargs) -> Dict`
- [ ] Extract inputs safely using helpers:
  - [ ] Use `get_scalar(inputs, port)` for scalar extraction
  - [ ] Use `get_vector(inputs, port, dim)` for vector extraction
  - [ ] Use `InitStateManager(params)` for state initialization
- [ ] Return dictionary with:
  - [ ] Output values: `{0: value1, 1: value2}`
  - [ ] Error flag: `'E': False` (True if error occurred)
  - [ ] Optional error message: `'error': 'message'` (if E=True)

Example:
```python
def execute(self, time, inputs, params, **kwargs):
    from blocks.input_helpers import get_scalar
    
    u = get_scalar(inputs, 0, default=0.0)
    gain = params.get("gain", 1.0)
    
    try:
        y = u * gain
        return {0: y, 'E': False}
    except Exception as e:
        return {'E': True, 'error': str(e)}
```

## Optional Features

### Custom Icon (Visual Representation)

- [ ] Implement `draw_icon(self, block_rect)` if needed
- [ ] Use normalized 0-1 coordinates (0,0 = top-left, 1,1 = bottom-right)
- [ ] Return `QPainterPath` or None to use default

Example:
```python
def draw_icon(self, block_rect):
    from PyQt5.QtGui import QPainterPath
    path = QPainterPath()
    path.moveTo(0.2, 0.5)
    path.lineTo(0.8, 0.5)
    return path
```

### Symbolic Execution (Equation Extraction)

- [ ] Implement `symbolic_execute(self, inputs, params)` if applicable
- [ ] Return dict of SymPy expressions or None
- [ ] Used for automatic equation generation

Example:
```python
def symbolic_execute(self, inputs, params):
    try:
        from sympy import Symbol
        K = Symbol('K')
        u = inputs.get(0, Symbol('u'))
        return {0: K * u}
    except ImportError:
        return None
```

### Optional Ports

- [ ] Define `optional_inputs = {0, 2}` if some ports don't require connections
- [ ] Define `optional_outputs = {1}` for outputs not requiring connections
- [ ] Document which ports are optional in the `doc` property

Example:
```python
class MyBlock(BaseBlock):
    optional_inputs = {0}      # Input 0 is optional
    optional_outputs = {1}     # Output 1 is optional
```

### Special Block Types

- [ ] If feedthrough (output depends on input at same timestep): `b_type = 2`
- [ ] If state-based (no direct feedthrough): `b_type = 0` (default)
- [ ] If needs custom port grid behavior: override `use_port_grid_snap`
- [ ] If source block (no inputs required): override `requires_inputs` to return False
- [ ] If sink block (no outputs required): override `requires_outputs` to return False

## Parameter Template Reuse

When applicable, use reusable templates to reduce code duplication:

- [ ] `init_conds_param()` - Initial condition value
- [ ] `init_flag_param()` - Initialization flag
- [ ] `limit_params(min, max)` - Min/max saturation
- [ ] `method_param(choices, default)` - Method selection
- [ ] `domain_params_1d(L, N)` - 1D spatial domain
- [ ] `domain_params_2d(Lx, Ly, Nx, Ny)` - 2D spatial domain
- [ ] `slew_rate_params()` - Rate limiter rates
- [ ] PDE-specific templates (diffusivity, wave_speed, etc.)

Example:
```python
from blocks.param_templates import init_conds_param, method_param

@property
def params(self):
    return {
        **init_conds_param(default=0.0),
        **method_param(["RK45", "Euler"], default="RK45"),
        "custom_param": {"type": "float", "default": 1.0}
    }
```

## Code Quality Checklist

- [ ] Class inherits from `BaseBlock`
- [ ] All properties are @property decorated
- [ ] No hardcoded magic numbers (use defaults in params)
- [ ] Error handling in execute() (try/except)
- [ ] Proper numpy usage (np.atleast_1d, np.array, etc.)
- [ ] Type hints in docstrings (optional but recommended)
- [ ] Input validation (check for None, NaN, etc.)
- [ ] State management uses InitStateManager pattern

## Testing Checklist

- [ ] Create unit test in `tests/unit/test_myblock.py`
- [ ] Test cases include:
  - [ ] Default parameter execution
  - [ ] Custom parameter values
  - [ ] Edge cases (zero, negative, large values)
  - [ ] Error conditions
  - [ ] Vector/array inputs if applicable
- [ ] All tests pass: `pytest tests/unit/test_myblock.py -v`
- [ ] Run full test suite: `pytest tests/ -v`

Example test:
```python
import pytest
from blocks.myblock import MyBlockClass

def test_myblock_default_params():
    block = MyBlockClass()
    params = block.params
    inputs = {0: 2.0}
    result = block.execute(0, inputs, params)
    assert result['E'] == False
    assert result[0] == pytest.approx(2.0)

def test_myblock_custom_gain():
    block = MyBlockClass()
    params = {**block.params}
    params["gain"]["default"] = 5.0
    inputs = {0: 2.0}
    result = block.execute(0, inputs, params)
    assert result[0] == pytest.approx(10.0)
```

## Subdirectory Package Setup

If creating a block in `blocks/pde/`, `blocks/optimization_primitives/`, or `blocks/optimization/`:

- [ ] Ensure `blocks/<subdir>/__init__.py` exists
- [ ] Add import: `from blocks.<subdir>.myblock import MyBlockClass`
- [ ] Update `__all__` list with new block name
- [ ] No other changes needed - loader will auto-discover

Example (in `blocks/pde/__init__.py`):
```python
from blocks.pde.myblock import MyBlockClass

__all__ = [
    'MyBlockClass',
    # ... other blocks
]
```

## Documentation

- [ ] `doc` property has clear description
- [ ] `doc` includes parameter explanations
- [ ] `doc` includes input/output descriptions
- [ ] Example usage in `doc` (optional but helpful)
- [ ] Complex blocks: create wiki page in `docs/wiki/`

Example:
```python
@property
def doc(self):
    return (
        "Custom Block: Does X using Y."
        "\n\nParameters:"
        "\n- gain: Scaling factor"
        "\n- method: Algorithm (linear, cubic)"
        "\n\nInputs:"
        "\n- u: Input signal"
        "\n\nOutputs:"
        "\n- y: Scaled output"
    )
```

## Pre-Commit Checklist

Before committing:

- [ ] All tests pass
- [ ] No lint errors (if linter configured)
- [ ] Documentation complete
- [ ] Block appears in palette when app runs
- [ ] No import errors
- [ ] Code follows project style (PEP 8)
- [ ] Commit message: "Add [BlockName] block for [description]"

## Common Pitfalls to Avoid

❌ **Don't**:
- Forget to implement required properties
- Use bare `except:` without handling
- Modify global state in execute()
- Assume inputs are always connected (use defaults)
- Create deeply nested inheritance hierarchies
- Hardcode array dimensions (use params)
- Return None from execute() (return error dict instead)

✓ **Do**:
- Use parameter templates when available
- Use input helpers (get_scalar, get_vector)
- Document every parameter and port
- Include error handling in execute()
- Test with edge cases
- Follow existing block patterns
- Use `optional_inputs`/`optional_outputs` for optional connections

## After Creating Block

- [ ] Run app: `python main.py`
- [ ] Check block appears in palette
- [ ] Try placing block on canvas
- [ ] Try connecting to other blocks
- [ ] Try editing parameters
- [ ] Run a simple simulation with block
- [ ] Commit with clear message
- [ ] Consider adding to wiki examples if complex

---

## Quick Templates

### Minimal Block (copy & paste)
```python
from blocks.base_block import BaseBlock
import numpy as np

class MyNewBlock(BaseBlock):
    @property
    def block_name(self):
        return "MyBlock"
    
    @property
    def category(self):
        return "Math"
    
    @property
    def color(self):
        return "yellow"
    
    @property
    def doc(self):
        return "Block description"
    
    @property
    def params(self):
        return {
            "param1": {"type": "float", "default": 1.0, "doc": "Description"}
        }
    
    @property
    def inputs(self):
        return [{"name": "u", "type": "any"}]
    
    @property
    def outputs(self):
        return [{"name": "y", "type": "any"}]
    
    def execute(self, time, inputs, params, **kwargs):
        u = inputs.get(0, 0.0)
        y = u * params.get("param1", 1.0)
        return {0: y, 'E': False}
```

### With State Management
```python
from blocks.base_block import BaseBlock
from blocks.input_helpers import InitStateManager
import numpy as np

class MyStatefulBlock(BaseBlock):
    @property
    def block_name(self):
        return "MyStateful"
    
    @property
    def category(self):
        return "Control"
    
    @property
    def color(self):
        return "magenta"
    
    @property
    def doc(self):
        return "Stateful block description"
    
    @property
    def params(self):
        from blocks.param_templates import init_conds_param, init_flag_param
        return {
            **init_conds_param(default=0.0),
            **init_flag_param(),
            "tau": {"type": "float", "default": 1.0, "doc": "Time constant"}
        }
    
    @property
    def inputs(self):
        return [{"name": "u", "type": "float"}]
    
    @property
    def outputs(self):
        return [{"name": "y", "type": "float"}]
    
    def execute(self, time, inputs, params, **kwargs):
        init_mgr = InitStateManager(params)
        
        if init_mgr.needs_init():
            params['state'] = params.get('init_conds', 0.0)
            init_mgr.mark_initialized()
        
        u = inputs.get(0, 0.0)
        dt = kwargs.get('dtime', 0.01)
        tau = params.get('tau', 1.0)
        
        # Simple first-order response
        state = params['state']
        new_state = state + (dt / tau) * (u - state)
        params['state'] = new_state
        
        return {0: new_state, 'E': False}
```

## Help Resources

- **Base Class**: `/blocks/base_block.py`
- **Templates**: `/blocks/param_templates.py`
- **Helpers**: `/blocks/input_helpers.py`
- **Loader**: `/lib/block_loader.py`
- **Examples**: `/blocks/*.py` (review 3-4 similar blocks)
- **Tests**: `/tests/unit/test_*_block.py` (for test patterns)
- **Wiki**: `/docs/wiki/` (for complex block documentation)
