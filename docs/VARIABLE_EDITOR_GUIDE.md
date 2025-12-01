# Variable Editor Usage Guide

## Accessing the Variable Editor

**macOS**: `Cmd+Shift+V` or View → Show/Hide Variable Editor
**Windows/Linux**: `Ctrl+Shift+V` or View → Show/Hide Variable Editor

## Defining Variables

The Variable Editor accepts Python syntax for defining variables:

```python
# Numbers
K = 10
amplitude = 5.5

# Lists (for transfer function numerators/denominators)
num = [1, 2]
den = [1, 3, 2]

# More complex lists
coefficients = [1.5, 2.3, 3.7, 4.2]
```

## Using Variables in Blocks

1. Define your variables in the Variable Editor
2. Click "Update Workspace"  
3. In block parameters, type the variable name instead of the value:
   - Transfer Function numerator: `num` (instead of `[1, 2]`)
   - Transfer Function denominator: `den` (instead of `[1, 3, 2]`)
   - Gain value: `K` (instead of `10`)

## How It Works

- Variables are stored in `WorkspaceManager` (singleton)
- During simulation initialization (`execution_init`), variable names in `block.params` are resolved
- Resolved values are stored in `block.exec_params` and used during execution
- This allows you to change values in one place and have all blocks update automatically

## Example Workflow

```python
# In Variable Editor:
K = 10
num = [1, 2]
den = [1, 3, 2]
```

Then in your blocks:
- **Gain Block**: Set `gain` parameter to `K`
- **Transfer Function Block**: Set `num` to `num` and `den` to `den`

When you update `K = 20` in the Variable Editor and click "Update Workspace", all gain blocks using `K` will use the new value in the next simulation!
