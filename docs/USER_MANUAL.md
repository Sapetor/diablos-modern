# DiaBloS User Manual

## Getting Started

### Running the Application
```bash
python main.py
```

### Interface Overview
- **Left Panel**: Block palette - drag blocks onto canvas
- **Center**: Canvas - build your diagram
- **Right Panel**: Property editor - configure selected block
- **Bottom**: Variable editor - define workspace variables

---

## Basic Workflow

### 1. Add Blocks
Drag blocks from the palette onto the canvas.

### 2. Connect Blocks
Click an output port and drag to an input port.

### 3. Configure Parameters
Click a block, then edit its parameters in the property editor.

### 4. Define Variables
Use the variable editor to define constants:
```python
K = 2.5
A = [1, 2, 3]
M = [[1, 0], [0, 1]]
```

### 5. Run Simulation
- **F5** or **Run → Start Simulation**
- Configure simulation time and step size
- View results in Scope blocks

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+C | Copy |
| Ctrl+V | Paste |
| Ctrl+D | Duplicate |
| Delete | Delete selected |
| Ctrl+S | Save |
| Ctrl+O | Open |
| F5 | Run simulation |

---

## MIMO Support

DiaBloS supports vector and matrix signals:

### Vector Signals
- Constant block: Set `value = [1, 2, 3]`
- Connections automatically carry vector data
- Thicker lines indicate vector signals (during simulation)

### Matrix Gain
- Gain block supports matrix multiplication: `gain = [[1,0],[0,1]]`
- Output: `y = K @ u` (matrix-vector multiplication)

### Selector Block
- Extract specific elements from vectors
- Use indices like `0,2` or `1:3`

---

## Tips

- **Flip blocks**: Right-click → Flip
- **Resize blocks**: Drag corner handles
- **Toggle routing**: Right-click connection → Toggle Routing Mode
- **Test incrementally**: Build and test small sections first
