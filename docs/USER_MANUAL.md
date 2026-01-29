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

### General
| Key | Action |
|-----|--------|
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+C | Copy |
| Ctrl+V | Paste |
| Ctrl+A | Select all |
| Delete | Delete selected |
| Ctrl+S | Save |
| Ctrl+O | Open |
| Ctrl+G | Create subsystem from selection |

### Simulation
| Key | Action |
|-----|--------|
| F5 | Run simulation |
| F6 | Pause simulation |
| F7 | Stop simulation |
| F8 | Single step (advance one timestep) |

### Alignment (when 2+ blocks selected)
| Key | Action |
|-----|--------|
| Ctrl+Shift+L | Align left |
| Ctrl+Shift+R | Align right |
| Ctrl+Shift+H | Align center (horizontal) |
| Ctrl+Shift+T | Align top |
| Ctrl+Shift+B | Align bottom |

### View
| Key | Action |
|-----|--------|
| Ctrl+Shift+M | Toggle minimap |
| Ctrl+Shift+V | Toggle variable editor |
| Ctrl+Shift+W | Toggle workspace viewer |
| Ctrl+T | Toggle theme (dark/light) |

---

## Subsystems

Subsystems allowed you to group blocks together to simplify large diagrams.

### Creating a Subsystem
1. Select multiple blocks using the selection rectangle or Shift+Click.
2. Press **Ctrl+G** (Group).
3. Selected blocks are replaced by a single "Subsystem" block.

### Editing a Subsystem
- **Double-click** a subsystem to enter it.
- **Press Esc** or click "Up" in the breadcrumb bar to exit.

### Adding Ports (MIMO)
- Inside a subsystem, drag **Inport** and **Outport** blocks from the palette.
- When you exit the subsystem, the parent block automatically updates to match the number and order of internal ports.
- Top-to-bottom order of internal ports corresponds to top-to-bottom order of external pins.

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

## Alignment Tools

Keep your diagrams tidy with alignment and distribution tools.

### Aligning Blocks
1. Select 2 or more blocks (Shift+Click or rectangle selection)
2. Right-click → **Align** submenu
3. Choose alignment option:
   - **Align Left/Right**: Align to leftmost/rightmost block
   - **Align Center (Horizontal)**: Align to horizontal center
   - **Align Top/Bottom**: Align to topmost/bottommost block
   - **Align Center (Vertical)**: Align to vertical center

### Distributing Blocks
With 3+ blocks selected:
- **Distribute Horizontally**: Equal horizontal spacing
- **Distribute Vertically**: Equal vertical spacing

---

## Single-Step Simulation

Debug your simulations one timestep at a time.

### How to Use
1. Press **F8** to start stepping (no need to press F5 first)
2. Simulation initializes at t=0 and advances one step
3. Press **F8** again to advance another timestep
4. Check Scope plots after each step
5. Press **F7** to stop when done

### Use Cases
- Debug unexpected behavior
- Understand signal flow
- Verify initial conditions
- Step through short simulations

---

## Minimap

Navigate large diagrams with the minimap overview.

### Enabling the Minimap
- **View → Minimap** or press **Ctrl+Shift+M**

### Features
- Shows scaled overview of entire diagram
- Blue rectangle shows current viewport
- Click anywhere on minimap to pan canvas
- Drag on minimap for continuous panning
- Dock on left or right side of window

---

## Tips

- **Flip blocks**: Right-click → Flip
- **Resize blocks**: Drag corner handles
- **Toggle routing**: Right-click connection → Toggle Routing Mode
- **Test incrementally**: Build and test small sections first
- **Use minimap**: For large diagrams, enable minimap for quick navigation
- **Step through**: Use F8 to debug simulations step-by-step
