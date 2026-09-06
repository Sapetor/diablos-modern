# Quick Start

Building and running a first diagram takes about a minute.

## Start the application

=== "Prebuilt app"

    Launch **DiaBloS** from Applications (macOS), `DiaBloS.exe` (Windows) or
    `./DiaBloS/DiaBloS` (Linux).

=== "From source"

    ```bash
    python diablos_modern.py
    ```

The window opens on an empty canvas with the **Block Palette** on the left and
the **Properties** panel on the right.

## A first-order lag, in six steps

We will build `Step → 1/(s+1) → Scope`.

1. **Add a source.** In the palette, expand **SOURCES** and drag **Step** onto
   the canvas. Can't find it? Press <kbd>Ctrl</kbd>+<kbd>K</kbd> for the command
   palette, or type in the palette's search box (`s`, `sum`, `pid`… all work).

2. **Add the dynamics.** From **CONTROL**, drag a **TranFn** block to the right
   of the Step.

3. **Add a sink.** From **SINKS**, drag a **Scope** to the right of the TranFn.

4. **Wire them up.** Press on the Step's output port (the small disc on its
   right edge), drag to the TranFn's input port, and release. The preview snaps
   to the port under the cursor and turns green when the target is accepted, red
   when it is not. Click-then-click also works if you prefer. Repeat from TranFn
   to Scope.

5. **Set the transfer function.** Click the TranFn block. In the Properties
   panel set `numerator` to `[1]` and `denominator` to `[1, 1]`, giving
   `1/(s+1)`.

6. **Run.** Press <kbd>F5</kbd>. The **Simulation Configuration** dialog opens —
   set **Simulation Duration** to `10`, leave the rest alone and press
   **Simulate**. The scope window opens with the step response when the run
   finishes.

Save with <kbd>Ctrl</kbd>+<kbd>S</kbd>; DiaBloS writes a `.diablos` file (JSON).

## Open an example instead

**File ▸ Examples** lists everything in `examples/`. Good ones to start with:

| Example | Shows |
|---|---|
| `c01_tank_feedback` | A classic single-loop feedback controller |
| `c03_bode_frequency_response` | Frequency response and the analysis markers |
| `c05_lqr_vs_open_loop` | State feedback designed with the LQR block |
| `heat_equation_demo` | A 1D PDE rendered in FieldScope |
| `library_block_demo` | A masked subsystem loaded from the user library |

## What to try next

- **Right-click a block.** Parameters, live tuning, flip, rename, wrap in a
  subsystem, detach wires, alignment — it is all there.
- **Right-click a wire.** Switch it between **Bezier (curved)** and
  **Orthogonal (Manhattan)**, auto-route it, or reset its routing. Drag a wire to
  create a bend; double-click a bend handle to remove it.
- **Click empty canvas.** The Properties panel turns into a diagram inspector
  showing the solver settings and the last runs.
- **Analysis ▸ Linearize & Analyze…** on a closed loop gives you poles, a Bode
  plot and stability margins without writing anything down. See
  [Analysis & Experiments](../user-guide/analysis.md).

## Keyboard shortcuts

Press <kbd>F1</kbd> for the full, always-current list. The ones worth learning
first:

| Shortcut | Action |
|----------|--------|
| <kbd>F5</kbd> | Run simulation |
| <kbd>F6</kbd> / <kbd>F7</kbd> | Pause / stop |
| <kbd>F8</kbd> | Single step |
| <kbd>Ctrl</kbd>+<kbd>K</kbd> | Command palette |
| <kbd>Ctrl</kbd>+<kbd>N</kbd> / <kbd>O</kbd> / <kbd>S</kbd> | New / open / save |
| <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd> | Undo / redo |
| <kbd>Ctrl</kbd>+<kbd>C</kbd> / <kbd>Ctrl</kbd>+<kbd>V</kbd> | Copy / paste |
| <kbd>Ctrl</kbd>+<kbd>A</kbd> | Select all |
| <kbd>Del</kbd> or <kbd>Backspace</kbd> | Delete selection |
| <kbd>Ctrl</kbd>+<kbd>G</kbd> | Wrap selection in a subsystem |
| <kbd>Ctrl</kbd>+<kbd>F</kbd> | Flip selected blocks |
| <kbd>Ctrl</kbd>+<kbd>0</kbd> | Fit diagram to window |
| <kbd>Ctrl</kbd>+<kbd>T</kbd> | Toggle light/dark theme |
| <kbd>Esc</kbd> | Cancel a wire, clear the selection, or leave a subsystem |

Alignment uses <kbd>Ctrl</kbd>+<kbd>Shift</kbd> plus <kbd>L</kbd>/<kbd>R</kbd>/<kbd>H</kbd>/<kbd>T</kbd>/<kbd>B</kbd>,
and the dockable panels use <kbd>Ctrl</kbd>+<kbd>Shift</kbd> plus
<kbd>M</kbd> (minimap), <kbd>V</kbd> (variable editor), <kbd>W</kbd> (workspace
variables) and <kbd>T</kbd> (tuning panel).

## Next steps

- [Creating diagrams](../user-guide/creating-diagrams.md)
- [Running simulations](../user-guide/running-simulations.md) — solvers, tolerances, zero crossings
- [Analysis & experiments](../user-guide/analysis.md)
- [Block reference](../wiki/Home.md)
