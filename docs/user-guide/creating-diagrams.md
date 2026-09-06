# Creating Diagrams

## The block palette

The palette on the left lists every block by category, with **Favorites** and
**Recent** pinned at the top once you have used something. Category headers are
collapsible and remember their state. Type in the search box to filter — `s`,
`sum` and `pid` all work.

Blocks you have saved yourself appear under **USER LIBRARY**; see
[Masks and user library blocks](#masks-and-user-library-blocks).

## Adding blocks

- **Drag and drop** a block from the palette onto the canvas.
- **Right-click empty canvas** for an *Add commonly used* shortlist plus
  **add here**.
- **<kbd>Ctrl</kbd>+<kbd>K</kbd>** opens the command palette, which can also add
  a block by name.

## Connecting blocks

**Drag to connect.** Press on an output port (the disc on a block's right edge),
drag, and release on an input port. The preview snaps to whatever port is under
the cursor and turns **green** when the target will be accepted, **red** when it
will not.

**Click to connect** still works if you prefer: click the output port, then
click the input port. A wire can also be started from a free *input* port and
dragged back to a source.

<kbd>Esc</kbd> cancels a wire in progress.

### Wire routing

Each wire has its own routing mode. Right-click a wire ▸ **Routing**:

- **Bezier (curved)** — a smooth curve. This is the default; set the
  application-wide default under **View ▸ Default Connection Routing**.
- **Orthogonal (Manhattan)** — axis-aligned segments with small rounded corners,
  routed around blocks by a grid A\* router that keeps clear of block name
  labels.

Also on the wire menu: **Auto-route wire**, **Reset routing**, **Highlight
path**, **Edit label…** and **Delete wire**. Auto-routing switches the wire to
orthogonal mode so the menu reflects what is drawn.

**Bends.** Drag a wire to create a three-segment bend; bends snap to the grid,
and double-clicking a bend handle removes it. Hand-made bends survive undo/redo
and are not thrown away when you move a block — only orthogonal, auto-routed
wires get re-routed, and bezier wires stay curved.

Where wires cross, the one in front is drawn with a small gap so the crossing
reads correctly.

### Routing by tag instead of by wire

For long or repeated connections, use `Goto` and `From` blocks. Give both the
same `tag` and they form a virtual link with no wire drawn. Tags auto-link, are
validated as part of diagram integrity checking, and a small HUD shows the tag
counts. Both are drawn as pointed tag shapes displaying `[tag]`.

## Editing parameters

Select a block and edit it in the **Properties** panel on the right; changes
apply live. Right-click ▸ **Edit parameters…** opens the same thing as a dialog.

Numeric fields also accept **workspace variable names** — define `K` in the
[Variable Editor](../VARIABLE_EDITOR_GUIDE.md) and type `K` into a gain field.
The name is what gets saved, so the diagram stays parametric.

## Selecting and moving

- **Click** selects a block; **Ctrl/Cmd+click** adds to the selection.
- **Drag on empty canvas** rubber-band selects.
- **<kbd>Ctrl</kbd>+<kbd>A</kbd>** selects everything; <kbd>Esc</kbd> clears.
- Dragging shows **smart alignment guides** against nearby blocks.
- Right-click with 2+ blocks selected for **Align && Distribute**, or use
  <kbd>Ctrl</kbd>+<kbd>Shift</kbd> plus <kbd>L</kbd>/<kbd>R</kbd>/<kbd>H</kbd>/<kbd>T</kbd>/<kbd>B</kbd>.
- **<kbd>Ctrl</kbd>+<kbd>F</kbd>** flips the selection horizontally, so feedback
  paths read right to left.
- **<kbd>F2</kbd>** renames a block.

Blocks can be resized by their handles; ports rescale with them. Delete with
<kbd>Del</kbd> or <kbd>Backspace</kbd>.

## Block shapes

Blocks declare their own outline rather than all being rectangles:

- **Gain** and **MatrixGain** are triangles showing the gain value inside.
- **Sum** and **Product** are circles (with up to three inputs), each input port
  marked `+`/`-` or `×`/`÷`.
- **Goto** and **From** are pointed tags showing `[tag]`.
- Everything else is a rectangle with its icon drawn inside.

A masked subsystem can pick its own outline — see below. The available shapes
are `rect`, `triangle`, `circle` and `tag`.

## Subsystems

Select some blocks and press <kbd>Ctrl</kbd>+<kbd>G</kbd> (or right-click ▸
**Wrap in subsystem**) to group them. Double-click a subsystem to descend into
it; <kbd>Esc</kbd> with nothing selected goes back up. The breadcrumb above the
canvas shows where you are.

Subsystem ports synchronize automatically from the `Inport` and `Outport` blocks
inside, so a subsystem can have any number of inputs and outputs. Subsystems are
flattened recursively before a run, so the compiled fast solver handles them at
full speed.

See [Subsystems](../wiki/Subsystems_Architecture.md) for the architecture.

## Masks and user library blocks

A **mask** turns a subsystem into something that behaves like a built-in block:
its own name, icon, outline, palette category, documentation, and a chosen set
of parameters instead of the raw internals.

Right-click a subsystem (or use the **Edit** menu):

| Action | What it does |
|---|---|
| **Edit mask…** | Open the **Edit Mask** dialog |
| **Look under mask** | Descend into the masked subsystem's contents |
| **Save as library block…** | Write it to your library so it appears in the palette |
| **Reload from library** | Re-read an instance from its source file |
| **Refresh Block Library** (Edit menu) | Rescan the library folders |

The **Edit Mask** dialog has three tabs:

- **Parameters** — the ordered list of parameters the mask exposes, with
  **Add**, **Remove**, **Move up** and **Move down**. A mask parameter can be
  referenced by name inside the subsystem, so `gain = "K"` on an inner block
  picks up the mask's `K`.
- **Icon & Appearance** — **Display name**, **Palette category** (defaults to
  `User Library`), **Icon text** (a short label or emoji drawn inside the block)
  and **Outline shape**.
- **Documentation** — text shown in the property panel and the palette tooltip.

### Where library blocks live

A library block is an ordinary `.diablos` file with an extra `library_block`
section, so it also opens as a diagram. DiaBloS looks for them in this order,
highest priority first:

1. Every entry of the `DIABLOS_LIBRARY_PATH` environment variable
2. A `library/` folder **next to the diagram you have open** — the way to ship a
   project's blocks alongside it
3. Your per-user library folder

The per-user folder is the project root in a source checkout, and in a packaged
build:

| Platform | Folder |
|---|---|
| macOS | `~/Library/Application Support/DiaBloS/library/` |
| Windows | `%APPDATA%\DiaBloS\library\` |
| Linux | `~/.local/share/DiaBloS/library/` |

**Save as library block…** writes to the first `DIABLOS_LIBRARY_PATH` entry if
that variable is set, otherwise to the per-user folder.

`examples/library_block_demo.diablos` is a worked example. The file format is
documented in the [Developer Guide](../DEVELOPER_GUIDE.md) under *Masks and the
Library File Format*.

## Saving

<kbd>Ctrl</kbd>+<kbd>S</kbd> writes a `.diablos` file — JSON, so it diffs and
merges. The file carries the diagram, its solver settings and its subsystem
contents. DiaBloS autosaves before every run.

## Next steps

- [Running simulations](running-simulations.md)
- [Analysis & experiments](analysis.md)
- [Block reference](block-reference.md)
