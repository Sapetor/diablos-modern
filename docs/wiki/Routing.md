# Routing Blocks

List of available blocks in the **Routing** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Demux](#demux) | Demultiplexer (Demux). |
| [From](#from) | From Tag. |
| [Goto](#goto) | Goto Tag. |
| [Mux](#mux) | Multiplexer (Mux). |
| [Selector](#selector) | Selector / Indexer. |
| [Switch](#switch) | Signal Switch. |

---

### Demux

Demultiplexer (Demux).

Splits a vector input signal into individual scalar/vector components.

Parameters:
- Outputs: Number of output ports.

Usage:
Use to extract signals from a bus or Mux.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `output_shape` | int | `1` | The size of each output vector. |

**Ports**: 1 In, 2 Out

---

### From

From Tag.

Receives a signal from a matching 'Goto' block.

Parameters:
- Tag: Identifier of the source 'Goto' block.

Usage:
Reduces visual clutter.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tag` | string | `A` | Tag name to link Goto/From. |
| `signal_name` | string | `` | Optional label; defaults to tag when empty. |

**Ports**: 0 In, 1 Out

---

### Goto

Goto Tag.

Sends a signal to a matching 'From' block without a visible wire.

Parameters:
- Tag: Unique identifier (string) to match with 'From'.

Usage:
Reduces visual clutter by hiding long connections.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tag` | string | `A` | Tag name to link Goto/From. |
| `signal_name` | string | `` | Optional label; defaults to tag when empty. |

**Ports**: 1 In, 0 Out

---

### Mux

Multiplexer (Mux).

Combines multiple scalar or vector signals into a single vector output.

Parameters:
- Inputs: Number of signals to combine.

Usage:
Use to bundle signals for Scope plotting or bus routing.

**Ports**: 2 In, 1 Out

---

### Selector

Selector / Indexer.

Picks specific elements from a vector input.

Parameters:
- Indices: List of 0-based indices to extract.
  Example: [0, 2] extracts 1st and 3rd elements.
- Input Width: (Optional) Expected size of input vector.

Usage:
Reordering or subsetting signals.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `indices` | string | `0` | Comma-separated indices to select (0-based). E.g., '0,2,4' or '1:3' for range. |

**Ports**: 1 In, 1 Out

---

### Switch

Signal Switch.

Passes one of the inputs based on the Control signal (middle port).

Criteria:
- u2 >= Threshold: Output = u1 (Top port)
- u2 < Threshold:  Output = u3 (Bottom port)

Parameters:
- Threshold: Switching value.

Usage:
Conditional logic or selecting between valid signals.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `threshold` | float | `0.0` | Control threshold (threshold mode). |
| `n_inputs` | int | `2` | Number of data inputs (>=2). |
| `mode` | string | `threshold` | 'threshold' or 'index'. |

**Ports**: 3 In, 1 Out

---
