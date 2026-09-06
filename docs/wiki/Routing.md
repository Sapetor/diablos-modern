# Routing Blocks

List of available blocks in the **Routing** category.

You can find detailed information about parameters and usage below.

| Block | Description |
|-------|-------------|
| [Demux](#demux) | Demultiplexer (Demux). |
| [From](#from) | From Tag. |
| [Goto](#goto) | Goto Tag. |
| [Inport](#inport) | Subsystem input port. |
| [Mux](#mux) | Multiplexer (Mux). |
| [NetworkChannel](#networkchannel) | Lossy, jittery communication link (loss + random latency). |
| [Outport](#outport) | Subsystem output port. |
| [PacketLoss](#packetloss) | Lossy channel with Bernoulli or Gilbert-Elliott drops. |
| [Selector](#selector) | Selector / Indexer. |
| [Subsystem](#subsystem) | Hierarchical container block. |
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

### Inport

Subsystem Input Port.

Represents an input terminal of a Subsystem.
When placed inside a subsystem, it creates an input port on the parent Subsystem block.

Usage:
Place inside a Subsystem to define external inputs.
The port name (e.g., "In1", "In2") determines the label shown on the parent block.

**Ports**: 0 In, 1 Out

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

### Outport

Subsystem Output Port.

Represents an output terminal of a Subsystem.
When placed inside a subsystem, it creates an output port on the parent Subsystem block.

Usage:
Place inside a Subsystem to define external outputs.
The port name (e.g., "Out1", "Out2") determines the label shown on the parent block.

**Ports**: 1 In, 0 Out

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

### Subsystem

Hierarchical Container Block.

A container block that holds other blocks and connections, used to simplify complex diagrams by grouping related functionality.

Operations:
- **Double-click** to enter the subsystem and edit its contents
- **Add Inport blocks** inside to create input ports on the parent
- **Add Outport blocks** inside to create output ports on the parent

Usage:
Organize complex diagrams into logical modules.
Reuse functionality by copy-pasting subsystems.
Create hierarchical multi-level designs.

See [Subsystems Architecture](Subsystems_Architecture.md) for technical details on flattening and execution.

**Ports**: Dynamic (based on internal Inport/Outport blocks)

---
### NetworkChannel

Unreliable, jittery communication link: random packet loss plus a random
per-packet transport delay.

At each sample instant a Bernoulli trial decides whether the packet is dropped.
A surviving packet is given a latency drawn uniformly from
`[min_delay, max_delay]` and is enqueued for delivery at `time + delay`. The
output holds the most recently delivered packet (zero-order hold).

Because the block exposes a `seed`, it takes a derived sub-seed in a
[Monte Carlo ensemble](../user-guide/analysis.md#monte-carlo-ensembles).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `loss_prob` | float | `0.1` | Probability (0..1) that a packet is dropped. |
| `min_delay` | float | `0.0` | Minimum per-packet transport delay (s). |
| `max_delay` | float | `0.1` | Maximum per-packet transport delay (s). |
| `sample_time` | float | `0.0` | Channel sample period (s). 0 = every step. |
| `seed` | int | `0` | RNG seed (0 = non-reproducible, nonzero = reproducible). |
| `drop_mode` | choice | `hold` | Output before any packet is delivered: `hold`, `zero`, `nan`. |
| `initial_value` | float | `0.0` | Held value before the first delivered packet. |

**Ports**: 1 In, 1 Out

---

### PacketLoss

Lossy communication link. At each sample instant a seeded random trial decides
whether the packet is delivered (output = input) or dropped (output falls back
per `drop_mode`).

Two loss models:

- `bernoulli` — i.i.d. drops with probability `loss_prob`.
- `gilbert_elliott` — a bursty two-state Markov chain: `p_bg` and `p_gb` set the
  good↔bad transition probabilities, `loss_prob` applies in the good state and
  `loss_prob_bad` in the bad one.

Because the block exposes a `seed`, it takes a derived sub-seed in a
[Monte Carlo ensemble](../user-guide/analysis.md#monte-carlo-ensembles).

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `loss_model` | choice | `bernoulli` | `bernoulli` (i.i.d.) or `gilbert_elliott` (bursty). |
| `loss_prob` | float | `0.1` | Bernoulli / good-state drop probability (0..1). |
| `p_bg` | float | `0.1` | Gilbert-Elliott good→bad transition probability. |
| `p_gb` | float | `0.5` | Gilbert-Elliott bad→good transition probability. |
| `loss_prob_bad` | float | `0.9` | Gilbert-Elliott drop probability in the bad state. |
| `sample_time` | float | `0.0` | Channel sample period (s). 0 = every step. |
| `seed` | int | `0` | RNG seed (0 = non-reproducible, nonzero = reproducible). |
| `drop_mode` | choice | `hold` | Output on a dropped packet: `hold`, `zero`, `nan`. |
| `initial_value` | float | `0.0` | Held value before the first delivered packet. |

**Ports**: 1 In, 1 Out

---
