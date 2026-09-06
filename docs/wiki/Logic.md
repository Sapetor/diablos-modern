# Logic Blocks

List of available blocks in the **Logic** category.

Logic blocks treat any nonzero input as *true* and emit `1.0` for true and `0.0`
for false, element-wise for vector signals. They are the usual way to build a
switching law: feed the output into the control port of a
[Switch](Routing.md#switch), or into a `Sum` to gate a signal.

All three are supported by the [compiled fast solver](../FAST_SOLVER.md).

| Block | Description |
|-------|-------------|
| [CompareToConstant](#comparetoconstant) | Compares the input against a fixed constant. |
| [LogicalOperator](#logicaloperator) | Boolean combination of the inputs (AND, OR, XOR, ...). |
| [RelationalOperator](#relationaloperator) | Compares two input signals. |

---

### CompareToConstant

Compare an input signal against a fixed constant: `y = (in OP constant)`.

A one-input convenience form of `RelationalOperator`.

Usage:
Threshold detection — trigger when a signal exceeds a setpoint.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `operator` | choice | `>` | Comparison applied as `input OP constant`: `>`, `>=`, `<`, `<=`, `==`, `!=`. |
| `constant` | float | `0.0` | Constant value compared against the input. |

**Ports**: 1 In, 1 Out

---

### LogicalOperator

Boolean logic over the inputs; any nonzero input counts as true.

`NOT` uses only the first input. `XOR` over more than two inputs is cascaded, so
it computes odd parity. The number of input ports is editable from the property
panel.

Usage:
Gate events, combine threshold detectors, build switching logic.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `operator` | choice | `AND` | Boolean operator: `AND`, `OR`, `NAND`, `NOR`, `XOR`, `NOT`. |

**Ports**: 2 In (editable count), 1 Out

---

### RelationalOperator

Compare two input signals: `y = (in1 OP in2)`.

Usage:
Drive a `Switch` control port, or build bang-bang / threshold logic for hybrid
system examples.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `operator` | choice | `>` | Comparison applied as `in1 OP in2`: `>`, `>=`, `<`, `<=`, `==`, `!=`. |

**Ports**: 2 In, 1 Out

---
