# Other Blocks

List of available blocks in the **Other** category.

| Block | Description |
|-------|-------------|
| [External](#external) | External Function Block (not implemented). |

The frequency-response marker blocks that used to be listed here -- `BodeMag`,
`BodePhase`, `Nyquist` and `RootLocus` -- now declare the **Analysis** category
and are documented on the [Analysis](Analysis.md) page.

---

### External

External Function Block.

> **NOT IMPLEMENTED**: this block is a stub. It returns an error when executed,
> and it is hidden from the block palette. For custom behaviour use the
> [Function](Math.md#function) block (a sandboxed Python expression) or add a
> block of your own -- see the [Developer Guide](../DEVELOPER_GUIDE.md).

Intended to execute custom Python code loaded from an external file. The `exec`
path was deliberately disabled in 1.0.0 as part of removing `eval`/`exec` from
the codebase.

#### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `filename` | string | `` | Path to external Python script |
| `function` | string | `execute` | Function name to call |

**Ports**: 1 In, 1 Out

**Status**: returns `{'E': True, 'error': 'External file not loaded: ...'}` when executed.

---
