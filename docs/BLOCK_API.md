# DiaBloS Block API

**`BLOCK_API_VERSION = 1`** (`blocks/base_block.py`)

This is the reference for writing a block — either a built-in one inside the
repository, or a *user block* you drop into a folder without touching DiaBloS
itself. Everything here is checked mechanically by
`blocks.base_block.validate_block_class`, so if your block loads without a
warning it satisfies this document.

Start from the copy-paste templates:

- [`examples/custom_block_template.py`](examples/custom_block_template.py) — a
  stateless block and a stateful block, fully commented.
- [`examples/custom_kernel_template.py`](examples/custom_kernel_template.py) —
  the same plus a compiled-path kernel.

---

## 1. The minimal block

```python
import numpy as np
from blocks.base_block import BaseBlock


class ScaleBlock(BaseBlock):          # class convention: <BlockName>Block
    @property
    def block_name(self):             # palette name + key in saved diagrams
        return "Scale"

    @property
    def params(self):                 # spec dict: name -> {type, default, doc}
        return {"gain": {"type": "float", "default": 1.0, "doc": "Scale factor"}}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        return {0: np.atleast_1d(inputs.get(0, 0.0)) * params["gain"]}
```

The four required members are **`@property` methods**, not plain class
attributes — `BaseBlock` declares them abstract, so a class that assigns
`block_name = "Scale"` instead is still abstract and will not load.

A block class must be constructible as `cls()`: no required constructor
arguments.

---

## 2. `block_name`

A non-empty string. It is:

- the name shown in the palette and used by the command palette;
- the `block_fn` value persisted in every `.diablos` file that uses the block.

Consequences:

- **Renaming it breaks saved diagrams.** They will open with the block missing
  and a "Missing block types" warning naming the search paths.
- **It must be unique.** A user block whose `block_name` collides with a
  built-in is skipped with a warning — built-ins always win, precisely because
  the name is a persisted key. Prefix your own blocks if in doubt
  (`AcmeScale`).

---

## 3. `params` — the parameter spec

`params` returns a **spec** dict, `name -> {…}`:

```python
{
    "gain":  {"type": "float", "default": 1.0, "doc": "Scale factor"},
    "mode":  {"type": "choice", "default": "up", "choices": ["up", "down"]},
    "limit": {"type": "float", "default": 1.0, "range": (0.0, 10.0)},
}
```

A bare value is accepted as shorthand for `{"default": value}`, but the full
form is what gives you a tooltip and an appropriate editor.

Recognised keys:

| Key | Meaning |
|---|---|
| `default` | **Required** in a spec dict. The value a new block starts with. |
| `type` | Documentation token. Values in use: `float`, `int`, `bool`, `string`, `list`, `array`, `vector`, `matrix`, `choice`, `any`. |
| `doc` | Tooltip text. Written in English and translated at *display* time. |
| `choices` / `options` | List of allowed values → renders a combo box. |
| `range` | `(min, max)` pair → renders a slider (floats) or bounds a spin box (ints). |
| `accepts_array` | `True` for a float parameter that also accepts `[1, 2, 3]`. |
| `advanced` | `True` files the parameter under the "Advanced" group. |
| `group` | Explicit property-panel group name. |
| `no_slider` | Suppress the slider for a float parameter. |

**The property editor picks its widget from the Python type of the current
*value*, not from `type`** (`modern_ui/widgets/property_editor.py`): `bool` →
check box, a spec with `choices` → combo box, `int` → spin box, `float` →
slider/spin box, everything else → line edit. So the practical rule is: make
`default` the exact Python type you want edited (`1.0`, not `1`, for a float
parameter).

Two conventions:

- Keys wrapped in underscores (`_init_start_`, `_t_old_`) are **internal
  state**, hidden from the property editor.
- `sampling_time` (or `sample_time`) is the rate convention shared by the
  sampled-data blocks: `-1` continuous, `0` inherited, `> 0` a discrete period
  in seconds. Declare it only if your block really is sampled; a block with
  `sampling_time > 0` forces the whole diagram onto the interpreted solver, by
  design. `BaseBlock.requires_sample_time` (override to `True`) makes the
  engine warn when a block that is meaningless without a rate does not resolve
  one.

At **run time** `execute()` receives the *flattened* parameters: `params["gain"]`
is `1.0`, not `{"type": …, "default": 1.0}`.

---

## 4. Ports

`inputs` and `outputs` return lists of dicts, one per port, in port-index
order:

```python
[{"name": "in", "type": "any"}, {"name": "reset", "type": "float"}]
```

- `name` — a non-empty string, shown in the palette tooltip and on hover.
- `type` — a documentation token. In use: `any` (default and by far the most
  common), `float`, `vector`, `matrix`. Types are **not** enforced by the
  engine; wiring is not type-checked.

`PortDefinition` in `base_block.py` is only a type alias for `Dict[str, str]` —
ports are plain dicts, never `PortDefinition(...)`.

Port counts are fixed unless you override `io_editable` (below).

---

## 5. `execute()`

```python
def execute(self, time, inputs, params, **kwargs) -> dict:
```

- `time` — current simulation time in seconds.
- `inputs` — `{port_index: value}`. **A port may be absent**: it is optional, it
  is unconnected, or the engine is doing an *output-only* probe. Always use
  `inputs.get(idx, default)`, and treat "no input at all" as "return the held
  output without advancing state".
- `params` — the flattened parameter values, and the place your state lives.
- `**kwargs` — **mandatory in the signature.** The engine passes extra keywords
  to some blocks (`dtime`, the step used for a probe; `output_only=True`;
  `next_add_in_memory`). A signature without `**kwargs` raises `TypeError` mid
  simulation.

Return either

- `{port_index: value}` — one entry per output port; values are normally
  `np.ndarray` (use `np.atleast_1d`), or
- `{"E": True, "error": "message"}` — abort the run and show the message.

Returning `None` is treated as a failure.

### State lives in `params`

> **Critical rule.** Every value that must persist across time steps goes into
> `params` (`params["_t_old_"] = time`), never onto `self`.

The engine's `reset_memblocks()` (`lib/engine/simulation_engine.py`)
re-initialises blocks between runs by setting `_init_start_ = True` in
`params`/`exec_params` and clearing stale accumulators. It never touches
instance attributes, so state hidden on `self` survives a reset invisibly and
leaks into the next run — a bug that only shows up on the *second* run.

The first call of a run has `params["_init_start_"] is True`; build your state
there and set the flag to `False`:

```python
def execute(self, time, inputs, params, **kwargs):
    if params.get("_init_start_", True):
        params["_last_"] = np.zeros(1)
        params["_t_old_"] = time
        params["_init_start_"] = False

    if 0 not in inputs:                       # output-only probe
        return {0: np.array(params["_last_"])}

    dt = time - params["_t_old_"]
    ...
```

Declare `_init_start_` in your `params` spec (or reuse
`blocks/param_templates.py: init_flag_param()`).

---

## 6. Optional hooks

All of these have working defaults in `BaseBlock`; override only what you need.

| Hook | Default | What it does |
|---|---|---|
| `category` | `"Other"` | Palette section, and the default for the port requirements below. Existing sections: `Sources`, `Math`, `Control`, `Filters`, `Sinks`, `Routing`, `Analysis`, `PDE`, `Optimization`, `Other`. A new name simply creates a new section. Translated at display time. |
| `doc` | — | Long description for the palette tooltip and property panel. |
| `requires_inputs` | `category != "Sources"` | Whether every input must be wired for the diagram to run. |
| `requires_outputs` | `category not in ("Sinks", "Other")` | Same for outputs. |
| `optional_inputs` / `optional_outputs` | `set()` | Port **indices** that may stay unconnected, e.g. `{1}`. |
| `shape` | `"rect"` | Outline: `rect`, `triangle` (amplifier), `circle` (Sum/Product), `tag` (Goto/From). Unknown tokens are rejected by the validator. |
| `use_port_grid_snap` | `True` | Set `False` when exact port geometry matters (triangles). |
| `io_editable` | `None` | `"input"`, `"output"` or `"both"` to let the user change the port count. |
| `draw_icon(block_rect)` | `None` | Return a `QPainterPath` in 0..1 normalized coordinates for the block glyph. See [`SYMBOL_DRAWING_GUIDE.md`](SYMBOL_DRAWING_GUIDE.md). |
| `symbolic_execute(inputs, params)` | `None` | Return `{port: sympy_expr}` so the block participates in equation extraction / transfer-function export. Without it the block is opaque to the symbolic engine (numeric linearization still works). |
| `get_symbolic_params(params)` | `{}` | Which parameters become symbols during extraction. |
| `output_is_post_update` | `False` | `True` when `execute()` returns the *already advanced* state (like Integrator), so a sample-time gate holds the pre-update value instead. |
| `requires_sample_time` | `False` | `True` for a block that is a pure recursion in the sample index; the engine warns when no rate resolves. |
| `hidden` (plain class attribute) | absent | `True` keeps the block out of the palette. |

---

## 7. The compiled (fast solver) path

DiaBloS has two execution paths (see [`FAST_SOLVER.md`](FAST_SOLVER.md)):

1. the **interpreted** path calls `execute()` once per block per step. Every
   block works here; this is the only path a custom block needs;
2. the **compiled** path turns the whole diagram into one ODE right-hand side
   for `scipy.integrate.solve_ivp`. It is much faster, and it requires each
   block to provide a *kernel*.

A kernel is registered next to its block family in
`lib/engine/compiler_kernels/`:

```python
from lib.engine.compiler_kernels import kernel

@kernel("Softclip")                      # the canonical fn-name
def build_soft_clip(ctx):
    b_name = ctx.b_name                  # this block's signal key
    src = ctx.input_sources[0]           # signal key feeding input port 0
    limit = abs(float(ctx.params.get("limit", 1.0)))

    def exec_soft_clip(t, y, dy_vec, signals):
        signals[b_name] = min(max(signals.get(src, 0.0), -limit), limit)

    return exec_soft_clip
```

The builder runs **once per compile** and bakes parameters into a closure that
runs thousands of times per solve — do all lookups and conversions in the
builder. The registration name is the *canonical* form of `block_name`:
`lib.engine.block_names.canonical_fn(block_name)`, essentially `str.title()`
plus a few historical overrides.

A discontinuous block declares its switching surfaces with `@events(...)`
returning `EventSpec`s, so the solver lands exactly on each corner instead of
stepping across it:

```python
from lib.engine.compiler_kernels import EventSpec, events, signal_scalar

@events("Softclip")
def events_soft_clip(ctx):
    src = ctx.input_sources[0]
    return [
        EventSpec(
            block=ctx.b_name,
            label="upper_limit",
            func=lambda t, y, signals: signal_scalar(signals, src) - 1.0,
        )
    ]
```

### What happens if your block has no kernel

Nothing breaks. `SystemCompiler.check_compilability` gates the whole diagram on
an allowlist (`COMPILABLE_BLOCKS`); a diagram containing a block that is not on
it runs on the interpreted path, which is slower but produces the same answer.

**Today that allowlist is a hard-coded set inside
`lib/engine/system_compiler.py` and a user module cannot extend it**, so a
diagram containing a user block always uses the interpreted solver, whether or
not you register a kernel. Registering one is still meaningful if you intend to
contribute the block upstream (or you maintain your own build): add the name to
`COMPILABLE_BLOCKS` there and the kernel takes effect.

---

## 8. Localization

The UI is translated through `lib/i18n.py`; blocks participate in two narrow
ways:

- a block's `category` name and every param `doc` string are translated **at
  display time** (`tr(category)`, `tr(doc)`), because the stored values are
  registry keys. Write them in English and they are picked up automatically by
  `python scripts/extract_strings.py --update` for blocks inside the repo.
- **never** translate identifiers: `block_name`, parameter keys, port names,
  `.diablos` format strings. They are persisted and compared against.

Strings in a *user* block are not extracted into the shipped catalogs (only
`blocks/`, `lib/`, `modern_ui/` are scanned), so ship your own text in whatever
language you like — or in English, so it reads consistently.

---

## 9. Packaging and installation

### Where the file goes

DiaBloS scans these folders, **highest priority first**:

1. every folder listed in the `DIABLOS_BLOCKS_PATH` environment variable
   (`os.pathsep`-separated: `:` on macOS/Linux, `;` on Windows);
2. a `blocks/` folder next to the `.diablos` file you have open
   (project-local — ship a model together with its blocks);
3. the per-user folder `<user data dir>/blocks`:
   - macOS `~/Library/Application Support/DiaBloS/blocks/`
   - Windows `%APPDATA%/DiaBloS/blocks/`
   - Linux `~/.local/share/DiaBloS/blocks/`

The first folder that provides a given *file name* wins, so a project-local
copy shadows the user's own. In a development checkout the per-user folder
resolves to the repository's own `blocks/` package and is skipped (it is
already the built-in registry); use `DIABLOS_BLOCKS_PATH` or a project-local
folder there.

```bash
DIABLOS_BLOCKS_PATH=~/my-diablos-blocks python diablos_modern.py
```

Only top-level `*.py` files are picked up. Names starting with `_` are skipped,
so a shared helper can live next to your blocks as `_helpers.py` (import it
with a normal `import` only if it is on `sys.path`; otherwise keep the block
self-contained).

### What the loader does

- each module is imported in isolation, under `diablos_user_blocks.<stem>`. A
  module that raises is logged **with its traceback** and skipped; the other
  modules still load;
- every `BaseBlock` subclass *defined* in the module (imported ones are
  ignored) is run through `validate_block_class`. A class that fails is skipped
  with the validator's message;
- a `block_name` that collides with a built-in — or with an earlier user block
  — is skipped;
- classes that pass are appended after the built-ins and behave exactly like
  them: palette, property panel, save/load, undo, analysis.

*Edit ▸ Reload User Blocks* re-imports the modules from disk, so you can edit a
block and see the change without restarting. *Edit ▸ Open User Blocks Folder…*
opens the per-user folder, creating it if needed.

### Frozen builds

Loading is by file path through `importlib`, so it works identically in a
PyInstaller bundle — where the bundled `blocks/` package is *not* scanned and
built-ins come from the static `_BLOCK_MODULES` registry. A user block may
import anything the bundle ships (numpy, scipy, PyQt5); an import of a
third-party package that is *not* in the bundle fails cleanly with a logged
traceback and the block is skipped.

### Diagrams that use a user block

The block's `block_name` is stored in the file. Opening the diagram on a
machine that has the block works normally; on one that does not, the block is
left out and DiaBloS shows a translated warning naming the missing block types
and every folder it searched.

---

## 10. Validating your block

```python
from blocks.base_block import validate_block_class, block_contract_errors
from my_module import MyBlock

validate_block_class(MyBlock)          # raises BlockContractError with details
print(block_contract_errors(MyBlock))  # or get the list of problems
```

`validate_block_class` checks that the class

- derives from `BaseBlock` and is not abstract (all required members
  implemented as properties);
- is constructible as `cls()`;
- has a non-empty string `block_name`;
- returns a well-formed `params` spec (dict, string keys, every spec dict has a
  `default`, `type`/`doc` are strings, `choices`/`options` are lists, `range` is
  a numeric pair);
- returns well-formed `inputs`/`outputs` (lists of dicts with a non-empty
  string `name` and a string `type`);
- lists only in-range integer port indices in `optional_inputs`/
  `optional_outputs`;
- has an `execute()` accepting `time`, `inputs`, `params` **and** `**kwargs`;
- has a string `category`, a `shape` from the supported set, and an
  `io_editable` from `{None, "input", "output", "both"}`;
- defines `draw_icon` / `symbolic_execute` as callables when it defines them.

The same check runs automatically: on every built-in block in a development
checkout (violations are logged), and on every user block always (violations
skip the block).

---

## 11. Versioning policy

`BLOCK_API_VERSION` lives in `blocks/base_block.py` and is currently **1**.

- It is bumped **only** for a change that breaks an existing third-party block:
  a new required member, a changed `execute()` signature or return shape, a
  removed hook.
- Additive changes — a new *optional* hook, a new `type` token, a new palette
  category — do not bump it.
- A user module may declare its own module-level `BLOCK_API_VERSION`. DiaBloS
  refuses to load a module that asks for a **newer** API than the running build
  implements, with a message naming both versions, instead of letting it fail
  obscurely at run time. Declaring an older (or no) version always loads.

---

## Related documents

- [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) — repository workflow, adding a
  built-in block, testing.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — where blocks sit in the engine.
- [`FAST_SOLVER.md`](FAST_SOLVER.md) — the compiled path in detail.
- [`SYMBOL_DRAWING_GUIDE.md`](SYMBOL_DRAWING_GUIDE.md) — `draw_icon` conventions.
- [`USER_MANUAL.md`](USER_MANUAL.md) — the user library (masked subsystems saved
  as reusable palette blocks), the no-code sibling of this API.
