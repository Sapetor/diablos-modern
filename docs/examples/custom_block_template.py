"""Copy-paste template for a DiaBloS custom (user) block module.

Drop a copy of this file into your user blocks folder and edit it:

* macOS (installed app):  ~/Library/Application Support/DiaBloS/blocks/
* Windows (installed):    %APPDATA%/DiaBloS/blocks/
* Linux (installed):      ~/.local/share/DiaBloS/blocks/
* any checkout or build:  a folder listed in ``DIABLOS_BLOCKS_PATH``
                          (os.pathsep-separated), or a ``blocks/`` folder next
                          to the ``.diablos`` file you are working on.

Then use *Edit > Reload User Blocks* (no restart needed) and the blocks appear
in the palette under their ``category``, marked with a small diamond.

The full contract is documented in ``docs/BLOCK_API.md``.  Two blocks are shown
here: a stateless one and a stateful one, which is where most of the
non-obvious rules live.
"""

import numpy as np

from blocks.base_block import BaseBlock

# Declaring the API revision this module was written against is optional but
# recommended: DiaBloS refuses to load a module that asks for a newer API than
# the running build implements, instead of failing later inside execute().
BLOCK_API_VERSION = 1


class ScaledOffsetBlock(BaseBlock):
    """A stateless block: ``y = gain * u + offset``.

    Class name convention is ``<BlockName>Block``.  Only the five members
    below are required; everything else has a sensible default in BaseBlock.
    """

    @property
    def block_name(self):
        """Palette name AND the key stored in saved diagrams.

        Pick something unlikely to collide: a user block whose name is already
        taken by a built-in is skipped (built-ins win), and renaming it later
        breaks diagrams that already reference the old name.
        """
        return "ScaledOffset"

    @property
    def category(self):
        """Palette section. Also drives port-requirement defaults (see docs)."""
        return "Math"

    @property
    def doc(self):
        """Optional: the palette tooltip / property-panel blurb."""
        return "Affine map y = gain * u + offset."

    @property
    def params(self):
        """Parameter *spec*: name -> {type, default, doc, ...}.

        ``default`` is what a freshly dropped block gets; its Python type is
        what the property editor renders (float -> spin box, bool -> check box,
        a spec with ``choices`` -> combo box).  ``doc`` is translated at
        display time, so write it in English.
        """
        return {
            "gain": {
                "type": "float",
                "default": 1.0,
                "doc": "Multiplies the input",
                "range": (-10.0, 10.0),  # optional: renders a slider
            },
            "offset": {
                "type": "float",
                "default": 0.0,
                "doc": "Added after the gain",
            },
        }

    @property
    def inputs(self):
        """One dict per input port, in port order (index 0, 1, ...)."""
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        """Compute the outputs for the current step.

        Args:
            time: current simulation time (seconds).
            inputs: ``{port_index: value}``; a port may be missing when it is
                optional or during an output-only probe, so always use
                ``inputs.get(idx, default)``.
            params: the *flattened* parameter values (``params["gain"]`` is a
                float here, not the spec dict above).
            **kwargs: extra engine keywords (``dtime``, ``output_only``, ...).
                Always accept ``**kwargs`` -- the engine passes them to some
                blocks and a signature without it raises TypeError at run time.

        Returns:
            ``{port_index: value}``, or ``{"E": True, "error": "..."}`` to fail
            the run with a message.
        """
        u = np.atleast_1d(inputs.get(0, 0.0))
        return {0: u * float(params["gain"]) + float(params["offset"])}


class MovingAverageBlock(BaseBlock):
    """A stateful block: the mean of the last ``window`` samples.

    THE RULE for stateful blocks: every value that must survive from one time
    step to the next lives in ``params``, never on ``self``.  The engine's
    ``reset_memblocks()`` re-initialises blocks between runs through ``params``
    only; state hidden on the instance survives a reset invisibly and leaks
    into the next run.
    """

    @property
    def block_name(self):
        return "MovingAverage"

    @property
    def category(self):
        return "Filters"

    @property
    def doc(self):
        return "Mean of the last N samples (simple FIR smoother)."

    @property
    def params(self):
        return {
            "window": {
                "type": "int",
                "default": 5,
                "doc": "Number of samples averaged",
            },
            # The engine sets this back to True before every run; use it to
            # (re)build your state. Keys starting and ending with "_" are
            # internal and hidden from the property editor.
            "_init_start_": {
                "type": "bool",
                "default": True,
                "doc": "Internal: initialization flag",
            },
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        window = max(1, int(params.get("window", 5)))

        # First call of a run: build the state inside params.
        if params.get("_init_start_", True):
            params["_buffer_"] = []
            params["_last_"] = np.zeros(1)
            params["_init_start_"] = False

        # An output-only probe passes no inputs: return the held value and do
        # NOT advance the state, or the block would consume an extra sample.
        if 0 not in inputs:
            return {0: np.array(params["_last_"])}

        sample = np.atleast_1d(inputs[0]).astype(float)
        buffer = params["_buffer_"]
        buffer.append(sample)
        del buffer[:-window]

        mean = np.mean(np.stack(buffer, axis=0), axis=0)
        params["_last_"] = mean
        return {0: mean}
