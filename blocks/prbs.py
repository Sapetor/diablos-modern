import numpy as np
from blocks.base_block import BaseBlock


class PRBSBlock(BaseBlock):
    """
    Pseudo-Random Binary Sequence (PRBS) source.

    Generates a binary sequence that flips every `bit_time` seconds using a seeded RNG.
    """

    @property
    def block_name(self):
        return "PRBS"

    @property
    def category(self):
        return "Sources"

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return "Generates a pseudo-random binary sequence."

    @property
    def params(self):
        return {
            "high": {"type": "float", "default": 1.0, "doc": "Value for logic high."},
            "low": {"type": "float", "default": 0.0, "doc": "Value for logic low."},
            "bit_time": {"type": "float", "default": 0.1, "doc": "Seconds each bit is held."},
            "seed": {"type": "int", "default": 0, "doc": "Seed for deterministic sequence."},
            "_init_start_": {"type": "bool", "default": True, "doc": "Internal init flag."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params):
        bit_time = float(params.get("bit_time", 0.1))
        if bit_time <= 0:
            return {"E": True, "error": "bit_time must be positive"}

        # Lazy init
        if params.get("_init_start_", True):
            rng_seed = int(params.get("seed", 0))
            params["_rng"] = np.random.default_rng(rng_seed)
            params["_next_flip"] = bit_time
            params["_current"] = float(params["high"]) if params["_rng"].integers(0, 2) else float(params["low"])
            params["_init_start_"] = False

        # Advance sequence on bit boundaries
        while time >= params["_next_flip"]:
            bit = params["_rng"].integers(0, 2)
            params["_current"] = float(params["high"]) if bit else float(params["low"])
            params["_next_flip"] += bit_time

        return {0: np.atleast_1d(params["_current"])}
