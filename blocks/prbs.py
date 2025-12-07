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
        return "Generates a pseudo-random binary sequence (LFSR-based)."

    @property
    def params(self):
        return {
            "high": {"type": "float", "default": 1.0, "doc": "Value for logic high."},
            "low": {"type": "float", "default": 0.0, "doc": "Value for logic low."},
            "bit_time": {"type": "float", "default": 0.1, "doc": "Seconds each bit is held."},
            "order": {"type": "int", "default": 7, "doc": "LFSR order (sequence length 2^order-1)."},
            "seed": {"type": "int", "default": 1, "doc": "Non‑zero initial LFSR state."},
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

        order = int(params.get("order", 7))
        if order < 2 or order > 24:
            return {"E": True, "error": "order must be between 2 and 24"}

        # Primitive tap sets for maximal-length LFSR (Galois form)
        primitive_taps = {
            2: [1, 0],
            3: [2, 0],
            4: [3, 0],
            5: [4, 2],
            6: [5, 0],
            7: [6, 5],
            8: [7, 5, 4, 3],
            9: [8, 4],
            10: [9, 6],
            11: [10, 8],
            12: [11, 10, 9, 3],
            13: [12, 11, 8, 6],
            14: [13, 11, 9, 8],
            15: [14, 13],
            16: [15, 13, 12, 10],
            17: [16, 13],
            18: [17, 10],
            19: [18, 17, 16, 13],
            20: [19, 16],
            21: [20, 18],
            22: [21, 20],
            23: [22, 17],
            24: [23, 22, 21, 16],
        }
        taps = primitive_taps.get(order)
        if not taps:
            return {"E": True, "error": f"Unsupported order {order}"}

        if params.get("_init_start_", True):
            seed = int(params.get("seed", 1)) & ((1 << order) - 1)
            if seed == 0:
                seed = 1  # LFSR cannot start at zero
            params["_lfsr"] = seed
            params["_next_flip"] = bit_time
            params["_taps"] = taps
            params["_mask"] = (1 << order) - 1
            # Initial output uses current LFSR LSB
            params["_current_bit"] = params["_lfsr"] & 1
            params["_init_start_"] = False

        # Advance sequence on bit boundaries
        while time >= params["_next_flip"]:
            # XOR of tap bits for feedback
            lfsr = params["_lfsr"]
            feedback = 0
            for p in params["_taps"]:
                feedback ^= (lfsr >> p) & 1
            # Shift left, inject feedback into LSB
            lfsr = ((lfsr << 1) & params["_mask"]) | feedback
            params["_lfsr"] = lfsr
            params["_current_bit"] = lfsr & 1
            params["_next_flip"] += bit_time

        params["_current"] = float(params["high"]) if params["_current_bit"] else float(params["low"])

        return {0: np.atleast_1d(params["_current"])}
