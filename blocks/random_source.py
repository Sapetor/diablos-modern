import numpy as np
from blocks.base_block import BaseBlock


class RandomSourceBlock(BaseBlock):
    """
    Random signal source.

    A pure source (zero inputs, one output) that draws a single random value
    from a seeded RNG at each sample instant and holds it between instants.
    Supported distributions:

    - ``uniform``   : U[low, high]
    - ``bernoulli`` : 1.0 with probability ``p`` else 0.0
    - ``normal``    : N(mu, sigma)
    - ``randint``   : uniform integer in [low, high] (both inclusive)

    Typical uses: drive a Switch control port for packet gating, or feed a
    VariableTransportDelay ``tau`` port to model random latency.

    All cross-timestep state (RNG, held value, next sample time) lives in
    ``params`` so the engine's ``reset_memblocks`` can reset it cleanly.
    """

    @property
    def block_name(self):
        return "RandomSource"

    @property
    def category(self):
        return "Sources"

    @property
    def b_type(self):
        """Source block - generates output without requiring input."""
        return 0

    @property
    def requires_inputs(self):
        return False

    @property
    def color(self):
        return "blue"

    @property
    def doc(self):
        return (
            "Random Source."
            "\n\nGenerates a random value from a seeded RNG, held between sample "
            "instants."
            "\n\nDistributions:"
            "\n- uniform:   U[low, high]."
            "\n- bernoulli: 1.0 with probability p, else 0.0."
            "\n- normal:    N(mu, sigma)."
            "\n- randint:   integer in [low, high] (inclusive)."
            "\n\nParameters:"
            "\n- Distribution: Which distribution to sample from."
            "\n- p: Bernoulli success probability (0..1)."
            "\n- low / high: Range for uniform and randint."
            "\n- mu / sigma: Mean and std-dev for normal."
            "\n- Sample Time: Sample period (s). 0 = every solver step (use dtime)."
            "\n- Seed: RNG seed (0 = non-reproducible; nonzero = reproducible)."
            "\n\nUsage:"
            "\nDrive a Switch control port (packet gating) or a "
            "VariableTransportDelay tau port (random latency)."
        )

    @property
    def params(self):
        return {
            "distribution": {"type": "choice", "default": "uniform",
                             "choices": ["uniform", "bernoulli", "normal", "randint"],
                             "doc": "Distribution to sample from."},
            "p": {"type": "float", "default": 0.5,
                  "doc": "Bernoulli success probability (0..1)."},
            "low": {"type": "float", "default": 0.0,
                    "doc": "Lower bound for uniform / randint."},
            "high": {"type": "float", "default": 1.0,
                     "doc": "Upper bound for uniform / randint."},
            "mu": {"type": "float", "default": 0.0,
                   "doc": "Mean for the normal distribution."},
            "sigma": {"type": "float", "default": 1.0,
                      "doc": "Standard deviation for the normal distribution."},
            "sample_time": {"type": "float", "default": 0.0,
                            "doc": "Sample period (s). 0 = every step (use dtime)."},
            "seed": {"type": "int", "default": 0,
                     "doc": "RNG seed (0 = non-reproducible, nonzero = reproducible)."},
            "_init_start_": {"type": "bool", "default": True,
                             "doc": "Internal init flag."},
        }

    @property
    def inputs(self):
        return []

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a 'dice / random samples' icon in normalized 0-1 coordinates."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # scattered random stems on a baseline
        path.moveTo(0.10, 0.80)
        path.lineTo(0.10, 0.45)
        path.moveTo(0.30, 0.80)
        path.lineTo(0.30, 0.25)
        path.moveTo(0.50, 0.80)
        path.lineTo(0.50, 0.55)
        path.moveTo(0.70, 0.80)
        path.lineTo(0.70, 0.30)
        path.moveTo(0.90, 0.80)
        path.lineTo(0.90, 0.50)
        # baseline
        path.moveTo(0.05, 0.80)
        path.lineTo(0.95, 0.80)
        return path

    def _draw(self, params):
        """Draw a single value from the seeded RNG per the chosen distribution."""
        rng = params["_rng"]
        distribution = params.get("distribution", "uniform")

        if distribution == "bernoulli":
            p = float(params.get("p", 0.5))
            return 1.0 if rng.random() < p else 0.0

        if distribution == "normal":
            mu = float(params.get("mu", 0.0))
            sigma = float(params.get("sigma", 1.0))
            return sigma * rng.standard_normal() + mu

        if distribution == "randint":
            low = int(round(float(params.get("low", 0.0))))
            high = int(round(float(params.get("high", 1.0))))
            if high < low:
                low, high = high, low
            # inclusive of both endpoints
            return float(rng.integers(low, high + 1))

        # default: uniform U[low, high]
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        return low + (high - low) * rng.random()

    def execute(self, time, inputs, params, **kwargs):
        # Initialize cross-timestep state in params (never on self).
        if params.get("_init_start_", True):
            seed = int(params.get("seed", 0))
            # seed == 0 -> entropy (non-reproducible); nonzero -> reproducible.
            params["_rng"] = np.random.default_rng(seed if seed != 0 else None)
            params["_held_"] = None
            params["_next_sample_time_"] = float(time)
            params["_init_start_"] = False

        # Step size: sample_time if > 0, else the solver step (dtime).
        sample_time = float(params.get("sample_time", 0.0))
        dtime = kwargs.get("dtime", params.get("dtime"))
        dtime = float(dtime) if dtime else 0.0
        step = sample_time if sample_time > 0.0 else dtime

        # Sample instant? (advance schedule, draw once per grid point).
        # Force a draw on the very first call so the output is never None.
        if params["_held_"] is None or time >= params["_next_sample_time_"] - 1e-12:
            params["_held_"] = float(self._draw(params))

            if step > 0.0:
                while params["_next_sample_time_"] <= time + 1e-12:
                    params["_next_sample_time_"] += step
            else:
                # No step info: sample on every call.
                params["_next_sample_time_"] = time

        return {0: np.atleast_1d(params["_held_"])}
