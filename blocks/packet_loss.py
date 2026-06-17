import numpy as np
from blocks.base_block import BaseBlock


class PacketLossBlock(BaseBlock):
    """
    Lossy communication channel / packet-loss block.

    At each sample instant the block draws a uniform random number from a
    seeded RNG. With probability ``loss_prob`` the incoming "packet" is dropped
    and the output falls back to a held value according to ``drop_mode``;
    otherwise the packet is delivered (output = input) and the held value is
    updated. Between sample instants the current held value is repeated.

    All cross-timestep state (RNG, held value, next sample time) lives in
    ``params`` so the engine's ``reset_memblocks`` can reset it cleanly.
    """

    @property
    def block_name(self):
        return "PacketLoss"

    @property
    def category(self):
        return "Routing"

    @property
    def b_type(self):
        """Memory block (like Delay) — safe inside feedback loops."""
        return 1

    @property
    def color(self):
        return "darkred"

    @property
    def doc(self):
        return (
            "Packet-Loss / Lossy Channel."
            "\n\nModels an unreliable communication link. At each sample instant a "
            "random trial decides whether the packet is delivered or dropped."
            "\n\nDelivered: output = input (and the held value is updated)."
            "\nDropped:   output falls back per Drop Mode."
            "\n\nLoss Models:"
            "\n- bernoulli: i.i.d. drops with probability Loss Probability."
            "\n- gilbert_elliott: two-state Markov chain (good/bad). Drops are bursty: "
            "in the good state the drop prob is Loss Probability; in the bad state it is "
            "Loss Prob (Bad). Transitions: good->bad with prob p(bg), bad->good with prob p(gb)."
            "\n\nParameters:"
            "\n- Loss Model: 'bernoulli' (i.i.d.) or 'gilbert_elliott' (bursty)."
            "\n- Loss Probability: Bernoulli / good-state drop probability (0..1)."
            "\n- p(bg): good->bad transition probability (Gilbert-Elliott)."
            "\n- p(gb): bad->good transition probability (Gilbert-Elliott)."
            "\n- Loss Prob (Bad): drop probability while in the bad state (Gilbert-Elliott)."
            "\n- Sample Time: Channel sample period (s). 0 = every solver step."
            "\n- Seed: RNG seed (0 = non-reproducible; nonzero = reproducible)."
            "\n- Drop Mode: 'hold' (keep last delivered), 'zero', or 'nan'."
            "\n- Initial Value: Held value before the first delivery."
            "\n\nUsage:"
            "\nSimulate dropped samples in networked / sampled-data control."
        )

    @property
    def params(self):
        return {
            "loss_model": {"type": "choice", "default": "bernoulli",
                           "options": ["bernoulli", "gilbert_elliott"],
                           "doc": "Loss model: 'bernoulli' (i.i.d.) or "
                                  "'gilbert_elliott' (bursty two-state Markov)."},
            "loss_prob": {"type": "float", "default": 0.1,
                          "doc": "Bernoulli / good-state drop probability (0..1)."},
            "p_bg": {"type": "float", "default": 0.1,
                     "doc": "Gilbert-Elliott good->bad transition probability."},
            "p_gb": {"type": "float", "default": 0.5,
                     "doc": "Gilbert-Elliott bad->good transition probability."},
            "loss_prob_bad": {"type": "float", "default": 0.9,
                              "doc": "Gilbert-Elliott drop probability in the bad state."},
            "sample_time": {"type": "float", "default": 0.0,
                            "doc": "Channel sample period (s). 0 = every step (use dtime)."},
            "seed": {"type": "int", "default": 0,
                     "doc": "RNG seed (0 = non-reproducible, nonzero = reproducible)."},
            "drop_mode": {"type": "choice", "default": "hold",
                          "options": ["hold", "zero", "nan"],
                          "doc": "Output on a dropped packet: hold / zero / nan."},
            "initial_value": {"type": "float", "default": 0.0,
                              "doc": "Held value before the first delivered packet."},
            "_init_start_": {"type": "bool", "default": True,
                             "doc": "Internal init flag."},
        }

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def draw_icon(self, block_rect):
        """Draw a 'lossy pulse train' icon (one pulse missing) in 0-1 coords."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # pulse 1
        path.moveTo(0.10, 0.75)
        path.lineTo(0.10, 0.30)
        path.lineTo(0.25, 0.30)
        path.lineTo(0.25, 0.75)
        # gap (dropped pulse) — lift to next pulse
        path.moveTo(0.55, 0.75)
        path.lineTo(0.55, 0.30)
        path.lineTo(0.70, 0.30)
        path.lineTo(0.70, 0.75)
        # baseline
        path.moveTo(0.05, 0.75)
        path.lineTo(0.95, 0.75)
        return path

    def execute(self, time, inputs, params, **kwargs):
        # Initialize cross-timestep state in params (never on self).
        if params.get("_init_start_", True):
            seed = int(params.get("seed", 0))
            # seed == 0 -> entropy (non-reproducible); nonzero -> reproducible.
            params["_rng"] = np.random.default_rng(seed if seed != 0 else None)
            params["_held_"] = np.atleast_1d(
                np.asarray(params.get("initial_value", 0.0), dtype=float)
            )
            params["_next_sample_time_"] = float(time)
            # Gilbert-Elliott two-state Markov chain starts in the "good" state.
            params["_ge_state_"] = "good"
            params["_init_start_"] = False

        loss_prob = float(params.get("loss_prob", 0.1))
        drop_mode = params.get("drop_mode", "hold")
        loss_model = params.get("loss_model", "bernoulli")

        # Step size: sample_time if > 0, else the solver step (dtime).
        sample_time = float(params.get("sample_time", 0.0))
        dtime = kwargs.get("dtime", params.get("dtime"))
        dtime = float(dtime) if dtime else 0.0
        step = sample_time if sample_time > 0.0 else dtime

        current_input = np.atleast_1d(np.asarray(inputs.get(0, 0.0), dtype=float))

        # Sample instant? (advance the schedule, draw once per grid point)
        if time >= params["_next_sample_time_"] - 1e-12:
            rng = params["_rng"]
            if loss_model == "gilbert_elliott":
                # 1) Transition the Markov state first.
                if params["_ge_state_"] == "good":
                    if rng.random() < float(params.get("p_bg", 0.1)):
                        params["_ge_state_"] = "bad"
                else:  # "bad"
                    if rng.random() < float(params.get("p_gb", 0.5)):
                        params["_ge_state_"] = "good"
                # 2) Drop probability depends on the (post-transition) state.
                if params["_ge_state_"] == "good":
                    drop_prob = loss_prob
                else:
                    drop_prob = float(params.get("loss_prob_bad", 0.9))
            else:  # "bernoulli" (default) — byte-identical to the original path.
                drop_prob = loss_prob

            u = rng.random()
            if u < drop_prob:
                # DROPPED -> output per drop_mode.
                if drop_mode == "zero":
                    out = np.zeros_like(params["_held_"])
                elif drop_mode == "nan":
                    out = np.full_like(params["_held_"], np.nan)
                else:  # "hold"
                    out = params["_held_"]
            else:
                # DELIVERED -> pass through and update held value.
                params["_held_"] = current_input
                out = current_input

            # Advance the sample schedule past the current time.
            if step > 0.0:
                while params["_next_sample_time_"] <= time + 1e-12:
                    params["_next_sample_time_"] += step
            else:
                # No step info: sample on every call.
                params["_next_sample_time_"] = time

            return {0: np.atleast_1d(out)}

        # Between sample instants: repeat the held value.
        return {0: np.atleast_1d(params["_held_"])}
