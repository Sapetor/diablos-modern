import numpy as np
from blocks.base_block import BaseBlock


class NetworkChannelBlock(BaseBlock):
    """
    Network communication link / lossy + jittery channel block.

    Combines random PACKET LOSS and random DELAY (jitter) into a single
    "communication link". At each sample instant a Bernoulli trial decides
    whether the incoming packet is delivered. A delivered packet is given a
    per-packet random latency drawn uniformly from ``[min_delay, max_delay]``
    and is enqueued for delivery at ``time + delay``. The output is the value
    of the most-recently-delivered packet whose delivery time has already
    passed (zero-order hold between deliveries).

    Before any packet has been delivered the output is the held / initial
    value, or - for the no-delivery case - whatever ``drop_mode`` dictates.

    All cross-timestep state (RNG, held value, packet buffer, next sample
    time) lives in ``params`` so the engine's ``reset_memblocks`` can reset it
    cleanly. Nothing persistent is stored on ``self``.
    """

    @property
    def block_name(self):
        return "NetworkChannel"

    @property
    def category(self):
        return "Routing"

    @property
    def b_type(self):
        """Memory block (like Delay) - safe inside feedback loops."""
        return 1

    @property
    def color(self):
        return "darkred"

    @property
    def doc(self):
        return (
            "Network Channel / Communication Link."
            "\n\nModels an unreliable, jittery communication link combining random "
            "packet loss with a random per-packet transport delay (latency jitter)."
            "\n\nAt each sample instant a Bernoulli trial (Loss Probability) decides "
            "whether the packet is dropped. A surviving packet is assigned a random "
            "latency drawn uniformly from [Min Delay, Max Delay] and is delivered "
            "that many seconds later. The output holds the most recently delivered "
            "packet (zero-order hold) until the next one arrives."
            "\n\nParameters:"
            "\n- Loss Probability: Probability (0..1) that a packet is dropped."
            "\n- Min Delay / Max Delay: Latency window (s) for delivered packets."
            "\n- Sample Time: Channel sample period (s). 0 = every solver step."
            "\n- Seed: RNG seed (0 = non-reproducible; nonzero = reproducible)."
            "\n- Drop Mode: Output before any delivery: 'hold', 'zero', or 'nan'."
            "\n- Initial Value: Held value before the first delivered packet."
            "\n\nUsage:"
            "\nSimulate networked / sampled-data control over a lossy link with"
            "\nvariable latency (jitter)."
        )

    @property
    def params(self):
        return {
            "loss_prob": {"type": "float", "default": 0.1,
                          "doc": "Probability (0..1) that a packet is dropped."},
            "min_delay": {"type": "float", "default": 0.0,
                          "doc": "Minimum per-packet transport delay (s)."},
            "max_delay": {"type": "float", "default": 0.1,
                          "doc": "Maximum per-packet transport delay (s)."},
            "sample_time": {"type": "float", "default": 0.0,
                            "doc": "Channel sample period (s). 0 = every step (use dtime)."},
            "seed": {"type": "int", "default": 0,
                     "doc": "RNG seed (0 = non-reproducible, nonzero = reproducible)."},
            "drop_mode": {"type": "choice", "default": "hold",
                          "options": ["hold", "zero", "nan"],
                          "doc": "Output before any packet is delivered: hold / zero / nan."},
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
        """Draw a 'lossy + delayed pulse' icon (two nodes, one dropped) in 0-1 coords."""
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        # transmit node
        path.addEllipse(0.08, 0.42, 0.16, 0.16)
        # receive node
        path.addEllipse(0.76, 0.42, 0.16, 0.16)
        # delivered (delayed) link - lower arc-ish path
        path.moveTo(0.24, 0.50)
        path.lineTo(0.45, 0.70)
        path.lineTo(0.76, 0.50)
        # dropped link - dashed upper segments with a gap
        path.moveTo(0.26, 0.42)
        path.lineTo(0.40, 0.30)
        path.moveTo(0.55, 0.30)
        path.lineTo(0.74, 0.42)
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
            # Buffer of delivered (but not-yet-due) packets as a list of
            # (delivery_time, value) tuples, kept sorted by delivery_time.
            params["_buffer_"] = []
            params["_delivered_any_"] = False
            params["_next_sample_time_"] = float(time)
            params["_init_start_"] = False

        loss_prob = float(params.get("loss_prob", 0.1))
        drop_mode = params.get("drop_mode", "hold")

        min_delay = float(params.get("min_delay", 0.0))
        max_delay = float(params.get("max_delay", 0.1))
        # Guard against an inverted window (min > max).
        if max_delay < min_delay:
            min_delay, max_delay = max_delay, min_delay
        min_delay = max(0.0, min_delay)
        max_delay = max(0.0, max_delay)

        # Step size: sample_time if > 0, else the solver step (dtime).
        sample_time = float(params.get("sample_time", 0.0))
        dtime = kwargs.get("dtime", params.get("dtime"))
        dtime = float(dtime) if dtime else 0.0
        step = sample_time if sample_time > 0.0 else dtime

        current_input = np.atleast_1d(np.asarray(inputs.get(0, 0.0), dtype=float))

        buffer = params["_buffer_"]

        # ---- Sample instant: maybe enqueue a new packet -------------------
        if time >= params["_next_sample_time_"] - 1e-12:
            u = params["_rng"].random()
            if u >= loss_prob:
                # DELIVERED -> draw a per-packet latency and enqueue.
                if max_delay > min_delay:
                    d = float(params["_rng"].uniform(min_delay, max_delay))
                else:
                    d = min_delay  # degenerate window -> fixed delay
                delivery_time = float(time) + d
                # Keep buffer sorted by delivery time (latencies can reorder).
                idx = len(buffer)
                while idx > 0 and buffer[idx - 1][0] > delivery_time:
                    idx -= 1
                buffer.insert(idx, (delivery_time, current_input.copy()))
            # (dropped packets are simply never enqueued)

            # Advance the sample schedule past the current time.
            if step > 0.0:
                while params["_next_sample_time_"] <= time + 1e-12:
                    params["_next_sample_time_"] += step
            else:
                # No step info: sample on every call.
                params["_next_sample_time_"] = time

        # ---- Deliver: take the most-recent packet whose time has passed ---
        # Buffer is sorted by delivery_time; consume everything that is due.
        keep_from = 0
        for dtime_pkt, value in buffer:
            if dtime_pkt <= time + 1e-12:
                params["_held_"] = value
                params["_delivered_any_"] = True
                keep_from += 1
            else:
                break
        # Prune all packets that have already been delivered (their value is
        # now captured in _held_); only future-due packets remain.
        if keep_from > 0:
            del buffer[:keep_from]
        params["_buffer_"] = buffer

        # ---- Output ------------------------------------------------------
        if params["_delivered_any_"]:
            # Zero-order hold on the last delivered packet.
            out = params["_held_"]
        else:
            # No packet delivered yet -> per drop_mode fallback.
            if drop_mode == "zero":
                out = np.zeros_like(params["_held_"])
            elif drop_mode == "nan":
                out = np.full_like(params["_held_"], np.nan)
            else:  # "hold"
                out = params["_held_"]

        return {0: np.atleast_1d(out)}
