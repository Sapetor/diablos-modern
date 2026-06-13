"""Unit tests for the NetworkChannel (lossy + jittery link) block."""

import numpy as np
import pytest

from blocks.network_channel import NetworkChannelBlock


def _fresh_params(**overrides):
    """Build a default param dict from the block's declared defaults."""
    block = NetworkChannelBlock()
    params = {k: v["default"] for k, v in block.params.items()}
    params.update(overrides)
    return params


def _run(params, input_seq, dtime=0.1):
    """Drive the block over a sequence of scalar inputs.

    Returns a list of 1-D output arrays, one per step. A fresh block instance
    is used (state lives in ``params``, which the caller owns).
    """
    block = NetworkChannelBlock()
    outputs = []
    for i, val in enumerate(input_seq):
        t = i * dtime
        res = block.execute(
            time=t,
            inputs={0: np.atleast_1d(float(val))},
            params=params,
            dtime=dtime,
        )
        outputs.append(np.atleast_1d(res[0]).copy())
    return outputs


@pytest.mark.unit
class TestNetworkChannel:
    def test_block_contract(self):
        block = NetworkChannelBlock()
        assert block.block_name == "NetworkChannel"
        assert block.category == "Routing"
        assert block.b_type == 1  # memory block (feedback-safe)
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1

    def test_fixed_delay_then_hold_of_step(self):
        """loss_prob=0, min_delay==max_delay==D => pure delay-then-hold.

        A step (0 -> 1 at t=0.3s) should appear at the output ~D seconds
        later and then hold.
        """
        D = 0.3
        params = _fresh_params(loss_prob=0.0, min_delay=D, max_delay=D,
                               seed=7, initial_value=0.0)
        dtime = 0.1
        # input: 0 for t in [0,0.2], then 1 from t=0.3 onward
        n = 12
        inputs = [0.0] * 3 + [1.0] * (n - 3)
        outputs = _run(params, inputs, dtime=dtime)

        # Before the step's packet is delivered (step sent at t=0.3, arrives
        # at ~0.6 => index 6) the output stays at the initial 0.
        for i in range(0, 6):
            assert np.allclose(outputs[i], [0.0]), f"step {i} should be 0, got {outputs[i]}"
        # At/after delivery the output becomes 1 and holds.
        for i in range(6, n):
            assert np.allclose(outputs[i], [1.0]), f"step {i} should be 1, got {outputs[i]}"

    def test_fixed_delay_value_at_arrival_matches_sent(self):
        """Each delivered value equals the input that was sent D earlier."""
        D = 0.2
        params = _fresh_params(loss_prob=0.0, min_delay=D, max_delay=D,
                               seed=3, initial_value=-99.0)
        dtime = 0.1
        # ramp 0,1,2,3,...,9 ; with D=0.2 (2 steps), input sent at step k
        # is delivered at step k+2.
        n = 10
        inputs = list(range(n))
        outputs = _run(params, inputs, dtime=dtime)
        # steps 0,1: nothing delivered yet -> initial value
        assert np.allclose(outputs[0], [-99.0])
        assert np.allclose(outputs[1], [-99.0])
        # step k>=2 delivers the value sent at step k-2 == (k-2)
        for k in range(2, n):
            assert np.allclose(outputs[k], [float(k - 2)]), \
                f"step {k}: expected {k-2}, got {outputs[k]}"

    def test_loss_prob_one_never_delivers(self):
        """loss_prob=1 => nothing is ever delivered; output stays initial."""
        params = _fresh_params(loss_prob=1.0, min_delay=0.0, max_delay=0.05,
                               seed=11, initial_value=2.0, drop_mode="hold")
        inputs = [3.0, -1.5, 7.0, 0.25, 42.0, 8.0, 9.0]
        outputs = _run(params, inputs)
        for out in outputs:
            assert np.allclose(out, [2.0])

    def test_loss_prob_one_drop_mode_zero(self):
        """drop_mode='zero' outputs 0 while nothing has been delivered."""
        params = _fresh_params(loss_prob=1.0, seed=11,
                               drop_mode="zero", initial_value=5.0)
        outputs = _run(params, [4.0, 4.0, 4.0])
        for out in outputs:
            assert np.allclose(out, [0.0])

    def test_loss_prob_one_drop_mode_nan(self):
        """drop_mode='nan' outputs NaN while nothing has been delivered."""
        params = _fresh_params(loss_prob=1.0, seed=11, drop_mode="nan")
        outputs = _run(params, [4.0, 4.0, 4.0])
        for out in outputs:
            assert np.all(np.isnan(out))

    def test_loss_fraction_matches_probability(self):
        """~(1-loss_prob) fraction of packets should be delivered.

        Count distinct delivered packets via the buffer enqueue rate: with
        zero delay, the number of unique delivered values over many steps
        approximates (1-loss_prob)*n.
        """
        loss_prob = 0.3
        params = _fresh_params(loss_prob=loss_prob, min_delay=0.0,
                               max_delay=0.0, seed=42)
        n = 5000
        # Unique increasing inputs; with zero delay each delivered packet
        # shows up immediately, so a change in output == a delivery.
        inputs = np.arange(1, n + 1, dtype=float)
        outputs = _run(params, inputs)
        # Count steps where the output equals the just-sent input (delivered
        # this step with zero latency).
        delivered = sum(
            1 for i, out in enumerate(outputs)
            if np.allclose(out, [inputs[i]])
        )
        frac_delivered = delivered / n
        assert abs(frac_delivered - (1.0 - loss_prob)) < 0.05

    def test_same_seed_reproducible(self):
        """Two fresh blocks with the same seed produce identical output."""
        inputs = np.arange(1, 401, dtype=float)
        o1 = _run(_fresh_params(loss_prob=0.4, min_delay=0.0,
                                max_delay=0.3, seed=42), inputs)
        o2 = _run(_fresh_params(loss_prob=0.4, min_delay=0.0,
                                max_delay=0.3, seed=42), inputs)
        assert all(np.allclose(a, b) for a, b in zip(o1, o2))

    def test_different_seed_differs(self):
        """Different seeds produce a different output trace."""
        inputs = np.arange(1, 401, dtype=float)
        o1 = _run(_fresh_params(loss_prob=0.4, min_delay=0.0,
                                max_delay=0.3, seed=42), inputs)
        o2 = _run(_fresh_params(loss_prob=0.4, min_delay=0.0,
                                max_delay=0.3, seed=123), inputs)
        assert any(not np.allclose(a, b) for a, b in zip(o1, o2))

    def test_seed_zero_is_nondeterministic_but_valid(self):
        """seed=0 => entropy-seeded; runs still produce finite output."""
        inputs = np.arange(1, 51, dtype=float)
        out = _run(_fresh_params(loss_prob=0.2, seed=0), inputs)
        assert len(out) == 50
        # outputs are finite (no NaN with default hold once delivered)
        assert all(np.all(np.isfinite(o)) for o in out)

    def test_vector_input_preserved(self):
        """Vector packets are delivered/held elementwise."""
        params = _fresh_params(loss_prob=0.0, min_delay=0.0,
                               max_delay=0.0, seed=42)
        block = NetworkChannelBlock()
        vec = np.array([1.0, 2.0, 3.0])
        res = block.execute(time=0.0, inputs={0: vec}, params=params, dtime=0.1)
        assert np.allclose(np.atleast_1d(res[0]), vec)

    def test_state_lives_in_params(self):
        """No persistent simulation state on the instance (only in params)."""
        params = _fresh_params(loss_prob=0.0, seed=42)
        block = NetworkChannelBlock()
        block.execute(time=0.0, inputs={0: np.array([1.0])}, params=params, dtime=0.1)
        # The RNG / buffer / held value must be in params, not on self.
        assert "_rng" in params
        assert "_buffer_" in params
        assert "_held_" in params
        assert not hasattr(block, "_rng")
        assert not hasattr(block, "_buffer_")
