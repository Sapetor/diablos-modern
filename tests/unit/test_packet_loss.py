"""Unit tests for the PacketLoss (lossy channel) block."""

import numpy as np
import pytest

from blocks.packet_loss import PacketLossBlock


def _fresh_params(**overrides):
    """Build a default param dict from the block's declared defaults."""
    block = PacketLossBlock()
    params = {k: v["default"] for k, v in block.params.items()}
    params.update(overrides)
    return params


def _run(params, input_seq, dtime=0.1):
    """Drive the block over a sequence of inputs; return (outputs, deliveries).

    ``deliveries[i]`` is True when sample i was delivered (output == input),
    False when it was dropped.
    """
    block = PacketLossBlock()
    outputs = []
    deliveries = []
    for i, val in enumerate(input_seq):
        t = i * dtime
        res = block.execute(
            time=t,
            inputs={0: np.atleast_1d(float(val))},
            params=params,
            dtime=dtime,
        )
        out = np.atleast_1d(res[0])
        outputs.append(out.copy())
        deliveries.append(bool(np.allclose(out, np.atleast_1d(float(val)))))
    return outputs, deliveries


@pytest.mark.unit
class TestPacketLoss:
    def test_block_contract(self):
        block = PacketLossBlock()
        assert block.block_name == "PacketLoss"
        assert block.category == "Routing"
        assert block.b_type == 1  # memory block (feedback-safe)
        assert len(block.inputs) == 1
        assert len(block.outputs) == 1

    def test_loss_fraction_matches_probability(self):
        """Over many samples ~loss_prob fraction should be dropped."""
        params = _fresh_params(loss_prob=0.5, seed=42)
        n = 4000
        # Use a unique nonzero input each step so a "delivery" is unambiguous.
        inputs = np.arange(1, n + 1, dtype=float)
        _, deliveries = _run(params, inputs)
        dropped = deliveries.count(False)
        frac = dropped / n
        assert abs(frac - 0.5) < 0.05  # within 5% of the loss probability

    def test_hold_repeats_previous_delivered_value(self):
        """In hold mode a dropped sample repeats the last delivered value."""
        params = _fresh_params(loss_prob=0.5, seed=42, drop_mode="hold")
        n = 200
        inputs = np.arange(1, n + 1, dtype=float)
        outputs, deliveries = _run(params, inputs)

        last_delivered = np.atleast_1d(float(params["initial_value"]))
        found_drop = False
        for out, delivered, val in zip(outputs, deliveries, inputs):
            if delivered:
                last_delivered = np.atleast_1d(float(val))
            else:
                found_drop = True
                # Dropped output equals the previously delivered (held) value.
                assert np.allclose(out, last_delivered)
        assert found_drop  # sanity: at least one drop happened

    def test_loss_prob_zero_passes_everything(self):
        """loss_prob=0 => every packet delivered unchanged."""
        params = _fresh_params(loss_prob=0.0, seed=42)
        inputs = np.array([3.0, -1.5, 7.0, 0.25, 42.0])
        outputs, deliveries = _run(params, inputs)
        assert all(deliveries)
        for out, val in zip(outputs, inputs):
            assert np.allclose(out, np.atleast_1d(val))

    def test_loss_prob_one_always_holds(self):
        """loss_prob=1 => every packet dropped; hold keeps the initial value."""
        params = _fresh_params(loss_prob=1.0, seed=42,
                               drop_mode="hold", initial_value=2.0)
        inputs = np.array([3.0, -1.5, 7.0, 0.25, 42.0])
        outputs, deliveries = _run(params, inputs)
        assert not any(deliveries)
        for out in outputs:
            assert np.allclose(out, np.atleast_1d(2.0))

    def test_same_seed_reproducible(self):
        """Two fresh blocks with the same seed drop identically."""
        inputs = np.arange(1, 501, dtype=float)
        _, d1 = _run(_fresh_params(loss_prob=0.5, seed=42), inputs)
        _, d2 = _run(_fresh_params(loss_prob=0.5, seed=42), inputs)
        assert d1 == d2

    def test_different_seed_differs(self):
        """Different seeds produce a different drop pattern."""
        inputs = np.arange(1, 501, dtype=float)
        _, d1 = _run(_fresh_params(loss_prob=0.5, seed=42), inputs)
        _, d2 = _run(_fresh_params(loss_prob=0.5, seed=123), inputs)
        assert d1 != d2

    def test_drop_mode_zero(self):
        """drop_mode='zero' outputs 0 on a dropped packet."""
        params = _fresh_params(loss_prob=1.0, seed=42,
                               drop_mode="zero", initial_value=9.0)
        outputs, _ = _run(params, np.array([5.0, 5.0, 5.0]))
        for out in outputs:
            assert np.allclose(out, np.atleast_1d(0.0))

    def test_drop_mode_nan(self):
        """drop_mode='nan' outputs NaN on a dropped packet."""
        params = _fresh_params(loss_prob=1.0, seed=42, drop_mode="nan")
        outputs, _ = _run(params, np.array([5.0, 5.0, 5.0]))
        for out in outputs:
            assert np.all(np.isnan(out))

    def test_vector_input_preserved(self):
        """Vector packets are held/delivered elementwise."""
        params = _fresh_params(loss_prob=0.0, seed=42)
        block = PacketLossBlock()
        vec = np.array([1.0, 2.0, 3.0])
        res = block.execute(time=0.0, inputs={0: vec}, params=params, dtime=0.1)
        assert np.allclose(np.atleast_1d(res[0]), vec)

    def test_sample_time_holds_between_samples(self):
        """With sample_time>dtime the block holds between sample instants."""
        params = _fresh_params(loss_prob=0.0, seed=42, sample_time=0.5)
        block = PacketLossBlock()
        dtime = 0.1
        # t=0 samples and delivers 10.0; t=0.1..0.4 should hold 10.0.
        out0 = np.atleast_1d(block.execute(time=0.0, inputs={0: np.array([10.0])},
                                           params=params, dtime=dtime)[0])
        assert np.allclose(out0, [10.0])
        for k in range(1, 5):  # t = 0.1, 0.2, 0.3, 0.4
            out = np.atleast_1d(block.execute(
                time=k * dtime, inputs={0: np.array([99.0])},
                params=params, dtime=dtime)[0])
            assert np.allclose(out, [10.0])  # held, not the new 99.0
