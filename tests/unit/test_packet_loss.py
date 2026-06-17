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

    def test_default_loss_model_is_bernoulli(self):
        """The block defaults to the i.i.d. Bernoulli model."""
        block = PacketLossBlock()
        assert block.params["loss_model"]["default"] == "bernoulli"
        spec = block.params["loss_model"]
        opts = spec.get("options") or spec.get("choices")
        assert opts == ["bernoulli", "gilbert_elliott"]


def _drop_seq(deliveries):
    """Drop indicator sequence: 1 if dropped, 0 if delivered."""
    return [0 if d else 1 for d in deliveries]


def _max_run_length(drop_seq):
    """Longest run of consecutive drops (value==1)."""
    best = run = 0
    for v in drop_seq:
        if v == 1:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


@pytest.mark.unit
class TestGilbertElliott:
    def test_ge_is_bursty_vs_bernoulli(self):
        """Gilbert-Elliott yields markedly more clustered (bursty) drops than
        i.i.d. Bernoulli at a comparable average loss rate."""
        n = 6000
        inputs = np.arange(1, n + 1, dtype=float)

        # GE: rarely enters the bad state (p_bg small) but stays a while
        # (p_gb small) dropping almost everything there -> bursts.
        ge_params = _fresh_params(
            loss_model="gilbert_elliott", seed=7,
            loss_prob=0.0,        # good state never drops
            loss_prob_bad=1.0,    # bad state always drops
            p_bg=0.02,            # good->bad
            p_gb=0.1,             # bad->good (stays bad ~10 samples)
        )
        _, ge_deliv = _run(ge_params, inputs)
        ge_drops = _drop_seq(ge_deliv)
        ge_rate = sum(ge_drops) / n

        # Bernoulli at the SAME average loss rate.
        bern_params = _fresh_params(
            loss_model="bernoulli", seed=7, loss_prob=ge_rate)
        _, bern_deliv = _run(bern_params, inputs)
        bern_drops = _drop_seq(bern_deliv)

        # Sanity: a meaningful number of drops actually occurred.
        assert sum(ge_drops) > 50
        # Burstiness: GE's longest consecutive-drop run is much larger than
        # i.i.d. Bernoulli at the same mean rate.
        assert _max_run_length(ge_drops) > 2 * _max_run_length(bern_drops)

    def test_ge_drops_cluster_in_bad_state(self):
        """With good-state loss 0 and bad-state loss 1, every drop must occur
        while the chain is in the bad state, and drops come in contiguous runs
        (length > 1 on average)."""
        n = 4000
        inputs = np.arange(1, n + 1, dtype=float)
        params = _fresh_params(
            loss_model="gilbert_elliott", seed=11,
            loss_prob=0.0, loss_prob_bad=1.0,
            p_bg=0.03, p_gb=0.2)
        _, deliveries = _run(params, inputs)
        drops = _drop_seq(deliveries)
        total_drops = sum(drops)
        assert total_drops > 20
        # Count bursts (maximal runs of consecutive drops).
        bursts = 0
        prev = 0
        for v in drops:
            if v == 1 and prev == 0:
                bursts += 1
            prev = v
        # Average burst length clearly > 1 (bursty, not isolated drops).
        assert total_drops / bursts > 1.5

    def test_ge_reproducible_with_fixed_seed(self):
        """Two fresh GE blocks with the same seed drop identically."""
        inputs = np.arange(1, 801, dtype=float)
        kw = dict(loss_model="gilbert_elliott", seed=99,
                  loss_prob=0.05, loss_prob_bad=0.95, p_bg=0.1, p_gb=0.4)
        _, d1 = _run(_fresh_params(**kw), inputs)
        _, d2 = _run(_fresh_params(**kw), inputs)
        assert d1 == d2

    def test_ge_different_seed_differs(self):
        """Different seeds give a different GE drop pattern."""
        inputs = np.arange(1, 801, dtype=float)
        base = dict(loss_model="gilbert_elliott",
                    loss_prob=0.05, loss_prob_bad=0.95, p_bg=0.1, p_gb=0.4)
        _, d1 = _run(_fresh_params(seed=1, **base), inputs)
        _, d2 = _run(_fresh_params(seed=2, **base), inputs)
        assert d1 != d2

    def test_ge_honors_drop_mode_and_hold(self):
        """GE reuses the existing drop_mode/hold logic: a held value repeats
        the last delivered packet on a dropped sample."""
        params = _fresh_params(
            loss_model="gilbert_elliott", seed=3, drop_mode="hold",
            loss_prob=0.0, loss_prob_bad=1.0, p_bg=0.5, p_gb=0.1,
            initial_value=0.0)
        n = 300
        inputs = np.arange(1, n + 1, dtype=float)
        outputs, deliveries = _run(params, inputs)

        last_delivered = np.atleast_1d(0.0)
        found_drop = False
        for out, delivered, val in zip(outputs, deliveries, inputs):
            if delivered:
                last_delivered = np.atleast_1d(float(val))
            else:
                found_drop = True
                assert np.allclose(out, last_delivered)
        assert found_drop

    def test_bernoulli_path_unchanged_by_new_params(self):
        """Adding GE params must not change the Bernoulli RNG stream: an
        explicit bernoulli model with the default loss_prob reproduces the
        same drop pattern as a plain default-model run."""
        inputs = np.arange(1, 501, dtype=float)
        _, d_default = _run(_fresh_params(loss_prob=0.5, seed=42), inputs)
        _, d_explicit = _run(
            _fresh_params(loss_model="bernoulli", loss_prob=0.5, seed=42),
            inputs)
        assert d_default == d_explicit
