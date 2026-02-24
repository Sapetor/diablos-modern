import pytest
from blocks.prbs import PRBSBlock


def test_prbs_deterministic_sequence():
    block = PRBSBlock()
    params = {
        "high": 1.0,
        "low": 0.0,
        "bit_time": 0.2,
        "order": 4,
        "seed": 1,
        "_init_start_": True,
    }

    times = [0.0, 0.2, 0.4, 0.6, 0.8]
    outputs = [block.execute(t, {}, params)[0][0] for t in times]

    # Expected LFSR (order 4, taps [3,0], seed 1): 1,1,1,1,0,...
    assert outputs == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_prbs_requires_positive_bit_time():
    block = PRBSBlock()
    params = {
        "bit_time": 0.0,
        "_init_start_": True,
        "high": 1.0,
        "low": 0.0,
        "seed": 1,
        "order": 4,
    }
    result = block.execute(0.0, {}, params)
    assert result.get("E") is True


def test_prbs_validates_order():
    block = PRBSBlock()
    params = {"bit_time": 0.1, "order": 1, "_init_start_": True, "high": 1.0, "low": 0.0, "seed": 1}
    result = block.execute(0.0, {}, params)
    assert result.get("E") is True


@pytest.mark.parametrize("order", range(2, 17))
def test_prbs_maximal_length_sequence(order):
    """Each LFSR order must produce a maximal-length sequence of period 2^n - 1."""
    block = PRBSBlock()
    params = {
        "high": 1.0,
        "low": 0.0,
        "bit_time": 1.0,
        "order": order,
        "seed": 1,
        "_init_start_": True,
    }

    expected_period = (1 << order) - 1

    # Run one full period plus 1 to check it cycles back
    # Track LFSR states
    states = []
    for i in range(expected_period + 1):
        t = float(i)
        block.execute(t, {}, params)
        states.append(params["_lfsr"])

    # A maximal-length LFSR visits all 2^n - 1 non-zero states exactly once
    unique_states = set(states[:expected_period])
    assert len(unique_states) == expected_period, (
        f"Order {order}: expected {expected_period} unique states, got {len(unique_states)}"
    )
    # State after one full period should equal the initial state (cycle)
    assert states[expected_period] == states[0], (
        f"Order {order}: LFSR did not cycle back after {expected_period} steps"
    )


def test_prbs_default_order7_produces_nonzero():
    """Default order=7 PRBS must produce both high and low values."""
    block = PRBSBlock()
    params = {
        "high": 1.0,
        "low": -1.0,
        "bit_time": 0.05,
        "order": 7,
        "seed": 1,
        "_init_start_": True,
    }

    outputs = set()
    for i in range(127):
        t = i * 0.05
        result = block.execute(t, {}, params)
        outputs.add(float(result[0][0]))

    assert 1.0 in outputs, "PRBS never produces high value"
    assert -1.0 in outputs, "PRBS never produces low value"


def test_prbs_high_low_values():
    """PRBS output should only be high or low, nothing else."""
    block = PRBSBlock()
    params = {
        "high": 5.0,
        "low": -3.0,
        "bit_time": 0.1,
        "order": 4,
        "seed": 1,
        "_init_start_": True,
    }

    for i in range(15):
        result = block.execute(i * 0.1, {}, params)
        val = float(result[0][0])
        assert val in (5.0, -3.0), f"Unexpected PRBS value {val}"
