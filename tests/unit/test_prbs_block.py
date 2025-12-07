import numpy as np
from blocks.prbs import PRBSBlock


def test_prbs_deterministic_sequence():
    block = PRBSBlock()
    params = {
        "high": 1.0,
        "low": 0.0,
        "bit_time": 0.2,
        "seed": 42,
        "_init_start_": True,
    }

    times = [0.0, 0.1, 0.21, 0.39, 0.41]
    outputs = [block.execute(t, {}, params)[0][0] for t in times]

    # With seed 42 and bit_time 0.2 we expect: low, low, high, high, high
    assert outputs == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_prbs_requires_positive_bit_time():
    block = PRBSBlock()
    params = {"bit_time": 0.0, "_init_start_": True, "high": 1.0, "low": 0.0, "seed": 0}
    result = block.execute(0.0, {}, params)
    assert result.get("E") is True
