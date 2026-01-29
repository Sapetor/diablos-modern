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
