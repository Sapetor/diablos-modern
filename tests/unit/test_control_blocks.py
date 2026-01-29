import numpy as np
from blocks.saturation import SaturationBlock
from blocks.rate_limiter import RateLimiterBlock
from blocks.pid import PIDBlock
from blocks.hysteresis import HysteresisBlock
from blocks.deadband import DeadbandBlock
from blocks.switch import SwitchBlock
from blocks.goto import GotoBlock
from blocks.from_block import FromBlock


def test_saturation_clips_values():
    block = SaturationBlock()
    params = {"min": -1.0, "max": 1.0}
    out = block.execute(0.0, {0: np.array([-2.0, 0.5, 2.0])}, params)[0]
    assert np.allclose(out, [-1.0, 0.5, 1.0])


def test_rate_limiter_limits_slew():
    block = RateLimiterBlock()
    params = {"rising_slew": 1.0, "falling_slew": 1.0, "_init_start_": True, "dtime": 0.1}
    # First call initializes state
    y0 = block.execute(0.0, {0: np.array([0.0])}, params)[0]
    assert y0 == 0.0
    # Second call attempts a step to 10, but should limit to 0.1
    y1 = block.execute(0.1, {0: np.array([10.0])}, params)[0]
    assert np.isclose(y1, 0.1)


def test_pid_proportional_only():
    block = PIDBlock()
    params = {
        "Kp": 2.0,
        "Ki": 0.0,
        "Kd": 0.0,
        "_init_start_": True,
        "dtime": 0.1,
    }
    out = block.execute(0.0, {0: np.array([1.0]), 1: np.array([0.25])}, params)[0]
    # error = 0.75; u = 2 * 0.75 = 1.5
    assert np.isclose(out, 1.5)


def test_hysteresis_switches_and_holds():
    block = HysteresisBlock()
    p = {"upper": 1.0, "lower": -1.0, "high": 2.0, "low": -2.0, "_init_start_": True}
    out1 = block.execute(0.0, {0: np.array([0.0])}, p)[0][0]
    assert out1 == -2.0
    out2 = block.execute(0.1, {0: np.array([1.2])}, p)[0][0]
    assert out2 == 2.0
    out3 = block.execute(0.2, {0: np.array([0.5])}, p)[0][0]
    # Should hold high state inside band
    assert out3 == 2.0


def test_deadband_zeroes_small_signal():
    block = DeadbandBlock()
    # DeadbandBlock uses start/end params, not deadband/center
    p = {"start": -0.5, "end": 0.5}
    out = block.execute(0.0, {0: np.array([-0.4, 0.0, 0.6])}, p)[0]
    # Input -0.4: inside deadzone [-0.5, 0.5] -> 0.0
    # Input 0.0: inside deadzone -> 0.0  
    # Input 0.6: above end (0.5) -> 0.6 - 0.5 = 0.1
    assert np.allclose(out, [0.0, 0.0, 0.1])


def test_switch_selects_true_branch():
    block = SwitchBlock()
    p = {"threshold": 0.0}
    out = block.execute(0.0, {0: np.array([0.5]), 1: np.array([10.0]), 2: np.array([-10.0])}, p)[0][0]
    assert out == 10.0  # ctrl>=thr -> first data input
    out2 = block.execute(0.0, {0: np.array([-0.1]), 1: np.array([10.0]), 2: np.array([-10.0])}, p)[0][0]
    assert out2 == -10.0


def test_switch_index_mode_multiway():
    block = SwitchBlock()
    p = {"mode": "index", "n_inputs": 3}
    out0 = block.execute(0.0, {0: np.array([0.1]), 1: np.array([1.0]), 2: np.array([2.0]), 3: np.array([3.0])}, p)[0][0]
    assert out0 == 1.0  # round(0.1)=0 -> in0
    out2 = block.execute(0.0, {0: np.array([2.2]), 1: np.array([1.0]), 2: np.array([2.0]), 3: np.array([3.0])}, p)[0][0]
    assert out2 == 3.0  # clamp to last


def test_goto_from_params_defaults():
    goto = GotoBlock()
    frm = FromBlock()
    assert goto.params["tag"]["default"] == "A"
    assert frm.params["tag"]["default"] == "A"


def test_link_goto_from_adds_line(simulation_model):
    from PyQt5.QtCore import QRect, QPoint
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine
    from blocks.step import StepBlock

    # Create a source block that feeds into Goto
    src = DBlock("Step", 0, QRect(0, 0, 70, 60), "green", block_class=StepBlock, in_ports=0, out_ports=1)
    
    # Create Goto block
    goto = DBlock("Goto", 0, QRect(100, 0, 70, 60), "orange", block_class=GotoBlock, in_ports=1, out_ports=0)
    goto.params['tag'] = "X"
    
    # Create From block
    frm = DBlock("From", 0, QRect(200, 0, 70, 60), "orange", block_class=FromBlock, in_ports=0, out_ports=1)
    frm.params['tag'] = "X"

    simulation_model.blocks_list = [src, goto, frm]
    
    # Create the required line from source to Goto (link_goto_from needs this)
    line_to_goto = DLine(
        sid=0,
        srcblock=src.name,
        srcport=0,
        dstblock=goto.name,
        dstport=0,
        points=[QPoint(70, 30), QPoint(100, 30)]
    )
    simulation_model.line_list = [line_to_goto]

    simulation_model.link_goto_from()

    # Expect TWO lines: original line + virtual hidden line to From
    assert len(simulation_model.line_list) == 2
    # Find the new virtual line (hidden=True)
    virtual_lines = [ln for ln in simulation_model.line_list if getattr(ln, 'hidden', False)]
    assert len(virtual_lines) == 1
    vline = virtual_lines[0]
    assert vline.srcblock == src.name  # Source of goto's input
    assert vline.dstblock == frm.name
