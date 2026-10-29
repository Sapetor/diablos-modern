"""PDE and optimization blocks: loading, execution and icons.

These used to print "[FAIL]" and ``return False``, which pytest counts as a
pass; they now assert.
"""

import numpy as np
import pytest

from lib.block_loader import load_blocks

FAMILIES = ("PDE", "Optimization")


def _instances():
    return [b() for b in load_blocks()]


def test_block_loading():
    """Every block class instantiates, and both families are present."""
    errors = []
    categories = set()
    for b in load_blocks():
        try:
            categories.add(getattr(b(), "category", "Unknown"))
        except Exception as e:  # report every failure, not just the first
            errors.append(f"{b.__name__}: {e}")
    assert not errors
    assert set(FAMILIES) <= categories


def test_heat_equation():
    """Heat flows in from the hot left boundary; both boundaries are held."""
    from blocks.pde.heat_equation_1d import HeatEquation1DBlock

    block = HeatEquation1DBlock()
    params = {
        "alpha": 0.1,
        "L": 1.0,
        "N": 10,
        "bc_type_left": "Dirichlet",
        "bc_type_right": "Dirichlet",
        "init_conds": [0.0],
        "_init_start_": True,
        "dtime": 0.01,
    }
    inputs = {0: 0.0, 1: 100.0, 2: 0.0}  # q_src=0, bc_left=100, bc_right=0
    for i in range(11):
        result = block.execute(0.01 * i, inputs, params)

    T_field, T_avg = result[0], result[1]
    assert T_field.shape == (10,)
    assert T_field[0] == 100.0
    assert T_field[-1] == 0.0
    assert T_avg > 0


def test_wave_equation():
    """An undamped Gaussian pulse keeps a finite, nonzero field and energy."""
    from blocks.pde.wave_equation_1d import WaveEquation1DBlock

    block = WaveEquation1DBlock()
    params = {
        "c": 1.0,
        "damping": 0.0,
        "L": 1.0,
        "N": 20,
        "bc_type_left": "Dirichlet",
        "bc_type_right": "Dirichlet",
        "init_displacement": "gaussian",
        "init_velocity": [0.0],
        "_init_start_": True,
        "dtime": 0.01,
    }
    inputs = {0: 0.0, 1: 0.0, 2: 0.0}  # force=0, bc_left=0, bc_right=0
    for i in range(11):
        result = block.execute(0.01 * i, inputs, params)

    u_field, energy = result[0], result[2]
    assert u_field.shape == (20,)
    assert np.all(np.isfinite(u_field))
    assert np.max(np.abs(u_field)) > 0
    assert np.isfinite(energy) and energy > 0


def test_field_probe():
    """Probing f(x) = 10x at the midpoint interpolates 5."""
    from blocks.pde.field_processing import FieldProbeBlock

    params = {"position": 0.5, "position_mode": "normalized", "L": 1.0}
    result = FieldProbeBlock().execute(0.0, {0: np.linspace(0, 10, 11)}, params)
    assert result[0] == pytest.approx(5.0, abs=0.01)


def test_field_integral():
    """The integral of f(x) = 2 over [0, 1] is 2."""
    from blocks.pde.field_processing import FieldIntegralBlock

    params = {"L": 1.0, "normalize": False}
    result = FieldIntegralBlock().execute(0.0, {0: np.full(11, 2.0)}, params)
    assert result[0] == pytest.approx(2.0, abs=0.1)


def test_parameter_block():
    """A Parameter block outputs its value and reports it for optimization."""
    from blocks.optimization.parameter import ParameterBlock

    block = ParameterBlock()
    params = {
        "name": "Kp",
        "value": 2.5,
        "lower": 0.0,
        "upper": 10.0,
        "scale": "linear",
        "fixed": False,
    }
    assert block.execute(0.0, {}, params)[0] == 2.5
    assert block.get_optimization_info(params)["name"] == "Kp"


def test_cost_function():
    """ISE of a decaying error accumulates a positive cost."""
    from blocks.optimization.cost_function import CostFunctionBlock

    block = CostFunctionBlock()
    params = {"cost_type": "ISE", "target": 0.0, "weight": 1.0, "_init_start_": True}
    params["dtime"] = 0.1
    for i, err in enumerate([1.0, 0.8, 0.5, 0.3, 0.1]):
        block.execute(i * 0.1, {0: err}, params)
    assert params.get("_accumulated_cost_", 0) > 0


def test_draw_icons():
    """Every PDE/Optimization block draws an icon (a QPainterPath or None)."""
    from PyQt6.QtGui import QPainterPath

    problems = []
    for inst in _instances():
        if getattr(inst, "category", "Unknown") not in FAMILIES:
            continue
        if not hasattr(inst, "draw_icon"):
            problems.append(f"{inst.block_name}: no draw_icon")
            continue
        try:
            path = inst.draw_icon(None)
        except Exception as e:
            problems.append(f"{inst.block_name}: {e}")
            continue
        if path is not None and not isinstance(path, QPainterPath):
            problems.append(f"{inst.block_name}: returned {type(path)}")
    assert not problems
