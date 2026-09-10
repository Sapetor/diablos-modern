"""A block whose ``execute()`` raises must surface as a block error, not a TypeError.

``SimulationEngine.execute_block`` used to return ``False`` when the block
raised, and ``DSim._block_failed`` then did ``"E" in out_value`` on a bool
("argument of type 'bool' is not iterable"), hiding the block's message.
Reproduced by ``examples/van_der_pol_stiff.diablos`` on the interpreted path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lib.engine.simulation_engine import SimulationEngine
from lib.lib import DSim


class _RaisingBlock:
    def execute(self, **kwargs):
        raise ValueError("array must not contain infs or NaNs")


class _GoodBlock:
    def execute(self, **kwargs):
        return {0: np.array([1.0])}


def _engine():
    engine = SimulationEngine(SimpleNamespace(blocks_list=[], line_list=[]))
    engine.time_step = 0.0
    return engine


def _dblock(name, instance):
    return SimpleNamespace(
        name=name,
        b_type=0,
        block_fn="Stub",
        external=False,
        block_instance=instance,
        input_queue={},
        exec_params={},
    )


def _dsim():
    d = DSim.__new__(DSim)
    d.engine = MagicMock()
    d.execution_initialized = True
    d.pbar = MagicMock()
    d.dynamic_plot = False
    return d


@pytest.mark.unit
class TestExecuteBlockErrors:
    def test_raising_block_returns_error_dict(self):
        engine = _engine()
        out = engine.execute_block(_dblock("integrator0", _RaisingBlock()))
        assert isinstance(out, dict)
        assert out["E"] is True
        assert out["error"] == "Block 'integrator0' failed: array must not contain infs or NaNs"
        assert engine.error_msg == out["error"]

    def test_good_block_output_is_passed_through(self):
        engine = _engine()
        out = engine.execute_block(_dblock("src", _GoodBlock()))
        assert set(out) == {0}
        assert np.array_equal(out[0], [1.0])
        assert engine.error_msg == ""

    def test_missing_instance_still_returns_false(self):
        # The "cannot run at all" paths keep their bool contract for the
        # engine's own init loops, which test ``out_value is False``.
        assert _engine().execute_block(_dblock("ghost", None)) is False


@pytest.mark.unit
class TestBlockFailedOnEngineResults:
    def test_error_dict_from_raising_block_stops_the_run(self):
        out = _engine().execute_block(_dblock("integrator0", _RaisingBlock()))
        d = _dsim()
        assert d._block_failed(out) is True
        assert d.error_msg == "Block 'integrator0' failed: array must not contain infs or NaNs"
        assert d.execution_initialized is False

    def test_normal_output_is_not_a_failure(self):
        d = _dsim()
        assert d._block_failed({0: np.array([1.0])}) is False
        assert d.execution_initialized is True

    def test_bool_result_is_a_failure_not_a_type_error(self):
        d = _dsim()
        assert d._block_failed(False) is True
        assert d.error_msg == "Block returned False"
