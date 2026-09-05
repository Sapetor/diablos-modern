"""Unit tests for the Optimizer meta-block (blocks/optimization/optimizer.py).

``OptimizerBlock`` is a configuration carrier, not a signal processor: it has no
ports, never runs during simulation, and exists so ``get_optimizer_config()`` can
hand a settings dict to the optimization engine. Two things are worth pinning:

* the *coercion* in ``get_optimizer_config`` -- everything arrives from the
  property editor as a string, and passing ``max_iter="50"`` straight to
  ``scipy.optimize.minimize`` is an error the block is supposed to prevent;
* the "meta-block" flags, since a block with zero ports that still required
  connections would make every diagram containing it fail validation.
"""

import pytest

from blocks.optimization.optimizer import OptimizerBlock


def _defaults(**overrides):
    params = {name: spec["default"] for name, spec in OptimizerBlock().params.items()}
    params.update(overrides)
    return params


@pytest.mark.unit
class TestOptimizerContract:
    def test_identity(self):
        block = OptimizerBlock()
        assert block.block_name == "Optimizer"
        assert block.category == "Optimization"

    def test_is_a_portless_meta_block(self):
        block = OptimizerBlock()
        assert block.inputs == []
        assert block.outputs == []
        # Both False is what marks it a meta-block for
        # DiagramValidator._check_isolated_blocks -- an unconnected Optimizer
        # must not be reported as an isolated block.
        assert block.requires_inputs is False
        assert block.requires_outputs is False

    def test_declares_the_documented_settings(self):
        params = OptimizerBlock().params
        assert set(params) == {
            "method",
            "max_iter",
            "tol",
            "use_constraints",
            "use_penalty",
            "penalty_factor",
            "verbose",
            "multistart",
            "popsize",
            "mutation",
            "recombination",
        }
        assert params["method"]["default"] == "L-BFGS-B"
        for name, spec in params.items():
            assert "doc" in spec, name

    def test_every_documented_method_is_named_in_the_doc(self):
        doc = OptimizerBlock().doc
        for method in ("L-BFGS-B", "SLSQP", "Nelder-Mead", "differential_evolution", "Powell"):
            assert method in doc

    def test_execute_is_inert(self):
        block = OptimizerBlock()
        params = _defaults()
        snapshot = dict(params)

        result = block.execute(time=0.0, inputs={}, params=params, dtime=0.01)

        assert result == {"E": False}, "the meta-block must not report an error"
        assert params == snapshot

    def test_draw_icon_returns_a_painter_path(self, qapp):
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QPainterPath

        path = OptimizerBlock().draw_icon(QRect(0, 0, 100, 60))
        assert isinstance(path, QPainterPath)
        assert not path.isEmpty()


@pytest.mark.unit
class TestOptimizerConfig:
    def test_defaults_round_trip(self):
        config = OptimizerBlock().get_optimizer_config(_defaults())

        assert config["method"] == "L-BFGS-B"
        assert config["max_iter"] == 100
        assert config["tol"] == pytest.approx(1e-6)
        assert config["use_constraints"] is True
        assert config["use_penalty"] is False
        assert config["multistart"] == 1

    def test_config_covers_every_declared_parameter(self):
        """A setting added to `params` but forgotten in `get_optimizer_config`
        is silently ignored by the engine."""
        config = OptimizerBlock().get_optimizer_config(_defaults())
        declared = set(OptimizerBlock().params)

        assert declared == set(config)

    def test_string_params_from_the_property_editor_are_coerced(self):
        """The property editor hands back strings; scipy needs numbers."""
        params = _defaults(
            max_iter="250",
            tol="1e-9",
            penalty_factor="500",
            multistart="4",
            popsize="20",
            mutation="0.5",
            recombination="0.9",
        )

        config = OptimizerBlock().get_optimizer_config(params)

        assert config["max_iter"] == 250 and isinstance(config["max_iter"], int)
        assert config["tol"] == pytest.approx(1e-9) and isinstance(config["tol"], float)
        assert config["penalty_factor"] == pytest.approx(500.0)
        assert config["multistart"] == 4 and isinstance(config["multistart"], int)
        assert config["popsize"] == 20 and isinstance(config["popsize"], int)
        assert config["mutation"] == pytest.approx(0.5)
        assert config["recombination"] == pytest.approx(0.9)

    def test_a_float_iteration_count_is_truncated_to_an_int(self):
        config = OptimizerBlock().get_optimizer_config(_defaults(max_iter=99.7))
        assert config["max_iter"] == 99

    def test_missing_params_fall_back_to_the_documented_defaults(self):
        """``get_optimizer_config`` is called with resolved runtime params,
        which may be a partial dict for an older saved diagram."""
        config = OptimizerBlock().get_optimizer_config({})

        assert config["method"] == "L-BFGS-B"
        assert config["max_iter"] == 100
        assert config["popsize"] == 15

    def test_selecting_a_different_method_is_passed_through_verbatim(self):
        config = OptimizerBlock().get_optimizer_config(_defaults(method="SLSQP"))
        assert config["method"] == "SLSQP"
