"""
Unit tests for the logical / relational operator blocks:

  - RelationalOperator  (in1 OP in2)
  - CompareToConstant   (in OP constant)
  - LogicalOperator     (AND / OR / NAND / NOR / XOR / NOT, nonzero = True)

All blocks emit 1.0 for true and 0.0 for false, element-wise.
"""

import numpy as np
import pytest

from blocks.relational_operator import RelationalOperatorBlock
from blocks.compare_to_constant import CompareToConstantBlock
from blocks.logical_operator import LogicalOperatorBlock


@pytest.mark.unit
class TestRelationalOperator:
    def test_metadata(self):
        b = RelationalOperatorBlock()
        assert b.block_name == "RelationalOperator"
        assert b.category == "Logic"
        assert len(b.inputs) == 2

    @pytest.mark.parametrize("op,a,b,expected", [
        (">", 3.0, 2.0, 1.0),
        (">", 1.0, 2.0, 0.0),
        (">=", 2.0, 2.0, 1.0),
        ("<", 1.0, 2.0, 1.0),
        ("<=", 3.0, 2.0, 0.0),
        ("==", 2.0, 2.0, 1.0),
        ("!=", 2.0, 2.0, 0.0),
    ])
    def test_comparisons(self, op, a, b, expected):
        block = RelationalOperatorBlock()
        result = block.execute(
            time=0.0, inputs={0: np.array([a]), 1: np.array([b])}, params={"operator": op}
        )
        assert np.isclose(result[0][0], expected)

    def test_elementwise_vectors(self):
        block = RelationalOperatorBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0, 5.0, 3.0]), 1: np.array([2.0, 2.0, 3.0])},
            params={"operator": ">"},
        )
        assert np.allclose(result[0], [0.0, 1.0, 0.0])

    def test_unknown_operator_errors(self):
        block = RelationalOperatorBlock()
        result = block.execute(time=0.0, inputs={0: np.array([1.0]), 1: np.array([2.0])},
                               params={"operator": "<<"})
        assert result.get("E") is True


@pytest.mark.unit
class TestCompareToConstant:
    def test_metadata(self):
        b = CompareToConstantBlock()
        assert b.block_name == "CompareToConstant"
        assert b.category == "Logic"
        assert len(b.inputs) == 1

    def test_greater_than_constant(self):
        block = CompareToConstantBlock()
        result = block.execute(time=0.0, inputs={0: np.array([5.0])},
                               params={"operator": ">", "constant": 3.0})
        assert np.isclose(result[0][0], 1.0)

    def test_less_equal_constant_false(self):
        block = CompareToConstantBlock()
        result = block.execute(time=0.0, inputs={0: np.array([5.0])},
                               params={"operator": "<=", "constant": 3.0})
        assert np.isclose(result[0][0], 0.0)

    def test_elementwise_threshold(self):
        block = CompareToConstantBlock()
        result = block.execute(time=0.0, inputs={0: np.array([-1.0, 0.0, 2.0])},
                               params={"operator": ">", "constant": 0.0})
        assert np.allclose(result[0], [0.0, 0.0, 1.0])


@pytest.mark.unit
class TestLogicalOperator:
    def test_metadata(self):
        b = LogicalOperatorBlock()
        assert b.block_name == "LogicalOperator"
        assert b.category == "Logic"
        assert b.io_editable == "input"

    @pytest.mark.parametrize("op,a,b,expected", [
        ("AND", 1.0, 1.0, 1.0),
        ("AND", 1.0, 0.0, 0.0),
        ("OR", 0.0, 1.0, 1.0),
        ("OR", 0.0, 0.0, 0.0),
        ("NAND", 1.0, 1.0, 0.0),
        ("NOR", 0.0, 0.0, 1.0),
        ("XOR", 1.0, 0.0, 1.0),
        ("XOR", 1.0, 1.0, 0.0),
    ])
    def test_two_input_ops(self, op, a, b, expected):
        block = LogicalOperatorBlock()
        result = block.execute(time=0.0, inputs={0: np.array([a]), 1: np.array([b])},
                               params={"operator": op, "_inputs_": 2})
        assert np.isclose(result[0][0], expected)

    def test_nonzero_is_true(self):
        # Any nonzero value counts as True.
        block = LogicalOperatorBlock()
        result = block.execute(time=0.0, inputs={0: np.array([2.5]), 1: np.array([-3.0])},
                               params={"operator": "AND", "_inputs_": 2})
        assert np.isclose(result[0][0], 1.0)

    def test_not_uses_first_input(self):
        block = LogicalOperatorBlock()
        result = block.execute(time=0.0, inputs={0: np.array([0.0])},
                               params={"operator": "NOT", "_inputs_": 1})
        assert np.isclose(result[0][0], 1.0)

    def test_three_input_and(self):
        block = LogicalOperatorBlock()
        result = block.execute(
            time=0.0,
            inputs={0: np.array([1.0]), 1: np.array([1.0]), 2: np.array([0.0])},
            params={"operator": "AND", "_inputs_": 3},
        )
        assert np.isclose(result[0][0], 0.0)

    def test_unknown_operator_errors(self):
        block = LogicalOperatorBlock()
        result = block.execute(time=0.0, inputs={0: np.array([1.0])},
                               params={"operator": "MAYBE", "_inputs_": 1})
        assert result.get("E") is True
