"""The block contract (``blocks/base_block.py``) and its validator.

``validate_block_class`` is the mechanical statement of what ``docs/BLOCK_API.md``
promises third-party authors.  Two things must stay true:

* **every built-in block satisfies it** -- otherwise the document lies, and the
  warning the loader logs for user blocks would be noise;
* **it actually catches** the mistakes it claims to (a missing ``default``, a
  port without a name, an ``execute()`` without ``**kwargs``, ...), or it is
  worse than no check at all.

The shipped templates under ``docs/examples/`` are validated here too, so a
copy-paste starting point can never be broken.
"""

import importlib.util
import os

import numpy as np
import pytest

from blocks.base_block import (
    BLOCK_API_VERSION,
    BaseBlock,
    BlockContractError,
    block_contract_errors,
    validate_block_class,
)

DOCS_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs",
    "examples",
)


class GoodBlock(BaseBlock):
    """A minimal, valid block used as the baseline for the negative cases."""

    @property
    def block_name(self):
        return "TestGood"

    @property
    def params(self):
        return {"gain": {"type": "float", "default": 1.0, "doc": "Scale factor"}}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        return {0: np.atleast_1d(inputs.get(0, 0.0)) * params["gain"]}


def _import_example(file_name):
    path = os.path.join(DOCS_EXAMPLES, file_name)
    spec = importlib.util.spec_from_file_location("docs_example_" + file_name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blocks_in(module):
    return [
        obj
        for name in dir(module)
        for obj in (getattr(module, name),)
        if isinstance(obj, type)
        and issubclass(obj, BaseBlock)
        and obj is not BaseBlock
        and obj.__module__ == module.__name__
    ]


@pytest.mark.unit
class TestApiVersion:
    def test_version_is_an_int(self):
        assert isinstance(BLOCK_API_VERSION, int)
        assert BLOCK_API_VERSION >= 1


@pytest.mark.unit
class TestEveryBuiltInBlock:
    def test_all_registered_blocks_satisfy_the_contract(self):
        """Every block the app registers must pass ``validate_block_class``."""
        from lib.block_loader import load_builtin_blocks

        failures = {}
        for cls in load_builtin_blocks():
            problems = block_contract_errors(cls)
            if problems:
                failures["{}.{}".format(cls.__module__, cls.__name__)] = problems

        assert not failures, "blocks break the contract in docs/BLOCK_API.md:\n" + "\n".join(
            "  {}:\n    - {}".format(name, "\n    - ".join(problems))
            for name, problems in sorted(failures.items())
        )

    def test_block_names_are_unique(self):
        """A duplicate name would make saved diagrams ambiguous."""
        from lib.block_loader import load_builtin_blocks

        seen = {}
        for cls in load_builtin_blocks():
            name = cls().block_name
            seen.setdefault(name, []).append(cls.__name__)
        duplicates = {n: c for n, c in seen.items() if len(c) > 1}
        assert not duplicates, "duplicate block_name(s): {}".format(duplicates)


@pytest.mark.unit
class TestValidatorAcceptsValidBlocks:
    def test_baseline_block_passes(self):
        assert block_contract_errors(GoodBlock) == []
        validate_block_class(GoodBlock)  # must not raise

    def test_bare_default_values_are_accepted(self):
        cls = type("BareParams", (GoodBlock,), {"params": property(lambda self: {"gain": 1.0})})
        assert block_contract_errors(cls) == []

    def test_optional_ports_in_range_are_accepted(self):
        cls = type("Opt", (GoodBlock,), {"optional_inputs": {0}})
        assert block_contract_errors(cls) == []


@pytest.mark.unit
class TestValidatorRejectsBrokenBlocks:
    def _errors(self, name, **members):
        return block_contract_errors(type(name, (GoodBlock,), members))

    def test_not_a_class(self):
        problems = block_contract_errors("not a class")
        assert problems and "not a class" in problems[0]

    def test_not_a_baseblock_subclass(self):
        problems = block_contract_errors(type("Foreign", (object,), {}))
        assert problems and "BaseBlock" in problems[0]

    def test_abstract_block_names_the_missing_members(self):
        class Incomplete(BaseBlock):
            @property
            def block_name(self):
                return "Incomplete"

        problems = block_contract_errors(Incomplete)
        assert problems and "abstract" in problems[0]
        assert "execute" in problems[0]

    def test_constructor_with_required_arguments(self):
        class NeedsArgs(GoodBlock):
            def __init__(self, required):
                self.required = required

        problems = block_contract_errors(NeedsArgs)
        assert problems and "instantiated" in problems[0]

    def test_empty_block_name(self):
        problems = self._errors("NoName", block_name=property(lambda self: ""))
        assert any("block_name" in p for p in problems)

    def test_params_spec_without_default(self):
        problems = self._errors(
            "NoDefault", params=property(lambda self: {"gain": {"type": "float"}})
        )
        assert any("default" in p for p in problems)

    def test_params_not_a_dict(self):
        problems = self._errors("BadParams", params=property(lambda self: ["gain"]))
        assert any("params must return a dict" in p for p in problems)

    def test_params_choices_must_be_a_list(self):
        problems = self._errors(
            "BadChoices",
            params=property(lambda self: {"mode": {"default": "a", "choices": "abc"}}),
        )
        assert any("choices" in p for p in problems)

    def test_params_range_must_be_a_numeric_pair(self):
        problems = self._errors(
            "BadRange", params=property(lambda self: {"g": {"default": 1.0, "range": (1.0,)}})
        )
        assert any("range" in p for p in problems)

    def test_port_without_a_name(self):
        problems = self._errors("BadPort", inputs=property(lambda self: [{"type": "any"}]))
        assert any("inputs[0]" in p and "name" in p for p in problems)

    def test_ports_not_a_list(self):
        problems = self._errors("BadPorts", outputs=property(lambda self: {"name": "out"}))
        assert any("outputs must return a list" in p for p in problems)

    def test_optional_port_index_out_of_range(self):
        problems = self._errors("BadOptional", optional_inputs={5})
        assert any("optional_inputs" in p and "5" in p for p in problems)

    def test_execute_without_kwargs(self):
        def execute(self, time, inputs, params):
            return {0: 0.0}

        problems = self._errors("NoKwargs", execute=execute)
        assert any("**kwargs" in p for p in problems)

    def test_execute_with_wrong_argument_names(self):
        def execute(self, t, u, p, **kwargs):
            return {0: 0.0}

        problems = self._errors("WrongArgs", execute=execute)
        assert any("'time'" in p for p in problems)

    def test_unknown_shape_token(self):
        problems = self._errors("BadShape", shape=property(lambda self: "hexagon"))
        assert any("shape" in p for p in problems)

    def test_bad_io_editable(self):
        problems = self._errors("BadIo", io_editable=property(lambda self: "sometimes"))
        assert any("io_editable" in p for p in problems)

    def test_property_that_raises_is_reported_not_propagated(self):
        def boom(self):
            raise RuntimeError("kaboom")

        problems = self._errors("Boom", outputs=property(boom))
        assert any("kaboom" in p for p in problems)

    def test_validate_raises_with_every_problem_listed(self):
        cls = type(
            "TwoProblems",
            (GoodBlock,),
            {
                "block_name": property(lambda self: ""),
                "inputs": property(lambda self: [{"type": "any"}]),
            },
        )
        with pytest.raises(BlockContractError) as excinfo:
            validate_block_class(cls)
        message = str(excinfo.value)
        assert "block_name" in message and "inputs[0]" in message
        assert "BLOCK_API_VERSION" in message


@pytest.mark.unit
class TestShippedTemplates:
    """``docs/examples/*.py`` must import and pass the validator."""

    @pytest.mark.parametrize("file_name", ["custom_block_template.py", "custom_kernel_template.py"])
    def test_template_blocks_are_valid(self, file_name):
        module = _import_example(file_name)
        classes = _blocks_in(module)
        assert classes, "{} defines no block".format(file_name)
        for cls in classes:
            validate_block_class(cls)

    def test_templates_declare_the_api_version(self):
        for file_name in ("custom_block_template.py", "custom_kernel_template.py"):
            module = _import_example(file_name)
            assert getattr(module, "BLOCK_API_VERSION", None) == BLOCK_API_VERSION

    def test_stateless_template_block_runs(self):
        module = _import_example("custom_block_template.py")
        block = module.ScaledOffsetBlock()
        out = block.execute(
            time=0.0, inputs={0: np.array([2.0])}, params={"gain": 3.0, "offset": 1.0}
        )
        assert np.isclose(out[0][0], 7.0)

    def test_stateful_template_block_keeps_state_in_params(self):
        module = _import_example("custom_block_template.py")
        block = module.MovingAverageBlock()
        params = {"window": 2, "_init_start_": True}
        block.execute(time=0.0, inputs={0: np.array([0.0])}, params=params)
        out = block.execute(time=0.1, inputs={0: np.array([2.0])}, params=params)
        assert np.isclose(out[0][0], 1.0)
        # State must live in params, not on the instance: a fresh params dict
        # (what reset_memblocks effectively hands the block) restarts it.
        assert "_buffer_" in params
        fresh = {"window": 2, "_init_start_": True}
        out = block.execute(time=0.0, inputs={0: np.array([4.0])}, params=fresh)
        assert np.isclose(out[0][0], 4.0)

    def test_kernel_template_registers_a_kernel_and_events(self):
        from lib.engine.compiler_kernels import get_event_builder, get_kernel_builder

        _import_example("custom_kernel_template.py")
        assert get_kernel_builder("Softclip") is not None
        assert get_event_builder("Softclip") is not None

    def test_kernel_template_name_matches_canonical_fn(self):
        from lib.engine.block_names import canonical_fn

        module = _import_example("custom_kernel_template.py")
        assert canonical_fn(module.SoftClipBlock().block_name) == "Softclip"
