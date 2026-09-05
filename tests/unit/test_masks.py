"""Unit tests for the subsystem mask model (``lib/masks.py``).

Covers the definition (normalize/validate), the block accessors that store a
mask in ``params``, and -- the part the engine depends on -- resolving mask
parameter values into a scope and rewriting inner block parameters against it.
"""

import pytest

from lib.masks import (
    MASK_KEY,
    MaskError,
    child_scope_for,
    clear_mask,
    default_mask,
    get_mask,
    is_masked,
    mask_values,
    normalize_mask,
    resolve_mask_scope,
    resolve_params_in_scope,
    set_mask,
)


class _FakeBlock:
    """Minimal stand-in for a Subsystem: just the attributes masks touch."""

    def __init__(self, name="Subsystem1", params=None):
        self.name = name
        self.username = name
        self.block_fn = "Subsystem"
        self.block_type = "Subsystem"
        self.params = dict(params or {})
        self.init_params_list = []
        self.sub_blocks = []
        self.sub_lines = []


def _mask(*params, **kwargs):
    mask = default_mask(kwargs.pop("name", "Vehicle"))
    mask["parameters"] = list(params)
    mask.update(kwargs)
    return mask


@pytest.mark.unit
class TestNormalizeMask:
    def test_defaults_are_filled_in(self):
        mask = normalize_mask({"name": "Vehicle", "parameters": [{"name": "m"}]})
        assert mask["category"] == "User Library"
        assert mask["shape"] == "rect"
        spec = mask["parameters"][0]
        assert spec["type"] == "float"
        assert spec["default"] == 0.0
        assert spec["options"] == []

    def test_name_is_required(self):
        with pytest.raises(MaskError):
            normalize_mask({"parameters": []})

    def test_parameter_name_must_be_an_identifier(self):
        with pytest.raises(MaskError) as exc:
            normalize_mask({"name": "V", "parameters": [{"name": "2m"}]})
        assert "identifier" in str(exc.value)

    def test_duplicate_parameter_rejected(self):
        with pytest.raises(MaskError) as exc:
            normalize_mask({"name": "V", "parameters": [{"name": "m"}, {"name": "m"}]})
        assert "Duplicate" in str(exc.value)

    def test_reserved_parameter_rejected(self):
        with pytest.raises(MaskError):
            normalize_mask({"name": "V", "parameters": [{"name": "_name_"}]})

    def test_choice_needs_options(self):
        with pytest.raises(MaskError):
            normalize_mask({"name": "V", "parameters": [{"name": "mode", "type": "choice"}]})

    def test_unknown_type_rejected(self):
        with pytest.raises(MaskError):
            normalize_mask({"name": "V", "parameters": [{"name": "m", "type": "complex"}]})

    def test_unknown_shape_rejected(self):
        with pytest.raises(MaskError):
            normalize_mask({"name": "V", "shape": "hexagon", "parameters": []})


@pytest.mark.unit
class TestMaskOnBlock:
    def test_set_mask_seeds_values_and_keeps_them_saveable(self):
        block = _FakeBlock()
        set_mask(block, _mask({"name": "m", "type": "float", "default": 1500.0}))

        assert is_masked(block)
        assert block.params["m"] == 1500.0
        # saving_params() only writes keys listed here.
        assert MASK_KEY in block.init_params_list
        assert "m" in block.init_params_list

    def test_set_mask_adopts_the_display_name(self):
        block = _FakeBlock(name="Subsystem1")
        set_mask(block, _mask(name="Vehicle"))
        assert block.username == "Vehicle"

    def test_set_mask_keeps_a_user_renamed_instance(self):
        block = _FakeBlock(name="Subsystem1")
        block.username = "Front axle"
        set_mask(block, _mask(name="Vehicle"))
        assert block.username == "Front axle"

    def test_editing_a_mask_retires_removed_parameters(self):
        block = _FakeBlock()
        set_mask(block, _mask({"name": "m"}, {"name": "b"}))
        block.params["m"] = 1200.0
        set_mask(block, _mask({"name": "m"}))
        assert "b" not in block.params
        assert "b" not in block.init_params_list
        assert block.params["m"] == 1200.0  # surviving value is preserved

    def test_clear_mask_removes_definition_and_values(self):
        block = _FakeBlock()
        set_mask(block, _mask({"name": "m"}))
        clear_mask(block)
        assert get_mask(block) is None
        assert "m" not in block.params
        assert block.init_params_list == []

    def test_malformed_stored_mask_is_ignored_not_raised(self):
        block = _FakeBlock(params={MASK_KEY: {"parameters": "not-a-list"}})
        assert get_mask(block) is None
        assert is_masked(block) is False

    def test_mask_values_fall_back_to_defaults(self):
        block = _FakeBlock()
        set_mask(block, _mask({"name": "m", "default": 3.0}, {"name": "b", "default": 4.0}))
        del block.params["b"]
        assert mask_values(block) == {"m": 3.0, "b": 4.0}


@pytest.mark.unit
class TestResolveMaskScope:
    def test_scalar_values(self):
        mask = _mask({"name": "m", "type": "float", "default": 1500.0})
        assert resolve_mask_scope(mask, {"m": 1200.0}, workspace={}) == {"m": 1200.0}

    def test_expression_over_workspace_variables(self):
        mask = _mask({"name": "m", "type": "float", "default": 1.0})
        scope = resolve_mask_scope(mask, {"m": "2 * base"}, workspace={"base": 700.0})
        assert scope["m"] == 1400.0

    def test_list_parameter(self):
        mask = _mask({"name": "den", "type": "list", "default": [1.0, 1.0]})
        scope = resolve_mask_scope(mask, {"den": "[m, b]"}, workspace={"m": 2.0, "b": 3.0})
        assert scope["den"] == [2.0, 3.0]

    def test_choice_and_string_values_are_not_evaluated(self):
        mask = _mask(
            {"name": "mode", "type": "choice", "options": ["fast", "slow"], "default": "fast"},
            {"name": "label", "type": "string", "default": "speed"},
        )
        scope = resolve_mask_scope(mask, {"mode": "slow", "label": "m"}, workspace={"m": 9.0})
        assert scope == {"mode": "slow", "label": "m"}

    def test_a_later_parameter_may_reference_an_earlier_one(self):
        mask = _mask(
            {"name": "m", "type": "float", "default": 1500.0},
            {"name": "b", "type": "float", "default": 0.0},
        )
        scope = resolve_mask_scope(mask, {"b": "m / 30"}, workspace={})
        assert scope["b"] == 50.0

    def test_missing_variable_names_the_parameter_and_the_options(self):
        mask = _mask({"name": "b", "type": "float", "default": 0.0}, name="Vehicle")
        with pytest.raises(MaskError) as exc:
            resolve_mask_scope(mask, {"b": "damping"}, workspace={"m": 1.0}, block_label="Vehicle")
        message = str(exc.value)
        assert "'b'" in message
        assert "Vehicle" in message
        assert "damping" in message
        assert "Available names" in message

    def test_outer_scope_wins_over_workspace(self):
        mask = _mask({"name": "k", "type": "float", "default": 0.0})
        scope = resolve_mask_scope(
            mask, {"k": "gain"}, outer_scope={"gain": 5.0}, workspace={"gain": 1.0}
        )
        assert scope["k"] == 5.0


@pytest.mark.unit
class TestResolveParamsInScope:
    def test_no_scope_returns_the_same_dict_object(self):
        params = {"gain": "K"}
        assert resolve_params_in_scope(params, None) is params
        assert resolve_params_in_scope(params, {}) is params

    def test_expression_strings_are_resolved(self):
        out = resolve_params_in_scope({"gain": "K", "den": "[m, b]"}, {"K": 2.0, "m": 3, "b": 4})
        assert out["gain"] == 2.0
        assert out["den"] == [3, 4]

    def test_unresolvable_strings_are_kept_verbatim(self):
        out = resolve_params_in_scope({"labels": "speed", "method": "RK45"}, {"K": 1.0})
        assert out == {"labels": "speed", "method": "RK45"}

    def test_internal_keys_are_never_touched(self):
        out = resolve_params_in_scope({"_name_": "K", "gain": "K"}, {"K": 2.0})
        assert out["_name_"] == "K"
        assert out["gain"] == 2.0

    def test_the_input_dict_is_not_mutated(self):
        params = {"gain": "K"}
        resolve_params_in_scope(params, {"K": 7.0})
        assert params["gain"] == "K"


@pytest.mark.unit
class TestNestedMasks:
    def test_child_scope_of_an_unmasked_subsystem_passes_through(self):
        block = _FakeBlock()
        outer = {"K": 3.0}
        assert child_scope_for(block, outer) is outer

    def test_nested_masks_resolve_outer_to_inner(self):
        """An inner mask's values are evaluated in the outer mask's scope."""
        outer_block = _FakeBlock(name="Outer")
        set_mask(outer_block, _mask({"name": "K", "type": "float", "default": 4.0}, name="Outer"))

        inner_block = _FakeBlock(name="Inner")
        set_mask(inner_block, _mask({"name": "g", "type": "float", "default": 1.0}, name="Inner"))
        inner_block.params["g"] = "2 * K"

        outer_scope = child_scope_for(outer_block, None, workspace={})
        inner_scope = child_scope_for(inner_block, outer_scope, workspace={})

        assert outer_scope == {"K": 4.0}
        assert inner_scope == {"g": 8.0}
        # The inner scope is isolated: the outer mask's names do not leak in.
        assert "K" not in inner_scope

    def test_inner_mask_missing_outer_variable_reports_the_inner_parameter(self):
        inner_block = _FakeBlock(name="Inner")
        set_mask(inner_block, _mask({"name": "g", "type": "float", "default": 1.0}, name="Inner"))
        inner_block.params["g"] = "2 * K"
        with pytest.raises(MaskError) as exc:
            child_scope_for(inner_block, None, workspace={})
        assert "'g'" in str(exc.value)
