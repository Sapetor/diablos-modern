"""Unit tests for the extracted compiled-system cache fingerprinting helpers.

These functions were private methods on SimulationEngine; extracting them into
``lib/engine/compile_cache.py`` lets us pin the (subtle) normalization and
key-stability behavior directly. The key invariant the cache relies on: equal
inputs produce equal keys, structurally different inputs produce different keys,
and volatile runtime/history params never enter the key.
"""

import types

import numpy as np
import pytest

from lib.engine.compile_cache import (
    _COMPILE_CACHE_IGNORED_PARAM_KEYS,
    compile_param_items,
    compiled_system_fingerprint,
    normalize_cache_value,
    source_params_fingerprint,
)


@pytest.mark.unit
class TestNormalizeCacheValue:
    def test_scalar_passthrough(self):
        assert normalize_cache_value(3) == 3
        assert normalize_cache_value("x") == "x"

    def test_equal_arrays_hash_equal_distinct_differ(self):
        a = normalize_cache_value(np.array([1.0, 2.0, 3.0]))
        b = normalize_cache_value(np.array([1.0, 2.0, 3.0]))
        c = normalize_cache_value(np.array([1.0, 2.0, 4.0]))
        assert a == b
        assert a != c
        # Result must be hashable (it becomes part of a dict/tuple cache key).
        assert hash(a) == hash(b)

    def test_shape_and_dtype_matter(self):
        row = normalize_cache_value(np.array([[1.0, 2.0]]))
        col = normalize_cache_value(np.array([[1.0], [2.0]]))
        assert row != col

    def test_nan_and_inf_are_stable(self):
        assert normalize_cache_value(float("nan")) == ("float", "nan")
        assert normalize_cache_value(float("inf")) == ("float", "inf")
        assert normalize_cache_value(float("-inf")) == ("float", "-inf")

    def test_numpy_scalar_matches_python_scalar(self):
        assert normalize_cache_value(np.float64(2.5)) == normalize_cache_value(2.5)

    def test_dict_is_order_independent(self):
        assert normalize_cache_value({"a": 1, "b": 2}) == normalize_cache_value({"b": 2, "a": 1})

    def test_unhashable_falls_back_to_repr(self):
        class Weird:
            __hash__ = None  # unhashable

            def __repr__(self):
                return "WEIRD"

        assert normalize_cache_value(Weird()) == ("Weird", "WEIRD")


@pytest.mark.unit
class TestCompileParamItems:
    def test_ignored_and_underscore_keys_dropped(self):
        params = {
            "gain": 2.0,
            "_init_start_": True,  # in the ignored set
            "_x_": [1, 2, 3],  # in the ignored set
            "_secret_": 1,  # underscore, not allowlisted -> dropped
            "_inputs_": [0],  # underscore BUT allowlisted -> kept
        }
        items = dict(compile_param_items(params))
        assert "gain" in items
        assert "_inputs_" in items
        assert "_init_start_" not in items
        assert "_x_" not in items
        assert "_secret_" not in items

    def test_ignored_set_covers_fingerprint_key(self):
        # The fingerprint we stash in exec_params must not feed back into itself.
        assert "_source_params_fingerprint" in _COMPILE_CACHE_IGNORED_PARAM_KEYS

    def test_none_params(self):
        assert compile_param_items(None) == ()

    def test_source_params_fingerprint_is_compile_param_items(self):
        p = {"a": 1, "b": [1, 2]}
        assert source_params_fingerprint(p) == compile_param_items(p)


def _blk(name, fn="Gain", **attrs):
    base = dict(
        name=name,
        block_fn=fn,
        in_ports=1,
        out_ports=1,
        b_type=0,
        hierarchy=0,
        effective_sample_time=-1.0,
        params={"gain": 1.0},
        exec_params={},
    )
    base.update(attrs)
    return types.SimpleNamespace(**base)


def _line(src, dst):
    return types.SimpleNamespace(srcblock=src, srcport=0, dstblock=dst, dstport=0, hidden=False)


@pytest.mark.unit
class TestCompiledSystemFingerprint:
    def test_identical_systems_match(self):
        blocks = [_blk("g1"), _blk("g2")]
        lines = [_line("g1", "g2")]
        k1 = compiled_system_fingerprint(blocks, blocks, lines, 0.01)
        k2 = compiled_system_fingerprint(
            [_blk("g1"), _blk("g2")], [_blk("g1"), _blk("g2")], [_line("g1", "g2")], 0.01
        )
        assert k1 == k2
        assert hash(k1) == hash(k2)  # must be usable as a dict key

    def test_param_change_changes_key(self):
        base = [_blk("g1")]
        changed = [_blk("g1", params={"gain": 2.0})]
        assert compiled_system_fingerprint(base, base, [], 0.01) != (
            compiled_system_fingerprint(changed, changed, [], 0.01)
        )

    def test_dt_and_order_change_key(self):
        blocks = [_blk("g1"), _blk("g2")]
        assert compiled_system_fingerprint(blocks, blocks, [], 0.01) != (
            compiled_system_fingerprint(blocks, blocks, [], 0.02)
        )
        assert compiled_system_fingerprint(blocks, blocks, [], 0.01) != (
            compiled_system_fingerprint(blocks, list(reversed(blocks)), [], 0.01)
        )

    def test_volatile_param_does_not_change_key(self):
        base = [_blk("g1")]
        with_history = [_blk("g1", params={"gain": 1.0, "_x_": [9, 9, 9]})]
        assert compiled_system_fingerprint(base, base, [], 0.01) == (
            compiled_system_fingerprint(with_history, with_history, [], 0.01)
        )
