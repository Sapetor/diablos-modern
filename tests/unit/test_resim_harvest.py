"""``harvest_scope_signals`` names every Scope channel by its label on both
execution paths.

The interpreter leaves ``params['vector']`` as a flat, ``vec_dim``-strided
buffer; the compiled replay preallocates an ``(n_samples, vec_dim)`` array. The
harvester used to key a single-channel Scope by its *label* only for the 2-D
layout and by its *block name* for the 1-D one, so the same diagram produced
different signal names in an ensemble depending on which solver ran it (and a
user's ``labels`` entry silently vanished on the interpreter). These tests pin
the layout-independent rule with stub blocks; the end-to-end check lives in
``tests/regression/test_harvest_scope_signals.py``.
"""

import numpy as np
import pytest

from lib.analysis.resim import harvest_scope_signals


class _Scope:
    block_fn = "Scope"

    def __init__(self, name, vector, vec_dim, vec_labels):
        self.name = name
        self.exec_params = {"vector": vector, "vec_dim": vec_dim}
        if vec_labels is not None:
            self.exec_params["vec_labels"] = vec_labels


class _DSim:
    def __init__(self, blocks, timeline=(0.0, 0.1, 0.2)):
        self.timeline = np.asarray(timeline)
        self.blocks_list = list(blocks)
        self.engine = type("E", (), {"active_blocks_list": list(blocks)})()


def _harvest(*blocks):
    out = harvest_scope_signals(_DSim(blocks))
    assert out is not None
    return out["signals"]


@pytest.mark.unit
class TestHarvestScopeSignals:
    def test_single_channel_keyed_by_label_on_both_layouts(self):
        flat = _Scope("scope1", np.array([1.0, 2.0, 3.0]), 1, ["y"])
        stacked = _Scope("scope1", np.array([[1.0], [2.0], [3.0]]), 1, ["y"])
        assert list(_harvest(flat)) == ["y"]
        assert list(_harvest(stacked)) == ["y"]
        np.testing.assert_array_equal(_harvest(flat)["y"], _harvest(stacked)["y"])

    def test_default_label_used_not_block_name(self):
        flat = _Scope("scope2", np.array([1.0, 2.0]), 1, ["scope2-0"])
        assert list(_harvest(flat)) == ["scope2-0"]

    def test_multi_channel_split_identically(self):
        flat = _Scope("s", np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0]), 2, ["a", "b"])
        stacked = _Scope("s", np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), 2, ["a", "b"])
        hf, hs = _harvest(flat), _harvest(stacked)
        assert list(hf) == list(hs) == ["a", "b"]
        np.testing.assert_array_equal(hf["a"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(hf["b"], hs["b"])

    def test_missing_labels_fall_back_to_block_name(self):
        single = _Scope("lone", np.array([1.0, 2.0]), 1, None)
        multi = _Scope("pair", np.array([[1.0, 2.0], [3.0, 4.0]]), 2, ["a"])
        sig = _harvest(single, multi)
        assert list(sig) == ["lone", "a", "pair[1]"]

    def test_string_label_accepted(self):
        blk = _Scope("s", np.array([1.0, 2.0]), 1, "y")
        assert list(_harvest(blk)) == ["y"]

    def test_duplicate_labels_disambiguated(self):
        a = _Scope("s1", np.array([1.0, 2.0]), 1, ["y"])
        b = _Scope("s2", np.array([[3.0], [4.0]]), 1, ["y"])
        sig = _harvest(a, b)
        assert list(sig) == ["y", "y#1"]
        np.testing.assert_array_equal(sig["y#1"], [3.0, 4.0])

    def test_skips_scopes_without_vector_and_non_scopes(self):
        empty = _Scope("s", None, 1, ["y"])
        empty.exec_params.pop("vector")
        gain = _Scope("g", np.array([1.0]), 1, ["g"])
        gain.block_fn = "Gain"
        assert _harvest(empty, gain) == {}

    def test_returns_none_without_timeline(self):
        d = _DSim([_Scope("s", np.array([1.0]), 1, ["y"])], timeline=())
        assert harvest_scope_signals(d) is None
