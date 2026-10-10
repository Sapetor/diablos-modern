"""Scope reference run: parameter snapshots and change lists."""

import types

import numpy as np
import pytest

from lib.plotting.run_compare import param_changes, snapshot_params


def _blk(name, username="", **params):
    return types.SimpleNamespace(name=name, username=username or name, params=params)


@pytest.mark.unit
class TestRunCompare:
    def test_snapshot_skips_internal_keys_and_copies(self):
        num = [1.0, 2.0]
        snap = snapshot_params([_blk("tf1", "plant", numerator=num, _name_="tf1")])
        assert snap == {"plant": {"numerator": [1.0, 2.0]}}
        num.append(3.0)
        assert snap["plant"]["numerator"] == [1.0, 2.0]

    def test_changes_list_old_and_new(self):
        ref = {"Gain1": {"gain": 2.0, "x": "a"}, "Step": {"value": 1.0}}
        cur = {"Gain1": {"gain": 3.0, "x": "a"}, "Step": {"value": 1.0}}
        assert param_changes(ref, cur) == ["Gain1.gain: 2.0 -> 3.0"]

    def test_added_removed_and_arrays(self):
        ref = {"A": {"k": np.array([1, 2])}, "Old": {}}
        cur = {"A": {"k": np.array([1, 3])}, "New": {}}
        assert param_changes(ref, cur) == ["A.k: [1, 2] -> [1, 3]", "New: added", "Old: removed"]

    def test_identical(self):
        snap = {"A": {"k": [1, 2]}}
        assert param_changes(snap, {"A": {"k": [1, 2]}}) == []
