"""Robustness gate for the block library.

Every registered block's ``execute()`` must answer bad input with either a
normal output dict or an ``{'E': True, 'error': ...}`` dict -- never an
exception.  The engine wraps ``execute()`` in a try/except, but an escaping
exception there aborts the run with a stack trace instead of naming the block,
and direct callers (replay, analysis, user scripts) get no such net.

Input variants: missing key, zeros, None, empty array, NaN, and a per-port
"one None among zeros" mix for multi-input blocks.
"""

import warnings

import numpy as np
import pytest

from lib.block_loader import load_builtin_blocks

# Blocks allowed to raise on a given variant.  Keep empty unless a raise is
# intentional; every entry needs a reason.
RAISE_ALLOWLIST = {
    # (class name, variant): "reason",
}

_CLASSES = sorted(load_builtin_blocks(), key=lambda c: (c.__module__, c.__name__))


def _flat(spec):
    return {k: (v.get("default") if isinstance(v, dict) else v) for k, v in spec.items()}


def _variants(nin):
    out = {
        "missing": {},
        "zeros": {i: np.zeros(1) for i in range(nin)},
        "none": {i: None for i in range(nin)},
        "empty": {i: np.array([]) for i in range(nin)},
        "nan": {i: np.array([np.nan]) for i in range(nin)},
    }
    if nin > 1:
        for port in range(nin):
            mix = {i: np.zeros(1) for i in range(nin)}
            mix[port] = None
            out[f"none_at_{port}"] = mix
    return out


def _cases():
    for cls in _CLASSES:
        try:
            nin = len(cls().inputs)
        except Exception:
            nin = 0
        for vname in _variants(nin):
            yield pytest.param(cls, vname, id=f"{cls.__name__}-{vname}")


@pytest.mark.unit
@pytest.mark.parametrize("cls,variant", list(_cases()))
def test_execute_never_raises_on_bad_input(cls, variant):
    if (cls.__name__, variant) in RAISE_ALLOWLIST:
        pytest.skip(RAISE_ALLOWLIST[(cls.__name__, variant)])
    inst = cls()
    inputs = _variants(len(inst.inputs))[variant]
    params = _flat(inst.params)
    params["_init_start_"] = True
    params["_name_"] = "blk"  # injected by the engine (simulation_model.py)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = inst.execute(time=0.0, inputs=dict(inputs), params=params, dtime=0.01)

    assert isinstance(out, dict), f"{cls.__name__} returned {type(out).__name__}"
    if out.get("E") is True:
        assert isinstance(out.get("error"), str) and out["error"], "error dict needs a message"
