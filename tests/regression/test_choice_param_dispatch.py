"""Guardrail against dropdown ("choice") parameter drift.

A block's set of selectable values lives in TWO hand-maintained places that
nothing keeps in sync: the ``options`` list in the ``params`` property (drives
the dropdown in the property editor) and the dispatch logic in ``execute()``
(the ``if/elif <param> == "..."`` chain, often re-implemented in the compiled
solver and replay loop too). When they drift you get bugs like the historical
``MathFunction`` one: ``cube`` was implemented in every code path and documented
but missing from the dropdown ``options`` list, so the working feature was
unreachable from the UI.

This module locks that class of bug down three ways:

1. ``test_every_dropdown_option_executes`` -- run ``execute()`` once per dropdown
   option for every block we can cleanly stimulate, and assert it neither raises
   nor returns an ``{'E': True}`` error. Catches dropdown values the dispatch
   does not handle (and any option that crashes the block).
2. ``test_all_dropdown_params_are_covered`` -- discover *every* dropdown param in
   the whole block library and assert each is either exercised by ``SPECS`` or
   explicitly allow-listed with a reason. A NEW dropdown param that is neither
   fails the suite, forcing a conscious decision.
3. ``test_math_function_options_match_dispatch`` -- the precise ``cube`` guard:
   the option list and the dispatched function set must be exactly equal.

NB: the property editor renders a QComboBox for ANY param carrying an ``options``
or ``choices`` list regardless of its declared ``type`` (see
``modern_ui/widgets/property_editor.py``), so "dropdown param" here means "has an
options/choices list", not ``type == 'choice'``.
"""

import inspect
import re

import numpy as np
import pytest

from lib.block_loader import load_blocks

pytestmark = pytest.mark.regression


# --------------------------------------------------------------------------- #
# Stimulus specs: how to exercise each block's execute() once per option.
# Each spec supplies the inputs/params/time/dtime needed to reach the dispatch
# without tripping unrelated errors. Options themselves are read from the live
# block.params (the source of truth) -- never duplicated here.
# --------------------------------------------------------------------------- #
def _spec(factory, *, inputs=None, params=None, time=0.5, dtime=0.01, needs_csv=False):
    return {
        "factory": factory,
        "inputs": inputs or {},
        "params": params or {},
        "time": time,
        "dtime": dtime,
        "needs_csv": needs_csv,
    }


def _math_function():
    from blocks.math_function import MathFunctionBlock
    return MathFunctionBlock()


def _logical():
    from blocks.logical_operator import LogicalOperatorBlock
    return LogicalOperatorBlock()


def _relational():
    from blocks.relational_operator import RelationalOperatorBlock
    return RelationalOperatorBlock()


def _compare():
    from blocks.compare_to_constant import CompareToConstantBlock
    return CompareToConstantBlock()


def _chirp():
    from blocks.chirp import ChirpBlock
    return ChirpBlock()


def _wave():
    from blocks.wave_generator import WaveGeneratorBlock
    return WaveGeneratorBlock()


def _random_source():
    from blocks.random_source import RandomSourceBlock
    return RandomSourceBlock()


def _packet_loss():
    from blocks.packet_loss import PacketLossBlock
    return PacketLossBlock()


def _network_channel():
    from blocks.network_channel import NetworkChannelBlock
    return NetworkChannelBlock()


def _residual_norm():
    from blocks.optimization_primitives.residual_norm import ResidualNormBlock
    return ResidualNormBlock()


def _numerical_gradient():
    from blocks.optimization_primitives.numerical_gradient import NumericalGradientBlock
    return NumericalGradientBlock()


def _linear_solver():
    from blocks.optimization_primitives.linear_system_solver import LinearSystemSolverBlock
    return LinearSystemSolverBlock()


def _lookup_1d():
    from blocks.lookup_table import LookupTable1DBlock
    return LookupTable1DBlock()


def _lookup_2d():
    from blocks.lookup_table import LookupTable2DBlock
    return LookupTable2DBlock()


def _from_file():
    from blocks.from_file import FromFileBlock
    return FromFileBlock()


def _export():
    from blocks.export import ExportBlock
    return ExportBlock()


def _switch():
    from blocks.switch import SwitchBlock
    return SwitchBlock()


def _field_probe():
    from blocks.pde.field_processing import FieldProbeBlock
    return FieldProbeBlock()


def _field_max():
    from blocks.pde.field_processing import FieldMaxBlock
    return FieldMaxBlock()


def _field_probe_2d():
    from blocks.pde.field_processing_2d import FieldProbe2DBlock
    return FieldProbe2DBlock()


# Keyed by block_name so the coverage test can cross-reference the live library.
SPECS = {
    "MathFunction": _spec(_math_function, inputs={0: np.array([0.5])}),
    "LogicalOperator": _spec(_logical, inputs={0: np.array([1.0]), 1: np.array([0.0])}),
    "RelationalOperator": _spec(_relational, inputs={0: np.array([1.0]), 1: np.array([2.0])}),
    "CompareToConstant": _spec(_compare, inputs={0: np.array([1.0])}),
    # Chirp: f0 MUST be > 0 or the 'logarithmic' sweep raises in scipy.
    "Chirp": _spec(_chirp, params={"f0": 1.0, "f1": 10.0, "t1": 10.0}),
    "WaveGenerator": _spec(_wave, time=0.25),
    "RandomSource": _spec(_random_source, params={"seed": 42}),
    "PacketLoss": _spec(_packet_loss, inputs={0: np.array([1.0])}, params={"seed": 42}),
    "NetworkChannel": _spec(_network_channel, inputs={0: np.array([1.0])}, params={"seed": 42}),
    "ResidualNorm": _spec(_residual_norm, inputs={0: np.array([1.0, 2.0, 3.0])}),
    "NumericalGradient": _spec(
        _numerical_gradient,
        inputs={0: 1.0, 1: 1.1, 2: 1.2, 3: 0.9, 4: 0.8},
        params={"dimension": 2},
    ),
    "LinearSystemSolver": _spec(
        _linear_solver,
        inputs={0: [[2.0, 1.0], [1.0, 3.0]], 1: [5.0, 7.0]},
        params={"dimension": 2},
    ),
    "LookupTable1D": _spec(_lookup_1d, inputs={0: np.array([1.5])}),
    "LookupTable2D": _spec(_lookup_2d, inputs={0: np.array([1.0]), 1: np.array([0.5])}),
    "FromFile": _spec(
        _from_file, params={"time_col": "t", "signal_col": "y"}, needs_csv=True,
    ),
    "Export": _spec(_export, inputs={0: np.array([1.0])}),
    # Newly-constrained dropdowns (mode/position_mode gained explicit options).
    "Switch": _spec(
        _switch, inputs={0: np.array([0.0]), 1: np.array([1.0]), 2: np.array([2.0])},
    ),
    "FieldProbe": _spec(_field_probe, inputs={0: np.array([0.0, 1.0, 2.0, 3.0, 4.0])}),
    "FieldMax": _spec(_field_max, inputs={0: np.array([1.0, 2.0, 3.0])}),
    "FieldProbe2D": _spec(_field_probe_2d, inputs={0: np.ones((4, 4))}),
}


# Dropdown params we intentionally do NOT smoke-test here, each with a reason.
# Stateful / complex blocks whose dropdown values are already exercised by
# dedicated tests; a single-step execute() would be a poor (or misleading) probe.
ALLOWLIST = {
    ("Integrator", "method"):
        "stateful ODE block; methods covered by tests/integration + tests/test_blocks.py",
    ("Integrator", "ivp_method"):
        "solve_ivp method selector; covered by integration/solver tests",
    ("RateTransition", "transition_mode"):
        "stateful sampled-data block; covered by tests/unit/test_multirate.py",
    ("Scope", "verify_mode"):
        "GUI sink verification mode (no signal dispatch); covered by tests/unit/test_sink_blocks.py",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dropdown_params(block):
    """Return {param_name: [options]} for every dropdown param on a block.

    A dropdown is any param carrying an ``options`` or ``choices`` list -- the
    same rule the property editor uses to decide on a QComboBox.
    """
    out = {}
    for pname, meta in (block.params or {}).items():
        if not isinstance(meta, dict):
            continue
        opts = meta.get("options") or meta.get("choices")
        if opts and isinstance(opts, (list, tuple)):
            out[pname] = list(opts)
    return out


def _build_params(block, overrides, choice_name, option):
    """Flatten block param defaults, apply overrides, then pin the choice value."""
    params = {
        pname: meta["default"]
        for pname, meta in (block.params or {}).items()
        if isinstance(meta, dict) and "default" in meta
    }
    params.update(overrides)
    params[choice_name] = option
    params.setdefault("_init_start_", True)
    params["_name_"] = block.block_name + "_test"
    return params


# Flat list of (block_name, param_name, option) cases, built from SPECS.
def _build_cases():
    cases = []
    for label, spec in SPECS.items():
        block = spec["factory"]()
        for pname, opts in _dropdown_params(block).items():
            for opt in opts:
                cases.append((label, pname, opt))
    return cases


CASES = _build_cases()


@pytest.fixture(scope="session")
def from_file_csv(tmp_path_factory):
    """A small (t, y) CSV so FromFile.execute() has real data to interpolate."""
    path = tmp_path_factory.mktemp("choice_dispatch") / "series.csv"
    path.write_text("t,y\n0.0,0.0\n1.0,1.0\n2.0,4.0\n3.0,9.0\n")
    return str(path)


# --------------------------------------------------------------------------- #
# 1. Every dropdown option must execute cleanly.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,param,option",
    CASES,
    ids=[f"{lbl}.{p}={o}" for lbl, p, o in CASES],
)
def test_every_dropdown_option_executes(label, param, option, from_file_csv):
    spec = SPECS[label]
    block = spec["factory"]()
    overrides = dict(spec["params"])
    if spec["needs_csv"]:
        overrides["data_file"] = from_file_csv

    params = _build_params(block, overrides, param, option)
    result = block.execute(
        time=spec["time"], inputs=dict(spec["inputs"]), params=params, dtime=spec["dtime"]
    )

    assert isinstance(result, dict), (
        f"{label} {param}={option!r}: execute() returned {type(result).__name__}, not a dict"
    )
    assert not result.get("E"), (
        f"{label} {param}={option!r}: dropdown option not handled by execute() -- "
        f"error: {result.get('error')}"
    )
    assert 0 in result, (
        f"{label} {param}={option!r}: execute() produced no output on port 0 -- "
        f"the option likely fell through the dispatch chain"
    )


# --------------------------------------------------------------------------- #
# 2. Suite-level coverage guard: no dropdown param may go untracked.
# --------------------------------------------------------------------------- #
def _discover_all_dropdowns():
    """Every (block_name, param) in the library that renders as a dropdown."""
    found = set()
    for cls in load_blocks():
        try:
            block = cls()
            params = block.params
        except Exception:
            continue  # not instantiable without context; not our concern here
        for pname in _dropdown_params(block):
            found.add((block.block_name, pname))
    return found


def test_all_dropdown_params_are_covered():
    found = _discover_all_dropdowns()
    covered = {(label, pname) for label, pname, _ in CASES} | set(ALLOWLIST)

    missing = found - covered
    assert not missing, (
        "New dropdown param(s) are neither smoke-tested nor allow-listed: "
        f"{sorted(missing)}. Add a stimulus to SPECS (preferred) or, for a "
        "stateful/complex block already covered elsewhere, an ALLOWLIST entry "
        "with a reason."
    )

    # Keep the registries honest: every tracked param must still exist.
    stale = covered - found
    assert not stale, (
        f"SPECS/ALLOWLIST reference dropdown params that no longer exist: {sorted(stale)}"
    )


def test_specs_and_allowlist_are_disjoint():
    """A param should be smoke-tested OR allow-listed, never both (avoids rot)."""
    spec_params = {(label, pname) for label, pname, _ in CASES}
    overlap = spec_params & set(ALLOWLIST)
    assert not overlap, f"Params both tested and allow-listed: {sorted(overlap)}"


# --------------------------------------------------------------------------- #
# 3. The precise cube guard: options list == dispatched function set.
# --------------------------------------------------------------------------- #
def test_math_function_options_match_dispatch():
    from blocks.math_function import MathFunctionBlock

    src = inspect.getsource(MathFunctionBlock.execute)
    handled = set(re.findall(r'func == "([^"]+)"', src))
    options = set(MathFunctionBlock().params["function"]["options"])

    assert options == handled, (
        "MathFunction dropdown/dispatch drift -- "
        f"handled but not selectable: {sorted(handled - options)}; "
        f"selectable but not handled: {sorted(options - handled)}"
    )
