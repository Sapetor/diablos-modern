"""Integration test: every examples/*.diablos actually *runs* headlessly.

``test_example_runs.py`` is the load/init canary — it stops at
``engine.initialize_execution``. This module goes one step further and executes
each gallery diagram end to end through the production headless path
(``DSim.deserialize`` + ``DSim.run_tuning_simulation``), then checks that the
Scope traces it produced are non-empty and finite. A diagram that loads but
integrates to NaN, or whose scopes never receive a sample, fails here.

Two details matter for keeping this affordable in CI:

* ``DSim.deserialize`` (not ``file_service.apply_loaded_data``) is what pushes
  the file's ``solver_method``/``rtol``/``atol`` onto the facade. The stiff
  Van der Pol example needs its saved Radau method — under the default RK45 the
  same run takes ~80 s instead of ~1 s.
* Horizons are capped by *output steps*, not seconds. Several examples are
  iteration counters driven at ``sim_dt = 1.0`` (the optimization-primitives
  loops), where "cap at 2 seconds" would mean two iterations and prove nothing.
  ``_capped_time`` keeps at most ``MAX_STEPS`` output points of each diagram's
  own time step, which bounds the work while keeping every diagram recognisable.

Each DSim owns pyqtgraph-backed plotting state; deleting that from under a
widget mid-construction segfaults the interpreter, so every run is collected
while Qt is idle (see ``tests/unit/test_linearization_result_window.py`` and
``tests/regression/test_zero_crossing.py`` for the same guard).
"""

import gc
import json
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.diablos"))

# Cap on the number of output samples a gallery run is allowed to produce.
MAX_STEPS = 400

# Diagrams whose full horizon is worth keeping (they are cheap) but which are
# still slow enough to be worth marking, so `-m "not slow"` skips them.
SLOW_EXAMPLES = {
    "heat_equation_2d_demo.diablos",
    "heat_equation_2d_verification.diablos",
    "c12x_kuramoto_sync.diablos",
    "relay_thermostat_events.diablos",
}


def _sim_data(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("sim_data", {})


def _capped_time(path):
    """Horizon to run ``path`` for: at most ``MAX_STEPS`` of its own time step."""
    sim = _sim_data(path)
    sim_time = float(sim.get("sim_time", 1.0))
    sim_dt = float(sim.get("sim_dt", 0.01))
    return min(sim_time, MAX_STEPS * sim_dt)


def _has_scope(path):
    with open(path, "r", encoding="utf-8") as fh:
        blocks = json.load(fh).get("blocks_data", [])
    return any(b.get("block_fn") in ("Scope", "AgentScope") for b in blocks)


def _run(path, sim_time=None, use_fast_solver=True, zero_crossing=None):
    """Run one diagram headlessly and return ``(dsim, harvested_signals)``."""
    from lib.analysis.resim import harvest_scope_signals
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    prev_instance = WorkspaceManager._instance
    WorkspaceManager._instance = None
    try:
        dsim = DSim()
        data = dsim.file_service.load(filepath=str(path))
        assert data is not None, "file_service.load returned None for {}".format(path.name)
        # deserialize (not apply_loaded_data) so the diagram's saved solver
        # settings reach the engine.
        dsim.deserialize(data)
        dsim.use_fast_solver = use_fast_solver
        if zero_crossing is not None:
            dsim.zero_crossing = bool(zero_crossing)

        ok, err = dsim.run_tuning_simulation(
            dsim.sim_time if sim_time is None else float(sim_time), dsim.sim_dt
        )
        assert ok, "{}: simulation failed — {}".format(path.name, err)
        return dsim, harvest_scope_signals(dsim)
    finally:
        WorkspaceManager._instance = prev_instance


@pytest.fixture(autouse=True)
def _collect_dsims():
    """Reclaim each test's DSim while Qt is idle (pyqtgraph teardown guard)."""
    yield
    gc.collect()


@pytest.mark.integration
@pytest.mark.parametrize("example_file", EXAMPLE_FILES, ids=lambda f: f.name)
def test_example_runs_and_produces_finite_data(example_file, qapp, request):
    """Run each gallery diagram and assert its Scope data is finite and non-empty."""
    if example_file.name in SLOW_EXAMPLES:
        request.node.add_marker(pytest.mark.slow)

    _dsim, result = _run(example_file, sim_time=_capped_time(example_file))

    if not _has_scope(example_file):
        # Solver-style examples (LinearSystemSolver + Display) legitimately have
        # no Scope; reaching here means they ran without raising.
        assert result is None or not result.get("signals")
        return

    assert result is not None, "{}: no scope data harvested".format(example_file.name)
    signals = result.get("signals") or {}
    assert signals, "{}: diagram has a Scope but produced no traces".format(example_file.name)

    for label, values in signals.items():
        trace = np.asarray(values, dtype=float).reshape(-1)
        assert trace.size > 0, "{}: scope trace {!r} is empty".format(example_file.name, label)
        assert np.all(np.isfinite(trace)), "{}: scope trace {!r} contains NaN/inf".format(
            example_file.name, label
        )


@pytest.mark.integration
def test_readme_documents_every_example():
    """examples/README.md must name every shipped .diablos file."""
    readme = (EXAMPLES_DIR / "README.md").read_text(encoding="utf-8")
    missing = [f.name for f in EXAMPLE_FILES if f.name not in readme]
    assert not missing, "examples/README.md does not mention: {}".format(", ".join(missing))


@pytest.mark.integration
def test_wiki_examples_page_documents_every_example():
    """docs/wiki/Examples.md (in the mkdocs nav) must stay in step with the folder."""
    page = (EXAMPLES_DIR.parent / "docs" / "wiki" / "Examples.md").read_text(encoding="utf-8")
    missing = [f.name for f in EXAMPLE_FILES if f.name not in page]
    assert not missing, "docs/wiki/Examples.md does not mention: {}".format(", ".join(missing))


@pytest.mark.integration
def test_van_der_pol_uses_its_saved_stiff_solver(qapp):
    """The stiff example must keep its Radau setting — RK45 makes it ~70x slower."""
    sim = _sim_data(EXAMPLES_DIR / "van_der_pol_stiff.diablos")
    assert sim.get("solver_method") == "Radau"

    dsim, result = _run(
        EXAMPLES_DIR / "van_der_pol_stiff.diablos",
        sim_time=_capped_time(EXAMPLES_DIR / "van_der_pol_stiff.diablos"),
    )
    assert dsim.solver_method == "Radau"
    diagnostics = dsim.engine.get_solver_diagnostics() or {}
    assert diagnostics.get("method_used") == "Radau", (
        "compiled path did not honour the diagram's stiff solver: {}".format(diagnostics)
    )
    trace = np.asarray(list(result["signals"].values())[0], dtype=float).reshape(-1)
    # The relaxation oscillator rides the cubic nullcline at |x| ~ 2.
    assert 1.5 < np.max(np.abs(trace)) < 3.0


@pytest.mark.integration
def test_relay_scope_records_switching_on_compiled_path(qapp):
    path = EXAMPLES_DIR / "relay_thermostat_events.diablos"
    _dsim, result = _run(path, sim_time=200.0, use_fast_solver=True, zero_crossing=True)
    signals = result["signals"]
    key = next((k for k in signals if "heater" in str(k).lower()), list(signals)[-1])
    relay = np.asarray(signals[key], dtype=float).reshape(-1)
    assert np.unique(relay).size > 1, "relay trace never switches: {}".format(np.unique(relay))
