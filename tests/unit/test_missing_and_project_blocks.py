"""Project-local user blocks survive opening, saving, cloning and process workers.

A trusted ``blocks/`` folder next to a diagram registers extra block types. Two
reported data-loss bugs:

* the diagram used to be rebuilt *before* those types were registered, and
  the loader silently dropped every block of an unknown type (and with it its
  connections) -- so a save afterwards deleted them from the file;
* analysis copies (``DSim.clone_for_analysis``) and spawned experiment
  workers rebuilt the diagram from a fresh registry without the project's
  blocks, running a reduced diagram (in-process) or nothing (workers=2).

Unknown block types are now kept as placeholders (``MissingBlock``) that save
back unchanged and refuse to run, and clones/workers inherit the parent's
approved block registrations.
"""

import json
import sys

import numpy as np
import pytest

import lib.user_blocks as ub

TRIPLER = """\
import numpy as np
from blocks.base_block import BaseBlock


class ReviewTriplerBlock(BaseBlock):
    @property
    def block_name(self):
        return "ReviewTripler"

    @property
    def category(self):
        return "Math"

    @property
    def params(self):
        return {"factor": {"type": "float", "default": 3.0, "doc": "Factor"}}

    @property
    def inputs(self):
        return [{"name": "in", "type": "any"}]

    @property
    def outputs(self):
        return [{"name": "out", "type": "any"}]

    def execute(self, time, inputs, params, **kwargs):
        return {0: float(params["factor"]) * np.atleast_1d(inputs.get(0, 0.0))}
"""


def write_project(root, trusted):
    """Diagram Constant(2) -> ReviewTripler -> Scope with a blocks/ folder."""
    from lib.diagram_builder import DiagramBuilder

    blocks = root / "blocks"
    blocks.mkdir(parents=True)
    (blocks / "tripler.py").write_text(TRIPLER, encoding="utf-8")
    builder = DiagramBuilder(sim_time=0.5, sim_dt=0.01)
    c = builder.add_block("Constant", 50, 100, name="c", params={"value": 2.0})
    t = builder.add_block("ReviewTripler", 200, 100, name="t", params={"factor": 3.0})
    s = builder.add_block("Scope", 350, 100, name="s", in_ports=1, out_ports=0)
    builder.connect(c, 0, t, 0)
    builder.connect(t, 0, s, 0)
    path = root / "model.diablos"
    builder.save(str(path))
    if trusted:
        ub.trust_project_blocks(str(blocks))
    return str(path)


@pytest.fixture
def userdata(tmp_path, monkeypatch):
    """Private trust store / user data dir; no stray user modules."""
    import lib.app_paths

    monkeypatch.delenv(ub.BLOCKS_ENV_VAR, raising=False)
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(lib.app_paths, "get_user_data_dir", lambda: str(home))
    monkeypatch.setattr(ub, "_SESSION_TRUSTED", set(), raising=False)
    ub.purge_user_modules()
    yield home
    ub.purge_user_modules()


@pytest.fixture
def trusted_project(tmp_path, userdata):
    return write_project(tmp_path / "trusted", trusted=True)


@pytest.fixture
def untrusted_project(tmp_path, userdata):
    return write_project(tmp_path / "untrusted", trusted=False)


def gui_ordered_dsim(path):
    """What a correct GUI open does: register the project's blocks, then load."""
    from lib.cli import new_headless_dsim

    dsim = new_headless_dsim()
    dsim.model.reload_blocks(path)
    with open(path, encoding="utf-8") as fh:
        dsim.deserialize(json.load(fh))
    return dsim


def block_fns(dsim):
    return sorted(b.block_fn for b in dsim.blocks_list)


def placeholders(dsim):
    from lib.simulation.missing_block import find_missing_blocks

    return find_missing_blocks(dsim.blocks_list)


# ---------------------------------------------------------------- placeholder
@pytest.mark.unit
class TestUnknownTypesArePreserved:
    def test_unknown_block_is_kept_with_its_connections(self, untrusted_project):
        dsim = gui_ordered_dsim(untrusted_project)
        assert block_fns(dsim) == ["Constant", "ReviewTripler", "Scope"]
        assert [b.missing_block_type for b in placeholders(dsim)] == ["ReviewTripler"]
        assert len(dsim.line_list) == 2
        names = {b.name for b in dsim.blocks_list}
        assert all(ln.srcblock in names and ln.dstblock in names for ln in dsim.line_list)

    def test_placeholder_serializes_back_unchanged(self, untrusted_project):
        with open(untrusted_project, encoding="utf-8") as fh:
            original = json.load(fh)
        dsim = gui_ordered_dsim(untrusted_project)
        saved = dsim.serialize()
        before = next(b for b in original["blocks_data"] if b["block_fn"] == "ReviewTripler")
        after = next(b for b in saved["blocks_data"] if b["block_fn"] == "ReviewTripler")
        for key in (
            "block_fn",
            "sid",
            "username",
            "params",
            "in_ports",
            "out_ports",
            "coords_left",
            "coords_top",
            "coords_width",
            "b_type",
            "io_edit",
            "fn_name",
        ):
            assert after[key] == before[key], key
        assert len(saved["lines_data"]) == 2

    def test_a_diagram_with_a_placeholder_refuses_to_run(self, untrusted_project):
        dsim = gui_ordered_dsim(untrusted_project)
        ok, err = dsim.run_tuning_simulation(0.1, 0.01)
        assert not ok
        assert "ReviewTripler" in err and "not available" in err

    def test_copying_a_placeholder_gives_a_placeholder(self, untrusted_project):
        from lib.simulation.missing_block import copy_missing_block

        dsim = gui_ordered_dsim(untrusted_project)
        (original,) = placeholders(dsim)
        clone = copy_missing_block(original, 7, original.rect.translated(30, 30))
        assert clone.missing_block_type == "ReviewTripler"
        assert clone.name == "reviewtripler7"
        assert clone.saving_params() == original.saving_params()
        assert (clone.in_ports, clone.out_ports) == (original.in_ports, original.out_ports)

    def test_placeholder_is_rebuilt_once_its_type_is_registered(self, untrusted_project):
        dsim = gui_ordered_dsim(untrusted_project)
        ub.trust_project_blocks(ub.project_blocks_dir(untrusted_project))
        dsim.model.reload_blocks(untrusted_project)
        assert dsim.resolve_missing_blocks() == 1
        assert placeholders(dsim) == []
        ok, err = dsim.run_tuning_simulation(0.1, 0.01)
        assert ok, err


# ---------------------------------------------------------------- clones/workers
def run_mc(dsim, workers):
    from lib.analysis.monte_carlo import MonteCarloRunner

    return MonteCarloRunner(dsim).run(2, sim_time=0.2, sim_dt=0.01, workers=workers)


@pytest.mark.unit
class TestTrustedProjectAnalysis:
    def test_clone_keeps_the_project_block(self, trusted_project):
        dsim = gui_ordered_dsim(trusted_project)
        assert placeholders(dsim) == []
        clone = dsim.clone_for_analysis()
        assert block_fns(clone) == block_fns(dsim) == ["Constant", "ReviewTripler", "Scope"]
        assert placeholders(clone) == []
        assert len(clone.line_list) == 2

    def test_workers_get_the_trusted_project_folder(self, trusted_project):
        from lib.analysis.parallel import block_context_snapshot

        context = block_context_snapshot(gui_ordered_dsim(trusted_project).clone_for_analysis())
        assert context == {
            "diagram_path": trusted_project,
            "project_blocks": ub.project_blocks_dir(trusted_project),
        }

    def test_workers_1_and_2_agree(self, trusted_project):
        dsim = gui_ordered_dsim(trusted_project)
        one = run_mc(dsim.clone_for_analysis(), workers=1)
        two = run_mc(dsim.clone_for_analysis(), workers=2)
        assert one["n_ok"] == two["n_ok"] == 2
        assert set(one["signals"]) == set(two["signals"]) and one["signals"]
        for name, stats in one["signals"].items():
            np.testing.assert_allclose(stats["mean"], two["signals"][name]["mean"])
            np.testing.assert_allclose(stats["mean"], 6.0)

    def test_headless_load_registers_trusted_project_blocks(self, trusted_project):
        from lib.cli import load_diagram

        dsim, _ = load_diagram(trusted_project)
        assert placeholders(dsim) == []
        assert "ReviewTripler" in block_fns(dsim)


@pytest.mark.unit
class TestUntrustedProjectAnalysis:
    def test_untrusted_folder_is_never_handed_to_workers(self, untrusted_project):
        from lib.analysis.parallel import block_context_snapshot

        dsim = gui_ordered_dsim(untrusted_project)
        assert block_context_snapshot(dsim) is None

    @pytest.mark.parametrize("workers", [1, 2])
    def test_runs_fail_explicitly(self, untrusted_project, workers, caplog):
        dsim = gui_ordered_dsim(untrusted_project)
        clone = dsim.clone_for_analysis()
        assert [b.missing_block_type for b in placeholders(clone)] == ["ReviewTripler"]
        with caplog.at_level("WARNING"):
            result = run_mc(clone, workers=workers)
        assert result["n_ok"] == 0
        assert "not available" in caplog.text and "ReviewTripler" in caplog.text
        # The project module never ran in this process.
        assert ub.USER_MODULE_PREFIX + ".tripler" not in sys.modules

    def test_missing_type_message_names_the_type(self, untrusted_project):
        from lib.simulation.missing_block import missing_block_types

        dsim = gui_ordered_dsim(untrusted_project)
        names = missing_block_types(dsim.blocks_list)
        assert names == ["ReviewTripler"]
        assert "ReviewTripler" in ub.missing_blocks_message(names, untrusted_project)


@pytest.mark.unit
def test_analysis_refuses_a_diagram_with_a_missing_block(qapp, tmp_path):
    """Linearize/trim go straight to the engine, not through DSim's run check;
    they must still fail explicitly instead of analysing a reduced diagram."""
    from lib import api
    from lib.diagram_builder import DiagramBuilder

    b = DiagramBuilder(sim_time=1.0, sim_dt=0.01)
    u = b.add_block("Step", 0, 0, name="u", params={"value": 1.0, "delay": 0.0})
    k = b.add_block("Gain", 100, 0, name="k", params={"gain": 2.0})
    x = b.add_block("Integrator", 200, 0, name="x", params={"init_conds": 0.0})
    y = b.add_block("Scope", 300, 0, name="y", params={"labels": "x"})
    b.connect(u, 0, k, 0)
    b.connect(k, 0, x, 0)
    b.connect(x, 0, y, 0)
    path = tmp_path / "m.diablos"
    b.save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    for rec in data["blocks_data"]:
        if rec["block_fn"] == "Gain":
            rec["block_fn"] = "NoSuchBlock"
    path.write_text(json.dumps(data), encoding="utf-8")

    sim = api.load(str(path))
    for analyse in (sim.linearize, sim.find_operating_point):
        with pytest.raises(Exception, match="NoSuchBlock"):
            analyse()
