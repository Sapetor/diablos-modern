"""Semantic diff of .diablos diagrams (lib/diagram_diff.py, ``diablos diff``)."""

import copy
import json
import os

import pytest

from lib.cli import main as cli_main
from lib.diagram_diff import diff_diagrams, format_diff, is_empty, load

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "discrete_pi_zoh.diablos")
SUBSYS = os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "library_block_demo.diablos"
)


@pytest.fixture
def base():
    return load(EXAMPLE)


def _block(d, name):
    return next(b for b in d["blocks_data"] if b["name"] == name)


@pytest.mark.unit
class TestDiagramDiff:
    def test_identical_and_layout_only_changes_are_empty(self, base):
        moved = copy.deepcopy(base)
        _block(moved, "tranfn4")["coords_left"] += 40
        _block(moved, "tranfn4")["selected"] = True
        moved["lines_data"][0]["points"][0][1] += 7
        assert is_empty(diff_diagrams(base, base))
        assert is_empty(diff_diagrams(base, moved))
        assert format_diff(diff_diagrams(base, moved)) == "no semantic differences"

    def test_layout_flag_reports_moves_and_bends(self, base):
        moved = copy.deepcopy(base)
        _block(moved, "tranfn4")["coords_left"] += 40
        moved["lines_data"][0]["points"][0][1] += 7
        d = diff_diagrams(base, moved, layout=True)
        assert d["blocks_changed"][0]["changes"] == ["coords_left: 740 -> 780"]
        assert d["wires_changed"][0]["wire"] == "reference:0 -> error:0"

    def test_param_label_and_sim_changes(self, base):
        new = copy.deepcopy(base)
        _block(new, "discretetranfn3")["params"]["numerator"] = [2.0, -1.5]
        _block(new, "tranfn4")["username"] = "plant"
        new["sim_data"]["sim_time"] = 20.0
        d = diff_diagrams(base, new)
        changed = {c["path"]: c["changes"] for c in d["blocks_changed"]}
        assert changed["discretetranfn3"] == ["numerator: [1.2, -1.0] -> [2.0, -1.5]"]
        assert changed["tranfn4"][0] == "label: 'plant 2/((s+1)(s+2))' -> 'plant'"
        assert d["sim"] and d["sim"][0].startswith("sim_time:")
        text = format_diff(d)
        assert '~ block DiscreteTranFn "digital PI" (discretetranfn3)' in text

    def test_added_removed_blocks_and_rewired_connections(self, base):
        new = copy.deepcopy(base)
        new["blocks_data"] = [b for b in new["blocks_data"] if b["name"] != "scope5"]
        new["lines_data"] = [ln for ln in new["lines_data"] if ln["dstblock"] != "scope5"]
        gain = copy.deepcopy(_block(base, "sum1"))
        gain.update(name="gain9", block_fn="Gain", username="K", params={"gain": 3.0})
        new["blocks_data"].append(gain)
        new["lines_data"].append(
            {"srcblock": "zeroorderhold2", "srcport": 0, "dstblock": "gain9", "dstport": 0}
        )
        d = diff_diagrams(base, new)
        assert d["blocks_removed"] == ['Scope "sampled-data loop" (scope5)']
        assert d["blocks_added"] == ['Gain "K" (gain9)']
        assert "sampler Ts = 0.1 s:0 -> K:0" in d["wires_added"]
        assert len(d["wires_removed"]) == 3

    def test_type_change_under_same_name_is_remove_plus_add(self, base):
        new = copy.deepcopy(base)
        _block(new, "sum1")["block_fn"] = "Product"
        d = diff_diagrams(base, new)
        assert [b.split()[0] for b in d["blocks_removed"]] == ["Sum"]
        assert [b.split()[0] for b in d["blocks_added"]] == ["Product"]
        assert d["blocks_changed"] == []

    def test_subsystem_contents_are_compared_by_path(self):
        base = load(SUBSYS)
        new = copy.deepcopy(base)
        sub = next(b for b in new["blocks_data"] if b.get("sub_blocks"))
        inner = next(b for b in sub["sub_blocks"] if b.get("params"))
        key = next(iter(inner["params"]))
        inner["params"][key] = "changed"
        d = diff_diagrams(base, new)
        assert [c["path"] for c in d["blocks_changed"]] == [f"{sub['name']}/{inner['name']}"]


@pytest.mark.unit
class TestDiffCli:
    def test_exit_codes_and_json(self, base, tmp_path, capsys):
        new = copy.deepcopy(base)
        _block(new, "discretetranfn3")["params"]["sampling_time"] = 0.05
        p = tmp_path / "new.diablos"
        p.write_text(json.dumps(new), encoding="utf-8")

        assert cli_main(["diff", EXAMPLE, EXAMPLE]) == 0
        assert "no semantic differences" in capsys.readouterr().out
        assert cli_main(["diff", EXAMPLE, str(p)]) == 1
        assert "sampling_time: 0.1 -> 0.05" in capsys.readouterr().out
        assert cli_main(["diff", "--json", EXAMPLE, str(p)]) == 1
        assert json.loads(capsys.readouterr().out)["blocks_changed"][0]["path"] == "discretetranfn3"
        assert cli_main(["diff", EXAMPLE, str(tmp_path / "missing.diablos")]) == 2


@pytest.mark.unit
def test_old_format_without_names_matches_the_resaved_file(base):
    """Older saves have no block ``name`` (the loader derives block_fn+sid) and
    wire blocks by their canvas label; re-saving the same diagram in the new
    format must not show up as every wire removed and re-added."""
    old = copy.deepcopy(base)
    labels = {b["name"]: b["username"] for b in old["blocks_data"]}
    for b in old["blocks_data"]:
        b.pop("name")
    for ln in old["lines_data"]:
        ln["srcblock"], ln["dstblock"] = labels[ln["srcblock"]], labels[ln["dstblock"]]
    assert is_empty(diff_diagrams(old, base))
