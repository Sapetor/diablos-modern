"""Unit tests for user block libraries (``lib/library.py``).

Three things are pinned down here:

  * the on-disk shape of a library file (``.diablos``-compatible, plus a
    top-level ``library_block`` section) and its round-trip;
  * discovery -- which folders are scanned, in which order, and which copy
    wins when the same id appears twice;
  * that a masked Subsystem survives a full save/load through ``FileService``,
    which is what makes a library instance self-contained.
"""

import json
import os

import pytest
from PyQt5.QtCore import QRect

from lib.library import (
    LIBRARY_ENV_VAR,
    LIBRARY_FORMAT_VERSION,
    LIBRARY_SECTION,
    build_library_document,
    discover_library_blocks,
    find_library_block,
    get_library_ref,
    library_search_paths,
    read_library_file,
    slugify,
    write_library_file,
)
from lib.masks import MASK_KEY, get_mask, set_mask

VEHICLE_MASK = {
    "name": "Vehicle",
    "description": "Force in, speed out.",
    "icon": "1/(ms+b)",
    "shape": "rect",
    "category": "User Library",
    "parameters": [
        {"name": "m", "type": "float", "default": 1500.0, "doc": "Mass [kg]"},
        {"name": "b", "type": "float", "default": 50.0, "doc": "Damping [N s/m]"},
    ],
}


@pytest.fixture(scope="module")
def model(qapp):
    """One SimulationModel for the whole module.

    Building a model loads every block class and its palette pixmap; doing that
    once per test churns enough Qt objects to destabilise later widget tests in
    a full-suite run, so the model is built once and cleared between tests.
    """
    from lib.models.simulation_model import SimulationModel

    instance = SimulationModel()
    yield instance
    instance.blocks_list.clear()
    instance.line_list.clear()


@pytest.fixture(autouse=True)
def _clear_model(model):
    model.blocks_list.clear()
    model.line_list.clear()
    yield


@pytest.fixture
def masked_subsystem(qapp):
    """A Subsystem carrying the Vehicle mask and one inner TranFn."""
    from blocks.subsystem import Subsystem
    from lib.simulation.block import DBlock

    subsys = Subsystem(block_name="Vehicle", sid=1, coords=QRect(100, 100, 120, 90))
    subsys.name = "Subsystem1"
    inner = DBlock(
        block_fn="TranFn",
        sid=0,
        coords=QRect(10, 10, 80, 60),
        color="#4CAF50",
        in_ports=1,
        out_ports=1,
        b_type=1,
        io_edit="none",
        fn_name="tranfn",
        params={"numerator": [1.0], "denominator": "[m, b]"},
    )
    subsys.sub_blocks.append(inner)
    set_mask(subsys, VEHICLE_MASK)
    return subsys


def _serialize(model, block):
    from lib.services.file_service import FileService

    return FileService(model)._serialize_block(block)


@pytest.mark.unit
def test_slugify_makes_a_safe_stem():
    assert slugify("Vehicle Model!") == "vehicle_model"
    assert slugify("   ") == "library_block"


@pytest.mark.unit
def test_build_library_document_shape(qapp, model, masked_subsystem):
    data = build_library_document(_serialize(model, masked_subsystem), "vehicle")

    # It is still a loadable diagram...
    assert data["version"] == "2.0"
    assert isinstance(data["blocks_data"], list) and len(data["blocks_data"]) == 1
    assert data["lines_data"] == []
    assert isinstance(data["sim_data"], dict)

    # ...plus the library marker.
    section = data[LIBRARY_SECTION]
    assert section["format_version"] == LIBRARY_FORMAT_VERSION
    assert section["id"] == "vehicle"
    assert section["name"] == "Vehicle"
    assert section["category"] == "User Library"
    assert [p["name"] for p in section["mask"]["parameters"]] == ["m", "b"]
    # Inner blocks and their connections travel inside the serialized block.
    assert section["block"]["sub_blocks"][0]["block_fn"] == "TranFn"
    assert "sub_lines" in section["block"]


@pytest.mark.unit
@pytest.mark.file_io
def test_library_file_round_trip(tmp_path, qapp, model, masked_subsystem):
    path = write_library_file(
        _serialize(model, masked_subsystem), directory=str(tmp_path), block_id="vehicle"
    )
    assert os.path.basename(path) == "vehicle.diablos"

    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    assert LIBRARY_SECTION in raw

    lib_block = read_library_file(path)
    assert lib_block is not None
    assert lib_block.block_id == "vehicle"
    assert lib_block.name == "Vehicle"
    assert lib_block.description == "Force in, speed out."
    assert lib_block.mask["parameters"][1]["name"] == "b"
    # The inner block keeps the *unresolved* mask expression.
    assert lib_block.block_data["sub_blocks"][0]["params"]["denominator"] == "[m, b]"


@pytest.mark.unit
@pytest.mark.file_io
def test_instance_data_is_an_independent_copy(tmp_path, qapp, model, masked_subsystem):
    path = write_library_file(
        _serialize(model, masked_subsystem), directory=str(tmp_path), block_id="vehicle"
    )
    lib_block = read_library_file(path)

    first = lib_block.instance_block_data()
    second = lib_block.instance_block_data()
    first["params"]["m"] = 999.0
    assert second["params"].get("m", 1500.0) != 999.0
    assert first["params"][MASK_KEY]["name"] == "Vehicle"
    # Every instance is stamped with where it came from.
    assert first["params"]["_library_ref"]["id"] == "vehicle"


@pytest.mark.unit
def test_a_plain_diagram_is_not_a_library_block(tmp_path):
    plain = tmp_path / "plain.diablos"
    plain.write_text(json.dumps({"version": "2.0", "blocks_data": [], "lines_data": []}))
    assert read_library_file(str(plain)) is None


@pytest.mark.unit
@pytest.mark.file_io
def test_discovery_from_the_env_var(tmp_path, monkeypatch, qapp, model, masked_subsystem):
    lib_dir = tmp_path / "mylib"
    lib_dir.mkdir()
    write_library_file(
        _serialize(model, masked_subsystem), directory=str(lib_dir), block_id="vehicle"
    )
    monkeypatch.setenv(LIBRARY_ENV_VAR, str(lib_dir))

    found = discover_library_blocks()
    ids = [lb.block_id for lb in found]
    assert "vehicle" in ids
    assert find_library_block("vehicle").name == "Vehicle"
    assert find_library_block("nope") is None


@pytest.mark.unit
def test_search_path_order(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    project = tmp_path / "project" / "diagram.diablos"
    monkeypatch.setenv(LIBRARY_ENV_VAR, str(env_dir))

    paths = library_search_paths(str(project))
    assert paths[0] == str(env_dir)
    assert paths[1] == str(tmp_path / "project" / "library")
    assert paths[-1].endswith("library")  # the per-user folder comes last


@pytest.mark.unit
@pytest.mark.file_io
def test_higher_priority_folder_wins(tmp_path, monkeypatch, qapp, model, masked_subsystem):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    project_dir = tmp_path / "project" / "library"
    project_dir.mkdir(parents=True)

    block_data = _serialize(model, masked_subsystem)
    write_library_file(block_data, directory=str(project_dir), block_id="vehicle")

    shadowed = dict(VEHICLE_MASK, name="Vehicle (env)")
    write_library_file(block_data, directory=str(env_dir), block_id="vehicle", mask=shadowed)

    monkeypatch.setenv(LIBRARY_ENV_VAR, str(env_dir))
    found = discover_library_blocks(str(tmp_path / "project" / "diagram.diablos"))
    vehicles = [lb for lb in found if lb.block_id == "vehicle"]
    assert len(vehicles) == 1
    assert vehicles[0].name == "Vehicle (env)"


@pytest.mark.unit
@pytest.mark.file_io
def test_masked_subsystem_survives_a_diagram_save_load(tmp_path, qapp, model, masked_subsystem):
    """A saved diagram keeps the mask, its values and the library ref."""
    from lib.library import attach_library_ref
    from lib.services.file_service import FileService

    attach_library_ref(masked_subsystem, {"id": "vehicle", "file": "vehicle.diablos"})
    masked_subsystem.params["m"] = 1200.0

    model.blocks_list.append(masked_subsystem)
    service = FileService(model)
    target = tmp_path / "diagram.diablos"
    assert service.save_to_file(service.serialize(), str(target))

    service.apply_loaded_data(service.load(filepath=str(target)))
    reloaded = model.blocks_list[0]

    mask = get_mask(reloaded)
    assert mask is not None and mask["name"] == "Vehicle"
    assert reloaded.params["m"] == 1200.0
    assert get_library_ref(reloaded)["id"] == "vehicle"
    assert reloaded.sub_blocks[0].params["denominator"] == "[m, b]"

    # And a *second* save keeps them too (init_params_list must be rebuilt on
    # load, or saving_params silently drops everything restored above).
    again = tmp_path / "again.diablos"
    assert service.save_to_file(service.serialize(), str(again))
    with open(again, encoding="utf-8") as fp:
        raw = json.load(fp)
    params = raw["blocks_data"][0]["params"]
    assert MASK_KEY in params
    assert params["m"] == 1200.0
    assert params["_library_ref"]["id"] == "vehicle"
