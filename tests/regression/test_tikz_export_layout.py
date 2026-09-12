r"""TikZ export layout, driven by the real ``examples/*.diablos`` gallery.

The rest of the TikZ suite builds its diagrams out of ``SimpleNamespace`` mocks,
which is how a whole class of layout defects survived: wires drawn straight
through the blocks that sat between their endpoints, four arrowheads stacked on
one anchor, a single ``bpt`` node shared by every junction in the picture.  None
of those are visible in a string assertion on a two-block fixture -- they need a
real diagram with real fan-out, real multi-port Mux/Demux blocks and a real
closed loop.

These tests load the gallery through ``FileService`` (the same path the GUI
uses) and check the *geometry the .tex describes*, not its wording.  What they
cannot check is what the PDF looks like, so every change here is also compiled
with pdflatex and eyeballed; see the exporter's module docstring for the knobs.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"

#: Gallery diagrams that between them cover every routing case the exporter has:
#: a plain SISO loop, multi-port Mux/Demux, long captions, a masked subsystem,
#: a fan-out feeding several columns, and a discrete/continuous mix.
LAYOUT_EXAMPLES = [
    "c01_tank_feedback",
    "inverted_pendulum_lqr",
    "c06_observer_estimation",
    "c05_mass_spring_state_space",
    "discrete_pi_zoh",
    "test_demux_logic",
    "c06_lqr_state_feedback",
    "pi_loop_three_plants",
    "library_block_demo",
    "pid_second_order",
]


def _load(name):
    """Return (blocks_list, line_list) for an example, as the GUI would."""
    from lib.models.simulation_model import SimulationModel
    from lib.services.file_service import FileService

    model = SimulationModel()
    service = FileService(model)
    data = service.load(str(EXAMPLES_DIR / f"{name}.diablos"))
    assert data is not None, f"{name}.diablos failed to load"
    service.apply_loaded_data(data)
    return model.blocks_list, model.line_list


@pytest.fixture(scope="module")
def exports(qapp):
    """{name: (exporter, picture)} for every layout example, built once."""
    from lib.export.tikz_exporter import TikZExporter

    built = {}
    for name in LAYOUT_EXAMPLES:
        blocks, lines = _load(name)
        exporter = TikZExporter(blocks, lines)
        built[name] = (exporter, exporter.export_snippet())
    return built


#: A ``\draw`` whose path is nothing but ``--`` segments: it never leaves the
#: straight line between its endpoints.
_STRAIGHT_PATH = re.compile(r"\\draw\[[^]]*\]((?:[^;]|\n)*);")
_NODE_REF = re.compile(r"\((?!\s*[-\d.$])([A-Za-z][A-Za-z0-9_]*)(?:\.[A-Za-z0-9 ]+)?\)")


def _draws(picture):
    return [m.group(0) for m in _STRAIGHT_PATH.finditer(picture)]


def _strip_node_text(text):
    """Drop every ``{...}`` group: node captions are prose, not node ids."""
    out = []
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _defined_ids(picture):
    ids = set(re.findall(r"\\node\[[^]]*\]\s*\(([A-Za-z][A-Za-z0-9_]*)\)", picture))
    ids |= set(re.findall(r"\\coordinate\s*\(([A-Za-z][A-Za-z0-9_]*)\)", picture))
    return ids


class TestEveryWireStaysOutOfTheBlocks:
    """A straight wire may only join blocks that are actually neighbours.

    A forward connection from column *i* to column *j > i+1* drawn as a plain
    ``--`` runs straight over everything parked in between; the export has to
    lift it into a lane instead.
    """

    @pytest.mark.parametrize("name", LAYOUT_EXAMPLES)
    def test_no_straight_wire_skips_a_column(self, name, exports):
        exporter, picture = exports[name]
        order = exporter._block_order
        nid_to_order = {exporter._nid(b): order[b.name] for b in exporter.blocks if b.name in order}
        offenders = []
        for draw in _draws(picture):
            if "|-" in draw or "-|" in draw:
                continue  # already routed through a lane
            endpoints = [nid_to_order[n] for n in _NODE_REF.findall(draw) if n in nid_to_order]
            if len(endpoints) == 2 and abs(endpoints[0] - endpoints[1]) > 1:
                offenders.append(draw)
        assert not offenders, f"{name}: straight wire over intervening blocks:\n" + "\n".join(
            offenders
        )

    @pytest.mark.parametrize("name", LAYOUT_EXAMPLES)
    def test_every_referenced_node_exists(self, name, exports):
        """A typo'd id compiles to a point at the origin, which still 'works'."""
        _exporter, picture = exports[name]
        defined = _defined_ids(picture)
        body = _strip_node_text(picture.split("% --- Blocks ---", 1)[-1])
        referenced = {
            n
            for n in _NODE_REF.findall(body)
            if not n.isdigit() and n not in {"cm", "mm", "pt", "of"}
        }
        assert referenced <= defined, f"{name}: undefined node ids {sorted(referenced - defined)}"


class TestPortsAreDistinguishable:
    """Multi-port blocks must not collapse every wire onto one anchor."""

    def test_a_four_way_demux_gets_four_distinct_points(self, exports):
        exporter, picture = exports["inverted_pendulum_lqr"]
        demux = next(b for b in exporter.blocks if b.block_fn == "Demux")
        points = {exporter._port_point(demux, k, is_output=True) for k in range(demux.out_ports)}
        assert len(points) == demux.out_ports
        for point in points:
            assert point in picture

    def test_a_three_way_demux_feeds_three_separate_inputs(self, exports):
        exporter, picture = exports["test_demux_logic"]
        dm = next(b for b in exporter.blocks if b.block_fn == "Demux")
        lo = next(b for b in exporter.blocks if b.block_fn == "LogicalOperator")
        sources = {exporter._port_point(dm, k, True) for k in range(3)}
        sinks = {exporter._port_point(lo, k, False) for k in range(3)}
        assert len(sources) == 3 and len(sinks) == 3
        assert all(p in picture for p in sources | sinks)


class TestBranchDotsAreUnique:
    """One dot id per junction: a shared one silently misroutes every wire."""

    @pytest.mark.parametrize("name", LAYOUT_EXAMPLES)
    def test_no_branch_id_is_defined_twice(self, name, exports):
        _exporter, picture = exports[name]
        ids = re.findall(r"\\node\[branch\]\s*\(([A-Za-z0-9_]+)\)", picture)
        assert len(ids) == len(set(ids)), f"{name}: duplicate branch ids {ids}"
        assert all(re.match(r"^bpt\d+$", i) for i in ids), ids

    def test_a_diagram_with_several_junctions_numbers_them(self, exports):
        exporter, picture = exports["inverted_pendulum_lqr"]
        ids = re.findall(r"\\node\[branch\]\s*\(([A-Za-z0-9_]+)\)", picture)
        assert len(ids) >= 2, "this diagram fans out at more than one port"
        assert len(exporter._branch_ids) == len(ids)


class TestNodesAreChainedNotPlaced:
    """Relative placement is what keeps a long caption off its neighbour."""

    @pytest.mark.parametrize("name", LAYOUT_EXAMPLES)
    def test_spine_blocks_use_relative_positioning(self, name, exports):
        exporter, picture = exports[name]
        spine_ids = {
            exporter._nid(b)
            for b in exporter.blocks
            if b.name in exporter._block_order and b.name not in exporter._lane_blocks
        }
        placed_absolutely = re.findall(
            r"\\node\[[^]]*\]\s*\(([A-Za-z][A-Za-z0-9_]*)\)\s*at\s*\(", picture
        )
        assert not (set(placed_absolutely) & spine_ids), (
            f"{name}: {sorted(set(placed_absolutely) & spine_ids)} placed at fixed coordinates"
        )

    @pytest.mark.parametrize("name", LAYOUT_EXAMPLES)
    def test_node_distance_is_declared_once(self, name, exports):
        _exporter, picture = exports[name]
        assert picture.count("node distance=") == 1


class TestCaptionsAreNotDoubled:
    def test_a_masked_subsystem_is_named_once(self, exports):
        _exporter, picture = exports["library_block_demo"]
        # The mask name belongs inside the box; captioning it again says it twice.
        assert len(re.findall(r"\{Vehicle\}", picture)) == 1
