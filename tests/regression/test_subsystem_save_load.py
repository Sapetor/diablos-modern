"""
Regression test for subsystem save/load round-trip.

Bug: FileService.serialize() emitted nothing for Subsystem.sub_blocks /
sub_lines / ports / ports_map, and apply_loaded_data() never tried to
reconstruct them. Saving a diagram containing a subsystem and reopening
it produced an empty container, and the flattener silently saw no
primitives inside the subsystem.

Fix: serialize() now recurses into subsystems and apply_loaded_data()
rebuilds children using a real Subsystem instance.
"""

import pytest
from PyQt5.QtCore import QPoint, QRect


@pytest.mark.regression
class TestSubsystemSaveLoad:
    """The subsystem's internal blocks and connections must survive a save/load round-trip."""

    def _build_diagram_with_subsystem(self, model):
        """
        Build:  Step  ->  Subsystem(In1 -> Gain -> Out1)  ->  Scope
        """
        from blocks.subsystem import Subsystem
        from lib.simulation.block import DBlock
        from lib.simulation.connection import DLine

        # Look up palette templates the same way real UI code does so the
        # MenuBlock's category/color/io_edit are realistic.
        def menu_block(block_fn):
            for mb in model.menu_blocks:
                if mb.block_fn == block_fn:
                    return mb
            raise AssertionError(f"menu_block missing: {block_fn}")

        step_mb = menu_block('Step')
        gain_mb = menu_block('Gain')
        scope_mb = menu_block('Scope')

        step = model.add_block(step_mb, QPoint(100, 100))
        scope = model.add_block(scope_mb, QPoint(500, 100))

        # Manually construct a Subsystem with one Inport, one Gain, one Outport
        subsys = Subsystem(block_name="Subsystem1", sid=1,
                           coords=QRect(250, 80, 100, 80),
                           color=step.b_color)
        subsys.io_edit = 'none'
        subsys.fn_name = 'subsystem'
        subsys.params['_name_'] = subsys.name

        # Internal blocks — keep names matching the convention the flattener
        # falls back to ("inport1", "outport1", "<fn>0").
        from blocks.inport import Inport
        from blocks.outport import Outport

        inport = Inport(block_name="In1")
        inport.name = "inport1"
        gain = DBlock(
            block_fn='Gain', sid=0, coords=QRect(280, 100, 60, 40),
            color=gain_mb.b_color, in_ports=1, out_ports=1, b_type=2,
            io_edit=gain_mb.io_edit, fn_name=gain_mb.fn_name,
            params={'gain': 2.0}, block_class=gain_mb.block_class,
            colors=model.colors,
        )
        outport = Outport(block_name="Out1")
        outport.name = "outport1"

        subsys.sub_blocks = [inport, gain, outport]

        # Internal lines: inport1 -> gain0 -> outport1
        subsys.sub_lines = [
            DLine(0, "inport1", 0, gain.name, 0, [QPoint(0, 0), QPoint(10, 10)]),
            DLine(1, gain.name, 0, "outport1", 0, [QPoint(0, 0), QPoint(10, 10)]),
        ]

        # External port layout (one input, one output) — what subsystem_manager
        # populates after a "Create Subsystem" action.
        subsys.ports = {
            'in':  [{'pos': (0, 40), 'type': 'input',  'name': '1'}],
            'out': [{'pos': (100, 40), 'type': 'output', 'name': '1'}],
        }
        subsys.ports_map = {'input': {0: 'inport1'}, 'output': {0: 'outport1'}}
        subsys.update_Block()
        model.blocks_list.append(subsys)

        # Top-level wires: step -> subsys -> scope
        line_in = DLine(0, step.name, 0, subsys.name, 0,
                        [QPoint(0, 0), QPoint(10, 10)])
        line_out = DLine(1, subsys.name, 0, scope.name, 0,
                         [QPoint(0, 0), QPoint(10, 10)])
        model.line_list.extend([line_in, line_out])

        return subsys, gain

    def test_round_trip_preserves_subsystem_children(self, file_service, simulation_model, tmp_path):
        """sub_blocks, sub_lines, ports, and ports_map all survive save+load."""
        subsys, _ = self._build_diagram_with_subsystem(simulation_model)

        original_sub_block_names = [b.name for b in subsys.sub_blocks]
        original_sub_line_count = len(subsys.sub_lines)
        original_ports = subsys.ports
        original_ports_map = subsys.ports_map

        # Save then load
        path = tmp_path / "round_trip.dat"
        data = file_service.serialize()
        assert file_service.save_to_file(data, str(path))

        loaded = file_service.load(str(path))
        assert loaded is not None

        # Wipe model state and apply loaded
        file_service.apply_loaded_data(loaded)

        # Find the reloaded subsystem
        reloaded = next((b for b in simulation_model.blocks_list
                         if b.block_fn == 'Subsystem'), None)
        assert reloaded is not None, "Subsystem block missing after reload"

        # The fix: children must be reconstructed
        assert [b.name for b in reloaded.sub_blocks] == original_sub_block_names, \
            "sub_blocks did not survive save/load"
        assert len(reloaded.sub_lines) == original_sub_line_count, \
            "sub_lines did not survive save/load"
        assert reloaded.ports == original_ports, \
            "external ports layout did not survive save/load"

        # ports_map keys should be ints again (not stringified by JSON)
        for kind, mapping in original_ports_map.items():
            for idx, name in mapping.items():
                assert reloaded.ports_map[kind][idx] == name, \
                    f"ports_map[{kind}][{idx}] lost or mistyped after reload"

        # Reloaded subsystem must be a real Subsystem instance so the
        # flattener's class-name detection still works.
        assert reloaded.__class__.__name__ == 'Subsystem'

    def test_round_trip_flattens_to_primitives(self, file_service, simulation_model, tmp_path):
        """After reload, the flattener should still see the inner Gain block."""
        from lib.engine.flattener import Flattener

        self._build_diagram_with_subsystem(simulation_model)

        path = tmp_path / "flatten_after_reload.dat"
        file_service.save_to_file(file_service.serialize(), str(path))
        loaded = file_service.load(str(path))
        file_service.apply_loaded_data(loaded)

        flat_blocks, _ = Flattener().flatten(simulation_model.blocks_list,
                                             simulation_model.line_list)
        flat_fns = {b.block_fn for b in flat_blocks}

        # Pre-fix this set was missing 'Gain' because the subsystem reloaded
        # with empty sub_blocks.
        assert 'Gain' in flat_fns, \
            f"Inner Gain block lost after subsystem reload; flattener saw {flat_fns}"
