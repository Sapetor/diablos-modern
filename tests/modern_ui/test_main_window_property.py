"""
Characterization tests for ModernDiaBloSWindow's property/parameter cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt (with REAL DBlock objects) and pin
down the observable behavior of the property-editing cluster before it is
extracted into a dedicated controller:

  * ``_convert_param_value``   (pure type-coercion helper; variable/expression
                                strings are preserved verbatim)
  * ``_on_property_changed``   (block-param editing: username/port-count special
                                cases, type-coerced param updates, list/array
                                passthrough, Goto/From relink, dirty + repaint)
  * ``_add_to_tuning``         (forward a pinned param to the tuning panel)

An autouse fixture restores the canvas dsim block list so tests can't leak.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_property.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import pytest
from PyQt5.QtCore import QRect

from lib.simulation.block import DBlock


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_blocks(window):
    dsim = window.canvas.dsim
    saved = list(getattr(dsim, 'blocks_list', []) or [])
    saved_dirty = getattr(dsim, 'dirty', False)
    yield
    dsim.blocks_list[:] = saved
    dsim.dirty = saved_dirty


def _make_block(name, block_fn='Gain', params=None, in_ports=1, out_ports=1):
    block = DBlock(
        block_fn=block_fn,
        sid=0,
        coords=QRect(10, 10, 80, 60),
        color='#4CAF50',
        in_ports=in_ports,
        out_ports=out_ports,
        b_type=2,
        io_edit='none',
        fn_name=block_fn.lower(),
        params=params if params is not None else {'gain': 1.0},
        external=False,
        colors=None,
    )
    block.name = name
    return block


def _add(window, block):
    window.canvas.dsim.blocks_list.append(block)
    return block


# ---------------------------------------------------------------------------
# _convert_param_value  (pure helper)
# ---------------------------------------------------------------------------

class TestConvertParamValue:
    def test_int(self, window):
        assert window._convert_param_value("5", int) == 5

    def test_float(self, window):
        assert window._convert_param_value("3.14", float) == 3.14

    def test_bool_true_string(self, window):
        assert window._convert_param_value("True", bool) is True
        assert window._convert_param_value("true", bool) is True

    def test_bool_false_string(self, window):
        assert window._convert_param_value("False", bool) is False
        assert window._convert_param_value("anything", bool) is False

    def test_list(self, window):
        assert window._convert_param_value("[1, 2, 3]", list) == [1, 2, 3]

    def test_list_non_list_literal_falls_back_to_string(self, window):
        # ast.literal_eval('5') -> 5 (int), not a list -> TypeError -> str fallback
        assert window._convert_param_value("5", list) == "5"

    def test_str_passthrough(self, window):
        assert window._convert_param_value("hello", str) == "hello"

    def test_unconvertible_expression_kept_as_string(self, window):
        # Expressions / variable names are preserved verbatim for later
        # resolution by the WorkspaceManager.
        assert window._convert_param_value("2*K", float) == "2*K"
        assert window._convert_param_value("[K, K]", list) == "[K, K]"

    def test_bad_int_falls_back_to_string(self, window):
        assert window._convert_param_value("not_a_number", int) == "not_a_number"


# ---------------------------------------------------------------------------
# _on_property_changed
# ---------------------------------------------------------------------------

class TestOnPropertyChanged:
    def test_param_value_coerced_and_stored(self, window):
        b = _add(window, _make_block("Gain0", params={'gain': 1.0}))
        window.dsim.dirty = False
        window._on_property_changed("Gain0", "gain", "2.5")
        assert b.params['gain'] == 2.5
        assert isinstance(b.params['gain'], float)
        assert window.canvas.dsim.dirty is True

    def test_expression_string_preserved(self, window):
        b = _add(window, _make_block("Gain1", params={'gain': 1.0}))
        window._on_property_changed("Gain1", "gain", "2*K")
        assert b.params['gain'] == "2*K"

    def test_list_value_passthrough(self, window):
        b = _add(window, _make_block("Vec0", params={'vector': [0.0]}))
        window._on_property_changed("Vec0", "vector", [1, 2, 3])
        assert b.params['vector'] == [1, 2, 3]

    def test_username_change_marks_dirty_only(self, window):
        b = _add(window, _make_block("Blk0", params={'gain': 1.0}))
        window.canvas.dsim.dirty = False
        window._on_property_changed("Blk0", "_username_", "MyLabel")
        # Username is not a param; the handler just marks dirty and repaints.
        assert window.canvas.dsim.dirty is True
        assert 'gain' in b.params  # untouched

    def test_input_port_count_change(self, window):
        b = _add(window, _make_block("Sum0", block_fn='Sum',
                                     params={'gain': 1.0}, in_ports=1))
        window._on_property_changed("Sum0", "_inputs_", 3)
        assert b.in_ports == 3
        assert b.params['_inputs_'] == 3
        assert window.canvas.dsim.dirty is True

    def test_output_port_count_change(self, window):
        b = _add(window, _make_block("Dem0", block_fn='Demux',
                                     params={'gain': 1.0}, out_ports=1))
        window._on_property_changed("Dem0", "_outputs_", 4)
        assert b.out_ports == 4
        assert b.params['_outputs_'] == 4

    def test_unknown_block_name_is_noop(self, window):
        _add(window, _make_block("Real0", params={'gain': 1.0}))
        window.canvas.dsim.dirty = False
        # No block named "Ghost" -> loop finds nothing, no error, no dirty flip.
        window._on_property_changed("Ghost", "gain", "9.9")
        assert window.canvas.dsim.dirty is False


# ---------------------------------------------------------------------------
# _add_to_tuning
# ---------------------------------------------------------------------------

class TestAddToTuning:
    def test_forwards_to_tuning_panel(self, window, monkeypatch):
        captured = {}

        def fake_add_parameter(block, param_name):
            captured['block'] = block
            captured['param'] = param_name

        monkeypatch.setattr(window.tuning_panel, "add_parameter", fake_add_parameter)
        b = _make_block("Tun0", params={'gain': 1.0})
        window._add_to_tuning(b, "gain")
        assert captured == {'block': b, 'param': 'gain'}
