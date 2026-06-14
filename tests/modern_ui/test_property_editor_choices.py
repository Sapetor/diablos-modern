"""
Regression tests for enum-parameter rendering in the PropertyEditor.

Bug: the editor read only ``meta['choices']``, but most blocks declare their
enum values under ``meta['options']``. Those params silently degraded to a
free-text box (no validation). Both keys must now yield a QComboBox.
"""

import pytest

from PyQt5.QtWidgets import QComboBox, QLineEdit


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def editor():
    from modern_ui.widgets.property_editor import PropertyEditor
    return PropertyEditor()


class TestChoicesAndOptions:
    def test_options_key_yields_combobox(self, editor):
        meta = {"type": "choice", "default": "linear",
                "options": ["linear", "logarithmic", "quadratic"]}
        w = editor._create_editor_for_value("method", "linear", meta, "Parameters")
        assert isinstance(w, QComboBox)
        assert [w.itemText(i) for i in range(w.count())] == \
            ["linear", "logarithmic", "quadratic"]
        assert w.currentText() == "linear"

    def test_choices_key_still_yields_combobox(self, editor):
        meta = {"choices": ["a", "b", "c"]}
        w = editor._create_editor_for_value("mode", "b", meta, "Parameters")
        assert isinstance(w, QComboBox)
        assert w.currentText() == "b"

    def test_choices_takes_precedence_over_options(self, editor):
        # If a block ever declares both, 'choices' wins (it is the newer key).
        meta = {"choices": ["x", "y"], "options": ["1", "2", "3"]}
        w = editor._create_editor_for_value("k", "y", meta, "Parameters")
        assert isinstance(w, QComboBox)
        assert [w.itemText(i) for i in range(w.count())] == ["x", "y"]

    def test_string_without_enum_is_lineedit(self, editor):
        # A plain string param (no choices/options) must remain a free-text box.
        w = editor._create_editor_for_value("label", "hello", {}, "Parameters")
        assert isinstance(w, QLineEdit)


# ---------------------------------------------------------------------------
# Real blocks: every enum param must resolve to a combo box
# ---------------------------------------------------------------------------

_ENUM_BLOCKS = [
    ("blocks.chirp", "ChirpBlock"),
    ("blocks.wave_generator", "WaveGeneratorBlock"),
    ("blocks.logical_operator", "LogicalOperatorBlock"),
    ("blocks.relational_operator", "RelationalOperatorBlock"),
    ("blocks.compare_to_constant", "CompareToConstantBlock"),
]


class TestRealBlockEnums:
    @pytest.mark.parametrize("module_name,class_name", _ENUM_BLOCKS)
    def test_block_enum_params_yield_combobox(self, editor, module_name, class_name):
        import importlib
        mod = importlib.import_module(module_name)
        block = getattr(mod, class_name)()
        params = block.params

        enum_keys = [
            k for k, meta in params.items()
            if isinstance(meta, dict) and (meta.get("choices") or meta.get("options"))
        ]
        assert enum_keys, f"{class_name} declared no enum params to test"

        for key in enum_keys:
            meta = params[key]
            default = meta.get("default", (meta.get("choices") or meta.get("options"))[0])
            w = editor._create_editor_for_value(key, default, meta, "Parameters")
            assert isinstance(w, QComboBox), (
                f"{class_name}.{key} did not render as a QComboBox"
            )
