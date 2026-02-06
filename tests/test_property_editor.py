
import sys
import math
import unittest
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QCheckBox, QLineEdit, QComboBox, QLabel
from modern_ui.widgets.property_editor import PropertyEditor, CollapsibleSection, SliderSpinBox

# Mock Block class (minimal)
class MockBlock:
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.username = name

# Mock Block with full metadata (simulates a real DBlock)
class MockBlockWithMeta:
    def __init__(self, name, params, block_instance=None, block_fn=None,
                 category=None, color=None, doc=None):
        self.name = name
        self.params = params
        self.username = name
        self.block_instance = block_instance
        self.block_fn = block_fn or name
        self.category = category
        self.color = color
        self.doc = doc

class MockBlockInstance:
    """Simulates a block class instance with metadata-rich params."""
    def __init__(self, params_meta, inputs=None, outputs=None):
        self._params = params_meta
        self._inputs = inputs or [{"name": "in", "type": "any"}]
        self._outputs = outputs or [{"name": "out", "type": "any"}]

    @property
    def params(self):
        return self._params

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


class TestPropertyEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_float_param(self):
        block = MockBlock("Gain", {"gain": 1.5})
        self.editor.set_block(block)
        sb = self.editor.findChild(QDoubleSpinBox)
        self.assertIsNotNone(sb)
        self.assertAlmostEqual(sb.value(), 1.5)

    def test_int_param(self):
        block = MockBlock("Counter", {"count": 10})
        self.editor.set_block(block)
        sb = self.editor.findChild(QSpinBox)
        self.assertIsNotNone(sb)
        self.assertEqual(sb.value(), 10)

    def test_bool_param(self):
        block = MockBlock("Switch", {"enabled": True})
        self.editor.set_block(block)
        cb = self.editor.findChild(QCheckBox)
        self.assertIsNotNone(cb)
        self.assertTrue(cb.isChecked())

    def test_list_param(self):
        block = MockBlock("Mux", {"inputs": [1, 2, 3]})
        self.editor.set_block(block)
        line_edits = self.editor.findChildren(QLineEdit)
        self.assertGreaterEqual(len(line_edits), 2, "Expected at least 2 QLineEdits (Name + inputs)")
        le = line_edits[1]
        self.assertEqual(le.text(), "[1, 2, 3]")


class TestGenericChoices(unittest.TestCase):
    """Test #1: Generic choices support via param metadata."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_choices_creates_combobox(self):
        """Any param with 'choices' metadata should produce a QComboBox."""
        meta = {
            "verify_mode": {
                "type": "string", "default": "auto",
                "choices": ["auto", "objective", "comparison", "trajectory", "none"],
                "doc": "Verification mode"
            }
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("Scope1", {"verify_mode": "auto"},
                                  block_instance=inst, block_fn="Scope")
        self.editor.set_block(block)
        cb = self.editor.findChild(QComboBox)
        self.assertIsNotNone(cb, "Expected QComboBox for param with choices")
        self.assertEqual(cb.currentText(), "auto")
        self.assertEqual(cb.count(), 5)

    def test_method_param_still_works(self):
        """The 'method' param should still get a QComboBox."""
        meta = {
            "method": {
                "type": "string", "default": "SOLVE_IVP",
                "choices": ["FWD_EULER", "BWD_EULER", "TUSTIN", "RK45", "SOLVE_IVP"],
                "doc": "Integration method"
            }
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("Int1", {"method": "SOLVE_IVP"},
                                  block_instance=inst, block_fn="Integrator")
        self.editor.set_block(block)
        cb = self.editor.findChild(QComboBox)
        self.assertIsNotNone(cb)
        self.assertEqual(cb.currentText(), "SOLVE_IVP")


class TestParamTooltips(unittest.TestCase):
    """Test #2: Per-parameter doc tooltips."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_doc_shown_as_tooltip(self):
        meta = {
            "Kp": {"type": "float", "default": 1.0, "doc": "Proportional gain."}
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("PID1", {"Kp": 1.0}, block_instance=inst)
        self.editor.set_block(block)
        # The editor widget (SliderSpinBox or QDoubleSpinBox) should have the tooltip
        dsb = self.editor.findChild(QDoubleSpinBox)
        self.assertIsNotNone(dsb)
        self.assertIn("Proportional gain", dsb.toolTip())


class TestBlockHeader(unittest.TestCase):
    """Test #3: Block identity header."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_header_shows_block_type(self):
        inst = MockBlockInstance({"Kp": {"type": "float", "default": 1.0}},
                                 inputs=[{"name": "sp"}, {"name": "meas"}],
                                 outputs=[{"name": "u"}])
        block = MockBlockWithMeta("PID1", {"Kp": 1.0}, block_instance=inst,
                                  block_fn="PID", category="Control", color="magenta")
        self.editor.set_block(block)
        labels = self.editor.findChildren(QLabel)
        # Find the block type label (should contain "PID")
        type_labels = [l for l in labels if l.text() == "PID"]
        self.assertTrue(len(type_labels) > 0, "Expected label with block type name 'PID'")

    def test_header_shows_category(self):
        inst = MockBlockInstance({"Kp": {"type": "float", "default": 1.0}})
        block = MockBlockWithMeta("PID1", {"Kp": 1.0}, block_instance=inst,
                                  block_fn="PID", category="Control", color="magenta")
        self.editor.set_block(block)
        from PyQt5.QtWidgets import QLabel
        labels = self.editor.findChildren(QLabel)
        cat_labels = [l for l in labels if l.text() == "Control"]
        self.assertTrue(len(cat_labels) > 0, "Expected category badge label")

    def test_header_shows_ports(self):
        inst = MockBlockInstance({"Kp": {"type": "float", "default": 1.0}},
                                 inputs=[{"name": "sp"}, {"name": "meas"}],
                                 outputs=[{"name": "u"}])
        block = MockBlockWithMeta("PID1", {"Kp": 1.0}, block_instance=inst,
                                  block_fn="PID", category="Control", color="magenta")
        self.editor.set_block(block)
        from PyQt5.QtWidgets import QLabel
        labels = self.editor.findChildren(QLabel)
        port_labels = [l for l in labels if "\u2192" in l.text()]
        self.assertTrue(len(port_labels) > 0, "Expected port info label with arrow")
        self.assertIn("2 in", port_labels[0].text())
        self.assertIn("1 out", port_labels[0].text())


class TestCollapsibleSections(unittest.TestCase):
    """Test #4: Collapsible parameter groups."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_params_grouped_into_sections(self):
        meta = {
            "Kp": {"type": "float", "default": 1.0, "doc": "Proportional gain"},
            "u_min": {"type": "float", "default": float('-inf'), "doc": "Lower limit"},
            "u_max": {"type": "float", "default": float('inf'), "doc": "Upper limit"},
            "sampling_time": {"type": "float", "default": -1.0, "doc": "Sample time"},
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("PID1",
                                  {"Kp": 1.0, "u_min": float('-inf'),
                                   "u_max": float('inf'), "sampling_time": -1.0},
                                  block_instance=inst, block_fn="PID")
        self.editor.set_block(block)
        sections = self.editor.findChildren(CollapsibleSection)
        self.assertGreaterEqual(len(sections), 2, "Expected at least 2 sections")
        section_names = [s.toggle_btn.text().strip() for s in sections]
        self.assertIn("Parameters", section_names)
        # u_min/u_max should go to Limits, sampling_time to Advanced
        self.assertTrue("Limits" in section_names or "Advanced" in section_names)

    def test_section_toggle(self):
        section = CollapsibleSection("Test", expanded=True)
        # Use isHidden() since isVisible() checks parent chain (False if not on screen)
        self.assertFalse(section.content.isHidden())
        section._toggle(False)
        self.assertTrue(section.content.isHidden())
        section._toggle(True)
        self.assertFalse(section.content.isHidden())


class TestSliderSpinBox(unittest.TestCase):
    """Test #5: Slider + spinbox for bounded floats."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def test_slider_value_sync(self):
        ssb = SliderSpinBox(5.0, 0.0, 10.0)
        self.assertAlmostEqual(ssb.value(), 5.0, places=1)
        ssb.setValue(7.5)
        self.assertAlmostEqual(ssb.value(), 7.5, places=1)

    def test_slider_range(self):
        ssb = SliderSpinBox(2.0, 0.0, 20.0)
        self.assertEqual(ssb.slider.minimum(), 0)
        self.assertEqual(ssb.slider.maximum(), 1000)

    def test_positive_float_gets_slider(self):
        """Positive finite float params should get SliderSpinBox."""
        meta = {
            "Kp": {"type": "float", "default": 1.0, "doc": "Gain"}
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("G1", {"Kp": 1.0}, block_instance=inst)
        editor = PropertyEditor()
        editor.set_block(block)
        sliders = editor.findChildren(SliderSpinBox)
        self.assertEqual(len(sliders), 1, "Expected SliderSpinBox for positive float")

    def test_negative_float_no_slider(self):
        """Negative float params should get regular QDoubleSpinBox, no slider."""
        meta = {
            "sampling_time": {"type": "float", "default": -1.0, "doc": "Sample time"}
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("B1", {"sampling_time": -1.0}, block_instance=inst)
        editor = PropertyEditor()
        editor.set_block(block)
        sliders = editor.findChildren(SliderSpinBox)
        self.assertEqual(len(sliders), 0, "Negative float should not get slider")

    def test_inf_float_no_slider(self):
        """Infinite float params should get regular QDoubleSpinBox."""
        meta = {
            "u_max": {"type": "float", "default": float('inf'), "doc": "Upper limit"}
        }
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("B1", {"u_max": float('inf')}, block_instance=inst)
        editor = PropertyEditor()
        editor.set_block(block)
        sliders = editor.findChildren(SliderSpinBox)
        self.assertEqual(len(sliders), 0, "Infinite float should not get slider")


class TestResetToDefault(unittest.TestCase):
    """Test #6: Reset-to-default buttons."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_reset_button_hidden_at_default(self):
        meta = {"Kp": {"type": "float", "default": 1.0, "doc": "Gain"}}
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("G1", {"Kp": 1.0}, block_instance=inst)
        self.editor.set_block(block)
        _, reset_btn, _ = self.editor._widgets["Kp"]
        self.assertTrue(reset_btn.isHidden(), "Reset hidden when value == default")

    def test_reset_button_shown_when_different(self):
        meta = {"Kp": {"type": "float", "default": 1.0, "doc": "Gain"}}
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("G1", {"Kp": 5.0}, block_instance=inst)
        self.editor.set_block(block)
        _, reset_btn, _ = self.editor._widgets["Kp"]
        self.assertFalse(reset_btn.isHidden(), "Reset shown when value != default")

    def test_reset_restores_default(self):
        meta = {"Kp": {"type": "float", "default": 1.0, "doc": "Gain"}}
        inst = MockBlockInstance(meta)
        block = MockBlockWithMeta("G1", {"Kp": 5.0}, block_instance=inst)
        self.editor.set_block(block)
        self.editor._reset_param("Kp")
        editor_widget, reset_btn, _ = self.editor._widgets["Kp"]
        self.assertAlmostEqual(editor_widget.value(), 1.0, places=2)
        self.assertTrue(reset_btn.isHidden())


class TestValueDiffers(unittest.TestCase):
    """Test value comparison helper."""
    @classmethod
    def setUpClass(cls):
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_same_float(self):
        self.assertFalse(self.editor._value_differs(1.0, 1.0))

    def test_different_float(self):
        self.assertTrue(self.editor._value_differs(1.0, 2.0))

    def test_inf_same(self):
        self.assertFalse(self.editor._value_differs(float('inf'), float('inf')))

    def test_inf_different_sign(self):
        self.assertTrue(self.editor._value_differs(float('inf'), float('-inf')))

    def test_nan_same(self):
        self.assertFalse(self.editor._value_differs(float('nan'), float('nan')))


if __name__ == '__main__':
    unittest.main()
