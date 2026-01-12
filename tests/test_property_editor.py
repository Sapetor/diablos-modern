
import sys
import unittest
from PyQt5.QtWidgets import QApplication
from modern_ui.widgets.property_editor import PropertyEditor

# Mock Block class
class MockBlock:
    def __init__(self, name, params):
        self.name = name
        self.params = params

class TestPropertyEditor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create application instance if not exists
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
        else:
            cls.app = QApplication.instance()

    def setUp(self):
        self.editor = PropertyEditor()

    def test_float_param(self):
        block = MockBlock("Gain", {"gain": 1.5})
        self.editor.set_block(block)
        # Find the QDoubleSpinBox
        from PyQt5.QtWidgets import QDoubleSpinBox
        sb = self.editor.findChild(QDoubleSpinBox)
        self.assertIsNotNone(sb)
        self.assertAlmostEqual(sb.value(), 1.5)
        print("Float param test passed")

    def test_int_param(self):
        block = MockBlock("Counter", {"count": 10})
        self.editor.set_block(block)
        from PyQt5.QtWidgets import QSpinBox
        sb = self.editor.findChild(QSpinBox)
        self.assertIsNotNone(sb)
        self.assertEqual(sb.value(), 10)
        print("Int param test passed")
        
    def test_bool_param(self):
        block = MockBlock("Switch", {"enabled": True})
        self.editor.set_block(block)
        from PyQt5.QtWidgets import QCheckBox
        cb = self.editor.findChild(QCheckBox)
        self.assertIsNotNone(cb)
        self.assertTrue(cb.isChecked())
        print("Bool param test passed")

    def test_list_param(self):
        block = MockBlock("Mux", {"inputs": [1, 2, 3]})
        self.editor.set_block(block)
        from PyQt5.QtWidgets import QLineEdit
        le = self.editor.findChild(QLineEdit)
        self.assertIsNotNone(le)
        self.assertEqual(le.text(), "[1, 2, 3]")
        print("List param test passed")

if __name__ == '__main__':
    unittest.main()
