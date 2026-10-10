"""Analysis > Transfer Function Between Blocks (GUI path)."""

import os

import pytest


@pytest.fixture
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    w.resize(1200, 800)
    w.show()
    w.open_example(os.path.abspath("examples/pid_second_order.diablos"))
    qapp.processEvents()
    yield w
    for win in list(w.experiment_controller._result_windows):
        win.close()
    w.close()


def _select(window, *names):
    for b in window.dsim.blocks_list:
        b.selected = b.name in names


def _results(window):
    return window.experiment_controller._result_windows


@pytest.mark.qt
def test_step_to_scope_lists_one_tf_per_scope_input(window):
    from modern_ui.widgets.transfer_function_window import TransferFunctionWindow

    _select(window, "scope5", "step0")  # order of selection does not matter
    window.transfer_function_between_blocks()
    win = _results(window)[-1]
    assert isinstance(win, TransferFunctionWindow)
    assert win.headings == [
        "From reference to tracking: reference",
        "From reference to tracking: output",
        "From reference to tracking: control u",
    ]
    texts = [v.toPlainText() for v in win.text_views]
    assert len(texts) == 3  # reference, output, control u
    assert texts[0].startswith("G(s) = 1\n")
    output = texts[1]
    # PID (filtered D adds a pole, I adds one) around a 2nd-order plant.
    assert "s^4 +" in output.splitlines()[0]
    assert "DC gain: 1\n" in output  # integral action: zero steady-state error
    assert "Mason" in output and "L1 = " in output and "Stable: yes" in output
    assert "reference -> PID controller -> plant input -> plant  4/(s^2+0.8s+4)" in output


@pytest.mark.qt
def test_needs_two_blocks_and_reports_unsupported(window, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    before = len(_results(window))

    _select(window, "step0")
    window.transfer_function_between_blocks()
    assert "exactly two blocks" in shown[-1]

    # Two sources: neither feeds the other, in either direction.
    _select(window, "step0", "step2")
    window.transfer_function_between_blocks()
    assert "no signal path" in shown[-1]
    assert len(_results(window)) == before
