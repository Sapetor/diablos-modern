"""
Tests for SignalPlot (lib/plotting/signal_plot.py).

Covers step-mode expansion, time-axis alignment, and the "Previous run"
overlay: dimmed dashed curves built from a held-run dict, hidden until the
checkbox is ticked, matched by label with index fallback, and re-penned
(still dimmed/dashed) on theme changes.

Run offscreen:
    $env:QT_QPA_PLATFORM="offscreen"
"""

import numpy as np
import pytest

from PyQt5.QtCore import Qt, QTimer

from lib.plotting.signal_plot import SignalPlot


def _previous_run(labels, vectors, step_modes=None, t=None):
    """Build a held-run dict in the format ScopePlotter._stash_run produces."""
    vectors = [np.asarray(v, dtype=float) for v in vectors]
    n = len(vectors[0]) if vectors else 0
    return {
        "sig": 12345,
        "t": np.linspace(0.0, 1.0, n) if t is None else np.asarray(t, dtype=float),
        "labels": list(labels),
        "vectors": vectors,
        "step_modes": list(step_modes) if step_modes is not None else [False] * len(vectors),
    }


@pytest.mark.unit
class TestStepModeExpansion:
    def test_bool_expands_to_all_plots(self, qapp):
        plot = SignalPlot(0.1, ["a", "b", "c"], 10, step_mode=True)
        assert plot.curve_step_modes == [True, True, True]
        plot.close()

    def test_short_list_is_padded_with_last_value(self, qapp):
        plot = SignalPlot(0.1, ["a", "b", "c"], 10, step_mode=[True])
        assert plot.curve_step_modes == [True, True, True]
        plot.close()

    def test_long_list_is_truncated(self, qapp):
        plot = SignalPlot(0.1, ["a"], 10, step_mode=[False, True, True])
        assert plot.curve_step_modes == [False]
        plot.close()


@pytest.mark.unit
class TestAlignTime:
    def test_equal_lengths_returned_unchanged(self, qapp):
        plot = SignalPlot(0.1, ["a"], 10)
        t = np.linspace(0.0, 0.4, 5)
        assert plot._align_time(t, 5, False) is t
        plot.close()

    def test_step_mode_pads_one_extra_point(self, qapp):
        plot = SignalPlot(0.1, ["a"], 10)
        t = np.linspace(0.0, 0.4, 5)
        aligned = plot._align_time(t, 5, True)
        assert len(aligned) == 6
        assert np.isclose(aligned[-1], t[-1] + plot.dt)
        plot.close()

    def test_longer_time_axis_is_truncated(self, qapp):
        plot = SignalPlot(0.1, ["a"], 10)
        t = np.linspace(0.0, 0.9, 10)
        assert len(plot._align_time(t, 4, False)) == 4
        assert len(plot._align_time(t, 4, True)) == 5
        plot.close()


@pytest.mark.unit
class TestPreviousRunOverlay:
    def test_no_held_data_disables_checkbox(self, qapp):
        plot = SignalPlot(0.1, ["a", "b"], 10)
        assert plot.prev_curves == [None, None]
        assert not plot.prev_run_checkbox.isChecked()
        assert not plot.prev_run_checkbox.isEnabled()
        plot.close()

    def test_overlay_curves_created_hidden_and_toggle(self, qapp):
        prev = _previous_run(["a", "b"], [np.arange(5.0), np.arange(5.0) * 2])
        plot = SignalPlot(0.1, ["a", "b"], 10, previous_run=prev)

        assert all(c is not None for c in plot.prev_curves)
        # Each subplot now has two data items: the live curve + the overlay
        for pw in plot.plot_items:
            assert len(pw.getPlotItem().listDataItems()) == 2
        # Hidden by default; checkbox unchecked but enabled
        assert all(not c.isVisible() for c in plot.prev_curves)
        assert not plot.prev_run_checkbox.isChecked()
        assert plot.prev_run_checkbox.isEnabled()

        plot.prev_run_checkbox.setChecked(True)
        assert all(c.isVisible() for c in plot.prev_curves)
        plot.prev_run_checkbox.setChecked(False)
        assert all(not c.isVisible() for c in plot.prev_curves)
        plot.close()

    def test_overlay_matches_by_label_before_index(self, qapp):
        y_a = np.arange(5.0)
        y_b = np.arange(5.0) * 10
        prev = _previous_run(["b", "a"], [y_b, y_a])
        plot = SignalPlot(0.1, ["a", "b"], 10, previous_run=prev)

        _, y0 = plot.prev_curves[0].getData()
        _, y1 = plot.prev_curves[1].getData()
        np.testing.assert_allclose(y0, y_a)
        np.testing.assert_allclose(y1, y_b)
        plot.close()

    def test_overlay_duplicate_labels_match_positionally(self, qapp):
        y0 = np.arange(5.0)
        y1 = np.arange(5.0) * 10
        prev = _previous_run(["out", "out"], [y0, y1])
        plot = SignalPlot(0.1, ["out", "out"], 10, previous_run=prev)

        # Both subplots share the label; each must get its own held trace,
        # not both the first occurrence.
        _, got0 = plot.prev_curves[0].getData()
        _, got1 = plot.prev_curves[1].getData()
        np.testing.assert_allclose(got0, y0)
        np.testing.assert_allclose(got1, y1)
        plot.close()

    def test_overlay_falls_back_to_index_and_ignores_missing(self, qapp):
        prev = _previous_run(["x", "y"], [np.arange(5.0), np.arange(5.0) * 2])
        plot = SignalPlot(0.1, ["a", "b", "c"], 10, previous_run=prev)

        # No label match: subplots 0/1 fall back to index, 2 has no held trace
        assert plot.prev_curves[0] is not None
        assert plot.prev_curves[1] is not None
        assert plot.prev_curves[2] is None
        _, y0 = plot.prev_curves[0].getData()
        np.testing.assert_allclose(y0, np.arange(5.0))
        plot.close()

    def test_overlay_respects_step_mode(self, qapp):
        prev = _previous_run(["a"], [np.arange(5.0)], step_modes=[True])
        plot = SignalPlot(0.1, ["a"], 10, previous_run=prev)

        x, y = plot.prev_curves[0].getData()
        assert len(x) == len(y) + 1
        plot.close()

    def test_overlay_drawn_behind_current_curve(self, qapp):
        prev = _previous_run(["a"], [np.arange(5.0)])
        plot = SignalPlot(0.1, ["a"], 10, previous_run=prev)
        assert plot.prev_curves[0].zValue() < plot.curves[0].zValue()
        plot.close()

    def test_theme_repaint_keeps_overlay_dimmed_and_dashed(self, qapp):
        prev = _previous_run(["a"], [np.arange(5.0)])
        plot = SignalPlot(0.1, ["a"], 10, previous_run=prev)

        plot._apply_theme()

        overlay_pen = plot.prev_curves[0].opts["pen"]
        assert overlay_pen.style() == Qt.DashLine
        assert overlay_pen.width() == 1
        assert overlay_pen.color().alpha() < 255
        # Current curve stays solid, full width and opaque
        current_pen = plot.curves[0].opts["pen"]
        assert current_pen.style() == Qt.SolidLine
        assert current_pen.width() == 2
        assert current_pen.color().alpha() == 255
        plot.close()

    def test_loop_updates_only_current_curves(self, qapp):
        prev = _previous_run(["a"], [np.arange(5.0)])
        plot = SignalPlot(0.1, ["a"], 10, previous_run=prev, step_mode=[True])

        t = np.linspace(0.0, 0.9, 10)
        y = np.arange(10.0).reshape(10, 1)  # (N, 1) is flattened
        plot.loop(t, [y])

        x_cur, y_cur = plot.curves[0].getData()
        assert len(x_cur) == len(y_cur) + 1  # step-mode alignment
        np.testing.assert_allclose(y_cur, np.arange(10.0))
        _, y_prev = plot.prev_curves[0].getData()
        np.testing.assert_allclose(y_prev, np.arange(5.0))  # untouched
        plot.close()


@pytest.mark.unit
class TestExportFigure:
    @staticmethod
    def _plot_with_data(qapp, step_mode=False):
        plot = SignalPlot(0.1, ["a", "b"], 10, step_mode=step_mode)
        t = np.linspace(0.0, 0.9, 10)
        plot.loop(t, [np.arange(10.0), np.arange(10.0).reshape(10, 1) * 2])
        return plot

    @staticmethod
    def _fake_save_dialog(monkeypatch, path, selected_filter="PDF (*.pdf)"):
        monkeypatch.setattr(
            "lib.plotting.signal_plot.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(path), selected_filter)),
        )

    def test_button_exists_next_to_csv_export(self, qapp):
        plot = SignalPlot(0.1, ["a"], 10)
        assert plot.export_figure_button.text() == "Export Figure..."
        assert plot.export_figure_button.parent() is plot.export_button.parent()
        plot.close()

    def test_no_data_skips_file_dialog(self, qapp, monkeypatch):
        plot = SignalPlot(0.1, ["a"], 10)

        def _fail(*_a, **_k):
            raise AssertionError("file dialog must not open without data")

        monkeypatch.setattr(
            "lib.plotting.signal_plot.QFileDialog.getSaveFileName", staticmethod(_fail)
        )
        plot.export_figure()  # warning box is neutralized by conftest
        plot.close()

    def test_exports_loadable_pdf(self, qapp, monkeypatch, tmp_path):
        plot = self._plot_with_data(qapp)
        path = tmp_path / "figure.pdf"
        self._fake_save_dialog(monkeypatch, path)

        plot.export_figure()

        assert path.exists()
        assert path.read_bytes()[:5] == b"%PDF-"
        # Success feedback mirrors the CSV export button
        assert "figure.pdf" in plot.export_figure_button.text()
        assert not plot.export_figure_button.isEnabled()
        plot.close()

    def test_exports_loadable_png(self, qapp, monkeypatch, tmp_path):
        plot = self._plot_with_data(qapp)
        path = tmp_path / "figure.png"
        self._fake_save_dialog(monkeypatch, path, selected_filter="PNG image (*.png)")

        plot.export_figure()

        assert path.exists()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        plot.close()

    def test_missing_extension_appended_from_selected_filter(self, qapp, monkeypatch, tmp_path):
        plot = self._plot_with_data(qapp)
        path = tmp_path / "figure"  # user typed no extension
        self._fake_save_dialog(monkeypatch, path, selected_filter="SVG (*.svg)")

        plot.export_figure()

        assert (tmp_path / "figure.svg").exists()
        plot.close()

    def test_export_success_shows_no_error_dialog(self, qapp, monkeypatch, tmp_path):
        # Regression: the success feedback once raised TypeError after saving
        # (PyQt5 lacks QTimer.singleShot's context-object overload), landing in
        # the except branch and showing "Export Failed" despite a written file.
        plot = self._plot_with_data(qapp)
        path = tmp_path / "figure.pdf"
        self._fake_save_dialog(monkeypatch, path)
        criticals = []
        monkeypatch.setattr(
            "lib.plotting.signal_plot.QMessageBox.critical",
            staticmethod(lambda *a, **k: criticals.append(a)),
        )

        plot.export_figure()

        assert path.exists()
        assert not criticals, "success path showed an error dialog"
        plot.close()

    def test_flash_restore_timer_is_button_parented_and_restores(self, qapp):
        plot = self._plot_with_data(qapp)
        btn = plot.export_figure_button
        original = btn.text()

        plot._flash_export_success(btn, "somewhere/fig.pdf")
        assert "fig.pdf" in btn.text()
        assert not btn.isEnabled()

        # Parented to the button (dies with the widget) and single-shot.
        timer = btn.findChild(QTimer)
        assert timer is not None
        assert timer.isSingleShot()

        timer.timeout.emit()  # fire the restore without waiting 3 s
        assert btn.text() == original
        assert btn.isEnabled()
        plot.close()

    def test_collect_figure_traces_flattens_and_keeps_step_flags(self, qapp):
        plot = self._plot_with_data(qapp, step_mode=[False, True])

        traces = plot._collect_figure_traces()

        assert [tr["name"] for tr in traces] == ["a", "b"]
        assert [tr["step"] for tr in traces] == [False, True]
        assert all(np.asarray(tr["y"]).ndim <= 2 for tr in traces)
        np.testing.assert_allclose(np.ravel(traces[1]["y"]), np.arange(10.0) * 2)
        plot.close()

    def test_collect_figure_traces_splits_list_labels(self, qapp):
        plot = SignalPlot(0.1, ["solo"], 10)
        plot.labels = [["s0", "s1"]]
        plot.timeline = np.linspace(0.0, 0.4, 5)
        plot.data_vectors = [np.column_stack([np.arange(5.0), np.arange(5.0) * 3])]

        traces = plot._collect_figure_traces()

        assert [tr["name"] for tr in traces] == ["s0", "s1"]
        np.testing.assert_allclose(traces[0]["y"], np.arange(5.0))
        np.testing.assert_allclose(traces[1]["y"], np.arange(5.0) * 3)
        plot.close()
