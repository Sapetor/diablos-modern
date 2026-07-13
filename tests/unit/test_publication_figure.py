"""
Tests for build_publication_figure (lib/plotting/publication_figure.py).

Pure matplotlib (no Qt / no qapp): the module builds a Figure directly, so
these tests run headless under MPLBACKEND=Agg (forced by tests/conftest.py).
"""

import numpy as np
import pytest

from lib.plotting.publication_figure import build_publication_figure


def _traces():
    t = np.linspace(0.0, 1.0, 50)
    return t, [
        {"name": "sine", "y": np.sin(t), "step": False},
        {"name": "square", "y": (t > 0.5).astype(float), "step": True},
    ]


@pytest.mark.unit
class TestBuildPublicationFigure:
    def test_single_axes_with_one_line_per_trace(self):
        t, traces = _traces()
        fig = build_publication_figure(t, traces)
        assert len(fig.axes) == 1
        assert len(fig.axes[0].get_lines()) == 2

    def test_axis_text_labels_and_legend(self):
        t, traces = _traces()
        fig = build_publication_figure(t, traces, {"title": "Demo"})
        ax = fig.axes[0]
        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_title() == "Demo"
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert legend_texts == ["sine", "square"]

    def test_step_traces_use_steps_post_drawstyle(self):
        t, traces = _traces()
        fig = build_publication_figure(t, traces)
        lines = fig.axes[0].get_lines()
        assert lines[0].get_drawstyle() == "default"
        assert lines[1].get_drawstyle() == "steps-post"

    def test_serif_labels(self):
        t, traces = _traces()
        fig = build_publication_figure(t, traces)
        ax = fig.axes[0]
        assert ax.xaxis.get_label().get_fontfamily() == ["serif"]

    def test_length_mismatch_truncates_to_shorter(self):
        t = np.linspace(0.0, 1.0, 10)
        fig = build_publication_figure(
            t,
            [
                {"name": "short_y", "y": np.arange(6.0), "step": False},
                {"name": "long_y", "y": np.arange(15.0), "step": False},
            ],
        )
        lines = fig.axes[0].get_lines()
        assert len(lines[0].get_xdata()) == 6
        assert len(lines[1].get_xdata()) == 10

    def test_column_vector_is_flattened(self):
        t = np.linspace(0.0, 1.0, 8)
        y = np.arange(8.0).reshape(8, 1)
        fig = build_publication_figure(t, [{"name": "col", "y": y, "step": False}])
        ydata = fig.axes[0].get_lines()[0].get_ydata()
        assert np.asarray(ydata).ndim == 1
        np.testing.assert_allclose(ydata, np.arange(8.0))

    @pytest.mark.parametrize("ext", ["pdf", "png", "svg"])
    def test_savefig_writes_nonempty_file(self, tmp_path, ext):
        t, traces = _traces()
        fig = build_publication_figure(t, traces)
        path = tmp_path / f"figure.{ext}"
        fig.savefig(str(path), dpi=300 if ext == "png" else None)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_options_figsize_and_dpi(self):
        t, traces = _traces()
        fig = build_publication_figure(t, traces, {"figsize": (4.0, 3.0), "dpi": 150})
        assert tuple(fig.get_size_inches()) == (4.0, 3.0)
        assert fig.dpi == 150
