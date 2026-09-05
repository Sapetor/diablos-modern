"""The control analyzers report errors through an injected GUI callback.

Two audit findings are covered:

1. ``lib/analysis/analyzers/lqr.py`` and ``root_locus.py`` built a
   ``QMessageBox`` themselves. ``lib/`` is the engine layer -- it must stay
   headless-safe and must not import the widget toolkit -- so the GUI now
   injects a ``(title, message) -> None`` sink (``ControlSystemAnalyzer(
   error_cb=...)``, supplied by ``ModernCanvas``) and the analyzers only log
   when there is none.

2. ``BaseAnalyzer`` records *why* a subsystem could not be reduced to an LTI
   model in ``last_error`` (branching, internal feedback, mixed rates, a
   non-LTI block), but the analyzers still showed the generic "no valid linear
   system found". The reason is now part of the user-facing message.
"""

import types

import pytest

from lib.analysis.analyzers.bode import BodeAnalyzer
from lib.analysis.analyzers.error_reporting import report_error
from lib.analysis.analyzers.lqr import LQRAnalyzer
from lib.analysis.analyzers.nyquist import NyquistAnalyzer
from lib.analysis.analyzers.root_locus import RootLocusAnalyzer

ANALYZERS = (BodeAnalyzer, NyquistAnalyzer, RootLocusAnalyzer, LQRAnalyzer)


class Recorder:
    """Stand-in for the GUI's message-box callback."""

    def __init__(self):
        self.calls = []

    def __call__(self, title, message):
        self.calls.append((title, message))


class _PlainBlock:
    """Hashable stand-in for a DBlock with nothing linear to extract."""

    def __init__(self, name="scope1", block_fn="Scope"):
        self.name = name
        self.block_fn = block_fn
        self.params = {}


def _canvas_without_lti_source():
    """A canvas whose only block has no transfer function to extract."""
    block = _PlainBlock()
    model = types.SimpleNamespace(line_list=[], get_block_by_name=lambda name: None)
    return block, types.SimpleNamespace(dsim=types.SimpleNamespace(model=model))


@pytest.mark.unit
class TestNoQtImportsInAnalysis:
    def test_analysis_package_does_not_import_qmessagebox(self):
        """No module under lib/analysis may import or construct a QMessageBox."""
        import pathlib
        import re

        import lib.analysis

        # Imports and constructions -- prose mentions in docstrings are fine.
        pattern = re.compile(r"import[^\n]*QMessageBox|QMessageBox\s*\(|QMessageBox\.")
        analysis_dir = pathlib.Path(lib.analysis.__file__).parent
        offenders = [
            str(path)
            for path in analysis_dir.rglob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []


@pytest.mark.unit
class TestReportError:
    def test_uses_the_callback_when_one_is_injected(self):
        recorder = Recorder()
        assert report_error(recorder, "T", "M") is True
        assert recorder.calls == [("T", "M")]

    def test_falls_back_to_logging_without_a_callback(self, caplog):
        with caplog.at_level("ERROR"):
            assert report_error(None, "T", "M") is False
        assert "M" in caplog.text

    def test_a_broken_callback_does_not_hide_the_error(self, caplog):
        def boom(title, message):
            raise RuntimeError("gui is gone")

        with caplog.at_level("ERROR"):
            assert report_error(boom, "T", "M") is False
        assert "M" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("analyzer_cls", ANALYZERS)
class TestInjectedCallback:
    def test_constructor_accepts_error_cb(self, analyzer_cls):
        recorder = Recorder()
        analyzer = analyzer_cls(None, error_cb=recorder)
        analyzer._show_error("something went wrong")
        assert recorder.calls == [(analyzer_cls.error_title, "something went wrong")]

    def test_defaults_to_logging(self, analyzer_cls, caplog):
        analyzer = analyzer_cls()
        with caplog.at_level("ERROR"):
            analyzer._show_error("something went wrong")
        assert "something went wrong" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("analyzer_cls", [BodeAnalyzer, NyquistAnalyzer, RootLocusAnalyzer])
class TestLastErrorReachesTheUser:
    def test_specific_reason_is_surfaced(self, analyzer_cls):
        recorder = Recorder()
        analyzer = analyzer_cls(None, error_cb=recorder)
        analyzer.last_error = "subsystem 'plant' does not reduce to a single signal path"
        analyzer._report_no_model("a Bode plot")

        assert len(recorder.calls) == 1
        _, message = recorder.calls[0]
        assert "does not reduce to a single signal path" in message

    def test_generic_message_when_no_reason_was_recorded(self, analyzer_cls):
        recorder = Recorder()
        analyzer = analyzer_cls(None, error_cb=recorder)
        analyzer.last_error = None
        analyzer._report_no_model("a Bode plot")

        _, message = recorder.calls[0]
        assert "no linear" in message.lower()

    def test_analyze_reports_instead_of_failing_silently(self, analyzer_cls):
        """A diagram with nothing linear upstream must tell the user why."""
        recorder = Recorder()
        analyzer = analyzer_cls(None, error_cb=recorder)
        block, canvas = _canvas_without_lti_source()

        assert analyzer.analyze(block, canvas) is None
        assert len(recorder.calls) == 1


@pytest.mark.unit
class TestFacadeForwardsTheCallback:
    def test_control_system_analyzer_passes_error_cb_to_each_analyzer(self):
        from lib.analysis.control_system_analyzer import ControlSystemAnalyzer

        recorder = Recorder()
        facade = ControlSystemAnalyzer(canvas=None, parent=None, error_cb=recorder)
        for name in ("bode_analyzer", "nyquist_analyzer", "root_locus_analyzer", "lqr_analyzer"):
            assert getattr(facade, name).error_cb is recorder

    def test_error_cb_is_optional(self):
        from lib.analysis.control_system_analyzer import ControlSystemAnalyzer

        facade = ControlSystemAnalyzer(canvas=None)
        assert facade.bode_analyzer.error_cb is None


@pytest.mark.unit
class TestLqrWorkspaceLookup:
    def test_matrix_resolves_through_workspace_get_value(self):
        """_resolve_matrix used the removed get_variable(), silently failing."""
        import numpy as np

        from lib.workspace import WorkspaceManager

        ws = WorkspaceManager()
        ws.set_variable("Amat", [[0.0, 1.0], [-2.0, -3.0]])
        try:
            resolved = LQRAnalyzer()._resolve_matrix("Amat", canvas=None)
            assert np.allclose(resolved, [[0.0, 1.0], [-2.0, -3.0]])
        finally:
            ws.delete_variable("Amat")
