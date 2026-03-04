"""
Unit tests for the blox exporter.

Uses the same SimpleNamespace mock pattern as test_tikz_exporter.py.
"""

import pytest
from types import SimpleNamespace

from lib.export.blox_exporter import BloxExporter


# ---------------------------------------------------------------------------
# Mock helpers (shared pattern with test_tikz_exporter.py)
# ---------------------------------------------------------------------------

def make_block(block_fn, sid=0, username='', category='Math',
               params=None, in_ports=1, out_ports=1, flipped=False,
               left=100, top=100, width=50, height=40):
    name = block_fn.lower() + str(sid)
    return SimpleNamespace(
        name=name,
        username=username or name,
        block_fn=block_fn,
        sid=sid,
        category=category,
        params=params or {},
        in_ports=in_ports,
        out_ports=out_ports,
        flipped=flipped,
        left=left, top=top, width=width, height=height,
    )


def make_line(srcblock, dstblock, srcport=0, dstport=0,
              hidden=False, label='', signal_width=1):
    return SimpleNamespace(
        srcblock=srcblock, srcport=srcport,
        dstblock=dstblock, dstport=dstport,
        hidden=hidden, label=label, signal_width=signal_width,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def simple_feedback_diagram():
    """Step -> Sum -> Gain -> TF -> Scope with TF->Sum feedback."""
    step = make_block('Step', sid=0, category='Sources', left=0)
    s = make_block('Sum', sid=1, category='Math',
                   params={'sign': '+-'}, in_ports=2, left=100)
    gain = make_block('Gain', sid=2, category='Math',
                      params={'gain': 2.0}, left=200)
    tf = make_block('TranFn', sid=3, category='Control',
                    params={'numerator': [1.0],
                            'denominator': [1.0, 1.0]},
                    left=300)
    scope = make_block('Scope', sid=4, category='Sinks', left=400)

    blocks = [step, s, gain, tf, scope]
    lines = [
        make_line(step.name, s.name, dstport=0),
        make_line(s.name, gain.name),
        make_line(gain.name, tf.name),
        make_line(tf.name, scope.name),
        make_line(tf.name, s.name, dstport=1),  # feedback
    ]
    return blocks, lines


def parallel_paths_diagram():
    """Diagram with parallel forward paths — should NOT be blox-compatible."""
    step = make_block('Step', sid=0, category='Sources', left=0)
    g1 = make_block('Gain', sid=1, category='Math', params={'gain': 2.0}, left=100)
    g2 = make_block('Gain', sid=2, category='Math', params={'gain': 3.0}, left=100)
    scope = make_block('Scope', sid=3, category='Sinks', left=200)

    blocks = [step, g1, g2, scope]
    lines = [
        make_line(step.name, g1.name),
        make_line(step.name, g2.name),  # parallel path
        make_line(g1.name, scope.name),
        make_line(g2.name, scope.name),
    ]
    return blocks, lines


# ---------------------------------------------------------------------------
# Tests: topology detection
# ---------------------------------------------------------------------------

class TestTopologyDetection:
    """Test can_export() for various diagram topologies."""

    def test_simple_feedback_is_compatible(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        assert exporter.can_export() is True

    def test_parallel_paths_not_compatible(self):
        blocks, lines = parallel_paths_diagram()
        exporter = BloxExporter(blocks, lines)
        assert exporter.can_export() is False

    def test_empty_diagram_not_compatible(self):
        exporter = BloxExporter([], [])
        assert exporter.can_export() is False

    def test_single_block_is_compatible(self):
        gain = make_block('Gain', sid=0, category='Math')
        exporter = BloxExporter([gain], [])
        assert exporter.can_export() is True

    def test_serial_chain_no_feedback(self):
        step = make_block('Step', sid=0, category='Sources', left=0)
        gain = make_block('Gain', sid=1, category='Math', left=100)
        scope = make_block('Scope', sid=2, category='Sinks', left=200)
        lines = [
            make_line(step.name, gain.name),
            make_line(gain.name, scope.name),
        ]
        exporter = BloxExporter([step, gain, scope], lines)
        assert exporter.can_export() is True


# ---------------------------------------------------------------------------
# Tests: export output
# ---------------------------------------------------------------------------

class TestBloxExport:
    """Test export() content for compatible diagrams."""

    def test_simple_feedback_exports(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export()
        assert result is not None
        assert r'\begin{tikzpicture}' in result
        assert r'\end{tikzpicture}' in result

    def test_contains_blox_macros(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export()
        assert r'\bXInput' in result
        assert r'\bXComp' in result or r'\bXSum' in result
        assert r'\bXBlocL' in result

    def test_standalone_has_documentclass(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export({'standalone': True})
        assert r'\documentclass' in result
        assert r'\usepackage{blox}' in result
        assert r'\begin{document}' in result
        assert r'\end{document}' in result

    def test_non_standalone_no_preamble(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export({'standalone': False})
        assert r'\documentclass' not in result
        assert r'\begin{tikzpicture}' in result

    def test_parallel_paths_returns_none(self):
        blocks, lines = parallel_paths_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export()
        assert result is None

    def test_feedback_emits_return(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export()
        assert r'\bXReturn' in result

    def test_gain_content_shows_value(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export({'show_values': True})
        assert '$2$' in result

    def test_tf_content_shows_fraction(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        result = exporter.export({'show_values': True})
        assert '\\dfrac' in result

    def test_get_info(self):
        blocks, lines = simple_feedback_diagram()
        exporter = BloxExporter(blocks, lines)
        info = exporter.get_info()
        assert info['block_count'] == 5
        assert info['connection_count'] == 5
