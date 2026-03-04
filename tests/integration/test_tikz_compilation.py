"""
Integration tests that compile TikZ/blox output with pdflatex.

Skipped automatically when pdflatex is not available.
"""

import os
import shutil
import subprocess
import pytest
from types import SimpleNamespace

from lib.export.tikz_exporter import TikZExporter
from lib.export.blox_exporter import BloxExporter


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

PDFLATEX = shutil.which('pdflatex')
BLOX_INSTALLED = bool(
    subprocess.run(
        ['kpsewhich', 'blox.sty'],
        capture_output=True, timeout=10
    ).stdout.strip()
) if shutil.which('kpsewhich') else False

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not PDFLATEX, reason='pdflatex not available'),
]


# ---------------------------------------------------------------------------
# Mock helpers
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


def simple_feedback_diagram():
    """Step -> Sum -> Gain -> TF -> Scope with feedback."""
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
        make_line(tf.name, s.name, dstport=1),
    ]
    return blocks, lines


def serial_chain_diagram():
    """Step -> Gain -> TF -> Scope (no feedback)."""
    step = make_block('Step', sid=0, category='Sources', left=0)
    gain = make_block('Gain', sid=1, category='Math',
                      params={'gain': 5.0}, left=100)
    tf = make_block('TranFn', sid=2, category='Control',
                    params={'numerator': [1.0],
                            'denominator': [1.0, 2.0, 1.0]},
                    left=200)
    scope = make_block('Scope', sid=3, category='Sinks', left=300)

    blocks = [step, gain, tf, scope]
    lines = [
        make_line(step.name, gain.name),
        make_line(gain.name, tf.name),
        make_line(tf.name, scope.name),
    ]
    return blocks, lines


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def compile_tex(tex_content, tmp_path):
    """Compile a .tex string with pdflatex. Returns (success, log)."""
    tex_file = tmp_path / 'test.tex'
    tex_file.write_text(tex_content, encoding='utf-8')
    result = subprocess.run(
        [PDFLATEX, '-interaction=nonstopmode', '-halt-on-error', str(tex_file)],
        cwd=str(tmp_path),
        capture_output=True,
        timeout=30,
    )
    log = result.stdout.decode('utf-8', errors='replace')
    return result.returncode == 0, log


# ---------------------------------------------------------------------------
# Tests: TikZ compilation
# ---------------------------------------------------------------------------

class TestTikZCompilation:
    """Verify that TikZ export produces compilable LaTeX."""

    def test_feedback_diagram_compiles(self, tmp_path):
        blocks, lines = simple_feedback_diagram()
        tex = TikZExporter(blocks, lines).export_document()
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'

    def test_serial_chain_compiles(self, tmp_path):
        blocks, lines = serial_chain_diagram()
        tex = TikZExporter(blocks, lines).export_document()
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'

    def test_all_options_compile(self, tmp_path):
        blocks, lines = simple_feedback_diagram()
        tex = TikZExporter(blocks, lines).export_document({
            'source_as_arrow': True,
            'sink_as_arrow': True,
            'show_signal_labels': True,
            'show_usernames': True,
            'show_values': True,
            'fill_blocks': True,
            'use_resizebox': False,
        })
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'

    def test_no_options_compile(self, tmp_path):
        blocks, lines = simple_feedback_diagram()
        tex = TikZExporter(blocks, lines).export_document({
            'source_as_arrow': False,
            'sink_as_arrow': False,
            'show_signal_labels': False,
            'show_usernames': False,
            'show_values': False,
            'fill_blocks': False,
        })
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'


# ---------------------------------------------------------------------------
# Tests: blox compilation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not BLOX_INSTALLED, reason='blox.sty not installed')
class TestBloxCompilation:
    """Verify that blox export produces compilable LaTeX."""

    def test_feedback_diagram_compiles(self, tmp_path):
        blocks, lines = simple_feedback_diagram()
        tex = BloxExporter(blocks, lines).export({'standalone': True})
        assert tex is not None, 'BloxExporter returned None for feedback diagram'
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'

    def test_serial_chain_compiles(self, tmp_path):
        blocks, lines = serial_chain_diagram()
        tex = BloxExporter(blocks, lines).export({'standalone': True})
        assert tex is not None
        ok, log = compile_tex(tex, tmp_path)
        assert ok, f'pdflatex failed:\n{log[-2000:]}'
