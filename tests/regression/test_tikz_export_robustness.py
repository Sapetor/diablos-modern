"""TikZ export defects found in the 2026-09-11 audit.

Each of these produced a .tex file the user only discovered was broken when
their paper failed to build -- or, worse, one that compiled into a wrong
picture. The existing suite missed them all because every fixture is a
hand-built mock; these drive the real exporter with adversarial names.
"""

import re

import pytest

from lib.export.tikz_exporter import TikZExporter, _escape_latex, _math_body_is_safe

# The exporter's mock factories already model every attribute it reads; a second
# set here would drift the next time the exporter grows one.
from tests.unit.test_tikz_exporter import make_block, make_line

pytestmark = pytest.mark.regression


def _Block(name, block_fn="TranFn", username=None):
    return make_block(block_fn, username=username if username is not None else name)


def _Line(src, dst, label=""):
    return make_line(src, dst, label=label)


def _export(blocks, lines):
    return TikZExporter(blocks, lines).export_document()


class TestEscaping:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # A bare $ opens math mode and breaks the whole document.
            ("cost $5", r"cost \$5"),
            # <, > and | compile, but render as the OT1 ligatures.
            ("Tools > LQR", r"Tools \ensuremath{>} LQR"),
            ("T < 5", r"T \ensuremath{<} 5"),
            ("a|b", r"a\ensuremath{|}b"),
        ],
    )
    def test_dangerous_characters_are_escaped(self, raw, expected):
        assert _escape_latex(raw) == expected

    def test_no_bare_dollar_survives(self):
        """A $ must always come out backslash-escaped, never raw."""
        escaped = _escape_latex("cost $5 per m^3")
        assert not re.search(r"(?<!\\)\$", escaped)

    def test_escaped_name_reaches_the_document_safely(self):
        blocks = [_Block("plant", username="cost $5 per m^3")]
        out = _export(blocks, [])
        assert r"cost \$5 per m\textasciicircum{}3" in out


class TestReservedNodeNames:
    @pytest.mark.parametrize("reserved", ["output", "bpt"])
    def test_a_block_cannot_claim_an_exporter_node_name(self, reserved):
        """Claiming it redefined the coordinate, silently misrouting wires."""
        blocks = [_Block("g", block_fn="Gain", username=reserved), _Block("p")]
        lines = [_Line("g", "p")]
        out = _export(blocks, lines)
        # The block must have been renamed away from the reserved id.
        assert not re.search(r"\\node\[[^]]*\]\s*\(" + reserved + r"\)", out)
        assert f"({reserved}_" in out


class TestMathLabelValidation:
    @pytest.mark.parametrize(
        "body", [r"\dfrac{1}{s+1}", r"\dot{x} = Ax + Bu", r"\alpha_1", "y", r"\frac{K}{s}"]
    )
    def test_real_math_is_still_passed_through(self, body):
        assert _math_body_is_safe(body) is True

    @pytest.mark.parametrize(
        "body",
        [
            r"y$ \immediate\write18{echo pwned} $z",
            r"\input{/etc/passwd}",
            r"\def\x{bad}",
            r"\csname foo\endcsname",
            r"\openout1=x",
        ],
    )
    def test_io_and_redefinition_commands_are_rejected(self, body):
        assert _math_body_is_safe(body) is False

    def test_injected_label_is_escaped_not_executed(self):
        evil = r"$y$ \immediate\write18{echo pwned} $z$"
        blocks = [_Block("a"), _Block("b")]
        out = _export(blocks, [_Line("a", "b", label=evil)])
        assert r"\immediate" not in out
        assert r"\textbackslash{}immediate" in out


class TestSnippetCarriesItsDependencies:
    def test_snippet_emits_the_tikz_libraries_it_needs(self):
        """calc / shapes.geometric / arrows.meta / positioning are all used."""
        snippet = TikZExporter([_Block("a")], []).export_snippet()
        assert r"\usetikzlibrary{" in snippet
        for lib in ("calc", "shapes.geometric", "arrows.meta", "positioning"):
            assert lib in snippet
        assert r"\documentclass" not in snippet

    def test_document_does_not_duplicate_the_library_line(self):
        out = _export([_Block("a")], [])
        assert out.count(r"\usetikzlibrary{") == 1


class TestStylesAreScopedToThePicture:
    """A fragment must not redefine styles in the document it is pasted into.

    The style names are generic (block, sum, signal, tf...), so a document-scope
    \\tikzset silently replaced a paper's own definitions from the point of the
    import onwards -- verified by rendering: the host's red star node became a
    DiaBloS blue rectangle.
    """

    def test_no_document_scope_tikzset(self):
        snippet = TikZExporter([_Block("a")], []).export_snippet()
        assert r"\tikzset{" not in snippet

    def test_styles_ride_in_the_picture_options(self):
        snippet = TikZExporter([_Block("a")], []).export_snippet()
        head = snippet.split(r"\end{tikzpicture}")[0]
        opening = head.index(r"\begin{tikzpicture}[")
        for style in ("block/.style", "sum/.style", "signal/.style", "tf/.style"):
            assert head.index(style) > opening, f"{style} escaped the picture scope"

    def test_two_figures_can_coexist(self):
        """Both fragments carry their own styles, so neither wins globally."""
        one = TikZExporter([_Block("a")], []).export_snippet({"fill_blocks": True})
        two = TikZExporter([_Block("b")], []).export_snippet({"fill_blocks": False})
        assert r"\tikzset{" not in one + two
        assert "fill=blue!5" in one
        assert "fill=blue!5" not in two


class TestNodeTextIsAlwaysPlaceable:
    @pytest.mark.parametrize("style", ["block/.style", "tf/.style", "source/.style"])
    def test_rectangular_styles_set_align(self, style):
        r"""Without align=, a node containing \\ fails to typeset."""
        snippet = TikZExporter([_Block("a")], []).export_snippet()
        body = snippet.split(style)[1].split("},")[0]
        assert "align=center" in body

    def test_newlines_in_a_name_are_collapsed(self):
        """A blank line inside \\node{} ends the paragraph mid-node."""
        assert _escape_latex("two\n\nlines") == "two lines"
        assert "\n" not in _escape_latex("a\tb\nc")


class TestMathGateIsStructural:
    """Delimiters alone do not make a body safe.

    A body can close the group and keep going, after which a command blocklist
    is decoration -- the attacker just picks a command that is not on it.
    """

    @pytest.mark.parametrize(
        "body",
        [
            r"x$} \renewcommand{\alpha}{pwned} \node{$z",  # escapes the node
            r"a} \node{b",  # unbalanced, closes the group
            r"a{b",  # unbalanced the other way
            r"a$b",  # leaves math mode
            r"\newcommand{\x}{y}",  # redefines for the rest of the document
            r"\renewcommand{\alpha}{pwned}",
        ],
    )
    def test_structural_escapes_are_rejected(self, body):
        assert _math_body_is_safe(body) is False

    @pytest.mark.parametrize(
        "body", [r"\dfrac{1}{s+1}", r"\dot{x} = Ax + Bu", r"K_{p} + \frac{K_i}{s}"]
    )
    def test_balanced_self_contained_math_still_passes(self, body):
        assert _math_body_is_safe(body) is True


class TestEscapingWorksInBothModes:
    r"""``_escape_latex`` feeds text mode AND contexts already in math mode.

    ``_name_to_math`` wraps its result in ``\text{}`` inside ``$...$``, and
    ``BloxExporter._latex_label`` wraps in ``$...$``, so a literal ``$<$``
    switched *out* of math there and reintroduced the ligature it was meant to
    fix. ``\ensuremath`` is correct in both.
    """

    @pytest.mark.parametrize("char", ["<", ">", "|"])
    def test_no_bare_dollar_in_the_replacement(self, char):
        assert "$" not in _escape_latex(char)

    def test_blox_applies_the_same_label_gate(self):
        """Both exporters are reachable from the same dialog."""
        from lib.export.blox_exporter import BloxExporter

        evil = r"$\immediate\write18{echo pwned}$"
        assert r"\immediate" not in BloxExporter._latex_label(evil)
        assert BloxExporter._latex_label(r"$\dfrac{1}{s+1}$") == r"$\dfrac{1}{s+1}$"
