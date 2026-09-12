r"""The math-label gate is an allowlist, not a blocklist.

A blocklist of dangerous TeX commands cannot be completed: ``\csname
input\endcsname`` and the ``^^5c`` byte notation both spell ``\input`` without
the word ever appearing, and ``\begin{filecontents}`` writes a file with no
"command" in sight. The set of commands a signal label legitimately needs is
small and known, so that is what the gate checks. Rejected labels are not
lost -- the exporters fall back to escaping them as text.
"""

import pytest

from lib.export.tex_safety import (
    SAFE_MATH_COMMANDS,
    SAFE_MATH_ENVIRONMENTS,
    math_body_is_safe,
)

pytestmark = pytest.mark.regression


class TestBlocklistBypassesAreRejected:
    @pytest.mark.parametrize(
        "body",
        [
            r"\csname input\endcsname{/etc/passwd}",  # spells \input indirectly
            r"^^5cinput{x}",  # ^^5c is a backslash
            r"\begin{filecontents}{evil.tex}x\end{filecontents}",  # writes a file
            r"\begin{ filecontents* }x\end{filecontents*}",
            r"\immediate\write18{echo pwned}",
            r"\input{x}",
            r"\newcommand{\x}{y}",
            r"\catcode`\@=0",
            r"\directlua{os.execute('id')}",
            r"\pwn",  # anything unknown, however innocent-looking
            r"\usetikzlibrary{external}",
            # structure: leaves math mode / unbalanced
            r"x$} \renewcommand{\alpha}{pwned} \node{$z",
            r"a} \node{b",
            r"a{b",
            r"a$b",
        ],
    )
    def test_rejected(self, body):
        assert math_body_is_safe(body) is False


class TestRealLabelsPass:
    @pytest.mark.parametrize(
        "body",
        [
            r"\dfrac{1}{s+1}",
            r"\dot{x} = Ax + Bu",
            r"K_{p} + \frac{K_i}{s}",
            r"\|x - x^*\|_2",
            r"\hat{\theta}(t)",
            r"e^{-\tau s}",
            r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}",
            r"\mathbf{u}_{\mathrm{ref}}",
            r"\sum_{i=1}^{N} \alpha_i \, y_i",
            r"\left\lVert \tilde{x} \right\rVert \le \varepsilon",
            r"u \to \infty",
            r"\text{error}",
            r"\operatorname{sat}(u)",
            r"x^{+}",  # no commands at all
            r"",
        ],
    )
    def test_accepted(self, body):
        assert math_body_is_safe(body) is True


class TestAllowlistHygiene:
    """Nothing that reads, writes, redefines or executes may be on the list."""

    FORBIDDEN = {
        "input", "include", "openin", "openout", "read", "write", "immediate",
        "special", "shipout", "directlua", "catcode", "csname", "def", "edef",
        "gdef", "xdef", "let", "newcommand", "renewcommand", "providecommand",
        "declarerobustcommand", "usepackage", "usetikzlibrary", "loop",
        "expandafter", "newread", "newwrite", "makeatletter", "futurelet",
        "uppercase", "lowercase", "scantokens", "jobname", "pdfobj",
    }  # fmt: skip

    def test_no_forbidden_command_is_allowed(self):
        assert not (self.FORBIDDEN & SAFE_MATH_COMMANDS)

    def test_filecontents_is_not_an_allowed_environment(self):
        assert "filecontents" not in SAFE_MATH_ENVIRONMENTS
        assert "filecontents*" not in SAFE_MATH_ENVIRONMENTS
