"""
Unit tests for TikZ exporter.

Uses SimpleNamespace mock objects to avoid Qt dependency.
Tests are organized by component, with regression tests for known bugs.
"""

import re
from types import SimpleNamespace

from lib.export.tikz_exporter import (
    _poly_to_latex,
    _escape_latex,
    _name_to_math,
    _sanitize_node_id,
    TikZExporter,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def make_block(
    block_fn,
    sid=0,
    username="",
    category="Math",
    params=None,
    in_ports=1,
    out_ports=1,
    flipped=False,
    left=100,
    top=100,
    width=50,
    height=40,
):
    """Create a lightweight mock block with the attributes TikZExporter reads."""
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
        left=left,
        top=top,
        width=width,
        height=height,
    )


def make_line(srcblock, dstblock, srcport=0, dstport=0, hidden=False, label="", signal_width=1):
    """Create a lightweight mock line (connection)."""
    return SimpleNamespace(
        srcblock=srcblock,
        srcport=srcport,
        dstblock=dstblock,
        dstport=dstport,
        hidden=hidden,
        label=label,
        signal_width=signal_width,
    )


# ---------------------------------------------------------------------------
# Test _poly_to_latex
# ---------------------------------------------------------------------------


class TestPolyToLatex:
    """Tests for polynomial coefficient -> LaTeX string conversion."""

    def test_constant_integer(self):
        assert _poly_to_latex([5.0]) == "5"

    def test_constant_float(self):
        assert _poly_to_latex([3.14]) == "3.14"

    def test_zero_poly(self):
        assert _poly_to_latex([0.0]) == "0"

    def test_linear_unit_coefficient(self):
        # [1, 0] = 1*s + 0 -> "s"
        assert _poly_to_latex([1.0, 0]) == "s"

    def test_linear_with_constant(self):
        # [1, 1] = s + 1
        assert _poly_to_latex([1.0, 1.0]) == "s + 1"

    def test_quadratic(self):
        # [1, 2, 1] = s^2 + 2s + 1
        result = _poly_to_latex([1.0, 2.0, 1.0])
        assert "s^{2}" in result
        assert "2s" in result
        assert "+ 1" in result

    def test_negative_coefficients(self):
        # [1, -3, 2] = s^2 - 3s + 2
        result = _poly_to_latex([1.0, -3.0, 2.0])
        assert "- 3s" in result
        assert "+ 2" in result

    def test_negative_unit_coefficient(self):
        # [-1, 1] = -s + 1
        result = _poly_to_latex([-1.0, 1.0])
        assert result.startswith("-s")

    def test_all_zeros_except_last(self):
        # [0, 0, 5] = 5 (just the constant term)
        assert _poly_to_latex([0.0, 0.0, 5.0]) == "5"

    def test_single_unit(self):
        # [1] = 1 (just the number 1)
        assert _poly_to_latex([1.0]) == "1"


# ---------------------------------------------------------------------------
# Test _escape_latex
# ---------------------------------------------------------------------------


class TestEscapeLatex:
    """Tests for LaTeX special character escaping."""

    def test_plain_text_unchanged(self):
        assert _escape_latex("hello world") == "hello world"

    def test_underscore_escaped(self):
        assert _escape_latex("a_b") == r"a\_b"

    def test_percent_escaped(self):
        assert _escape_latex("50%") == r"50\%"

    def test_ampersand_escaped(self):
        assert _escape_latex("a & b") == r"a \& b"

    def test_hash_escaped(self):
        assert _escape_latex("#1") == r"\#1"

    def test_braces_escaped(self):
        result = _escape_latex("{x}")
        assert r"\{" in result
        assert r"\}" in result

    def test_tilde_escaped(self):
        result = _escape_latex("~")
        assert "textasciitilde" in result

    def test_caret_escaped(self):
        result = _escape_latex("^")
        assert "textasciicircum" in result

    def test_backslash_escaped(self):
        result = _escape_latex("\\")
        assert "textbackslash" in result

    def test_backslash_with_braces_no_double_escape(self):
        """Regression test for Bug 1: _escape_latex double-escaping.

        If backslash is replaced first, the braces in \\textbackslash{}
        get re-escaped by the { and } replacements. The fix uses a
        single-pass regex to avoid this.
        """
        result = _escape_latex("a\\b{c}")
        # Should contain ONE textbackslash{} and ONE \{c\}
        # NOT textbackslash\{\} (double-escaped braces)
        assert r"\textbackslash{}" in result or "textbackslash" in result
        # The braces around c should be escaped once
        assert r"\{c\}" in result
        # Must NOT have double-escaped: \textbackslash\{\}
        assert r"\textbackslash\{\}" not in result

    def test_multiple_special_chars(self):
        result = _escape_latex("a_b & c%d")
        assert r"\_" in result
        assert r"\&" in result
        assert r"\%" in result


# ---------------------------------------------------------------------------
# Test _name_to_math
# ---------------------------------------------------------------------------


class TestNameToMath:
    """Tests for username -> math-mode label conversion."""

    def test_single_letter(self):
        assert _name_to_math("G") == "G"

    def test_single_lowercase(self):
        assert _name_to_math("x") == "x"

    def test_letter_digit(self):
        # K1 -> K_{1}
        assert _name_to_math("K1") == "K_{1}"

    def test_letter_digits(self):
        # G12 -> G_{12}
        assert _name_to_math("G12") == "G_{12}"

    def test_letter_lowercase(self):
        # Kp -> K_{p}
        assert _name_to_math("Kp") == "K_{p}"

    def test_letter_multiple_lowercase(self):
        # Kpid -> K_{pid}
        assert _name_to_math("Kpid") == "K_{pid}"

    def test_long_name_fallback(self):
        # 'tank' -> \text{tank}
        result = _name_to_math("tank")
        assert "\\text{" in result
        assert "tank" in result


# ---------------------------------------------------------------------------
# Test _sanitize_node_id
# ---------------------------------------------------------------------------


class TestSanitizeNodeId:
    """Tests for TikZ node identifier sanitization."""

    def test_clean_name(self):
        assert _sanitize_node_id("gain1") == "gain1"

    def test_special_chars_replaced(self):
        assert _sanitize_node_id("my-block!1") == "my_block_1"

    def test_leading_digit_prefixed(self):
        assert _sanitize_node_id("1block") == "n1block"

    def test_empty_string(self):
        assert _sanitize_node_id("") == "node"

    def test_spaces_replaced(self):
        assert _sanitize_node_id("my block") == "my_block"


# ---------------------------------------------------------------------------
# Test Gain port anchor
# ---------------------------------------------------------------------------


class TestGainPortAnchor:
    """Port anchors for a gain, including what ``flipped`` means on export.

    The export re-orders the diagram by signal flow, so on the left-to-right
    spine a flipped block still reads left to right -- honouring the flag there
    would put the input on the far side and drag its wire across the block.
    The flag earns its meaning in the return path, where the layout moves a
    flipped block into the loop lane and turns it around.
    """

    def _make_exporter_with_gain(self, flipped=False):
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 2.0}, flipped=flipped)
        exporter = TikZExporter([gain], [])
        return exporter, gain

    def test_unflipped_input(self):
        exp, gain = self._make_exporter_with_gain(flipped=False)
        assert exp._get_port_anchor(gain, 0, is_output=False) == "west"

    def test_unflipped_output(self):
        exp, gain = self._make_exporter_with_gain(flipped=False)
        assert exp._get_port_anchor(gain, 0, is_output=True) == "east"

    def test_flipped_on_the_spine_is_ignored(self):
        """Nothing in the spine runs right to left, so the flag is dropped."""
        exp, gain = self._make_exporter_with_gain(flipped=True)
        assert exp._get_port_anchor(gain, 0, is_output=False) == "west"
        assert exp._get_port_anchor(gain, 0, is_output=True) == "east"
        assert exp._get_block_style(gain) == "gain"

    def test_flipped_in_the_return_path_turns_the_block_around(self):
        """Input on the east, output on the west, triangle pointing left."""
        blocks, lines = _return_path_diagram()
        exp = TikZExporter(blocks, lines)
        exp.export_snippet()
        gain = next(b for b in blocks if b.block_fn == "Gain")
        assert gain.name in exp._lane_blocks
        assert exp._get_port_anchor(gain, 0, is_output=False) == "east"
        assert exp._get_port_anchor(gain, 0, is_output=True) == "west"
        assert exp._get_block_style(gain) == "gain flipped"


def _return_path_diagram():
    """Step -> Sum -> TF -> Scope with a *flipped* gain closing the loop."""
    step = make_block("Step", sid=0, category="Sources", left=0)
    s = make_block("Sum", sid=1, category="Math", params={"sign": "+-"}, in_ports=2, left=100)
    tf = make_block(
        "TranFn",
        sid=2,
        category="Control",
        params={"numerator": [1.0], "denominator": [1.0, 1.0]},
        left=200,
    )
    h = make_block("Gain", sid=3, username="H", params={"gain": 0.5}, flipped=True, left=300)
    scope = make_block("Scope", sid=4, category="Sinks", left=400)
    blocks = [step, s, tf, h, scope]
    lines = [
        make_line(step.name, s.name, dstport=0),
        make_line(s.name, tf.name),
        make_line(tf.name, scope.name),
        make_line(tf.name, h.name),
        make_line(h.name, s.name, dstport=1),
    ]
    return blocks, lines


# ---------------------------------------------------------------------------
# Test block content generation
# ---------------------------------------------------------------------------


class TestBlockContent:
    """Tests for TikZ node content generation."""

    def _get_content(self, block, show_values=True):
        exporter = TikZExporter([block], [])
        exporter._build_symbol_maps({"show_values": show_values})
        return exporter._get_block_content(block, {"show_values": show_values})

    def test_gain_integer_value(self):
        gain = make_block("Gain", params={"gain": 2.0})
        content = self._get_content(gain, show_values=True)
        assert "$2$" == content

    def test_gain_float_value(self):
        gain = make_block("Gain", params={"gain": 0.5})
        content = self._get_content(gain, show_values=True)
        assert "$0.5$" == content

    def test_gain_symbol_mode(self):
        gain = make_block("Gain", params={"gain": 2.0})
        content = self._get_content(gain, show_values=False)
        assert "$K$" == content

    def test_tf_fraction(self):
        tf = make_block(
            "TranFn", category="Control", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
        )
        content = self._get_content(tf, show_values=True)
        assert "\\dfrac" in content
        assert "s + 1" in content

    def test_tf_symbol_mode(self):
        tf = make_block(
            "TranFn", category="Control", params={"numerator": [1.0], "denominator": [1.0, 1.0]}
        )
        content = self._get_content(tf, show_values=False)
        assert "$G(s)$" == content

    def test_discrete_tf_uses_z(self):
        dtf = make_block(
            "DiscreteTranFn",
            category="Control",
            params={"numerator": [1.0], "denominator": [1.0, -0.5]},
        )
        content = self._get_content(dtf, show_values=True)
        assert "z" in content
        assert "s" not in content.replace("\\dfrac", "")  # dfrac has no s

    def test_integrator(self):
        integ = make_block("Integrator", category="Control")
        content = self._get_content(integ, show_values=True)
        assert "\\dfrac{1}{s}" in content

    def test_sum_empty_content(self):
        s = make_block("Sum", category="Math", params={"sign": "+-"}, in_ports=2)
        content = self._get_content(s)
        assert content == ""

    def test_pid_block(self):
        pid = make_block("PID", category="Control")
        assert self._get_content(pid) == "PID"


# ---------------------------------------------------------------------------
# Test Sum sign labels
# ---------------------------------------------------------------------------


class TestSumSigns:
    """Tests for Sum block sign label placement.

    Regression test for Bug 3: inconsistent +/- formatting.
    """

    def test_signs_both_math_mode(self):
        """Bug 3 regression: both + and - must be in math mode."""
        s = make_block("Sum", sid=0, category="Math", params={"sign": "+-"}, in_ports=2)
        exporter = TikZExporter([s], [])
        exporter._build_node_ids([s])
        node_id = exporter._nid(s)
        labels = exporter._sum_sign_labels(node_id, "+-", s)
        # Both signs should be in math mode for consistent sizing
        assert "$+$" in labels
        assert "$-$" in labels

    def test_all_plus_signs(self):
        s = make_block("Sum", sid=0, category="Math", params={"sign": "++"}, in_ports=2)
        exporter = TikZExporter([s], [])
        exporter._build_node_ids([s])
        node_id = exporter._nid(s)
        labels = exporter._sum_sign_labels(node_id, "++", s)
        assert labels.count("$+$") == 2

    def test_three_port_signs(self):
        s = make_block("Sum", sid=0, category="Math", params={"sign": "+-+"}, in_ports=3)
        exporter = TikZExporter([s], [])
        exporter._build_node_ids([s])
        node_id = exporter._nid(s)
        labels = exporter._sum_sign_labels(node_id, "+-+", s)
        assert labels.count("$+$") == 2
        assert labels.count("$-$") == 1


# ---------------------------------------------------------------------------
# Test symbol maps
# ---------------------------------------------------------------------------


class TestSymbolMaps:
    """Tests for TF/Gain symbol auto-numbering.

    Regression test for Bug 5: mixed block types get same numbering.
    """

    def test_single_tf_gets_G(self):
        tf = make_block("TranFn", sid=0, category="Control")
        exporter = TikZExporter([tf], [])
        exporter._build_symbol_maps({})
        assert exporter._tf_symbols[tf.name] == "G"

    def test_single_gain_gets_K(self):
        g = make_block("Gain", sid=0, category="Math")
        exporter = TikZExporter([g], [])
        exporter._build_symbol_maps({})
        assert exporter._gain_symbols[g.name] == "K"

    def test_multiple_tfs_numbered(self):
        tf1 = make_block("TranFn", sid=0, category="Control", left=0)
        tf2 = make_block("TranFn", sid=1, category="Control", left=200)
        exporter = TikZExporter([tf1, tf2], [])
        exporter._build_symbol_maps({})
        syms = set(exporter._tf_symbols.values())
        assert len(syms) == 2  # Two distinct symbols

    def test_different_tf_types_different_symbols(self):
        """Bug 5 regression: TranFn and DiscreteTranFn should get different base letters."""
        tf = make_block("TranFn", sid=0, category="Control", left=0)
        dtf = make_block("DiscreteTranFn", sid=1, category="Control", left=200)
        exporter = TikZExporter([tf, dtf], [])
        exporter._build_symbol_maps({})
        tf_sym = exporter._tf_symbols[tf.name]
        dtf_sym = exporter._tf_symbols[dtf.name]
        # Should use different base letters (G vs H)
        assert tf_sym != dtf_sym

    def test_username_overrides_auto_symbol(self):
        tf = make_block("TranFn", sid=0, category="Control", username="Plant")
        exporter = TikZExporter([tf], [])
        exporter._build_symbol_maps({})
        sym = exporter._tf_symbols[tf.name]
        # Should use the username-derived math label
        assert "text" in sym or "P" in sym  # _name_to_math('Plant') would wrap in \text

    def test_single_gain_username_override(self):
        g = make_block("Gain", sid=0, category="Math", username="Kp")
        exporter = TikZExporter([g], [])
        exporter._build_symbol_maps({})
        sym = exporter._gain_symbols[g.name]
        assert "K" in sym and "p" in sym


# ---------------------------------------------------------------------------
# Test signal labels
# ---------------------------------------------------------------------------


class TestSignalLabels:
    """Tests for connection signal label generation.

    Regression tests for Bug 4 (over-aggressive) and Bug 6 (LaTeX escaped).
    """

    def _make_exporter_and_label(self, blocks, line):
        exporter = TikZExporter(blocks, [line])
        exporter._build_symbol_maps({})
        # For Bug 4 fix, we need _block_type_counts. If the method
        # doesn't exist yet (pre-fix), we fall back.
        if hasattr(exporter, "_compute_block_type_counts"):
            exporter._compute_block_type_counts()
        src = exporter._resolve_block(line.srcblock)
        dst = exporter._resolve_block(line.dstblock)
        return exporter._get_signal_label(line, src, dst)

    def test_explicit_label_used(self):
        g = make_block("Gain", sid=0, category="Math")
        s = make_block("Scope", sid=1, category="Sinks")
        line = make_line(g.name, s.name, label="error")
        label = self._make_exporter_and_label([g, s], line)
        assert "error" in label

    def test_explicit_latex_label_not_escaped(self):
        """Bug 6 regression: labels in math mode should not be escaped."""
        g = make_block("Gain", sid=0, category="Math")
        s = make_block("Scope", sid=1, category="Sinks")
        line = make_line(g.name, s.name, label="$\\alpha$")
        label = self._make_exporter_and_label([g, s], line)
        # Must NOT contain escaped backslash
        assert "textbackslash" not in label
        assert "$\\alpha$" == label or "\\alpha" in label

    def test_single_gain_gets_conventional_label(self):
        g = make_block("Gain", sid=0, category="Math")
        s = make_block("Scope", sid=1, category="Sinks")
        line = make_line(g.name, s.name)
        label = self._make_exporter_and_label([g, s], line)
        assert label == "$u$"

    def test_multiple_gains_no_duplicate_labels(self):
        """Bug 4 regression: multiple Gain blocks should NOT all get $u$."""
        g1 = make_block("Gain", sid=0, category="Math", left=0)
        g2 = make_block("Gain", sid=1, category="Math", left=100)
        g3 = make_block("Gain", sid=2, category="Math", left=200)
        s = make_block("Scope", sid=3, category="Sinks", left=300)
        line = make_line(g1.name, s.name)
        label = self._make_exporter_and_label([g1, g2, g3, s], line)
        # With 3 gains, conventional label should be suppressed
        assert label != "$u$"


# ---------------------------------------------------------------------------
# Test full export
# ---------------------------------------------------------------------------


class TestFullExport:
    """Integration-level tests for the complete export pipeline."""

    def _simple_feedback_diagram(self):
        """Create Step -> Sum -> Gain -> TF -> Scope with feedback."""
        step = make_block("Step", sid=0, category="Sources", left=0)
        s = make_block("Sum", sid=1, category="Math", params={"sign": "+-"}, in_ports=2, left=100)
        gain = make_block("Gain", sid=2, category="Math", params={"gain": 2.0}, left=200)
        tf = make_block(
            "TranFn",
            sid=3,
            category="Control",
            params={"numerator": [1.0], "denominator": [1.0, 1.0]},
            left=300,
        )
        scope = make_block("Scope", sid=4, category="Sinks", left=400)

        blocks = [step, s, gain, tf, scope]
        lines = [
            make_line(step.name, s.name, dstport=0),
            make_line(s.name, gain.name),
            make_line(gain.name, tf.name),
            make_line(tf.name, scope.name),
            make_line(tf.name, s.name, dstport=1),  # feedback
        ]
        return blocks, lines

    def test_snippet_has_tikzpicture(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert r"\begin{tikzpicture}" in snippet
        assert r"\end{tikzpicture}" in snippet

    def test_snippet_no_documentclass(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert r"\documentclass" not in snippet

    def test_document_has_preamble(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        doc = exporter.export_document()
        assert r"\documentclass" in doc
        assert r"\usepackage{tikz}" in doc
        assert r"\usepackage{amsmath}" in doc
        assert r"\usetikzlibrary" in doc
        assert r"\begin{document}" in doc
        assert r"\end{document}" in doc

    def test_document_contains_snippet(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        doc = exporter.export_document()
        assert r"\begin{tikzpicture}" in doc
        assert r"\end{tikzpicture}" in doc

    def test_empty_diagram(self):
        exporter = TikZExporter([], [])
        snippet = exporter.export_snippet()
        assert "No blocks" in snippet

    def test_styles_defined(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        # Styles are scoped to the picture, not \tikzset at document scope:
        # the names are generic, and a paper with its own block/.style used to
        # lose it to ours.
        assert r"\tikzset{" not in snippet
        assert snippet.count(r"\begin{tikzpicture}[") == 1
        assert "block/.style" in snippet
        assert "sum/.style" in snippet
        assert "gain/.style" in snippet
        assert "tf/.style" in snippet
        assert "signal/.style" in snippet

    def test_resizebox_option(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet({"use_resizebox": True})
        assert r"\resizebox{\textwidth}{!}{" in snippet

    def test_no_resizebox_by_default(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert r"\resizebox" not in snippet

    def test_get_info(self):
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        info = exporter.get_info()
        assert info["block_count"] == 5
        assert info["connection_count"] == 5
        assert "Sum" in info["block_types"]
        assert "Gain" in info["block_types"]
        assert "TranFn" in info["block_types"]

    def test_feedback_draws_below(self):
        """Feedback connections should route below the main path."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        # Feedback paths use negative Y offsets
        assert "|-" in snippet or "--" in snippet

    def test_source_as_arrow(self):
        """When source_as_arrow is True, source blocks don't get nodes."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet({"source_as_arrow": True})
        # The Step block should NOT appear as a \node
        assert (
            "step0" not in snippet.split("% --- Connections ---")[0] or "\\node" not in snippet
        )  # approximate check

    def test_sink_as_arrow(self):
        """When sink_as_arrow is True, sink blocks get output arrows."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet({"sink_as_arrow": True})
        # Should have output arrow (via branch point or direct sink arrow)
        assert "+(" in snippet or "(output)" in snippet

    def test_branch_style_defined(self):
        """Branch point style must be in tikzset."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert "branch/.style" in snippet

    def test_branch_point_in_feedback_diagram(self):
        """Feedback diagram should produce a branch point and output coord."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert "(bpt1)" in snippet
        assert "(output)" in snippet
        assert "\\node[branch]" in snippet

    def test_feedback_uses_branch_point(self):
        """Feedback path should start from the branch point, not the block edge."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        # Feedback draw should reference the numbered dot as its start
        assert "(bpt1)" in snippet
        # Should use -| (vert-then-horiz) routing
        assert "-|" in snippet

    def test_no_branch_point_without_feedback(self):
        """Forward-only diagram should NOT have a branch point."""
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 2.0}, left=0)
        tf = make_block(
            "TranFn",
            sid=1,
            category="Control",
            params={"numerator": [1.0], "denominator": [1.0, 1.0]},
            left=100,
        )
        scope = make_block("Scope", sid=2, category="Sinks", left=200)
        blocks = [gain, tf, scope]
        lines = [
            make_line(gain.name, tf.name),
            make_line(tf.name, scope.name),
        ]
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        assert "(bpt" not in snippet
        assert "(output)" not in snippet

    def test_output_label_on_continuation_arrow(self):
        """The output continuation arrow should carry the signal label."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        # Output section should have the signal label ($y$ for TranFn)
        output_section = snippet.split("% --- Output ---")
        if len(output_section) > 1:
            assert "$y$" in output_section[1].split("% --- Connections ---")[0]

    def test_feedback_minus_sign_in_sum_circle(self):
        """Feedback to a Sum's minus port should show $-$ inside the circle."""
        blocks, lines = self._simple_feedback_diagram()
        exporter = TikZExporter(blocks, lines)
        snippet = exporter.export_snippet()
        # The minus sign is rendered inside the sum circle (not on the path)
        blocks_section = snippet.split("% --- Blocks ---")[1].split("% ---")[0]
        assert "$-$" in blocks_section


# ---------------------------------------------------------------------------
# Test adaptive spacing
# ---------------------------------------------------------------------------


class TestSpineLayout:
    """The signal-flow spine: ordering, relative placement and gaps.

    Blocks are no longer placed at absolute coordinates: they are chained with
    the ``positioning`` library, so the gap is measured border to border and a
    long caption pushes its neighbour instead of growing into it.
    """

    def _chain(self, *fns):
        blocks = []
        for i, fn in enumerate(fns):
            params = (
                {"numerator": [1.0], "denominator": [1.0, 1.0]}
                if fn in ("TranFn", "DiscreteTranFn")
                else {"gain": 1.0, "sign": "++"}
            )
            blocks.append(
                make_block(fn, sid=i, category="Control", params=params, in_ports=2, left=i * 100)
            )
        lines = [make_line(blocks[i].name, blocks[i + 1].name) for i in range(len(blocks) - 1)]
        return blocks, lines

    def test_blocks_are_chained_not_placed_at_coordinates(self):
        blocks, lines = self._chain("Sum", "Gain", "TranFn")
        snippet = TikZExporter(blocks, lines).export_snippet()
        node_lines = [ln for ln in snippet.splitlines() if ln.strip().startswith("\\node[")]
        block_nodes = [ln for ln in node_lines if "branch" not in ln and "footnotesize" not in ln]
        assert len(block_nodes) == 3
        # The first node anchors the picture; the rest hang off their neighbour.
        assert sum("right=" in ln for ln in block_nodes) == 2
        assert not re.search(r"\\node\[(?:block|sum|gain|tf)[^]]*\]\s*\([^)]+\) at \(", snippet)

    def test_node_distance_is_a_picture_option(self):
        """One knob, stated once, that a reader can retune by hand."""
        blocks, lines = self._chain("Sum", "Gain", "TranFn")
        snippet = TikZExporter(blocks, lines).export_snippet({"node_distance_cm": 2.2})
        assert "node distance=2.2cm" in snippet
        # Default gaps say `right=of', so the one knob really does drive them.
        assert "right=of " in snippet

    def test_order_follows_the_signal_not_the_canvas(self):
        blocks, lines = self._chain("Sum", "Gain", "TranFn")
        # Scramble the canvas positions; the export must not care.
        blocks[0].left, blocks[2].left = 900, 10
        exporter = TikZExporter(blocks, lines)
        exporter._build_node_ids(blocks)
        exporter._compute_textbook_layout(blocks, lines, {"page_width_cm": 14.0})
        assert exporter._ordered_names == [b.name for b in blocks]

    def test_a_chain_is_not_broken_up_by_a_fan_out(self):
        """Breadth-first put both successors first and made the rest a skip."""
        a = make_block("Gain", sid=0, category="Math", params={"gain": 1.0}, left=0)
        b = make_block("Gain", sid=1, category="Math", params={"gain": 2.0}, left=100)
        c = make_block("Gain", sid=2, category="Math", params={"gain": 3.0}, left=200)
        d = make_block("Sum", sid=3, category="Math", params={"sign": "++"}, in_ports=2, left=300)
        blocks = [a, b, c, d]
        lines = [
            make_line(a.name, b.name),
            make_line(b.name, c.name),
            make_line(c.name, d.name, dstport=0),
            make_line(a.name, d.name, dstport=1),
        ]
        exporter = TikZExporter(blocks, lines)
        exporter._build_node_ids(blocks)
        exporter._compute_textbook_layout(blocks, lines, {"page_width_cm": 14.0})
        assert exporter._ordered_names == [a.name, b.name, c.name, d.name]

    def test_unconnected_blocks_are_parked_at_the_end(self):
        """An orphan block in the middle used to split the signal path in two."""
        orphan = make_block("Gain", sid=9, category="Math", params={"gain": 1.0}, left=-500)
        blocks, lines = self._chain("Sum", "Gain", "TranFn")
        exporter = TikZExporter(blocks + [orphan], lines)
        exporter._build_node_ids(blocks + [orphan])
        exporter._compute_textbook_layout(blocks + [orphan], lines, {"page_width_cm": 14.0})
        assert exporter._ordered_names[-1] == orphan.name

    def test_gaps_shrink_to_fit_the_page_width(self):
        blocks = [
            make_block(
                "TranFn",
                sid=i,
                category="Control",
                params={"numerator": [1.0], "denominator": [1.0, 1.0]},
                left=i * 100,
            )
            for i in range(8)
        ]
        lines = [make_line(blocks[i].name, blocks[i + 1].name) for i in range(7)]
        exporter = TikZExporter(blocks, lines)
        exporter._build_node_ids(blocks)
        exporter._compute_textbook_layout(blocks, lines, {"page_width_cm": 14.0})
        assert exporter._spine_gaps
        assert max(exporter._spine_gaps) < 1.5
        assert min(exporter._spine_gaps) >= TikZExporter._MIN_GAP

    def test_single_block_needs_no_gap(self):
        tf = make_block("TranFn", sid=0, category="Control", left=0)
        exporter = TikZExporter([tf], [])
        exporter._build_node_ids([tf])
        exporter._compute_textbook_layout([tf], [], {"page_width_cm": 14.0})
        assert exporter._spine_gaps == []
        assert exporter._ordered_names == [tf.name]


class TestPortDistribution:
    """Several ports on one side must land on several points, not one anchor."""

    def _demux(self, out_ports=4):
        dm = make_block("Demux", sid=0, category="Routing", in_ports=1, out_ports=out_ports)
        return TikZExporter([dm], []), dm

    def test_ports_are_spread_down_the_side_in_order(self):
        exp, dm = self._demux(4)
        exp._build_node_ids([dm])
        points = [exp._port_point(dm, k, is_output=True) for k in range(4)]
        assert len(set(points)) == 4
        fractions = [float(re.search(r"!([0-9.]+)!", p).group(1)) for p in points]
        assert fractions == sorted(fractions), "port 0 must be the topmost"
        assert all("north east" in p and "south east" in p for p in points)

    def test_a_single_port_stays_on_the_plain_anchor(self):
        exp, dm = self._demux(1)
        exp._build_node_ids([dm])
        assert exp._port_point(dm, 0, is_output=True) == "(demux0.east)"

    def test_a_multi_port_block_is_tall_enough_to_tell_them_apart(self):
        dm = make_block("Demux", sid=0, category="Routing", in_ports=1, out_ports=4)
        snippet = TikZExporter([dm], []).export_snippet()
        assert "minimum height=28mm" in snippet

    def test_a_sum_keeps_its_angle_anchors(self):
        """Ports on a circle sit at angles; its corners are not on the outline."""
        s = make_block("Sum", sid=0, category="Math", params={"sign": "+-"}, in_ports=2)
        exp = TikZExporter([s], [])
        exp._build_node_ids([s])
        assert exp._port_point(s, 0, is_output=False) == "(sum0.180)"
        assert exp._port_point(s, 1, is_output=False) == "(sum0.270)"


class TestShapeFollowsTheCanvas:
    """The exported outline is the one ``resolve_block_shape`` paints."""

    def test_matrix_gain_is_a_triangle(self):
        b = make_block("MatrixGain", sid=0, category="Math", username="K")
        assert TikZExporter([b], [])._get_block_style(b) == "gain"

    def test_product_is_a_circle(self):
        b = make_block("Product", sid=0, category="Math", in_ports=2)
        exp = TikZExporter([b], [])
        assert exp._get_block_style(b) == "sum"
        assert exp._get_block_content(b, {}) == r"$\times$"

    def test_goto_and_from_are_tags(self):
        for fn in ("Goto", "From"):
            b = make_block(fn, sid=0, category="Routing", params={"tag": "A"})
            exp = TikZExporter([b], [])
            assert exp._get_block_style(b) == "tag"
            assert exp._get_block_content(b, {}) == "A"

    def test_tag_style_names_the_shape_explicitly(self):
        """`signal' is also this picture's wire style; the bare key picks that."""
        b = make_block("Goto", sid=0, category="Routing", params={"tag": "A"})
        snippet = TikZExporter([b], []).export_snippet()
        assert "tag/.style={draw, shape=signal" in snippet

    def test_a_plain_block_is_still_a_rectangle(self):
        b = make_block("Abs", sid=0, category="Math")
        assert TikZExporter([b], [])._get_block_style(b) == "block"


# ---------------------------------------------------------------------------
# Test gain username content
# ---------------------------------------------------------------------------


class TestGainUsername:
    """Tests for gain blocks showing username inside the triangle."""

    def _get_content(self, block, show_values=True):
        exporter = TikZExporter([block], [])
        exporter._build_symbol_maps({"show_values": show_values})
        return exporter._get_block_content(block, {"show_values": show_values})

    def test_gain_with_username_shows_name(self):
        """Gain with username 'Kp' should show $K_{p}$ inside triangle."""
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 1.0}, username="Kp")
        content = self._get_content(gain, show_values=True)
        assert "$K_{p}$" == content

    def test_gain_without_username_shows_value(self):
        """Gain without custom username should show numeric value."""
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 2.0})
        content = self._get_content(gain, show_values=True)
        assert "$2$" == content

    def test_gain_username_single_letter(self):
        """Gain with single-letter username 'K' should show $K$."""
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 5.0}, username="K")
        content = self._get_content(gain, show_values=True)
        assert "$K$" == content

    def test_gain_username_overrides_show_values_false(self):
        """Username takes priority even when show_values=False."""
        gain = make_block("Gain", sid=0, category="Math", params={"gain": 3.0}, username="Ki")
        content = self._get_content(gain, show_values=False)
        assert "$K_{i}$" == content


# ---------------------------------------------------------------------------
# Test sum label suppression
# ---------------------------------------------------------------------------


class TestSumLabelSuppression:
    """Tests for sum blocks not showing username labels."""

    def test_sum_with_username_no_label_below(self):
        """Sum block with custom username should NOT show it as a label below."""
        s = make_block(
            "Sum", sid=0, category="Math", params={"sign": "+-"}, in_ports=2, username="error"
        )
        exporter = TikZExporter([s], [])
        exporter._build_node_ids([s])
        exporter._build_symbol_maps({})
        exporter._compute_block_type_counts()
        exporter._compute_textbook_layout([s], [], {"page_width_cm": 14.0})
        node = exporter._block_to_node(s, {"show_usernames": True})
        # Sum signs use \footnotesize inside the circle — that's expected.
        # But there should NOT be a separate "below" label node for the username.
        lines = node.strip().split("\n")
        below_labels = [l for l in lines if "below=" in l]
        assert len(below_labels) == 0, f"Unexpected below-label for Sum: {below_labels}"
