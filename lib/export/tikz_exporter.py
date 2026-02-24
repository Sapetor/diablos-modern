"""
TikZ Exporter - Export block diagrams to TikZ code for LaTeX documents.

Generates publication-ready TikZ diagrams using standard control-systems
conventions: Sum=circle, Gain=triangle, TF=rectangle with fraction.
"""

import math
import re
import logging
from collections import Counter
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def _poly_to_latex(coeffs):
    """Convert polynomial coefficients to LaTeX string (highest power first)."""
    terms = []
    n = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        power = n - i
        if abs(c) < 1e-10:
            continue
        if abs(c - 1.0) < 1e-10 and power > 0:
            coef_str = ""
        elif abs(c + 1.0) < 1e-10 and power > 0:
            coef_str = "-"
        else:
            coef_str = f"{c:.4g}"
        if power == 0:
            term = coef_str if coef_str else "1"
        elif power == 1:
            term = f"{coef_str}s"
        else:
            term = f"{coef_str}s^{{{power}}}"
        terms.append(term)
    if not terms:
        return "0"
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" {term}"
        else:
            result += f" + {term}"
    return result


_ESCAPE_MAP = {
    '\\': r'\textbackslash{}',
    '{': r'\{',
    '}': r'\}',
    '_': r'\_',
    '&': r'\&',
    '%': r'\%',
    '#': r'\#',
    '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}',
}
_ESCAPE_RE = re.compile(r'[\\{}_&%#~^]')


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text (single-pass to avoid double-escaping)."""
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group()], text)


def _name_to_math(name: str) -> str:
    """Convert a block username to a math-mode label.

    Examples: 'Kp' -> 'K_p', 'tank' -> 'G_{\\text{tank}}',
    'K1' -> 'K_1', 'G' -> 'G(s)'.
    """
    # Already a single letter: return as-is
    if len(name) == 1 and name.isalpha():
        return name
    # Pattern: letter + digits (K1 -> K_1)
    m = re.match(r'^([A-Za-z])(\d+)$', name)
    if m:
        return f'{m.group(1)}_{{{m.group(2)}}}'
    # Pattern: letter + lowercase letters (Kp -> K_p, Ki -> K_i)
    m = re.match(r'^([A-Z])([a-z]+)$', name)
    if m:
        return f'{m.group(1)}_{{{m.group(2)}}}'
    # Fallback: wrap in \text
    return f'\\text{{{_escape_latex(name)}}}'


def _sanitize_node_id(name: str) -> str:
    """Convert block name/username to a valid TikZ node identifier."""
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'n' + sanitized
    return sanitized or 'node'


class TikZExporter:
    """Exports a DiaBloS block diagram to TikZ code."""

    def __init__(self, blocks_list, line_list):
        self.blocks = blocks_list
        self.lines = line_list
        self._block_map = {b.name: b for b in blocks_list}
        self._username_map = {b.username: b for b in blocks_list}
        self._node_ids: Dict[str, str] = {}

    def _build_node_ids(self, blocks):
        """Assign unique TikZ node IDs to all blocks, avoiding collisions."""
        self._node_ids = {}
        used: set = set()
        for b in blocks:
            raw = b.username if b.username != b.name else b.name
            base = _sanitize_node_id(raw)
            node_id = base
            counter = 2
            while node_id in used:
                node_id = f'{base}_{counter}'
                counter += 1
            used.add(node_id)
            self._node_ids[b.name] = node_id

    def _nid(self, block) -> str:
        """Return the TikZ node ID for *block*."""
        return self._node_ids.get(block.name, _sanitize_node_id(block.name))

    def export_document(self, options: Optional[Dict] = None) -> str:
        """Return a full standalone .tex document."""
        options = options or {}
        snippet = self.export_snippet(options)
        lines = [
            r'\documentclass[border=5mm]{standalone}',
            r'\usepackage{tikz}',
            r'\usepackage{amsmath}',
            r'\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc}',
            r'',
            r'\begin{document}',
            snippet,
            r'\end{document}',
        ]
        return '\n'.join(lines)

    def export_snippet(self, options: Optional[Dict] = None) -> str:
        """Return tikzset + tikzpicture (no document preamble).

        When used inside a LaTeX document with known ``\\textwidth``,
        wrap the output in ``\\resizebox{\\textwidth}{!}{...}`` to scale
        automatically.
        """
        opts = {
            'include_sinks': True,
            'sink_as_arrow': True,
            'source_as_arrow': True,
            'show_usernames': True,
            'show_values': True,
            'show_signal_labels': True,
            'fill_blocks': True,
            'page_width_cm': 14.0,
            'use_resizebox': False,
        }
        if options:
            opts.update(options)

        # Filter blocks
        blocks = list(self.blocks)
        if not opts['include_sinks']:
            blocks = [b for b in blocks if b.category != 'Sinks']

        if not blocks:
            return '% No blocks to export\n'

        sink_as_arrow = opts.get('sink_as_arrow', True)
        source_as_arrow = opts.get('source_as_arrow', True)

        # Rendered blocks = those that will appear as TikZ nodes
        rendered = [
            b for b in blocks
            if not (sink_as_arrow and self._is_sink_block_obj(b))
            and not (source_as_arrow and self._is_source_block(b))
        ]

        # Build maps using all blocks (connections may reference any)
        self._build_node_ids(blocks)
        self._build_symbol_maps(opts)
        self._compute_block_type_counts()

        # Topological sort for block ordering
        self._compute_textbook_layout(rendered, opts)

        # Compute arrow length from average layout gap
        avg_gap = (sum(self._layout_gaps) / len(self._layout_gaps)
                   if self._layout_gaps else 2.5)
        self._arrow_len = round(max(min(avg_gap * 0.3, 1.5), 0.6), 1)

        # Filter connections before drawing so feedback depths
        # can be pre-computed
        visible_lines = [ln for ln in self.lines if not ln.hidden]
        if not opts['include_sinks']:
            sink_names = {b.name for b in self.blocks if b.category == 'Sinks'}
            sink_usernames = {b.username for b in self.blocks if b.category == 'Sinks'}
            visible_lines = [
                ln for ln in visible_lines
                if ln.dstblock not in sink_names and ln.dstblock not in sink_usernames
                and ln.srcblock not in sink_names and ln.srcblock not in sink_usernames
            ]
        self._compute_feedback_depths(visible_lines, opts)

        # Build output
        parts = []
        parts.append(self._tikz_styles(opts))

        if opts.get('use_resizebox'):
            parts.append(r'\resizebox{\textwidth}{!}{%')

        parts.append(r'\begin{tikzpicture}')

        # Nodes — placed at computed coordinates
        parts.append('  % --- Blocks ---')
        for block in rendered:
            parts.append(self._block_to_node(block, opts))

        # Output continuation: branch point + output arrow (must come
        # before connections so feedback routing can reference the dot)
        output_cont = self._output_continuation(rendered, visible_lines, opts)
        if output_cont:
            parts.append('  % --- Output ---')
            parts.append(output_cont)

        if visible_lines:
            parts.append('  % --- Connections ---')
            for line in visible_lines:
                tikz_draw = self._line_to_tikz_draw(line, opts)
                if tikz_draw:
                    parts.append(tikz_draw)

        parts.append(r'\end{tikzpicture}')

        if opts.get('use_resizebox'):
            parts.append(r'}')

        return '\n'.join(parts)

    # ------------------------------------------------------------------
    # Symbol auto-numbering
    # ------------------------------------------------------------------

    def _build_symbol_maps(self, opts):
        """Build maps for generic symbol assignment (G_1, H_1, K_1, etc.).

        Different block types get different base letters:
        TranFn -> G, DiscreteTranFn -> H, StateSpace -> P, Gain -> K.
        Math-like usernames (Kp, G1, K) override auto-numbering;
        descriptive names (tank, plant, observer) fall through to auto.
        """
        self._tf_symbols = {}
        self._gain_symbols = {}
        self._symbol_from_username = set()

        cont_tf = [b for b in self.blocks if b.block_fn == 'TranFn']
        disc_tf = [b for b in self.blocks if b.block_fn == 'DiscreteTranFn']
        ss_blocks = [b for b in self.blocks if b.block_fn == 'StateSpace']
        gain_blocks = [b for b in self.blocks if b.block_fn == 'Gain']

        def _assign(block_list, base_letter, target_map):
            for i, b in enumerate(block_list):
                uname = b.username if b.username != b.name else None
                if uname:
                    math_label = _name_to_math(uname)
                    # Only use username as symbol if it's math-like
                    # (K_p, G_1, etc.), not descriptive (\text{tank})
                    if '\\text{' not in math_label:
                        target_map[b.name] = math_label
                        self._symbol_from_username.add(b.name)
                        continue
                # Auto-assign letter or numbered letter
                if len(block_list) == 1:
                    target_map[b.name] = base_letter
                else:
                    target_map[b.name] = f'{base_letter}_{{{i+1}}}'

        _assign(cont_tf, 'G', self._tf_symbols)
        _assign(disc_tf, 'H', self._tf_symbols)
        _assign(ss_blocks, 'P', self._tf_symbols)
        _assign(gain_blocks, 'K', self._gain_symbols)

    def _compute_block_type_counts(self):
        """Count blocks by type for smart signal labeling decisions."""
        self._block_type_counts = Counter(b.block_fn for b in self.blocks)

    # ------------------------------------------------------------------
    # Textbook layout — topology-based block placement
    # ------------------------------------------------------------------

    def _compute_textbook_layout(self, rendered_blocks, opts):
        """Place blocks using topological order with even spacing.

        All blocks go on a single horizontal line (Y=0) at evenly
        distributed X positions determined by BFS signal-flow order.
        This produces the clean, straight-line style used in
        control-systems textbooks.
        """
        self._textbook_pos = {}
        self._block_order = {}
        self._ordered_names = []
        self._layout_gaps = []

        if not rendered_blocks:
            return

        rendered_names = {b.name for b in rendered_blocks}
        block_by_name = {b.name: b for b in rendered_blocks}

        # Build forward adjacency among rendered blocks
        adj = {b.name: [] for b in rendered_blocks}
        has_incoming = set()

        for line in self.lines:
            if line.hidden:
                continue
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            if src.name in rendered_names and dst.name in rendered_names:
                adj[src.name].append(dst.name)
                has_incoming.add(dst.name)

        # Roots: blocks with no rendered incoming edges
        roots = [b.name for b in rendered_blocks
                 if b.name not in has_incoming]
        if not roots:
            roots = [min(rendered_blocks, key=lambda b: b.left).name]

        roots.sort(key=lambda n: block_by_name[n].left)

        # BFS to establish topological order
        visited = set()
        order = []
        queue = list(roots)

        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            order.append(name)
            neighbors = sorted(
                adj.get(name, []),
                key=lambda n: block_by_name[n].left if n in block_by_name else 0,
            )
            for nb in neighbors:
                if nb not in visited:
                    queue.append(nb)

        for b in rendered_blocks:
            if b.name not in visited:
                order.append(b.name)

        # Compute adaptive gaps based on block types
        gaps = []
        for i in range(1, len(order)):
            prev_fn = block_by_name[order[i - 1]].block_fn
            curr_fn = block_by_name[order[i]].block_fn
            gap = 2.8  # base
            if prev_fn in ('TranFn', 'DiscreteTranFn', 'StateSpace'):
                gap += 0.6
            if curr_fn in ('TranFn', 'DiscreteTranFn', 'StateSpace'):
                gap += 0.6
            if prev_fn == 'Sum' or curr_fn == 'Sum':
                gap -= 0.2
            if curr_fn == 'Gain' or prev_fn == 'Gain':
                gap -= 0.2
            gaps.append(max(gap, 1.8))

        # Scale down if total exceeds page_width
        total = sum(gaps) if gaps else 0
        page_width = opts.get('page_width_cm', 14.0)
        if total > page_width and gaps:
            scale = page_width / total
            gaps = [g * scale for g in gaps]

        # Assign positions
        x = 0.0
        for i, name in enumerate(order):
            self._textbook_pos[name] = (round(x, 2), 0.0)
            self._block_order[name] = i
            if i < len(gaps):
                x += gaps[i]
        self._ordered_names = order
        self._layout_gaps = gaps

    def _compute_feedback_depths(self, visible_lines, opts):
        """Assign Y-depths for feedback connections to avoid overlaps.

        Wider feedback loops (spanning more blocks) are placed deeper
        below the main path so that shorter inner loops don't cross them.
        """
        self._feedback_depth = {}

        feedback_info = []
        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            # Skip source/sink-as-arrow connections
            if self._is_source_block(src) and opts.get('source_as_arrow', True):
                continue
            if self._is_sink_block(dst) and opts.get('sink_as_arrow', True):
                continue

            src_order = self._block_order.get(src.name, -1)
            dst_order = self._block_order.get(dst.name, -1)

            if src_order >= 0 and dst_order >= 0 and dst_order <= src_order:
                span = src_order - dst_order
                key = (line.srcblock, line.dstblock, line.srcport, line.dstport)
                feedback_info.append((key, span))

        # Sort by span: wider loops get deeper placement
        feedback_info.sort(key=lambda x: x[1])

        base_depth = -1.8
        step = -0.9
        for i, (key, _span) in enumerate(feedback_info):
            self._feedback_depth[key] = round(base_depth + i * step, 1)

    # ------------------------------------------------------------------
    # Output continuation (branch point + output arrow)
    # ------------------------------------------------------------------

    def _output_continuation(self, rendered, visible_lines, opts):
        """Add output arrow + branch dot after the last forward block with feedback.

        Sets ``self._branch_point_id`` and ``self._branch_source`` so that
        feedback routing can branch from the dot instead of the block edge.
        Returns TikZ code for the output coordinate, arrow, and branch dot.
        """
        self._branch_point_id = None
        self._branch_source = None
        self._output_label = ''

        sink_as_arrow = opts.get('sink_as_arrow', True)

        # Find blocks that are feedback sources (connection goes backward)
        # Only consider connections between blocks that are in the layout
        # (skip source/sink-as-arrow endpoints that aren't rendered nodes).
        feedback_sources = set()
        for ln in visible_lines:
            src = self._resolve_block(ln.srcblock)
            dst = self._resolve_block(ln.dstblock)
            if not src or not dst:
                continue
            if src.name not in self._block_order or dst.name not in self._block_order:
                continue
            s_ord = self._block_order[src.name]
            d_ord = self._block_order[dst.name]
            if d_ord <= s_ord:
                feedback_sources.add(src.name)

        if not feedback_sources:
            return ''

        # Last block in topological order that feeds back
        last_fb = max(feedback_sources,
                      key=lambda n: self._block_order.get(n, 0))
        last_block = self._block_map.get(last_fb) or self._username_map.get(last_fb)
        if not last_block:
            return ''

        nid = self._nid(last_block)
        anchor = self._get_port_anchor(last_block, 0, is_output=True)
        arrow_len = self._arrow_len

        # Capture the signal label from the sink-as-arrow line for this block
        if sink_as_arrow:
            for ln in visible_lines:
                src = self._resolve_block(ln.srcblock)
                dst = self._resolve_block(ln.dstblock)
                if (src and dst and src.name == last_fb
                        and self._is_sink_block(dst)):
                    show_labels = opts.get('show_signal_labels', True)
                    if show_labels:
                        self._output_label = self._get_signal_label(
                            ln, src, dst)
                    break

        lines = []
        # Output coordinate
        lines.append(
            f'  \\coordinate (output) at '
            f'($({nid}.{anchor})+({arrow_len + 0.5},0)$);')
        # Output arrow with signal label
        label_node = ''
        if self._output_label:
            label_node = (f' node[midway, above, font=\\small] '
                          f'{{{self._output_label}}}')
        lines.append(
            f'  \\draw[signal] ({nid}.{anchor}) --{label_node} (output);')
        # Branch dot at 60% along the output segment
        lines.append(
            f'  \\node[branch] at '
            f'($({nid}.{anchor})!0.6!(output)$) (bpt) {{}};')

        self._branch_point_id = 'bpt'
        self._branch_source = last_fb
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # TikZ style definitions
    # ------------------------------------------------------------------

    def _tikz_styles(self, opts):
        """Return \\tikzset{...} block with style definitions."""
        fill_opt = ', fill=blue!5' if opts.get('fill_blocks') else ''
        source_fill = ', fill=green!8' if opts.get('fill_blocks') else ''
        sink_fill = ', fill=red!8' if opts.get('fill_blocks') else ''
        tf_fill = ', fill=blue!8' if opts.get('fill_blocks') else ''
        gain_fill = ', fill=blue!5' if opts.get('fill_blocks') else ''

        return (
            r'\tikzset{' + '\n'
            r'  block/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=10mm, minimum width=14mm, thick{fill_opt}' + '},\n'
            r'  sum/.style={draw, circle, minimum size=9mm, thick, inner sep=0pt},' + '\n'
            r'  gain/.style={draw, isosceles triangle, isosceles triangle apex angle=70,'
            r' shape border rotate=0, minimum height=10mm, thick,'
            f' inner sep=2pt{gain_fill}' + '},\n'
            r'  gain flipped/.style={draw, isosceles triangle, isosceles triangle apex angle=70,'
            r' shape border rotate=180, minimum height=10mm, thick,'
            f' inner sep=2pt{gain_fill}' + '},\n'
            r'  tf/.style={draw, rectangle, minimum height=12mm,'
            f' minimum width=16mm, thick{tf_fill}' + '},\n'
            r'  source/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=10mm, minimum width=12mm, thick{source_fill}' + '},\n'
            r'  sink/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=10mm, minimum width=12mm, thick{sink_fill}' + '},\n'
            r'  signal/.style={-{Stealth[length=2.5mm, width=2mm]}, semithick},' + '\n'
            r'  signal wide/.style={-{Stealth[length=2.5mm, width=2mm]}, thick},' + '\n'
            r'  branch/.style={fill, circle, minimum size=3.5pt, inner sep=0pt},' + '\n'
            r'}'
        )

    # ------------------------------------------------------------------
    # Block -> TikZ node
    # ------------------------------------------------------------------

    def _get_block_style(self, block):
        """Return the TikZ style name for a block."""
        fn = block.block_fn
        if fn == 'Sum':
            return 'sum'
        if fn == 'Gain':
            return 'gain flipped' if block.flipped else 'gain'
        if fn in ('TranFn', 'Integrator', 'Deriv', 'StateSpace',
                   'DiscreteTranFn'):
            return 'tf'
        if block.category == 'Sources':
            return 'source'
        if block.category == 'Sinks':
            return 'sink'
        return 'block'

    def _get_block_content(self, block, opts):
        """Return the text/math content to place inside the TikZ node."""
        fn = block.block_fn
        show_values = opts.get('show_values', True)

        if fn == 'Sum':
            return ''  # Signs placed separately

        if fn == 'Gain':
            # Prefer username inside the triangle when set
            uname = block.username
            has_custom_name = uname and uname != block.name
            if has_custom_name:
                return f'${_name_to_math(uname)}$'
            if show_values:
                gain = block.params.get('gain', 1.0)
                if isinstance(gain, (int, float)):
                    return f'${gain:.4g}$' if gain != int(gain) else f'${int(gain)}$'
                return '$K$'
            sym = self._gain_symbols.get(block.name, 'K')
            return f'${sym}$'

        if fn == 'TranFn':
            if show_values:
                num = block.params.get('numerator', [1.0])
                den = block.params.get('denominator', [1.0, 1.0])
                if isinstance(num, (list, tuple)) and isinstance(den, (list, tuple)):
                    num_tex = _poly_to_latex(num)
                    den_tex = _poly_to_latex(den)
                    return f'$\\dfrac{{{num_tex}}}{{{den_tex}}}$'
            sym = self._tf_symbols.get(block.name, 'G')
            return f'${sym}(s)$'

        if fn == 'Integrator':
            return r'$\dfrac{1}{s}$'

        if fn == 'Deriv':
            return '$s$'

        if fn == 'StateSpace':
            if show_values:
                return r'$\dot{x}{=}Ax{+}Bu$'
            sym = self._tf_symbols.get(block.name, 'P')
            return f'${sym}(s)$'

        if fn == 'DiscreteTranFn':
            if show_values:
                num = block.params.get('numerator', [1.0])
                den = block.params.get('denominator', [1.0, 1.0])
                if isinstance(num, (list, tuple)) and isinstance(den, (list, tuple)):
                    num_tex = _poly_to_latex(num)
                    den_tex = _poly_to_latex(den)
                    num_tex = num_tex.replace('s', 'z')
                    den_tex = den_tex.replace('s', 'z')
                    return f'$\\dfrac{{{num_tex}}}{{{den_tex}}}$'
            sym = self._tf_symbols.get(block.name, 'G')
            return f'${sym}(z)$'

        if fn == 'PID':
            return 'PID'

        if fn in ('Step', 'Constant'):
            if show_values and fn == 'Step':
                val = block.params.get('value', 1.0)
                return f'Step({val})'
            if show_values and fn == 'Constant':
                val = block.params.get('value', 0.0)
                return f'${val}$'
            return fn

        if block.category == 'Sources':
            return fn

        if block.category == 'Sinks':
            label = block.username if block.username != block.name else fn
            return _escape_latex(label)

        if fn == 'Subsystem':
            return _escape_latex(block.username if block.username != block.name else 'Subsystem')

        return _escape_latex(fn)

    def _block_to_node(self, block, opts):
        """Generate a \\node line for a block at its computed position."""
        style = self._get_block_style(block)
        content = self._get_block_content(block, opts)
        node_id = self._nid(block)

        tx, ty = self._textbook_pos.get(block.name, (0, 0))

        extra_opts = ''
        if block.block_fn == 'Subsystem':
            extra_opts = ', double'

        node_line = f'  \\node[{style}{extra_opts}] ({node_id}) at ({tx},{ty}) {{{content}}};'

        # Sum block: place +/- signs inside the circle near each input port
        if block.block_fn == 'Sum':
            sign_str = block.params.get('sign', block.params.get('signs', '++'))
            if isinstance(sign_str, dict):
                sign_str = sign_str.get('default', '++')
            node_line += self._sum_sign_labels(node_id, sign_str, block)

        # Username label below block — skip if username is already
        # reflected in the node content (e.g. TF/Gain symbol maps),
        # and never label Sum junctions.
        if (opts.get('show_usernames', True)
                and block.username and block.username != block.name
                and block.block_fn != 'Sum'):
            uname = block.username
            show_values = opts.get('show_values', True)
            # Symbol is "already shown" only when it's actually in the
            # node content AND derived from the username.  When show_values
            # is True, TF/StateSpace display the polynomial/equation, not
            # the symbol.  When the symbol is auto-assigned (G, G_1),
            # the username should still appear as a below-label.
            symbol_visible = (
                (block.name in self._symbol_from_username
                 and (not show_values or block.name in self._gain_symbols))
                or (block.block_fn == 'Gain' and uname != block.name)
            )
            # Skip if username is just a case-variant of block_fn
            # (e.g. username="mux" for block_fn="Mux")
            redundant = uname.lower() == block.block_fn.lower()
            if not symbol_visible and not redundant:
                label = _escape_latex(uname)
                node_line += f'\n  \\node[below=1mm of {node_id}, font=\\footnotesize] {{{label}}};'

        return node_line

    def _sum_port_angles(self, sign_str):
        """Return list of angles for Sum input ports.

        Standard control convention:
        - ``+-`` (feedback): + from left (180), - from bottom (270)
        - ``++`` : both from left, stacked (150, 210)
        - ``+-+``: upper-left, bottom, lower-left (150, 270, 210)
        """
        n = len(sign_str)
        if n == 1:
            return [180]
        if n == 2:
            # Feedback convention: if there's a minus, it enters from bottom
            if '-' in sign_str:
                plus_idx = sign_str.index('+') if '+' in sign_str else 0
                minus_idx = sign_str.index('-')
                angles = [None, None]
                angles[plus_idx] = 180
                angles[minus_idx] = 270
                return angles
            return [150, 210]
        if n == 3:
            return [150, 270, 210]
        # 4+: distribute on left half-circle
        step = 180 / (n + 1)
        return [int(90 + step * (i + 1)) for i in range(n)]

    def _sum_sign_labels(self, node_id, sign_str, block):
        """Place +/- signs inside the Sum circle near each input anchor."""
        labels = []
        angles = self._sum_port_angles(sign_str)

        for i, sign_char in enumerate(sign_str):
            if i < len(angles):
                angle = angles[i]
                sign = '$+$' if sign_char == '+' else '$-$'
                inner_dist = 0.27
                dx = round(inner_dist * math.cos(math.radians(angle)), 2)
                dy = round(inner_dist * math.sin(math.radians(angle)), 2)
                labels.append(
                    f'\n  \\node[font=\\footnotesize, inner sep=0pt] '
                    f'at ($({node_id}.center)+({dx},{dy})$) {{{sign}}};'
                )

        return ''.join(labels)

    # ------------------------------------------------------------------
    # Connection -> TikZ \draw
    # ------------------------------------------------------------------

    def _resolve_block(self, name):
        """Find a block by name or username."""
        if name in self._block_map:
            return self._block_map[name]
        if name in self._username_map:
            return self._username_map[name]
        return None

    def _get_port_anchor(self, block, port_idx, is_output):
        """Return TikZ anchor name for a port on a block."""
        fn = block.block_fn
        flipped = block.flipped

        if fn == 'Sum':
            if is_output:
                return 'east' if not flipped else 'west'
            sign_str = block.params.get('sign', block.params.get('signs', '++'))
            if isinstance(sign_str, dict):
                sign_str = sign_str.get('default', '++')
            angles = self._sum_port_angles(sign_str)
            if port_idx < len(angles):
                return str(angles[port_idx])
            return 'west'

        if fn == 'Gain':
            # Use west/east for horizontal alignment; 'left side' is on
            # the slanted edge (above center) and produces diagonal lines.
            return 'west' if not is_output else 'east'

        # Generic blocks
        if is_output:
            edge = 'east' if not flipped else 'west'
        else:
            edge = 'west' if not flipped else 'east'

        total = block.out_ports if is_output else block.in_ports
        if total <= 1:
            return edge

        if total == 2:
            if edge == 'east':
                return ['north east', 'south east'][port_idx] if port_idx < 2 else edge
            else:
                return ['north west', 'south west'][port_idx] if port_idx < 2 else edge
        if total == 3:
            if edge == 'east':
                return ['north east', 'east', 'south east'][port_idx] if port_idx < 3 else edge
            else:
                return ['north west', 'west', 'south west'][port_idx] if port_idx < 3 else edge

        return edge

    @staticmethod
    def _format_explicit_label(label):
        """Format an explicit line label, preserving LaTeX math mode."""
        label = label.strip()
        if label.startswith('$') and label.endswith('$'):
            return label
        if '\\' in label:
            return label
        return _escape_latex(label)

    _CONVENTIONAL_LABELS = {
        'Step': '$r$', 'Constant': '$r$', 'Sine': '$r$', 'Ramp': '$r$',
        'Sum': '$e$',
        'Gain': '$u$',
        'TranFn': '$y$', 'Integrator': '$y$',
        'StateSpace': '$y$', 'DiscreteTranFn': '$y$',
    }

    # Block types that should never get auto-generated signal labels
    _NO_SIGNAL_LABEL = {'Mux', 'Demux', 'Subsystem', 'Inport', 'Outport',
                        'Switch', 'Terminator'}

    def _get_signal_label(self, line, src_block, dst_block):
        """Generate a signal label for a connection.

        Priority:
        1. Explicit label on the line object (LaTeX math preserved)
        2. Conventional signal name — only when the diagram has exactly
           one block of this type (avoids ambiguous duplicate labels)
        3. Short math-like username (1-3 chars) as subscripted symbol
        4. Empty — never use long descriptive names as signal labels
        """
        if line.label:
            return self._format_explicit_label(line.label)

        fn = src_block.block_fn
        if fn in self._NO_SIGNAL_LABEL:
            return ''

        count = self._block_type_counts.get(fn, 0)
        if count <= 1 and fn in self._CONVENTIONAL_LABELS:
            return self._CONVENTIONAL_LABELS[fn]

        # Only use username as signal label if it looks like a math symbol
        # (short: 1-3 chars, starts with a letter). Skip descriptive names
        # like "plant", "observer", "controller" — they clutter the diagram.
        uname = src_block.username
        if uname and uname != src_block.name and len(uname) <= 3:
            return f'${_name_to_math(uname)}$'

        return ''

    def _is_sink_block(self, block):
        """Check if a block is a sink (Scope, Display, etc.)."""
        return block.category == 'Sinks'

    _is_sink_block_obj = _is_sink_block  # alias

    def _is_source_block(self, block):
        """Check if a block is a source (Step, Constant, etc.)."""
        return block.category == 'Sources'

    def _line_to_tikz_draw(self, line, opts):
        """Generate a \\draw command for a connection line.

        Uses textbook layout positions and topological order to
        produce clean straight (forward) or orthogonal (feedback) paths.
        """
        src_block = self._resolve_block(line.srcblock)
        dst_block = self._resolve_block(line.dstblock)

        if not src_block or not dst_block:
            return None

        src_is_arrow = (self._is_source_block(src_block)
                        and opts.get('source_as_arrow', True))
        dst_is_arrow = (self._is_sink_block(dst_block)
                        and opts.get('sink_as_arrow', True))

        # Skip connections where both endpoints are arrows (no TikZ nodes)
        if src_is_arrow and dst_is_arrow:
            return None

        src_id = self._nid(src_block)
        dst_id = self._nid(dst_block)

        src_anchor = self._get_port_anchor(src_block, line.srcport, is_output=True)
        dst_anchor = self._get_port_anchor(dst_block, line.dstport, is_output=False)

        style = 'signal wide' if line.signal_width > 1 else 'signal'

        # Signal label
        show_labels = opts.get('show_signal_labels', True)
        label_text = self._get_signal_label(line, src_block, dst_block) if show_labels else ''

        # --- Source-as-arrow: incoming arrow instead of drawing source node ---
        if src_is_arrow:
            label_node = ''
            if label_text:
                label_node = f' node[midway, above, font=\\small] {{{label_text}}}'
            arrow_len = self._arrow_len
            dst_is_south = dst_anchor in ('270', 'south')
            if dst_is_south:
                return (
                    f'  \\draw[{style}] ($({dst_id}.{dst_anchor})+(0,-{arrow_len})$) '
                    f'--{label_node} ({dst_id}.{dst_anchor});'
                )
            else:
                return (
                    f'  \\draw[{style}] ($({dst_id}.{dst_anchor})+(-{arrow_len},0)$) '
                    f'--{label_node} ({dst_id}.{dst_anchor});'
                )

        # --- Sink-as-arrow: output arrow instead of routing to scope node ---
        if dst_is_arrow:
            # Skip if output continuation already drew an arrow for this source
            if (getattr(self, '_branch_source', None)
                    and src_block.name == self._branch_source):
                return None
            label_node = ''
            if label_text:
                label_node = f' node[midway, above, font=\\small] {{{label_text}}}'
            arrow_len = self._arrow_len
            return (
                f'  \\draw[{style}] ({src_id}.{src_anchor}) --{label_node} '
                f'+({arrow_len},0);'
            )

        # --- Detect feedback by topological order ---
        src_order = self._block_order.get(src_block.name, 0)
        dst_order = self._block_order.get(dst_block.name, 0)
        is_feedback = dst_order <= src_order

        if not is_feedback:
            # Forward connection: all blocks at Y=0, straight line
            label_node = ''
            if label_text:
                label_node = f' node[midway, above, font=\\small] {{{label_text}}}'
            return (
                f'  \\draw[{style}] ({src_id}.{src_anchor})'
                f' --{label_node} ({dst_id}.{dst_anchor});'
            )

        # For feedback connections, only use explicitly-set line labels
        # (not auto-generated ones) to avoid duplicating the forward label.
        label_text = self._format_explicit_label(line.label) if line.label else ''

        # --- Feedback: route below the main path ---
        # When a branch point exists for this source, start from the dot;
        # otherwise fall back to a small rightward jut from the block edge.
        # Path goes down then left using -| (vert-then-horiz).
        key = (line.srcblock, line.dstblock, line.srcport, line.dstport)
        depth = self._feedback_depth.get(key, -1.5)

        if (getattr(self, '_branch_source', None)
                and src_block.name == self._branch_source
                and getattr(self, '_branch_point_id', None)):
            start = self._branch_point_id
        else:
            start = f'{src_id}.{src_anchor}'

        # Determine feedback label: skip redundant sign labels (already
        # drawn inside the Sum circle), but keep explicit signal labels.
        fb_label_text = ''
        if dst_block.block_fn == 'Sum':
            # Signs are already rendered inside the circle — only add
            # an explicit signal label if one exists on the line.
            if label_text:
                fb_label_text = label_text
        elif label_text:
            fb_label_text = label_text

        label_below = ''
        if fb_label_text:
            label_below = (f' node[pos=0.5, below, font=\\small] '
                           f'{{{fb_label_text}}}')

        return (
            f'  \\draw[{style}, rounded corners=4pt] ({start})'
            f' -- ++(0,{depth})'
            f' -|{label_below} ({dst_id}.{dst_anchor});'
        )

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def get_info(self):
        """Return summary info about the diagram."""
        return {
            'block_count': len(self.blocks),
            'connection_count': len([ln for ln in self.lines if not ln.hidden]),
            'block_types': sorted(set(b.block_fn for b in self.blocks)),
        }
