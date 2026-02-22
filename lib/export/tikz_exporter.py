"""
TikZ Exporter - Export block diagrams to TikZ code for LaTeX documents.

Generates publication-ready TikZ diagrams using standard control-systems
conventions: Sum=circle, Gain=triangle, TF=rectangle with fraction.
"""

import math
import re
import logging
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


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('_', r'\_'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('#', r'\#'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


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
            snippet,
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

        # Topological sort for block ordering
        self._compute_textbook_layout(rendered, opts)

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
        """Build maps for generic symbol assignment (G_1, K_1, etc.)."""
        self._tf_symbols = {}
        self._gain_symbols = {}

        tf_blocks = [b for b in self.blocks
                     if b.block_fn in ('TranFn', 'DiscreteTranFn', 'StateSpace')]
        gain_blocks = [b for b in self.blocks if b.block_fn == 'Gain']

        for i, b in enumerate(tf_blocks):
            if len(tf_blocks) == 1:
                self._tf_symbols[b.name] = 'G'
            else:
                # Use username if it looks like a symbol, otherwise number
                uname = b.username if b.username != b.name else None
                if uname:
                    self._tf_symbols[b.name] = _name_to_math(uname)
                else:
                    self._tf_symbols[b.name] = f'G_{{{i+1}}}'

        for i, b in enumerate(gain_blocks):
            if len(gain_blocks) == 1:
                self._gain_symbols[b.name] = 'K'
            else:
                uname = b.username if b.username != b.name else None
                if uname:
                    self._gain_symbols[b.name] = _name_to_math(uname)
                else:
                    self._gain_symbols[b.name] = f'K_{{{i+1}}}'

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

        # Assign positions: even X spacing, all at Y = 0
        n = len(order)
        page_width = opts.get('page_width_cm', 14.0)
        spacing = page_width / max(n - 1, 1) if n > 1 else 0

        for i, name in enumerate(order):
            self._textbook_pos[name] = (round(i * spacing, 2), 0.0)
            self._block_order[name] = i
        self._ordered_names = order

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

        base_depth = -1.5
        step = -0.8
        for i, (key, _span) in enumerate(feedback_info):
            self._feedback_depth[key] = round(base_depth + i * step, 1)

    # ------------------------------------------------------------------
    # TikZ style definitions
    # ------------------------------------------------------------------

    def _tikz_styles(self, opts):
        """Return \\tikzset{...} block with style definitions."""
        fill_opt = ', fill=blue!5' if opts.get('fill_blocks') else ''
        source_fill = ', fill=green!10' if opts.get('fill_blocks') else ''
        sink_fill = ', fill=red!10' if opts.get('fill_blocks') else ''
        tf_fill = ', fill=blue!8' if opts.get('fill_blocks') else ''

        return (
            r'\tikzset{' + '\n'
            r'  block/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=8mm, minimum width=12mm, thick{fill_opt}' + '},\n'
            r'  sum/.style={draw, circle, minimum size=6mm, thick, inner sep=0pt},' + '\n'
            r'  gain/.style={draw, isosceles triangle, isosceles triangle apex angle=60,'
            r' shape border rotate=0, minimum height=8mm, thick, inner sep=1pt' + '},\n'
            r'  gain flipped/.style={draw, isosceles triangle, isosceles triangle apex angle=60,'
            r' shape border rotate=180, minimum height=8mm, thick, inner sep=1pt' + '},\n'
            r'  tf/.style={draw, rectangle, minimum height=10mm,'
            f' minimum width=14mm, thick{tf_fill}' + '},\n'
            r'  source/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=8mm, minimum width=10mm, thick{source_fill}' + '},\n'
            r'  sink/.style={draw, rectangle, rounded corners=2pt,'
            f' minimum height=8mm, minimum width=10mm, thick{sink_fill}' + '},\n'
            r'  signal/.style={-{Latex[length=2.5mm]}, thick},' + '\n'
            r'  signal wide/.style={-{Latex[length=2.5mm]}, very thick},' + '\n'
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
            sym = self._tf_symbols.get(block.name, 'G')
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
        # reflected in the node content (e.g. TF/Gain symbol maps)
        if opts.get('show_usernames', True) and block.username and block.username != block.name:
            uname = block.username
            already_shown = (
                block.name in self._tf_symbols
                or block.name in self._gain_symbols
            )
            if not already_shown:
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
                sign = '+' if sign_char == '+' else '$-$'
                inner_dist = 0.18
                dx = round(inner_dist * math.cos(math.radians(angle)), 2)
                dy = round(inner_dist * math.sin(math.radians(angle)), 2)
                labels.append(
                    f'\n  \\node[font=\\scriptsize, inner sep=0pt] '
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
            if flipped:
                return 'apex' if not is_output else 'left side'
            return 'left side' if not is_output else 'apex'

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

    def _get_signal_label(self, line, src_block, dst_block):
        """Generate a signal label for a connection.

        Priority:
        1. Explicit label on the line object
        2. Conventional signal name by source block type ($r$, $e$, $u$, $y$)
        3. User-assigned username converted to math label
        """
        # Explicit label on the line takes priority
        if line.label:
            return _escape_latex(line.label)

        # Conventional names by block type (checked BEFORE username)
        fn = src_block.block_fn
        if fn in ('Step', 'Constant', 'Sine', 'Ramp'):
            return '$r$'
        if fn == 'Sum':
            return '$e$'
        if fn == 'Gain':
            return '$u$'
        if fn in ('TranFn', 'Integrator', 'StateSpace', 'DiscreteTranFn'):
            return '$y$'

        # If the user gave a custom name, derive a math label from it
        uname = src_block.username
        auto_name = src_block.name  # e.g. 'step0', 'gain2'
        if uname and uname != auto_name:
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
            arrow_len = 1.2
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
            label_node = ''
            if label_text:
                label_node = f' node[midway, above, font=\\small] {{{label_text}}}'
            arrow_len = 1.2
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

        # --- Feedback: route below the main path using |- operator ---
        # Path: src → right jut → down to depth → |- (horiz then vert) → dst
        # The |- operator goes horizontal first (left), then vertical (up)
        # to the destination anchor — no absolute coordinates needed.
        key = (line.srcblock, line.dstblock, line.srcport, line.dstport)
        depth = self._feedback_depth.get(key, -1.5)

        label_below = ''
        if label_text:
            label_below = f' node[pos=0.5, below, font=\\small] {{{label_text}}}'

        return (
            f'  \\draw[{style}] ({src_id}.{src_anchor})'
            f' -- ++(0.5,0) -- ++(0,{depth})'
            f' |-{label_below} ({dst_id}.{dst_anchor});'
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
