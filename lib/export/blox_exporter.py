"""
Blox Exporter - Export block diagrams using the LaTeX blox package macros.

The blox package provides high-level macros for control system block
diagrams (\bXInput, \bXBlocL, \bXComp, \bXReturn, etc.).  This exporter
produces much more readable LaTeX than raw TikZ, but only works for
serial-chain topologies with 0-2 feedback arcs.

Returns None from export() when the diagram is too complex.
"""

import logging
from typing import Dict, List, Optional

from lib.export.tikz_exporter import (
    _poly_to_latex,
    _escape_latex,
    _name_to_math,
    _sanitize_node_id,
)

logger = logging.getLogger(__name__)


class BloxExporter:
    """Export a DiaBloS block diagram using ``\\usepackage{blox}`` macros."""

    # Maximum feedback arcs for blox compatibility
    MAX_FEEDBACK = 3

    def __init__(self, blocks_list, line_list):
        self.blocks = blocks_list
        self.lines = line_list
        self._block_map = {b.name: b for b in blocks_list}
        self._username_map = {b.username: b for b in blocks_list}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_export(self) -> bool:
        """Check if diagram topology is blox-compatible.

        Requirements:
        - Single forward chain (each block has at most 1 successor)
        - At most MAX_FEEDBACK back-edges (feedback arcs)
        - No parallel forward paths
        """
        chain, feedback = self._analyse_topology()
        if chain is None:
            return False
        if len(feedback) > self.MAX_FEEDBACK:
            return False
        return True

    def export(self, options: Optional[Dict] = None) -> Optional[str]:
        """Generate blox output.  Returns None if topology is incompatible."""
        chain, feedback = self._analyse_topology()
        if chain is None or len(feedback) > self.MAX_FEEDBACK:
            return None

        opts = {
            'show_values': True,
            'show_signal_labels': True,
            'show_usernames': True,
            'page_width_cm': 14.0,
            'use_resizebox': False,
            'standalone': True,
        }
        if options:
            opts.update(options)

        parts = []

        # Preamble
        if opts.get('standalone', True):
            parts.append(r'\documentclass[border=5mm]{standalone}')
            parts.append(r'\usepackage{blox}')
            parts.append(r'\usepackage{tikz}')
            parts.append(r'\usepackage{amsmath}')
            parts.append('')
            parts.append(r'\begin{document}')

        if opts.get('use_resizebox'):
            parts.append(r'\resizebox{\textwidth}{!}{%')

        parts.append(r'\begin{tikzpicture}')

        # Compute inter-block distance (em)
        n = len(chain)
        dist = max(round(14.0 / max(n - 1, 1) * 1.5, 1), 2.0) if n > 1 else 3.0

        # Emit blocks along the forward chain
        prev_node = None
        node_ids = {}  # block.name -> blox node id

        for i, block in enumerate(chain):
            nid = _sanitize_node_id(
                block.username if block.username != block.name else block.name
            )
            node_ids[block.name] = nid

            if i == 0 and self._is_source(block):
                parts.append(f'  \\bXInput{{{nid}}}')
                prev_node = nid
                continue

            if self._is_source(block):
                parts.append(f'  \\bXInput{{{nid}}}')
                prev_node = nid
                continue

            if self._is_sink(block):
                parts.append(f'  \\bXOutput[{dist}]{{{nid}}}{{{prev_node}}}')
                # Link from previous
                label = self._find_label(prev_node, nid, chain, feedback, node_ids, opts)
                if label:
                    parts.append(f'  \\bXLink[{label}]{{{prev_node}}}{{{nid}}}')
                else:
                    parts.append(f'  \\bXLink{{{prev_node}}}{{{nid}}}')
                prev_node = nid
                continue

            if block.block_fn == 'Sum':
                sign_str = block.params.get('sign', block.params.get('signs', '+-'))
                if isinstance(sign_str, dict):
                    sign_str = sign_str.get('default', '+-')
                parts.append(
                    self._emit_sum(nid, prev_node, sign_str, dist)
                )
                # Auto-link from previous
                label = self._find_label(prev_node, nid, chain, feedback, node_ids, opts)
                if label:
                    parts.append(f'  \\bXLink[{label}]{{{prev_node}}}{{{nid}}}')
                else:
                    parts.append(f'  \\bXLink{{{prev_node}}}{{{nid}}}')
                prev_node = nid
                continue

            # Generic block (Gain, TranFn, etc.)
            content = self._block_content(block, opts)
            parts.append(
                f'  \\bXBlocL[{dist}]{{{nid}}}{{{content}}}{{{prev_node}}}'
            )
            prev_node = nid

        # Emit feedback arcs
        for fb in feedback:
            src_block, dst_block, line = fb
            src_nid = node_ids.get(src_block.name, '')
            dst_nid = node_ids.get(dst_block.name, '')
            if not src_nid or not dst_nid:
                continue

            fb_label = ''
            if opts.get('show_signal_labels') and line.label:
                fb_label = line.label

            # Check if there's a feedback block (e.g., H(s)) between src and dst
            fb_block = self._find_feedback_block(src_block, dst_block, feedback)
            if fb_block:
                fb_nid = node_ids.get(fb_block.name, _sanitize_node_id(fb_block.name))
                node_ids[fb_block.name] = fb_nid
                fb_content = self._block_content(fb_block, opts)
                # Branch point at output
                branch_nid = f'{src_nid}_branch'
                parts.append(f'  \\bXBranchy[{dist}]{{{src_nid}}}{{{branch_nid}}}')
                parts.append(
                    f'  \\bXBlocrL[{dist}]{{{fb_nid}}}{{{fb_content}}}{{{branch_nid}}}'
                )
                label_opt = f'[{fb_label}]' if fb_label else ''
                parts.append(
                    f'  \\bXReturn{label_opt}{{{fb_nid}}}{{{dst_nid}}}'
                )
            else:
                # Simple feedback with no intermediate block
                fb_dist = max(dist, 4.0)
                label_opt = f'[{fb_label}]' if fb_label else ''
                parts.append(
                    f'  \\bXReturn[{fb_dist}]{{{src_nid}}}{{{dst_nid}}}{{{fb_label}}}'
                )

        parts.append(r'\end{tikzpicture}')

        if opts.get('use_resizebox'):
            parts.append('}')

        if opts.get('standalone', True):
            parts.append(r'\end{document}')

        return '\n'.join(parts)

    def get_info(self):
        """Return summary info."""
        return {
            'block_count': len(self.blocks),
            'connection_count': len([ln for ln in self.lines if not ln.hidden]),
            'block_types': sorted(set(b.block_fn for b in self.blocks)),
        }

    # ------------------------------------------------------------------
    # Topology analysis
    # ------------------------------------------------------------------

    def _resolve(self, name):
        """Find a block by name or username."""
        if name in self._block_map:
            return self._block_map[name]
        return self._username_map.get(name)

    def _analyse_topology(self):
        """Detect forward chain and feedback arcs.

        Returns (chain, feedback) or (None, []) if incompatible.
        chain: ordered list of blocks on the main forward path
        feedback: list of (src_block, dst_block, line) tuples
        """
        if not self.blocks:
            return (None, [])

        # Build adjacency from visible lines
        fwd = {}   # block.name -> [(dst_block, line)]
        back_edges = []
        block_names = {b.name for b in self.blocks}

        # Find roots (sources or blocks with no incoming)
        has_incoming = set()

        for line in self.lines:
            if line.hidden:
                continue
            src = self._resolve(line.srcblock)
            dst = self._resolve(line.dstblock)
            if not src or not dst:
                continue
            if src.name not in block_names or dst.name not in block_names:
                continue
            has_incoming.add(dst.name)
            fwd.setdefault(src.name, []).append((dst, line))

        # Identify roots
        roots = [b for b in self.blocks if b.name not in has_incoming]
        if not roots:
            roots = [min(self.blocks, key=lambda b: b.left)]

        # Sort roots by position (leftmost first)
        roots.sort(key=lambda b: b.left)

        # BFS forward chain — must be linear (each node has at most 1 forward successor)
        chain = []
        visited = set()
        queue = [roots[0]]

        while queue:
            block = queue.pop(0)
            if block.name in visited:
                continue
            visited.add(block.name)
            chain.append(block)

            successors = fwd.get(block.name, [])
            forward_succs = []
            for dst, line in successors:
                if dst.name in visited:
                    back_edges.append((block, dst, line))
                else:
                    forward_succs.append((dst, line))

            if len(forward_succs) > 1:
                # Parallel paths — not blox compatible
                return (None, [])

            if forward_succs:
                queue.append(forward_succs[0][0])

        # Check all blocks are in chain
        if len(chain) < len(self.blocks):
            # Disconnected components — try to include them but flag if branching
            for b in self.blocks:
                if b.name not in visited:
                    # Skip sinks that are not in the chain
                    if b.category == 'Sinks':
                        continue
                    return (None, [])

        return (chain, back_edges)

    def _find_feedback_block(self, src_block, dst_block, feedback_list):
        """Check if there's an intermediate block on a feedback path.

        This is a simplification — for now we don't detect intermediate
        blocks. The blox \\bXReturn handles direct feedback.
        """
        return None

    # ------------------------------------------------------------------
    # Block content helpers
    # ------------------------------------------------------------------

    def _is_source(self, block):
        return block.category == 'Sources'

    def _is_sink(self, block):
        return block.category == 'Sinks'

    def _block_content(self, block, opts):
        """Generate the content string for a blox block node."""
        fn = block.block_fn
        show_values = opts.get('show_values', True)

        if fn == 'Gain':
            if show_values:
                gain = block.params.get('gain', 1.0)
                if isinstance(gain, (int, float)):
                    val = f'{gain:.4g}' if gain != int(gain) else str(int(gain))
                    return f'${val}$'
                return '$K$'
            return '$K$'

        if fn == 'TranFn':
            if show_values:
                num = block.params.get('numerator', [1.0])
                den = block.params.get('denominator', [1.0, 1.0])
                if isinstance(num, (list, tuple)) and isinstance(den, (list, tuple)):
                    return f'$\\dfrac{{{_poly_to_latex(num)}}}{{{_poly_to_latex(den)}}}$'
            return '$G(s)$'

        if fn == 'DiscreteTranFn':
            if show_values:
                num = block.params.get('numerator', [1.0])
                den = block.params.get('denominator', [1.0, 1.0])
                if isinstance(num, (list, tuple)) and isinstance(den, (list, tuple)):
                    num_tex = _poly_to_latex(num, var='z')
                    den_tex = _poly_to_latex(den, var='z')
                    return f'$\\dfrac{{{num_tex}}}{{{den_tex}}}$'
            return '$H(z)$'

        if fn == 'Integrator':
            return r'$\dfrac{1}{s}$'

        if fn == 'Deriv':
            return '$s$'

        if fn == 'StateSpace':
            if show_values:
                return r'$\dot{x}{=}Ax{+}Bu$'
            return '$P(s)$'

        if fn == 'PID':
            return 'PID'

        if fn == 'Subsystem':
            uname = block.username if block.username != block.name else 'Subsystem'
            return _escape_latex(uname)

        return _escape_latex(fn)

    def _emit_sum(self, nid, prev_node, sign_str, dist):
        """Emit the appropriate blox comparator/sum command."""
        if sign_str == '+-':
            return f'  \\bXComp[{dist}]{{{nid}}}{{{prev_node}}}'
        if sign_str == '-+':
            return f'  \\bXComp*[{dist}]{{{nid}}}{{{prev_node}}}'
        if len(sign_str) <= 2:
            return f'  \\bXSumb[{dist}]{{{nid}}}{{{prev_node}}}'
        # General case: use bXCompSum with per-port signs
        # bXCompSum{name}{prev}{n}{s}{w}{e}
        n_sign = sign_str[0] if len(sign_str) > 2 else ''
        s_sign = sign_str[1] if len(sign_str) > 1 else ''
        w_sign = sign_str[0] if len(sign_str) > 0 else '+'
        e_sign = ''
        return (
            f'  \\bXCompSum[{dist}]{{{nid}}}{{{prev_node}}}'
            f'{{{n_sign}}}{{{s_sign}}}{{{w_sign}}}{{{e_sign}}}'
        )

    def _find_label(self, src_nid, dst_nid, chain, feedback, node_ids, opts):
        """Find a signal label for the connection between two chain nodes."""
        if not opts.get('show_signal_labels', True):
            return ''

        # Look for a line that matches these two nodes
        for line in self.lines:
            if line.hidden:
                continue
            src = self._resolve(line.srcblock)
            dst = self._resolve(line.dstblock)
            if not src or not dst:
                continue
            s_nid = node_ids.get(src.name, '')
            d_nid = node_ids.get(dst.name, '')
            if s_nid == src_nid and d_nid == dst_nid:
                if line.label:
                    label = line.label.strip()
                    if label.startswith('$') and label.endswith('$'):
                        return label
                    return f'${_escape_latex(label)}$'
        return ''
