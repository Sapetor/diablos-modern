"""
TikZ Exporter - Export block diagrams to TikZ code for LaTeX documents.

Generates publication-ready TikZ diagrams using standard control-systems
conventions: Sum=circle, Gain=triangle, TF=rectangle with fraction.  The block
outline itself is not decided here: it comes from ``resolve_block_shape``
(``lib/models/block_shape.py``), the same rule the canvas paints with, so a
MatrixGain is a triangle and a Goto a tag in both places.

Layout
------
The canvas layout is deliberately discarded.  The diagram is redrawn as a
textbook figure: blocks are ordered by following the signal depth-first from
its sources (reverse postorder -- a real topological order on an acyclic
component, and a chain-following one otherwise), placed on a single row, and
chained with the ``positioning`` library so ``right=<gap> of <previous>``
measures the gap *border to border*.  Absolute coordinates are only used for
the loop lanes, whose offsets have nowhere else to come from.

Nothing may be drawn straight across that row except a wire between immediate
neighbours.  Everything else gets a lane, allocated greedily so overlapping
spans never share one:

* below the row -- wires that close a loop, and the *return-path* blocks
  themselves (a block the user flipped that really does carry signal backwards);
* above the row -- forward wires that skip a column, which drawn straight would
  run through whatever sits in between.

Sizing knobs (``node_distance_cm``, ``lane_base_cm``, ``lane_step_cm``,
``page_width_cm``) are export options and are echoed into the file's header
comment; ``\\resizebox`` is available but off by default, because scaling the
picture scales the type with it.
"""

import math
import re

from lib.export.tex_safety import math_body_is_safe
import logging
from collections import Counter
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _all_finite_numbers(coeffs) -> bool:
    """Return True if *coeffs* is a sequence of real, finite numbers.

    Used to guard _poly_to_latex against user-editable params that may
    contain strings or non-finite values, which would otherwise raise
    mid-export.
    """
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        return False
    for c in coeffs:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return False
        if not math.isfinite(c):
            return False
    return True


def _poly_to_latex(coeffs, var="s"):
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
            term = f"{coef_str}{var}"
        else:
            term = f"{coef_str}{var}^{{{power}}}"
        terms.append(term)
    if not terms:
        return "0"
    result = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            result += f" - {term[1:]}"
        else:
            result += f" + {term}"
    return result


_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # A block named e.g. "cost $5" used to emit a bare $, which opens math
    # mode and makes the whole document uncompilable.
    "$": r"\$",
    # <, > and | are not errors but render as the OT1 ligatures ż, ¡ and --.
    # \ensuremath rather than a bare $<$ because this escaper also feeds
    # contexts that are already in math mode (_name_to_math wraps its result in
    # \text{} inside $...$, and BloxExporter._latex_label wraps in $...$), where
    # a literal $ would switch *out* of math and reintroduce the ligature.
    # \ensuremath is right in both, and needs no font-encoding package.
    "<": r"\ensuremath{<}",
    ">": r"\ensuremath{>}",
    "|": r"\ensuremath{|}",
}
_ESCAPE_RE = re.compile(r"[\\{}_&%#~^$<>|]")


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text (single-pass to avoid double-escaping)."""
    # A newline inside a username becomes a blank line in the .tex, which ends
    # the paragraph mid-node: "Paragraph ended before \\node was complete".
    collapsed = " ".join(text.split())
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group()], collapsed)


def _name_to_math(name: str) -> str:
    """Convert a block username to a math-mode label.

    Examples: 'Kp' -> 'K_p', 'tank' -> 'G_{\\text{tank}}',
    'K1' -> 'K_1', 'G' -> 'G(s)'.
    """
    # Already a single letter: return as-is
    if len(name) == 1 and name.isalpha():
        return name
    # Pattern: letter + digits (K1 -> K_1)
    m = re.match(r"^([A-Za-z])(\d+)$", name)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    # Pattern: letter + lowercase letters (Kp -> K_p, Ki -> K_i)
    m = re.match(r"^([A-Z])([a-z]+)$", name)
    if m:
        return f"{m.group(1)}_{{{m.group(2)}}}"
    # Fallback: wrap in \text
    return f"\\text{{{_escape_latex(name)}}}"


#: Node and coordinate names the exporter emits for itself.
_OUTPUT_NODE = "output"
#: Branch dots are numbered (``bpt1``, ``bpt2``...) -- a diagram can have one
#: per fanned-out output port, and a single shared id silently made every
#: feedback wire leave from the last dot drawn.
_BRANCH_PREFIX = "bpt"
_BRANCH_ID_RE = re.compile(r"^bpt\d*$", re.IGNORECASE)

#: \usetikzlibrary is legal in the document body, so a snippet can carry the
#: libraries it needs and still paste into a paper that only loaded tikz.
#: shapes.symbols supplies the ``signal`` shape used for Goto/From tags.
_TIKZ_LIBRARIES = (
    r"\usetikzlibrary{shapes.geometric, shapes.symbols, arrows.meta, positioning, calc}"
)

#: amsmath (for \dfrac) can only be loaded in a preamble, hence the comment.
_SNIPPET_REQUIREMENTS = [
    r"% Requires \usepackage{tikz} and \usepackage{amsmath} in your preamble.",
    r"% Keep the next line outside any TeX group (a box, a minipage, a scaled",
    r"% wrapper): TeX marks a library loaded globally but defines it locally,",
    r"% so a second copy inside a group is a no-op and that picture then fails.",
    r"% Moving it to your preamble once is the safe way to paste several.",
    _TIKZ_LIBRARIES,
]

#: Opens an exported document. The output is meant to be hand-finished, so the
#: file says how to include it and which knobs matter, in order of usefulness.
_DOCUMENT_HEADER = [
    r"% ---------------------------------------------------------------",
    r"%  Block diagram exported from DiaBloS.",
    r"%",
    r"%  Compile on its own:  pdflatex <this file>",
    r"%",
    r"%  Use it in a paper:   \usepackage{standalone}   % in the preamble",
    r"%                       \includestandalone[width=\columnwidth]{<this file>}",
    r"%                       % plain \input{<this file>} also works",
    r"%",
    r"%  Hand-tuning, in order of usefulness:",
    r"%    node distance=... ...........  gap between neighbouring blocks",
    r"%                                   (blocks are placed border-to-border",
    r"%                                    with `right=of', so a wide caption",
    r"%                                    pushes its neighbour instead of",
    r"%                                    overlapping it)",
    r"%    x=..., y=... ................  the picture's coordinate units;",
    r"%                                   they scale the loop lanes too",
    r"%    block/.style, tf/.style .....  size and look of every such node",
    r"%    signal/.style ...............  wire weight and arrowhead",
    r"%",
    r"%  To fit a column, prefer widening node distance / shortening captions",
    r"%  over \resizebox: scaling the whole picture shrinks the font with it,",
    r"%  and most journals set a minimum figure font size (typically 6-8pt).",
    r"%  The styles are scoped to this picture, so they cannot collide",
    r"%  with your document's own. Nothing here depends on DiaBloS.",
    r"% ---------------------------------------------------------------",
]


def _sanitize_node_id(name: str) -> str:
    """Convert block name/username to a valid TikZ node identifier."""
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    if sanitized and not sanitized[0].isalpha():
        sanitized = "n" + sanitized
    return sanitized or "node"


class TikZExporter:
    """Exports a DiaBloS block diagram to TikZ code."""

    def __init__(self, blocks_list, line_list):
        self.blocks = blocks_list
        self.lines = line_list
        self._block_map = {b.name: b for b in blocks_list}
        self._username_map = {b.username: b for b in blocks_list}
        self._node_ids: Dict[str, str] = {}
        # Layout/routing state, rebuilt by every _picture() call.
        self._block_order: Dict[str, int] = {}
        self._ordered_names = []
        self._spine_gaps = []
        self._layout_gaps = []
        self._lane_blocks: Dict[str, tuple] = {}
        self._lane_depth: Dict[str, float] = {}
        self._feedback_depth: Dict[tuple, float] = {}
        self._detour_lane: Dict[tuple, float] = {}
        self._lane_base = 1.8
        self._lane_step = 0.9
        self._branch_ids: Dict[tuple, str] = {}
        self._branch_primary: Dict[tuple, tuple] = {}
        self._labelled_ports: set = set()
        self._drawn_stub_ports: set = set()
        self._node_driving_ports: set = set()
        self._auto_label_owner: Dict[str, tuple] = {}
        self._output_source = None
        self._output_label = ""
        self._arrow_len = 1.0

    #: Node/coordinate names the exporter emits itself (see _output_continuation).
    #: A block allowed to take one of these silently redefines it, and every
    #: wire that referenced it then resolves to the wrong point -- which still
    #: compiles, so the damage only shows up in the rendered PDF. Derived from
    #: the emitting code rather than restated, so the two cannot drift.
    #: ``bpt``/``bpt1``/``bpt2``... are all reserved, since branch dots are
    #: numbered on demand and the count is not known while ids are assigned.
    _RESERVED_NODE_IDS = frozenset({_OUTPUT_NODE, _BRANCH_PREFIX})

    @classmethod
    def _is_reserved_node_id(cls, node_id: str) -> bool:
        return node_id in cls._RESERVED_NODE_IDS or bool(_BRANCH_ID_RE.match(node_id))

    def _build_node_ids(self, blocks):
        """Assign unique TikZ node IDs to all blocks, avoiding collisions."""
        self._node_ids = {}
        used: set = set(self._RESERVED_NODE_IDS)
        for b in blocks:
            raw = b.username if b.username != b.name else b.name
            base = _sanitize_node_id(raw)
            node_id = base
            counter = 2
            while node_id in used or self._is_reserved_node_id(node_id):
                node_id = f"{base}_{counter}"
                counter += 1
            used.add(node_id)
            self._node_ids[b.name] = node_id

    def _nid(self, block) -> str:
        """Return the TikZ node ID for *block*."""
        return self._node_ids.get(block.name, _sanitize_node_id(block.name))

    def export_document(self, options: Optional[Dict] = None) -> str:
        """Return a full standalone .tex document."""
        lines = _DOCUMENT_HEADER + [
            r"\documentclass[border=5mm]{standalone}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{tikz}",
            r"\usepackage{amsmath}",
            _TIKZ_LIBRARIES,
            r"",
            r"\begin{document}",
            self._picture(options),
            r"\end{document}",
        ]
        return "\n".join(lines)

    def export_snippet(self, options: Optional[Dict] = None) -> str:
        """Return the tikzpicture plus the \\usetikzlibrary line it needs.

        Both entry points compose ``_picture``; neither un-composes the other.
        """
        picture = self._picture(options)
        if picture.startswith("% No blocks"):
            return picture
        return "\n".join(_SNIPPET_REQUIREMENTS + [picture])

    def _picture(self, options: Optional[Dict] = None) -> str:
        """Return the ``tikzpicture`` itself (styles included, no preamble).

        To fit a column, retune ``node_distance_cm`` (or the picture's x/y
        units by hand) rather than reaching for ``use_resizebox``: that option
        exists, but scaling the picture scales its type below the minimum font
        size most journals set for figures.
        """
        opts = {
            "include_sinks": True,
            "sink_as_arrow": True,
            "source_as_arrow": True,
            "show_usernames": True,
            "show_values": True,
            "show_signal_labels": True,
            "fill_blocks": True,
            "page_width_cm": 14.0,
            # \resizebox scales the fonts with the picture, and journals set a
            # minimum figure font size, so it stays off: node distance and the
            # picture's x/y units are the knobs that keep the type readable.
            "use_resizebox": False,
            # Border-to-border gap between neighbouring blocks on the spine.
            "node_distance_cm": 1.5,
            # First loop lane and the spacing between successive lanes.
            "lane_base_cm": 1.8,
            "lane_step_cm": 0.9,
        }
        if options:
            opts.update(options)

        # Filter blocks
        blocks = list(self.blocks)
        if not opts.get("include_sinks", True):
            blocks = [b for b in blocks if b.category != "Sinks"]

        if not blocks:
            return "% No blocks to export\n"

        sink_as_arrow = opts.get("sink_as_arrow", True)
        source_as_arrow = opts.get("source_as_arrow", True)

        # Rendered blocks = those that will appear as TikZ nodes
        rendered = [
            b
            for b in blocks
            if not (sink_as_arrow and self._is_sink_block_obj(b))
            and not (source_as_arrow and self._is_source_block(b))
        ]

        # Build maps using all blocks (connections may reference any)
        self._build_node_ids(blocks)
        self._build_symbol_maps(opts)
        self._compute_block_type_counts()

        # Filter connections before laying out, so the ordering, the loop
        # lanes and the fan-out dots all see exactly the edges that get drawn.
        visible_lines = [ln for ln in self.lines if not ln.hidden]
        if not opts.get("include_sinks", True):
            sink_names = {b.name for b in self.blocks if b.category == "Sinks"}
            sink_usernames = {b.username for b in self.blocks if b.category == "Sinks"}
            visible_lines = [
                ln
                for ln in visible_lines
                if ln.dstblock not in sink_names
                and ln.dstblock not in sink_usernames
                and ln.srcblock not in sink_names
                and ln.srcblock not in sink_usernames
            ]

        # Signal-flow ordering: spine blocks left to right, return-path blocks
        # (flipped, carrying signal backwards) into the loop lane below.
        self._compute_textbook_layout(rendered, visible_lines, opts)

        # Stub arrows (source/sink) are sized from the *tightest* gap in the
        # picture, so they read as part of the same grid and a stub can never
        # run into the block next door.
        gap = min([float(opts.get("node_distance_cm", 1.5))] + list(self._spine_gaps))
        self._arrow_len = round(max(min(gap * 0.6, 1.5), 0.5), 1)

        self._compute_feedback_depths(visible_lines, opts)
        self._compute_fanout(visible_lines, opts)

        # Build output
        parts = []
        if opts.get("use_resizebox"):
            parts.append(r"\resizebox{\textwidth}{!}{%")

        # Styles ride in the picture's own option list, so they are scoped to
        # this figure rather than redefined globally.
        parts.append(r"\begin{tikzpicture}[")
        parts.append(self._tikz_styles(opts))
        parts.append(r"  ]")

        # Nodes — the spine is chained with `right=of`, so a wide caption
        # pushes its neighbour along instead of overlapping it.
        parts.append("  % --- Blocks ---")
        by_name = {}
        for block in rendered:
            by_name.setdefault(block.name, block)
        spine = [by_name[name] for name in self._ordered_names if name in by_name]
        prev_id = None
        for i, block in enumerate(spine):
            gap = self._spine_gaps[i - 1] if 0 < i <= len(self._spine_gaps) else None
            parts.append(self._block_to_node(block, opts, prev_id=prev_id, gap=gap))
            prev_id = self._nid(block)
        for name in self._lane_blocks:
            if name in by_name:
                parts.append(self._block_to_node(by_name[name], opts))

        # Output continuation: branch point + output arrow (must come
        # before connections so feedback routing can reference the dot)
        output_cont = self._output_continuation(rendered, visible_lines, opts)
        if output_cont:
            parts.append("  % --- Output ---")
            parts.append(output_cont)

        dots = self._branch_dot_nodes()
        if dots:
            parts.append("  % --- Branch points ---")
            parts.extend(dots)

        if visible_lines:
            parts.append("  % --- Connections ---")
            for line in visible_lines:
                tikz_draw = self._line_to_tikz_draw(line, opts)
                if tikz_draw:
                    parts.append(tikz_draw)

        parts.append(r"\end{tikzpicture}")

        if opts.get("use_resizebox"):
            parts.append(r"}")

        return "\n".join(parts)

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

        cont_tf = [b for b in self.blocks if b.block_fn == "TranFn"]
        disc_tf = [b for b in self.blocks if b.block_fn == "DiscreteTranFn"]
        ss_blocks = [b for b in self.blocks if b.block_fn == "StateSpace"]
        gain_blocks = [b for b in self.blocks if b.block_fn in ("Gain", "MatrixGain")]

        def _assign(block_list, base_letter, target_map):
            for i, b in enumerate(block_list):
                uname = b.username if b.username != b.name else None
                if uname:
                    math_label = _name_to_math(uname)
                    # Only use username as symbol if it's math-like
                    # (K_p, G_1, etc.), not descriptive (\text{tank})
                    if "\\text{" not in math_label:
                        target_map[b.name] = math_label
                        self._symbol_from_username.add(b.name)
                        continue
                # Auto-assign letter or numbered letter
                if len(block_list) == 1:
                    target_map[b.name] = base_letter
                else:
                    target_map[b.name] = f"{base_letter}_{{{i + 1}}}"

        _assign(cont_tf, "G", self._tf_symbols)
        _assign(disc_tf, "H", self._tf_symbols)
        _assign(ss_blocks, "P", self._tf_symbols)
        _assign(gain_blocks, "K", self._gain_symbols)

    def _compute_block_type_counts(self):
        """Count blocks by type for smart signal labeling decisions."""
        self._block_type_counts = Counter(b.block_fn for b in self.blocks)

    # ------------------------------------------------------------------
    # Textbook layout — topology-based block placement
    # ------------------------------------------------------------------

    def _compute_textbook_layout(self, rendered_blocks, visible_lines=None, opts=None):
        """Order the rendered blocks into a left-to-right signal-flow spine.

        Blocks are chained with the ``positioning`` library rather than placed
        at absolute coordinates: ``right=<gap>cm of <previous>`` measures
        border to border, so a block carrying a long caption or a tall fraction
        pushes its neighbour along instead of growing into it.

        A *flipped* block that carries signal backwards (its predecessor sits
        to the right of its successor) is the textbook return-path element -- a
        gain in the feedback branch.  It leaves the spine and is placed in the
        loop lane underneath, between the two blocks it joins.
        """
        if opts is None:
            opts = {}
        if visible_lines is None:
            visible_lines = [ln for ln in self.lines if not ln.hidden]

        self._block_order = {}
        self._ordered_names = []
        self._spine_gaps = []
        self._layout_gaps = []
        self._lane_blocks = {}
        self._lane_depth = {}

        if not rendered_blocks:
            return

        rendered_names = {b.name for b in rendered_blocks}
        block_by_name = {b.name: b for b in rendered_blocks}

        # Forward adjacency among rendered blocks, plus the set of blocks fed
        # by a source drawn as a bare arrow (they are the natural entry points
        # of a loop that has no in-degree-zero block at all).
        adj = {b.name: [] for b in rendered_blocks}
        rev = {b.name: [] for b in rendered_blocks}
        has_incoming = set()
        source_fed = set()

        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            if src.name in rendered_names and dst.name in rendered_names:
                adj[src.name].append(dst.name)
                rev[dst.name].append(src.name)
                has_incoming.add(dst.name)
            elif dst.name in rendered_names and self._is_source_block(src):
                source_fed.add(dst.name)

        def _left(name):
            b = block_by_name.get(name)
            return getattr(b, "left", 0) if b is not None else 0

        # Entry points, best first: blocks nothing feeds that do feed someone,
        # then (for a closed loop) whatever a source arrow drives, then the
        # leftmost block on the canvas.  Blocks wired to nothing at all are
        # parked at the end so they cannot split the signal path in two.
        roots = sorted(
            (b.name for b in rendered_blocks if b.name not in has_incoming and adj[b.name]),
            key=_left,
        )
        isolated = [
            b.name for b in rendered_blocks if b.name not in has_incoming and not adj[b.name]
        ]

        visited = set()
        order = []

        def _walk(start):
            """Depth-first, emitting the reverse postorder of one component.

            Breadth-first put every successor of a fan-out next to each other,
            which pushed the rest of the chain past them and turned ordinary
            forward wires into long skips.  Following each chain to its end
            first keeps a signal path contiguous, and on an acyclic component
            reverse postorder is a genuine topological order, so no wire has
            to run backwards unless the diagram really does loop.
            """
            if start in visited:
                return
            visited.add(start)
            post = []
            stack = [(start, iter(sorted(adj.get(start, []), key=_left)))]
            while stack:
                node, successors = stack[-1]
                descended = False
                for nb in successors:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append((nb, iter(sorted(adj.get(nb, []), key=_left))))
                        descended = True
                        break
                if not descended:
                    stack.pop()
                    post.append(node)
            order.extend(reversed(post))

        for root in roots:
            _walk(root)
        # Remaining components are cycles (a closed loop has no in-degree-zero
        # block); enter each one where its reference signal comes in.
        remaining = [b.name for b in rendered_blocks if b.name not in visited]
        while remaining:
            start = min(remaining, key=lambda n: (n not in source_fed, _left(n)))
            _walk(start)
            remaining = [n for n in remaining if n not in visited]
        for name in sorted(isolated, key=_left):
            if name not in visited:
                visited.add(name)
                order.append(name)

        index = {name: i for i, name in enumerate(order)}

        # Return-path blocks: flipped, and the signal really does run backwards
        # through them (their feeder sits after their consumer).
        lane_names = []
        for name in order:
            block = block_by_name[name]
            if not getattr(block, "flipped", False):
                continue
            preds = [p for p in rev.get(name, []) if p != name]
            succs = [s for s in adj.get(name, []) if s != name]
            if not preds or not succs:
                continue
            feeder = max(preds, key=lambda n: index[n])
            consumer = min(succs, key=lambda n: index[n])
            if index[feeder] > index[consumer]:
                lane_names.append((name, feeder, consumer))

        lane_set = {name for name, _f, _c in lane_names}
        spine = [n for n in order if n not in lane_set]
        self._block_order = {name: i for i, name in enumerate(spine)}
        for name, feeder, consumer in lane_names:
            # Ordering value only decides feedback-vs-forward for edges that do
            # not touch the lane; the lane's own wires are routed explicitly.
            self._block_order[name] = self._block_order.get(feeder, 0)
            self._lane_blocks[name] = (feeder, consumer)

        # Blocks a loop or detour wire drops into from the side need room in
        # front of them for the vertical approach.
        approached = set()
        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst or dst.name not in self._block_order:
                continue
            if src.name in lane_set:
                approached.add(dst.name)
                continue
            if src.name not in self._block_order or dst.name in lane_set:
                continue
            s_ord = self._block_order[src.name]
            d_ord = self._block_order[dst.name]
            if d_ord <= s_ord or d_ord > s_ord + 1:
                approached.add(dst.name)

        node_distance = float(opts.get("node_distance_cm", 1.5))
        approach_room = round(self._APPROACH + 0.25, 2)
        # A caption wider than the block it names sticks out on both sides, so
        # two blocks that clear each other can still have colliding captions.
        overhang = {
            name: self._caption_overhang(block_by_name[name], opts)
            for name in spine + list(lane_set)
        }
        # Room a gap owes to something other than taste: caption overhang on
        # either side, and the vertical run a lane wire makes in front of a
        # block it drops into.  Squeezing below this is what put captions and
        # wires on top of their neighbours.
        required = []
        for i in range(1, len(spine)):
            room = overhang[spine[i - 1]] + overhang[spine[i]]
            if spine[i] in approached:
                room += approach_room
            required.append(round(room, 2))
        gaps = [round(node_distance + room, 2) for room in required]

        # Wider than the target column: squeeze the discretionary part of every
        # gap, never the type and never the room computed above.
        page_width = opts.get("page_width_cm", 14.0)
        body = sum(self._estimated_width(block_by_name[n]) for n in spine)
        total = body + sum(gaps)
        if gaps and page_width and total > page_width and node_distance:
            slack = max(page_width - body - sum(required), 0.0)
            squeezed = max(slack / len(gaps), self._MIN_GAP)
            gaps = [round(min(squeezed, node_distance) + room, 2) for room in required]

        self._ordered_names = spine
        self._spine_gaps = gaps
        self._layout_gaps = gaps

    #: Rough printed width (cm) of each node style, used only to decide whether
    #: the spine needs compressing to fit ``page_width_cm``.
    _WIDTH_BY_STYLE = {"sum": 0.9, "gain": 1.3, "gain flipped": 1.3, "tf": 2.0, "tag": 1.8}
    _MIN_GAP = 0.6
    #: Horizontal run a loop/detour wire makes before turning into a port.
    _APPROACH = 0.5

    #: Printed width of one footnotesize character, in cm (a caption is prose,
    #: so this only has to be close).
    _CAPTION_CHAR_CM = 0.145

    def _caption_overhang(self, block, opts) -> float:
        """How far *block*'s caption sticks out past one side of the block."""
        caption = self._caption_for(block, opts)
        if not caption:
            return 0.0
        width = len(caption) * self._CAPTION_CHAR_CM
        return round(max(0.0, (width - self._estimated_width(block)) / 2.0), 2)

    def _deepest_block(self, opts) -> float:
        """Distance from the spine to the bottom of the deepest node + caption."""
        deepest = 0.0
        for name in self._ordered_names:
            block = self._resolve_block(name)
            if block is None:
                continue
            ports = self._side_port_count(block)
            if ports > 1 and self._shape_of(block) == "rect" and block.block_fn != "Sum":
                height = max(1.0, 0.7 * ports)
            else:
                height = 1.2 if self._get_block_style(block) == "tf" else 1.0
            below = height / 2.0 + (0.55 if self._caption_for(block, opts) else 0.0)
            deepest = max(deepest, below)
        return round(deepest + 0.45, 2)

    def _estimated_width(self, block) -> float:
        """Approximate printed width of *block*'s node, in cm."""
        style = self._get_block_style(block)
        base = self._WIDTH_BY_STYLE.get(style, 1.6)
        try:
            text = self._get_block_content(block, {"show_values": True})
        except Exception:  # pragma: no cover - width is only a hint
            text = ""
        plain = re.sub(r"\\[a-zA-Z]+|[${}^_]", "", text or "")
        return max(base, 0.18 * len(plain) + 0.5)

    @staticmethod
    def _allocate_lanes(intervals):
        """Greedily pack ``[(key, lo, hi)]`` spans into non-overlapping lanes.

        Narrow spans land closest to the spine, so an inner loop nests inside
        the one that encloses it instead of crossing it.
        """
        lanes = {}
        occupied = []  # lane index -> list of (lo, hi)
        for key, lo, hi in sorted(intervals, key=lambda t: (t[2] - t[1], t[1])):
            placed = None
            for idx, spans in enumerate(occupied):
                if all(hi < a or lo > b for a, b in spans):
                    placed = idx
                    break
            if placed is None:
                placed = len(occupied)
                occupied.append([])
            occupied[placed].append((lo, hi))
            lanes[key] = placed
        return lanes

    def _compute_feedback_depths(self, visible_lines, opts):
        """Assign the Y offset of every wire that cannot run along the spine.

        Two lane stacks are allocated independently: feedback wires below the
        spine and forward wires that skip a column above it.  A forward wire
        drawn as a straight line from column *i* to column *j > i+1* runs
        straight through whatever sits in between, so it detours instead.
        """
        self._feedback_depth = {}
        self._detour_lane = {}
        # The first lane has to clear the tallest node *and* its caption; a
        # four-way Demux is nearly three centimetres deep, and the loop used to
        # be drawn straight through the caption underneath it.
        lane_base = max(float(opts.get("lane_base_cm", 1.8)), self._deepest_block(opts))
        lane_step = float(opts.get("lane_step_cm", 0.9)) or 0.9
        self._lane_base = lane_base
        self._lane_step = lane_step

        below = []
        above = []
        for name, (feeder, consumer) in getattr(self, "_lane_blocks", {}).items():
            lo = min(self._block_order.get(consumer, 0), self._block_order.get(feeder, 0))
            hi = max(self._block_order.get(consumer, 0), self._block_order.get(feeder, 0))
            below.append((("lane", name), lo, hi))

        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            # Skip source/sink-as-arrow connections and anything touching a
            # return-lane block (routed from the lane node itself).
            if self._is_source_block(src) and opts.get("source_as_arrow", True):
                continue
            if self._is_sink_block(dst) and opts.get("sink_as_arrow", True):
                continue
            if src.name in self._lane_blocks or dst.name in self._lane_blocks:
                continue

            src_order = self._block_order.get(src.name, -1)
            dst_order = self._block_order.get(dst.name, -1)
            if src_order < 0 or dst_order < 0:
                continue
            key = self._line_key(line)
            if dst_order <= src_order:
                below.append((key, dst_order, src_order))
            elif dst_order > src_order + 1:
                above.append((key, src_order, dst_order))

        for key, idx in self._allocate_lanes(below).items():
            depth = -round(lane_base + idx * lane_step, 2)
            if isinstance(key, tuple) and key and key[0] == "lane":
                self._lane_depth[key[1]] = depth
            else:
                self._feedback_depth[key] = depth
        for key, idx in self._allocate_lanes(above).items():
            self._detour_lane[key] = round(lane_base + idx * lane_step, 2)

    # ------------------------------------------------------------------
    # Fan-out (branch dots) and output continuation
    # ------------------------------------------------------------------

    @staticmethod
    def _line_key(line):
        return (line.srcblock, line.dstblock, line.srcport, line.dstport)

    def _compute_fanout(self, visible_lines, opts):
        """Number one branch dot per output port that drives several wires.

        A single shared ``bpt`` node meant the second junction in a diagram
        silently reused the first one's coordinate, so every wire that thought
        it left the second dot actually left the first.
        """
        self._branch_ids = {}
        self._branch_primary = {}
        self._labelled_ports = set()
        self._drawn_stub_ports = set()
        self._node_driving_ports = set()
        self._auto_label_owner = {}
        self._output_source = None
        self._output_label = ""

        source_as_arrow = opts.get("source_as_arrow", True)
        sink_as_arrow = opts.get("sink_as_arrow", True)

        # The block whose output leaves the picture: the last one in flow order
        # that also feeds something backwards.
        feedback_sources = set()
        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            if src.name in self._lane_blocks or dst.name in self._lane_blocks:
                if dst.name in self._lane_blocks:
                    feedback_sources.add(src.name)
                continue
            if src.name not in self._block_order or dst.name not in self._block_order:
                continue
            if self._block_order[dst.name] <= self._block_order[src.name]:
                feedback_sources.add(src.name)
        feedback_sources = {n for n in feedback_sources if n in self._block_order}
        if feedback_sources:
            self._output_source = max(feedback_sources, key=lambda n: self._block_order[n])

        # Group the wires that actually leave each output port.
        ports = {}
        for line in visible_lines:
            src = self._resolve_block(line.srcblock)
            dst = self._resolve_block(line.dstblock)
            if not src or not dst:
                continue
            if self._is_source_block(src) and source_as_arrow:
                continue
            if src.name not in self._block_order and src.name not in self._lane_blocks:
                continue
            ports.setdefault((src.name, line.srcport), []).append(line)
            if not (self._is_sink_block(dst) and sink_as_arrow) and (
                dst.name in self._block_order or dst.name in self._lane_blocks
            ):
                self._node_driving_ports.add((src.name, line.srcport))

        # A conventional signal name belongs to one port: the one furthest
        # down the chain, so `y' lands on the plant output and not also on the
        # controller output in front of it.
        for (src_name, port_idx), outgoing in ports.items():
            src = self._resolve_block(src_name)
            if src is None or any(ln.label for ln in outgoing):
                continue
            text = self._CONVENTIONAL_LABELS.get(src.block_fn)
            if not text or self._block_type_counts.get(src.block_fn, 0) > 1:
                continue
            best = self._auto_label_owner.get(text)
            if best is None or self._block_order.get(src_name, 0) > self._block_order.get(
                best[0], 0
            ):
                self._auto_label_owner[text] = (src_name, port_idx)

        # The output continuation is a wire too: the port it leaves also
        # carries the loop back, so it is a junction and needs its dot.
        out_port = (self._output_source, 0) if self._output_source is not None else None
        if out_port is not None:
            ports.setdefault(out_port, [])

        counter = 0
        for key in sorted(
            ports, key=lambda k: (self._block_order.get(k[0], 0), k[1], self._nid_or(k[0]))
        ):
            src_name, port_idx = key
            lines_out = ports[key]
            if len(lines_out) + (1 if key == out_port else 0) < 2:
                continue
            counter += 1
            self._branch_ids[key] = f"{_BRANCH_PREFIX}{counter}"
            if key == out_port:
                # The output arrow is the wire the dot sits on; every other
                # wire from this port starts at the dot.
                self._branch_primary[key] = ("__output__",)
                continue

            def _rank(ln):
                dst = self._resolve_block(ln.dstblock)
                if dst is None:
                    return (9, 0)
                if src_name == self._output_source and self._is_sink_block(dst) and sink_as_arrow:
                    return (0, 0)
                if self._is_sink_block(dst) and sink_as_arrow:
                    return (2, 0)
                s_ord = self._block_order.get(src_name, 0)
                d_ord = self._block_order.get(dst.name, 0)
                if dst.name in self._lane_blocks:
                    return (4, 0)
                if d_ord == s_ord + 1:
                    return (1, 0)
                if d_ord > s_ord:
                    return (3, d_ord)
                return (4, -d_ord)

            self._branch_primary[key] = self._line_key(min(lines_out, key=_rank))

    def _port_drives_a_node(self, port) -> bool:
        """True when this output port already feeds a block drawn in the picture."""
        return port in self._node_driving_ports

    def _nid_or(self, name):
        block = self._resolve_block(name)
        return self._nid(block) if block else name

    def _branch_dot_nodes(self):
        """``\\node[branch]`` for every fanned-out port, on its outgoing wire."""
        nodes = []
        for (src_name, port_idx), node_id in sorted(self._branch_ids.items(), key=lambda kv: kv[1]):
            block = self._resolve_block(src_name)
            if block is None:
                continue
            point = self._port_point(block, port_idx, is_output=True)
            jut = self._DOT_JUT * self._flow_sign(block)
            nodes.append(f"  \\node[branch] ({node_id}) at ($ {point} + ({jut},0) $) {{}};")
        return nodes

    def _branch_start(self, line, src_block):
        """Where *line* leaves its source: the branch dot, unless it is primary."""
        key = (src_block.name, line.srcport)
        node_id = self._branch_ids.get(key)
        if node_id and self._branch_primary.get(key) != self._line_key(line):
            return f"({node_id})"
        return self._port_point(src_block, line.srcport, is_output=True)

    def _output_continuation(self, rendered, visible_lines, opts):
        """Draw the arrow that carries the plant output out of the picture.

        Returns TikZ code for the output coordinate and arrow; the branch dot
        that feeds the loop back is emitted with the other dots.
        """
        last_fb = self._output_source
        if last_fb is None:
            return ""
        last_block = self._block_map.get(last_fb) or self._username_map.get(last_fb)
        if not last_block:
            return ""

        point = self._port_point(last_block, 0, is_output=True)
        arrow_len = (self._arrow_len + 0.5) * self._flow_sign(last_block)

        # Carry the signal label of the sink-as-arrow wire this arrow replaces.
        if opts.get("sink_as_arrow", True) and opts.get("show_signal_labels", True):
            for ln in visible_lines:
                src = self._resolve_block(ln.srcblock)
                dst = self._resolve_block(ln.dstblock)
                if src and dst and src.name == last_fb and self._is_sink_block(dst):
                    self._output_label = self._get_signal_label(ln, src, dst)
                    break
        if self._output_label:
            self._labelled_ports.add((last_fb, 0))

        lines = [f"  \\coordinate ({_OUTPUT_NODE}) at ($ {point} + ({arrow_len},0) $);"]
        label_node = ""
        if self._output_label:
            label_node = f" node[midway, above, font=\\small] {{{self._output_label}}}"
        lines.append(f"  \\draw[signal] {point} --{label_node} ({_OUTPUT_NODE});")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # TikZ style definitions
    # ------------------------------------------------------------------

    def _tikz_styles(self, opts):
        """Return the style definitions as a ``tikzpicture`` option list.

        Scoped to the picture rather than set globally, so the generic names
        (``block``, ``sum``, ``signal``...) cannot displace a host document's.
        """
        fill = ", fill=blue!5" if opts.get("fill_blocks") else ""
        tf_fill = ", fill=blue!8" if opts.get("fill_blocks") else ""
        source_fill = ", fill=green!8" if opts.get("fill_blocks") else ""
        sink_fill = ", fill=red!8" if opts.get("fill_blocks") else ""
        # align=center on every rectangular style: without it a node whose text
        # contains \\ does not typeset at all.
        rounded = "draw, rectangle, rounded corners=2pt, align=center"
        triangle = "draw, isosceles triangle, isosceles triangle apex angle=70"
        # `shape=signal', not a bare `signal': this picture also defines a
        # `signal' *style* for wires, and the bare key would pick that up.
        tag = "draw, shape=signal, signal pointer angle=100, align=center"
        arrow = "-{Stealth[length=2.5mm, width=2mm]}"
        node_distance = round(float(opts.get("node_distance_cm", 1.5)), 2)
        styles = [
            # Border-to-border gap for `right=of', the spine's only spacing knob.
            f"node distance={node_distance}cm",
            f"block/.style={{{rounded}, minimum height=10mm, minimum width=14mm, thick{fill}}}",
            "sum/.style={draw, circle, minimum size=9mm, thick, inner sep=0pt}",
            f"gain/.style={{{triangle}, shape border rotate=0,"
            f" minimum height=10mm, thick, inner sep=2pt{fill}}}",
            f"gain flipped/.style={{{triangle}, shape border rotate=180,"
            f" minimum height=10mm, thick, inner sep=2pt{fill}}}",
            f"tf/.style={{draw, rectangle, align=center, minimum height=12mm,"
            f" minimum width=16mm, thick{tf_fill}}}",
            f"source/.style={{{rounded},"
            f" minimum height=10mm, minimum width=12mm, thick{source_fill}}}",
            f"sink/.style={{{rounded}, minimum height=10mm, minimum width=12mm, thick{sink_fill}}}",
            f"tag/.style={{{tag}, minimum height=8mm, minimum width=14mm, thick{fill}}}",
            f"tag flipped/.style={{{tag}, shape border rotate=180,"
            f" minimum height=8mm, minimum width=14mm, thick{fill}}}",
            f"signal/.style={{{arrow}, semithick}}",
            f"signal wide/.style={{{arrow}, thick}}",
            "branch/.style={fill, circle, minimum size=3.5pt, inner sep=0pt}",
        ]
        return ",\n".join("  " + style for style in styles)

    # ------------------------------------------------------------------
    # Block -> TikZ node
    # ------------------------------------------------------------------

    def _shape_of(self, block) -> str:
        """Outline the canvas paints for *block* (``rect``/``triangle``/...)."""
        from lib.models.block_shape import resolve_block_shape

        try:
            return resolve_block_shape(block)
        except Exception:  # pragma: no cover - a broken block must still export
            logger.debug("Shape lookup failed for %r", getattr(block, "name", "?"), exc_info=True)
            return "rect"

    def _get_block_style(self, block):
        """Return the TikZ style name for a block.

        The outline comes from ``resolve_block_shape``, the same rule the
        canvas paints with, so a MatrixGain/Product/Goto exports as the shape
        the user drew rather than as a nondescript rectangle.
        """
        shape = self._shape_of(block)
        flipped = self._is_reversed(block)
        if shape == "circle":
            return "sum"
        if shape == "triangle":
            return "gain flipped" if flipped else "gain"
        if shape == "tag":
            return "tag flipped" if flipped else "tag"
        fn = block.block_fn
        if fn in ("TranFn", "Integrator", "Deriv", "StateSpace", "DiscreteTranFn"):
            return "tf"
        if block.category == "Sources":
            return "source"
        if block.category == "Sinks":
            return "sink"
        return "block"

    def _get_block_content(self, block, opts):
        """Return the text/math content to place inside the TikZ node."""
        fn = block.block_fn
        show_values = opts.get("show_values", True)

        if fn == "Sum":
            # Signs are placed separately inside the circle; a Sum with too
            # many ports for a circle is drawn as a box and needs a symbol.
            return "" if self._shape_of(block) == "circle" else r"$\Sigma$"

        if fn == "Product":
            return r"$\times$" if self._shape_of(block) == "circle" else _escape_latex(fn)

        if fn == "MatrixGain":
            # A matrix literal never fits in the triangle: name it.
            uname = block.username
            if uname and uname != block.name:
                label = _name_to_math(uname)
                if "\\text{" not in label:
                    return f"${label}$"
            return f"${self._gain_symbols.get(block.name, 'K')}$"

        if fn == "Gain":
            # Prefer username inside the triangle when set
            uname = block.username
            has_custom_name = uname and uname != block.name
            if has_custom_name:
                return f"${_name_to_math(uname)}$"
            if show_values:
                gain = block.params.get("gain", 1.0)
                if isinstance(gain, (int, float)) and math.isfinite(gain):
                    return f"${gain:.4g}$" if gain != int(gain) else f"${int(gain)}$"
                return "$K$"
            sym = self._gain_symbols.get(block.name, "K")
            return f"${sym}$"

        if fn == "TranFn":
            if show_values:
                num = block.params.get("numerator", [1.0])
                den = block.params.get("denominator", [1.0, 1.0])
                if _all_finite_numbers(num) and _all_finite_numbers(den):
                    num_tex = _poly_to_latex(num)
                    den_tex = _poly_to_latex(den)
                    return f"$\\dfrac{{{num_tex}}}{{{den_tex}}}$"
            sym = self._tf_symbols.get(block.name, "G")
            return f"${sym}(s)$"

        if fn == "Integrator":
            return r"$\dfrac{1}{s}$"

        if fn == "Deriv":
            return "$s$"

        if fn == "StateSpace":
            if show_values:
                return r"$\dot{x}{=}Ax{+}Bu$"
            sym = self._tf_symbols.get(block.name, "P")
            return f"${sym}(s)$"

        if fn == "DiscreteTranFn":
            if show_values:
                num = block.params.get("numerator", [1.0])
                den = block.params.get("denominator", [1.0, 1.0])
                if _all_finite_numbers(num) and _all_finite_numbers(den):
                    num_tex = _poly_to_latex(num, var="z")
                    den_tex = _poly_to_latex(den, var="z")
                    return f"$\\dfrac{{{num_tex}}}{{{den_tex}}}$"
            sym = self._tf_symbols.get(block.name, "G")
            return f"${sym}(z)$"

        if fn == "PID":
            return "PID"

        if fn in ("Goto", "From"):
            # The canvas prints the tag inside the pentagon; so does the paper.
            uname = block.username
            if uname and uname != block.name:
                return _escape_latex(uname)
            tag = block.params.get("tag", "")
            if isinstance(tag, dict):
                tag = tag.get("default", "")
            return _escape_latex(str(tag)) if tag else _escape_latex(fn)

        if fn in ("Step", "Constant"):
            if show_values and fn == "Step":
                val = block.params.get("value", 1.0)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    val_str = f"{val:.4g}" if val != int(val) else f"{int(val)}"
                    return f"Step({val_str})"
                return f"Step({_escape_latex(str(val))})"
            if show_values and fn == "Constant":
                val = block.params.get("value", 0.0)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    val_str = f"{val:.4g}" if val != int(val) else f"{int(val)}"
                    return f"${val_str}$"
                return f"$\\text{{{_escape_latex(str(val))}}}$"
            return fn

        if block.category == "Sources":
            return fn

        if block.category == "Sinks":
            label = block.username if block.username != block.name else fn
            return _escape_latex(label)

        if fn == "Subsystem":
            # A masked subsystem exports under its mask's display name; an
            # instance the user renamed keeps that name.
            from lib.masks import mask_display_name

            label = block.username if block.username != block.name else None
            label = label or mask_display_name(block) or "Subsystem"
            return _escape_latex(label)

        return _escape_latex(fn)

    def _block_to_node(self, block, opts, prev_id=None, gap=None):
        """Generate a ``\\node`` line for a block.

        Spine blocks are chained (``right=<gap>cm of <prev>``) so the spacing
        is measured between borders and a wide node never grows into its
        neighbour; a return-path block is dropped into the loop lane halfway
        between the two blocks it joins.
        """
        style = self._get_block_style(block)
        content = self._get_block_content(block, opts)
        node_id = self._nid(block)

        extra_opts = ""
        if block.block_fn == "Subsystem":
            extra_opts = ", double"
        # Spread-out ports need a side long enough to tell them apart: four
        # arrowheads on a 10mm edge read as one thick arrow.
        ports = self._side_port_count(block)
        if ports > 1 and self._shape_of(block) == "rect" and block.block_fn != "Sum":
            extra_opts += f", minimum height={max(10, 7 * ports)}mm"

        placement = ""
        if block.name in getattr(self, "_lane_blocks", {}):
            feeder, consumer = self._lane_blocks[block.name]
            depth = self._lane_depth.get(block.name, -1.8)
            f_id, c_id = self._nid_or(feeder), self._nid_or(consumer)
            placement = f" at ($ ({c_id})!0.5!({f_id}) + (0,{depth}) $)"
        elif prev_id is not None:
            node_distance = round(float(opts.get("node_distance_cm", 1.5)), 2)
            if gap is None or abs(float(gap) - node_distance) < 0.01:
                extra_opts += f", right=of {prev_id}"
            else:
                extra_opts += f", right={round(float(gap), 2)}cm of {prev_id}"

        node_line = f"  \\node[{style}{extra_opts}] ({node_id}){placement} {{{content}}};"

        # Sum block: place +/- signs inside the circle near each input port
        if block.block_fn == "Sum":
            sign_str = block.params.get("sign", block.params.get("signs", "++"))
            if isinstance(sign_str, dict):
                sign_str = sign_str.get("default", "++")
            node_line += self._sum_sign_labels(node_id, sign_str, block)

        caption = self._caption_for(block, opts)
        if caption:
            node_line += f"\n  \\node[below=1mm of {node_id}, font=\\footnotesize] {{{caption}}};"

        return node_line

    def _caption_for(self, block, opts) -> str:
        """The escaped username printed under *block*, or '' when it is not.

        Shared with the layout, which has to leave room for a caption wider
        than the block it names -- otherwise two captions collide even though
        their blocks do not.
        """
        if not opts.get("show_usernames", True):
            return ""
        uname = getattr(block, "username", "")
        if not uname or uname == block.name or block.block_fn == "Sum":
            return ""
        show_values = opts.get("show_values", True)
        # The symbol is "already shown" only when it is actually in the node
        # content AND derived from the username.  With show_values on, a
        # TF/StateSpace prints its polynomial, not its symbol; an auto-assigned
        # symbol (G, G_1) is not the username either, so the caption stays.
        symbol_visible = (
            block.name in self._symbol_from_username
            and (not show_values or block.name in self._gain_symbols)
        ) or (block.block_fn == "Gain" and uname != block.name)
        if symbol_visible:
            return ""
        # Skip a username that is just a case-variant of block_fn
        # (username="mux" for block_fn="Mux"), or one the node already prints --
        # a masked Subsystem puts its name in the box, and captioning it again
        # says it twice.
        content = self._get_block_content(block, opts)
        if (
            uname.lower() == block.block_fn.lower()
            or content.strip() == _escape_latex(uname)
            or block.block_fn == "Subsystem"
        ):
            return ""
        return _escape_latex(uname)

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
            if "-" in sign_str:
                plus_idx = sign_str.index("+") if "+" in sign_str else 0
                minus_idx = sign_str.index("-")
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
                sign = "$+$" if sign_char == "+" else "$-$"
                inner_dist = 0.27
                dx = round(inner_dist * math.cos(math.radians(angle)), 2)
                dy = round(inner_dist * math.sin(math.radians(angle)), 2)
                labels.append(
                    f"\n  \\node[font=\\footnotesize, inner sep=0pt] "
                    f"at ($({node_id}.center)+({dx},{dy})$) {{{sign}}};"
                )

        return "".join(labels)

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

    #: Horizontal run from an output port before the wire may turn, and the
    #: offset of the junction dot along it (the dot must sit on every wire
    #: leaving the port, so no route may turn before it).
    _DOT_JUT = 0.45
    _TURN_JUT = 0.65

    @staticmethod
    def _side_port_count(block) -> int:
        """Largest number of ports the block carries on one side."""
        try:
            return max(
                int(getattr(block, "in_ports", 1) or 1), int(getattr(block, "out_ports", 1) or 1)
            )
        except (TypeError, ValueError):
            return 1

    def _is_reversed(self, block) -> bool:
        """True when this block really does run right-to-left in the picture.

        ``flipped`` is a canvas fact, and the export re-orders the diagram by
        signal flow, so on the left-to-right spine it means nothing: honouring
        it there would put the input on the far side of the block and drag
        every wire across it.  It carries its meaning exactly where the user
        flips a block for -- a gain in the return path, which the layout puts
        in the loop lane pointing back towards the summing junction.
        """
        return bool(getattr(block, "flipped", False)) and block.name in self._lane_blocks

    def _flow_sign(self, block) -> int:
        """+1 when the block's output faces right, -1 when it is reversed."""
        return -1 if self._is_reversed(block) else 1

    def _port_side(self, block, is_output) -> str:
        """``east``/``west``: the side a port sits on, honouring ``flipped``.

        A flipped block takes its input on the east and emits on the west --
        that is the whole point of flipping a gain into a return path -- and
        the triangle/tag styles carry ``shape border rotate=180`` to match.
        """
        if self._flow_sign(block) < 0:
            return "west" if is_output else "east"
        return "east" if is_output else "west"

    def _get_port_anchor(self, block, port_idx, is_output):
        """Return the TikZ anchor name for a port on a block.

        For a Sum this is the angle its sign sits at; for everything else it is
        the side of the node.  Blocks with several ports on one side spread
        them out, which needs a coordinate rather than a named anchor -- see
        :meth:`_port_point`.
        """
        fn = block.block_fn

        if fn == "Sum":
            if is_output:
                return self._port_side(block, True)
            sign_str = block.params.get("sign", block.params.get("signs", "++"))
            if isinstance(sign_str, dict):
                sign_str = sign_str.get("default", "++")
            angles = self._sum_port_angles(sign_str)
            if port_idx < len(angles):
                angle = angles[port_idx]
                if self._flow_sign(block) < 0:
                    angle = (180 - angle) % 360
                return str(angle)
            return self._port_side(block, False)

        # Gain/tag: west/east are the base-midpoint and the tip, so a wire
        # meets them horizontally; the slanted edge would make it diagonal.
        return self._port_side(block, is_output)

    def _port_point(self, block, port_idx, is_output) -> str:
        """Return a TikZ coordinate (parenthesised) for one port of *block*.

        Ports on a rectangular block are spread evenly down its side in port
        order, so a Mux/Demux/StateSpace with four signals gets four distinct
        arrows instead of four arrowheads stacked on one anchor.
        """
        nid = self._nid(block)
        anchor = self._get_port_anchor(block, port_idx, is_output)
        try:
            total = int(block.out_ports if is_output else block.in_ports)
        except (TypeError, ValueError):
            total = 1
        if (
            total > 1
            and anchor in ("east", "west")
            and block.block_fn != "Sum"
            and self._shape_of(block) == "rect"
        ):
            k = min(max(int(port_idx), 0), total - 1)
            frac = round((k + 1) / (total + 1.0), 4)
            top = "north east" if anchor == "east" else "north west"
            bottom = "south east" if anchor == "east" else "south west"
            return f"($ ({nid}.{top})!{frac}!({nid}.{bottom}) $)"
        return f"({nid}.{anchor})"

    @staticmethod
    def _format_explicit_label(label):
        """Format an explicit line label, preserving LaTeX math mode.

        Only labels fully wrapped in ``$...$`` whose body passes the allowlist
        in ``lib.export.tex_safety`` are passed through verbatim (the user
        explicitly opted into math mode). Everything else is escaped so that
        user-supplied text containing backslashes, braces, or other special
        characters cannot inject raw LaTeX into the generated (and later
        compiled) .tex output.
        """
        label = label.strip()
        if (
            len(label) >= 2
            and label.startswith("$")
            and label.endswith("$")
            and math_body_is_safe(label[1:-1])
        ):
            return label
        return _escape_latex(label)

    _CONVENTIONAL_LABELS = {
        "Step": "$r$",
        "Constant": "$r$",
        "Sine": "$r$",
        "Ramp": "$r$",
        "Sum": "$e$",
        "Gain": "$u$",
        "TranFn": "$y$",
        "Integrator": "$y$",
        "StateSpace": "$y$",
        "DiscreteTranFn": "$y$",
    }

    # Block types that should never get auto-generated signal labels
    _NO_SIGNAL_LABEL = {"Mux", "Demux", "Subsystem", "Inport", "Outport", "Switch", "Terminator"}

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
            return ""

        count = self._block_type_counts.get(fn, 0)
        if count <= 1 and fn in self._CONVENTIONAL_LABELS:
            text = self._CONVENTIONAL_LABELS[fn]
            # Two block types can map to the same conventional name (a discrete
            # and a continuous transfer function are both "y"), which printed
            # the same letter on two different signals.  Only the port furthest
            # down the chain keeps it.
            owner = self._auto_label_owner.get(text)
            if owner is not None and owner != (src_block.name, line.srcport):
                return ""
            return text

        # Only use username as signal label if it looks like a math symbol
        # (short: 1-3 chars, starts with a letter). Skip descriptive names
        # like "plant", "observer", "controller" — they clutter the diagram.
        uname = src_block.username
        if uname and uname != src_block.name and len(uname) <= 3:
            return f"${_name_to_math(uname)}$"

        return ""

    def _is_sink_block(self, block):
        """Check if a block is a sink (Scope, Display, etc.)."""
        return block.category == "Sinks"

    _is_sink_block_obj = _is_sink_block  # alias

    def _is_source_block(self, block):
        """Check if a block is a source (Step, Constant, etc.)."""
        return block.category == "Sources"

    def _port_is_centred(self, block, port_idx, is_output) -> bool:
        """True when the port sits on the node's horizontal centre line."""
        anchor = self._get_port_anchor(block, port_idx, is_output)
        if anchor not in ("east", "west"):
            return anchor in ("0", "180")
        try:
            total = int(block.out_ports if is_output else block.in_ports)
        except (TypeError, ValueError):
            return True
        if total <= 1 or block.block_fn == "Sum" or self._shape_of(block) != "rect":
            return True
        return abs((min(max(int(port_idx), 0), total - 1) + 1) / (total + 1.0) - 0.5) < 1e-9

    def _turn_jut(self, block, port_idx, lane=0) -> float:
        """Run a wire makes straight out of a port before it may turn.

        Wires from different ports -- and different lanes off the same port --
        turn at different distances, so their vertical runs never land on top
        of one another.
        """
        offset = self._TURN_JUT + 0.15 * max(int(port_idx), 0) + 0.2 * max(int(lane), 0)
        return round(offset * self._flow_sign(block), 2)

    def _lane_index(self, offset) -> int:
        """Which lane an absolute Y offset belongs to (0 = closest to spine)."""
        step = getattr(self, "_lane_step", 0.9) or 0.9
        base = getattr(self, "_lane_base", 1.8)
        try:
            return max(int(round((abs(float(offset)) - base) / step)), 0)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0

    def _approach(self, dst_block, dst_point) -> str:
        """Point a lane wire drops to before running into *dst_point*."""
        offset = round(-self._APPROACH * self._flow_sign(dst_block), 2)
        return f"($ {dst_point} + ({offset},0) $)"

    @staticmethod
    def _label_node(text, where="midway", side="above"):
        if not text:
            return ""
        return f" node[{where}, {side}, font=\\small] {{{text}}}"

    def _signal_label_for(self, line, src_block, dst_block, opts):
        """Label for this wire, with auto-generated ones used once per port.

        Every wire leaving one port carries the same signal, so repeating the
        conventional name on each branch prints ``y`` twice side by side.
        An explicit label the user typed is always honoured.
        """
        if not opts.get("show_signal_labels", True):
            return ""
        text = self._get_signal_label(line, src_block, dst_block)
        if not text:
            return ""
        if line.label:
            return text
        key = (src_block.name, line.srcport)
        if key in self._labelled_ports:
            return ""
        self._labelled_ports.add(key)
        return text

    def _line_to_tikz_draw(self, line, opts):
        """Generate a ``\\draw`` command for a connection line.

        Wires along the spine are straight; anything that would cut across a
        block -- a loop closing backwards, or a forward wire skipping a column
        -- is routed orthogonally through a lane below or above the spine.
        """
        src_block = self._resolve_block(line.srcblock)
        dst_block = self._resolve_block(line.dstblock)

        if not src_block or not dst_block:
            return None

        src_is_arrow = self._is_source_block(src_block) and opts.get("source_as_arrow", True)
        dst_is_arrow = self._is_sink_block(dst_block) and opts.get("sink_as_arrow", True)

        # Skip connections where both endpoints are arrows (no TikZ nodes)
        if src_is_arrow and dst_is_arrow:
            return None

        dst_anchor = self._get_port_anchor(dst_block, line.dstport, is_output=False)
        dst_point = self._port_point(dst_block, line.dstport, is_output=False)
        style = "signal wide" if line.signal_width > 1 else "signal"
        arrow_len = self._arrow_len

        # --- Source-as-arrow: incoming arrow instead of drawing source node ---
        if src_is_arrow:
            label_node = self._label_node(self._signal_label_for(line, src_block, dst_block, opts))
            if dst_anchor in ("270", "south"):
                start = f"($ {dst_point} + (0,-{arrow_len}) $)"
            else:
                offset = round(-arrow_len * self._flow_sign(dst_block), 2)
                start = f"($ {dst_point} + ({offset},0) $)"
            return f"  \\draw[{style}] {start} --{label_node} {dst_point};"

        src_point = self._port_point(src_block, line.srcport, is_output=True)
        start = self._branch_start(line, src_block)

        # --- Sink-as-arrow: output arrow instead of routing to scope node ---
        if dst_is_arrow:
            # The output continuation already drew this block's main output.
            if src_block.name == self._output_source and line.srcport == 0:
                return None
            # A scope hung off a port whose signal already leaves the block
            # would stack a second arrowhead on the wire that is there.
            port = (src_block.name, line.srcport)
            if port in self._drawn_stub_ports or self._port_drives_a_node(port):
                return None
            self._drawn_stub_ports.add(port)
            label_node = self._label_node(self._signal_label_for(line, src_block, dst_block, opts))
            reach = round(arrow_len * self._flow_sign(src_block), 2)
            if self._port_is_centred(src_block, line.srcport, True):
                return f"  \\draw[{style}] {start} --{label_node} +({reach},0);"
            # A port off the centre line stays on its own row.
            return f"  \\draw[{style}] {src_point} --{label_node} +({reach},0);"

        label_text = self._signal_label_for(line, src_block, dst_block, opts)
        jut = self._turn_jut(src_block, line.srcport)

        # --- Return-path (flipped) block: it lives in the loop lane ---
        if dst_block.name in self._lane_blocks:
            return (
                f"  \\draw[{style}, rounded corners=4pt] {start} -- ++({jut},0)"
                f" |-{self._label_node(label_text, 'near end')} {dst_point};"
            )
        if src_block.name in self._lane_blocks:
            label_below = self._label_node(label_text, "midway", "below")
            if dst_anchor in ("270", "south"):
                return (
                    f"  \\draw[{style}, rounded corners=4pt] {src_point}"
                    f" -|{label_below} {dst_point};"
                )
            return (
                f"  \\draw[{style}, rounded corners=4pt] {src_point}"
                f" -|{label_below} {self._approach(dst_block, dst_point)} -- {dst_point};"
            )

        # --- Detect feedback / column-skipping by flow order ---
        key = self._line_key(line)
        src_order = self._block_order.get(src_block.name, 0)
        dst_order = self._block_order.get(dst_block.name, 0)

        if dst_order > src_order:
            if key not in self._detour_lane:
                # Neighbours on the spine: a straight wire, unless a port sits
                # off the centre line and the run has to step to reach it.
                if self._port_is_centred(src_block, line.srcport, True) and self._port_is_centred(
                    dst_block, line.dstport, False
                ):
                    return (
                        f"  \\draw[{style}] {start} --{self._label_node(label_text)} {dst_point};"
                    )
                # Sharp corners here on purpose: when the two ports happen to
                # sit at the same height the vertical run is zero, and a
                # rounded corner on a zero-length segment draws a visible kink.
                return (
                    f"  \\draw[{style}] {start} -- ++({jut},0)"
                    f" |-{self._label_node(label_text, 'near end')} {dst_point};"
                )
            # Skips at least one block: detour over the spine, never through it.
            lane = self._detour_lane[key]
            jut = self._turn_jut(src_block, line.srcport, self._lane_index(lane))
            back = round(-self._APPROACH * self._flow_sign(dst_block), 2)
            corner = f"($ {dst_point} + ({back},{lane}) $)"
            if dst_anchor in ("270", "south"):
                # Coming in on the underside: drop past the block first, so the
                # wire arrives pointing up instead of grazing the outline.
                below = f"($ {dst_point} + ({back},-{self._APPROACH}) $)"
                return (
                    f"  \\draw[{style}, rounded corners=4pt] {start} -- ++({jut},0)"
                    f" |-{self._label_node(label_text)} {corner} -- {below} -| {dst_point};"
                )
            return (
                f"  \\draw[{style}, rounded corners=4pt] {start} -- ++({jut},0)"
                f" |-{self._label_node(label_text)} {corner} |- {dst_point};"
            )

        # For feedback connections, only use explicitly-set line labels
        # (not auto-generated ones) to avoid duplicating the forward label.
        label_text = self._format_explicit_label(line.label) if line.label else ""
        label_below = self._label_node(label_text, "midway", "below")
        depth = self._feedback_depth.get(key, -1.8)
        jut = self._turn_jut(src_block, line.srcport, self._lane_index(depth))

        if dst_anchor in ("270", "south"):
            tail = f"-|{label_below} {dst_point};"
        else:
            tail = f"-|{label_below} {self._approach(dst_block, dst_point)} -- {dst_point};"
        return (
            f"  \\draw[{style}, rounded corners=4pt] {start} -- ++({jut},0) -- ++(0,{depth}) {tail}"
        )

    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------

    def get_info(self):
        """Return summary info about the diagram."""
        return {
            "block_count": len(self.blocks),
            "connection_count": len([ln for ln in self.lines if not ln.hidden]),
            "block_types": sorted(set(b.block_fn for b in self.blocks)),
        }
