"""Map simulation error text onto the diagram's blocks (Qt-free).

The engine reports failures as plain strings, usually naming the offending
block by its *flattened* name (``"Sub1/gain0"`` for ``gain0`` inside the
subsystem ``Sub1``). This module walks the real (unflattened) block tree, finds
the blocks such a message refers to, and wraps each message line in a
:class:`~lib.diagram_validator.ValidationError` so the error panel can show it
and the canvas can jump to / highlight the block, navigating into the
enclosing subsystem when needed.
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from lib.diagram_validator import ErrorSeverity, ValidationError


def iter_block_tree(blocks, prefix: str = "", chain=()) -> Iterator[Tuple[str, Any, tuple]]:
    """Yield ``(flat_name, block, enclosing_subsystems)`` for every block, nested too."""
    for block in blocks or []:
        name = f"{prefix}{block.name}"
        yield name, block, tuple(chain)
        sub_blocks = getattr(block, "sub_blocks", None)
        if sub_blocks:
            yield from iter_block_tree(sub_blocks, f"{name}/", tuple(chain) + (block,))


def find_block_chain(root_blocks, block) -> Optional[List[Any]]:
    """Subsystems to enter (outermost first) to reach ``block``; None if absent."""
    for _name, candidate, chain in iter_block_tree(root_blocks):
        if candidate is block:
            return list(chain)
    return None


def find_block_by_path(root_blocks, flat_name: str) -> Optional[Tuple[Any, List[Any]]]:
    """Resolve a flattened name (``"Sub/gain0"``) to ``(block, chain)``."""
    for name, block, chain in iter_block_tree(root_blocks):
        if name == flat_name:
            return block, list(chain)
    return None


def _aliases(name: str, block) -> List[str]:
    out = [name]
    username = getattr(block, "username", "") or ""
    if isinstance(username, str) and len(username) >= 2 and username != block.name:
        out.append(username)
    return out


def locate_blocks(root_blocks, message: str, hint: str = "") -> List[Tuple[str, Any]]:
    """Blocks a message refers to, as ``[(flat_name, block)]``.

    ``hint`` (the engine's ``error_block``) wins when it resolves; otherwise
    every block whose flattened name (or user label) occurs in ``message`` as a
    whole token is returned, in diagram order.
    """
    if hint:
        found = find_block_by_path(root_blocks, hint)
        if found:
            return [(hint, found[0])]
    hits = []
    for name, block, _chain in iter_block_tree(root_blocks):
        for alias in _aliases(name, block):
            if re.search(r"(?<![\w/])" + re.escape(alias) + r"(?![\w/])", message or ""):
                hits.append((name, block))
                break
    return hits


def build_run_errors(root_blocks, message: str, hint: str = "") -> List[ValidationError]:
    """One ERROR entry per non-empty message line, each tied to its blocks."""
    lines = [ln.strip() for ln in (message or "").splitlines() if ln.strip()]
    if not lines:
        return []
    errors = []
    for line in lines:
        # The engine hint only describes a single-line failure.
        hits = locate_blocks(root_blocks, line, hint if len(lines) == 1 else "")
        err = ValidationError(ErrorSeverity.ERROR, line, blocks=[b for _n, b in hits])
        err.block_name = hits[0][0] if hits else ""
        err.block_names = [n for n, _b in hits]
        errors.append(err)
    return errors


def error_blocks_with_ancestors(root_blocks, blocks) -> Dict[int, Any]:
    """``blocks`` plus every enclosing subsystem, keyed by ``id`` (dedup, ordered)."""
    out: Dict[int, Any] = {}
    for block in blocks:
        chain = find_block_chain(root_blocks, block)
        for sub in chain or []:
            out.setdefault(id(sub), sub)
        out.setdefault(id(block), block)
    return out
