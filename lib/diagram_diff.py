"""Semantic diff of two ``.diablos`` diagrams: by block and connection.

A line-based diff of the JSON is mostly noise (coordinates, selection flags,
wire bends). This compares what the diagram *means*:

* simulation settings (``sim_data``);
* blocks, matched by their internal ``name`` (stable across saves), with
  subsystem contents prefixed by the subsystem's path (``ctrl/gain2``):
  added / removed, type, label, parameters, port counts, flip;
* connections, matched by ``src:port -> dst:port``: added / removed.

Layout (positions, sizes, wire bends, routing mode) is ignored unless
``layout=True``. Pure JSON, no Qt and no block classes, so it runs anywhere.
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

_LAYOUT_KEYS = ("coords_left", "coords_top", "coords_width", "coords_height")


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, limit=40) -> str:
    text = json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else repr(v)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _block_name(b: dict) -> str:
    """Internal name as the loader derives it (``FileService``): the saved
    ``name``, else ``block_fn`` lower-cased + ``sid`` (older files)."""
    return str(b.get("name") or f"{str(b.get('block_fn', '')).lower()}{b.get('sid', '')}")


def _flatten(blocks, lines, prefix="") -> Tuple[Dict[str, dict], Dict[tuple, dict]]:
    """``{path: block}`` and ``{(src, sport, dst, dport): line}`` for all levels.

    Wire endpoints are resolved like the loader does: a line may name a block
    by its internal name or (older files) by its canvas label.
    """
    out_blocks: Dict[str, dict] = {}
    out_lines: Dict[tuple, dict] = {}
    resolve: Dict[str, str] = {}
    for b in blocks or []:
        name = _block_name(b)
        resolve.setdefault(name, name)
        if b.get("username"):
            resolve.setdefault(str(b["username"]), name)
        path = prefix + name
        out_blocks[path] = b
        if b.get("sub_blocks") is not None or b.get("sub_lines") is not None:
            sb, sl = _flatten(b.get("sub_blocks"), b.get("sub_lines"), path + "/")
            out_blocks.update(sb)
            out_lines.update(sl)
    for ln in lines or []:
        src, dst = str(ln.get("srcblock")), str(ln.get("dstblock"))
        key = (
            prefix + resolve.get(src, src),
            int(ln.get("srcport", 0)),
            prefix + resolve.get(dst, dst),
            int(ln.get("dstport", 0)),
        )
        out_lines[key] = ln
    return out_blocks, out_lines


def _label(path: str, block: dict) -> str:
    """``Gain "plant gain" (ctrl/gain2)``: type, canvas label, and the path
    when the label alone does not identify the block."""
    leaf = path.rsplit("/", 1)[-1]
    shown = block.get("username") or leaf
    text = f'{block.get("block_fn", "?")} "{shown}"'
    return text if (shown == leaf and "/" not in path) else f"{text} ({path})"


def _dict_changes(old: dict, new: dict) -> List[str]:
    out = []
    for k in new:
        if k not in old:
            out.append(f"{k}: added = {_fmt(new[k])}")
        elif old[k] != new[k]:
            out.append(f"{k}: {_fmt(old[k])} -> {_fmt(new[k])}")
    for k in old:
        if k not in new:
            out.append(f"{k}: removed (was {_fmt(old[k])})")
    return out


def diff_diagrams(a: dict, b: dict, layout: bool = False) -> dict:
    """Structured differences between diagrams ``a`` (old) and ``b`` (new)."""
    ba, la = _flatten(a.get("blocks_data"), a.get("lines_data"))
    bb, lb = _flatten(b.get("blocks_data"), b.get("lines_data"))

    removed = [p for p in ba if p not in bb or ba[p].get("block_fn") != bb[p].get("block_fn")]
    added = [p for p in bb if p not in ba or ba[p].get("block_fn") != bb[p].get("block_fn")]
    changed = []
    for path, new in bb.items():
        old = ba.get(path)
        if old is None or old.get("block_fn") != new.get("block_fn"):
            continue
        items = []
        if old.get("username") != new.get("username"):
            items.append(f"label: {_fmt(old.get('username'))} -> {_fmt(new.get('username'))}")
        items += _dict_changes(old.get("params") or {}, new.get("params") or {})
        for key in ("in_ports", "out_ports", "flipped"):
            if old.get(key) != new.get(key):
                items.append(f"{key}: {_fmt(old.get(key))} -> {_fmt(new.get(key))}")
        if layout:
            for key in _LAYOUT_KEYS:
                if old.get(key) != new.get(key):
                    items.append(f"{key}: {_fmt(old.get(key))} -> {_fmt(new.get(key))}")
        if items:
            changed.append({"block": _label(path, new), "path": path, "changes": items})

    def wire(key, blocks):
        def name(p):
            blk = blocks.get(p)
            user = (blk or {}).get("username")
            return user if user else p

        return f"{name(key[0])}:{key[1]} -> {name(key[2])}:{key[3]}"

    wires_removed = [wire(k, ba) for k in la if k not in lb]
    wires_added = [wire(k, bb) for k in lb if k not in la]
    wires_changed = []
    if layout:
        for k in lb:
            if k in la:
                old, new = la[k], lb[k]
                items = [
                    f"{key}: {_fmt(old.get(key))} -> {_fmt(new.get(key))}"
                    for key in ("points", "routing_mode")
                    if old.get(key) != new.get(key)
                ]
                if items:
                    wires_changed.append({"wire": wire(k, bb), "changes": items})

    return {
        "sim": _dict_changes(a.get("sim_data") or {}, b.get("sim_data") or {}),
        "blocks_added": [_label(p, bb[p]) for p in added],
        "blocks_removed": [_label(p, ba[p]) for p in removed],
        "blocks_changed": changed,
        "wires_added": wires_added,
        "wires_removed": wires_removed,
        "wires_changed": wires_changed,
    }


def is_empty(diff: dict) -> bool:
    return not any(diff.values())


def format_diff(diff: dict) -> str:
    """Human-readable report: ``+`` added, ``-`` removed, ``~`` changed."""
    lines = []
    for item in diff["sim"]:
        lines.append(f"~ sim {item}")
    for b in diff["blocks_removed"]:
        lines.append(f"- block {b}")
    for b in diff["blocks_added"]:
        lines.append(f"+ block {b}")
    for b in diff["blocks_changed"]:
        lines.append(f"~ block {b['block']}")
        lines.extend(f"    {c}" for c in b["changes"])
    for w in diff["wires_removed"]:
        lines.append(f"- wire {w}")
    for w in diff["wires_added"]:
        lines.append(f"+ wire {w}")
    for w in diff["wires_changed"]:
        lines.append(f"~ wire {w['wire']}")
        lines.extend(f"    {c}" for c in w["changes"])
    return "\n".join(lines) if lines else "no semantic differences"
