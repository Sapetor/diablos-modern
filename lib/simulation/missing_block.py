"""Placeholder for a saved block whose type is not available.

A ``.diablos`` file stores each block under its type name (``block_fn``).  When
that type is not registered -- a user block module that is not installed, or a
project ``blocks/`` folder the user has not trusted -- the loader used to skip
the record, so the block and every connection touching it vanished from the
open diagram, and the next save deleted them from the file for good.

:class:`MissingBlock` keeps such a record instead.  It is an ordinary
:class:`~lib.simulation.block.DBlock` that

* keeps the saved type name, ports, parameters and geometry, so connections
  still attach to it and ``FileService`` writes it back unchanged;
* is painted with a visible "missing: <type>" label (see ``block_renderer``);
* refuses to run: ``DSim`` stops before simulating a diagram that contains one
  (:func:`missing_block_error`), and the stub's ``execute`` reports an error as
  a second line of defence.

The original record is kept on ``raw_record`` for diagnostics.
"""

import copy
import functools
from typing import Any, Dict, Iterable, List, Optional

from blocks.base_block import BaseBlock
from lib.simulation.block import DBlock

__all__ = [
    "MissingBlock",
    "MissingBlockStub",
    "copy_missing_block",
    "is_missing_block",
    "find_missing_blocks",
    "missing_block_types",
    "missing_block_error",
]


def _port_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def missing_block_error(block: Any) -> str:
    """The error a run reports for the placeholder ``block``."""
    return (
        "Block '{}' is of type '{}', which is not available: its module is not "
        "installed, or the project's blocks/ folder is not trusted. The diagram "
        "cannot run until that block type is loaded.".format(
            getattr(block, "username", "") or getattr(block, "name", "?"),
            getattr(block, "missing_block_type", "?"),
        )
    )


class MissingBlockStub(BaseBlock):
    """``block_instance`` of a :class:`MissingBlock`: same ports, never runs."""

    def __init__(self, type_name: str = "?", n_in: int = 0, n_out: int = 0) -> None:
        self.type_name = type_name
        self.n_in = n_in
        self.n_out = n_out

    @property
    def block_name(self) -> str:
        return "MissingBlock"

    @property
    def params(self) -> Dict[str, Any]:
        return {}

    @property
    def inputs(self) -> List[Dict[str, str]]:
        return [{"name": "in{}".format(i), "type": "any"} for i in range(self.n_in)]

    @property
    def outputs(self) -> List[Dict[str, str]]:
        return [{"name": "out{}".format(i), "type": "any"} for i in range(self.n_out)]

    @property
    def requires_inputs(self) -> bool:
        return False

    @property
    def requires_outputs(self) -> bool:
        return False

    @property
    def doc(self) -> str:
        return "Placeholder for the unavailable block type '{}'.".format(self.type_name)

    def execute(self, time, inputs, params, **kwargs):
        return {
            "E": True,
            "error": "Block type '{}' is not available; the diagram cannot run.".format(
                self.type_name
            ),
        }


class MissingBlock(DBlock):
    """A saved block whose type could not be found, kept as-is."""

    def __init__(self, record: Dict[str, Any], coords: Any, color: Any = None, colors=None):
        block_fn = str(record["block_fn"])
        n_in = _port_count(record.get("in_ports"))
        n_out = _port_count(record.get("out_ports"))
        super().__init__(
            block_fn,
            record["sid"],
            coords,
            color,
            n_in,
            n_out,
            record.get("b_type", 2),
            record.get("io_edit", "none"),
            record.get("fn_name", block_fn.lower()),
            copy.deepcopy(record.get("params") or {}),
            record.get("external", False),
            username=record.get("username", "") or "",
            block_class=functools.partial(MissingBlockStub, block_fn, n_in, n_out),
            colors=colors,
            category="Other",
        )
        #: The type name the file asked for (also ``block_fn``).
        self.missing_block_type: str = block_fn
        #: The record as read from the file.
        self.raw_record: Dict[str, Any] = copy.deepcopy(record)


def copy_missing_block(block: "MissingBlock", sid: int, coords: Any, colors=None) -> "MissingBlock":
    """A new placeholder like ``block`` (same type, ports, params) at ``coords``.

    For copy/paste and duplicate, which would otherwise rebuild a plain
    ``DBlock`` with no block class -- neither a working block nor one that
    refuses to run.
    """
    record = copy.deepcopy(block.raw_record)
    record.pop("name", None)
    record.update(
        sid=sid,
        username="",
        in_ports=block.in_ports,
        out_ports=block.out_ports,
        params=copy.deepcopy(block.saving_params()),
        flipped=False,
    )
    return MissingBlock(record, coords, color=block.b_color, colors=colors)


def is_missing_block(block: Any) -> bool:
    """True for a :class:`MissingBlock` placeholder."""
    return bool(getattr(block, "missing_block_type", None))


def find_missing_blocks(blocks: Optional[Iterable[Any]]) -> List[Any]:
    """Every placeholder in ``blocks``, descending into subsystems."""
    found: List[Any] = []
    for block in blocks or []:
        if is_missing_block(block):
            found.append(block)
        found.extend(find_missing_blocks(getattr(block, "sub_blocks", None)))
    return found


def missing_block_types(blocks: Optional[Iterable[Any]]) -> List[str]:
    """Sorted, de-duplicated type names of the placeholders in ``blocks``."""
    return sorted({b.missing_block_type for b in find_missing_blocks(blocks)})
