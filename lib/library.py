"""User block libraries -- saving masked subsystems as reusable palette blocks.

A *library block* is one masked (or plain) Subsystem stored in its own file.
The file is ordinary ``.diablos`` JSON -- it opens in DiaBloS like any other
diagram -- plus a top-level ``library_block`` section that marks it as a
reusable component::

    {
      "version": "2.0",
      "sim_data":    { ... },
      "blocks_data": [ <the subsystem>  ],   # so the file opens as a diagram
      "lines_data":  [],
      "library_block": {
          "format_version": 1,
          "id":    "vehicle",                # = the file stem, the stable key
          "name":  "Vehicle",
          "category": "User Library",
          "description": "...",
          "mask":  { ... },                  # the mask definition
          "block": { ... }                   # the serialized Subsystem, whose
                                             # "sub_blocks" / "sub_lines" hold
                                             # the inner blocks and connections
      }
    }

Discovery
---------
:func:`library_search_paths` returns the folders that are scanned, highest
priority first:

1. every path in the ``DIABLOS_LIBRARY_PATH`` environment variable
   (``os.pathsep``-separated, so several folders can be listed);
2. a ``library/`` folder next to the currently open diagram (project-local);
3. the per-user library folder under :func:`lib.app_paths.user_data_path`.

The first folder that provides a given ``id`` wins, so a project-local or
env-pointed copy shadows the user's own.

Instances are copies
--------------------
Dragging a library block onto the canvas *copies* its contents into the
diagram, so the diagram stays self-contained: deleting or renaming the
library file never breaks a saved diagram.  The instance only keeps a
``library_ref`` (id, file name and version) so "Reload from library" can
re-sync the contents later while preserving the instance's mask parameter
values.
"""

import copy
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from lib.app_paths import get_user_data_dir
from lib.masks import MASK_KEY, normalize_mask

logger = logging.getLogger(__name__)

__all__ = [
    "LIBRARY_FORMAT_VERSION",
    "LIBRARY_ENV_VAR",
    "LIBRARY_DIR_NAME",
    "LIBRARY_SECTION",
    "LibraryBlock",
    "LibraryError",
    "user_library_dir",
    "default_library_dir",
    "library_search_paths",
    "discover_library_blocks",
    "read_library_file",
    "build_library_document",
    "write_library_file",
    "slugify",
    "library_ref_for",
    "attach_library_ref",
    "get_library_ref",
]

#: Bumped when the ``library_block`` section changes incompatibly.
LIBRARY_FORMAT_VERSION = 1

#: Environment variable holding extra library folders (os.pathsep separated).
LIBRARY_ENV_VAR = "DIABLOS_LIBRARY_PATH"

#: Folder name looked for next to the open diagram and under the user data dir.
LIBRARY_DIR_NAME = "library"

#: Top-level key that marks a ``.diablos`` file as a library block.
LIBRARY_SECTION = "library_block"

#: Key under which an instance remembers where it came from.
LIBRARY_REF_KEY = "_library_ref"

_LIBRARY_SUFFIXES = (".diablos", ".json")


class LibraryError(RuntimeError):
    """Raised when a library file cannot be written or read back."""


def slugify(text: str) -> str:
    """Turn a display name into a safe, stable file stem."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip()).strip("._-")
    return slug.lower() or "library_block"


# ---------------------------------------------------------------------------
# The in-memory record
# ---------------------------------------------------------------------------


class LibraryBlock:
    """One discovered library block (a parsed library file)."""

    __slots__ = (
        "block_id",
        "path",
        "mask",
        "block_data",
        "format_version",
        "name",
        "category",
        "description",
    )

    def __init__(
        self,
        block_id: str,
        path: str,
        mask: Optional[Dict[str, Any]],
        block_data: Dict[str, Any],
        format_version: int = LIBRARY_FORMAT_VERSION,
        name: str = "",
        category: str = "User Library",
        description: str = "",
    ) -> None:
        self.block_id = block_id
        self.path = path
        self.mask = mask
        self.block_data = block_data
        self.format_version = format_version
        self.name = name or block_id
        self.category = category or "User Library"
        self.description = description

    @property
    def file_name(self) -> str:
        return os.path.basename(self.path) if self.path else ""

    def instance_block_data(self) -> Dict[str, Any]:
        """A fresh copy of the serialized subsystem, tagged with a library ref.

        The copy is what gets instantiated on the canvas, so every instance is
        independent of the file it came from.
        """
        data = copy.deepcopy(self.block_data)
        params = data.setdefault("params", {})
        if self.mask is not None:
            params[MASK_KEY] = copy.deepcopy(self.mask)
        params[LIBRARY_REF_KEY] = self.reference()
        return data

    def reference(self) -> Dict[str, Any]:
        return {
            "id": self.block_id,
            "file": self.file_name,
            "format_version": self.format_version,
            "name": self.name,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<LibraryBlock {!r} from {!r}>".format(self.block_id, self.path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def user_library_dir(create: bool = False) -> str:
    """Path of the per-user library folder.

    Resolves through :func:`lib.app_paths.get_user_data_dir`, so it is the
    platform data directory in a frozen build and the project root in dev.
    """
    path = os.path.join(get_user_data_dir(), LIBRARY_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _env_library_paths() -> List[str]:
    raw = os.environ.get(LIBRARY_ENV_VAR, "")
    if not raw:
        return []
    return [os.path.abspath(os.path.expanduser(p)) for p in raw.split(os.pathsep) if p.strip()]


def project_library_dir(diagram_path: Optional[str]) -> Optional[str]:
    """``library/`` folder next to ``diagram_path`` (None when not applicable)."""
    if not diagram_path:
        return None
    folder = diagram_path if os.path.isdir(diagram_path) else os.path.dirname(diagram_path)
    if not folder:
        return None
    return os.path.join(os.path.abspath(folder), LIBRARY_DIR_NAME)


def default_library_dir() -> str:
    """Folder a newly saved library block is written to.

    The first ``DIABLOS_LIBRARY_PATH`` entry wins when that variable is set --
    that is the folder the user pointed the app at -- otherwise the per-user
    library folder, created on demand.
    """
    env_paths = _env_library_paths()
    if env_paths:
        return env_paths[0]
    return user_library_dir(create=True)


def library_search_paths(diagram_path: Optional[str] = None) -> List[str]:
    """Folders scanned for library blocks, highest priority first.

    Order: ``DIABLOS_LIBRARY_PATH`` entries, then a ``library/`` folder beside
    the open diagram, then the per-user library folder.  Duplicates are
    removed while preserving priority; non-existent folders are kept in the
    list (the caller skips them) so the order is easy to reason about.
    """
    paths: List[str] = []
    for candidate in _env_library_paths():
        if candidate not in paths:
            paths.append(candidate)
    project = project_library_dir(diagram_path)
    if project and project not in paths:
        paths.append(project)
    user = user_library_dir()
    if user not in paths:
        paths.append(user)
    return paths


def read_library_file(path: str) -> Optional[LibraryBlock]:
    """Parse one library file, or return None when it is not a library block."""
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read library file %s: %s", path, exc)
        return None
    return library_block_from_document(data, path)


def library_block_from_document(data: Any, path: str = "") -> Optional[LibraryBlock]:
    """Build a :class:`LibraryBlock` from a parsed ``.diablos`` document."""
    if not isinstance(data, dict):
        return None
    section = data.get(LIBRARY_SECTION)
    if not isinstance(section, dict):
        return None

    block_data = section.get("block")
    if not isinstance(block_data, dict):
        # Tolerate a document that only carries blocks_data.
        blocks = data.get("blocks_data") or []
        block_data = blocks[0] if blocks and isinstance(blocks[0], dict) else None
    if not isinstance(block_data, dict):
        logger.warning("Library file %s has no subsystem payload", path)
        return None

    mask = section.get("mask") or (block_data.get("params") or {}).get(MASK_KEY)
    if mask:
        try:
            mask = normalize_mask(mask)
        except Exception as exc:
            logger.warning("Library file %s has an invalid mask: %s", path, exc)
            mask = None

    stem = os.path.splitext(os.path.basename(path))[0] if path else ""
    block_id = str(section.get("id") or stem or slugify(section.get("name", "")))
    name = str(section.get("name") or (mask or {}).get("name") or block_id)
    category = str(section.get("category") or (mask or {}).get("category") or "User Library")
    description = str(section.get("description") or (mask or {}).get("description") or "")

    try:
        format_version = int(section.get("format_version", LIBRARY_FORMAT_VERSION))
    except (TypeError, ValueError):
        format_version = LIBRARY_FORMAT_VERSION
    if format_version > LIBRARY_FORMAT_VERSION:
        logger.warning(
            "Library file %s declares format_version %s, newer than the supported %s; "
            "loading it anyway",
            path,
            format_version,
            LIBRARY_FORMAT_VERSION,
        )

    return LibraryBlock(
        block_id=block_id,
        path=path,
        mask=mask,
        block_data=block_data,
        format_version=format_version,
        name=name,
        category=category,
        description=description,
    )


def discover_library_blocks(diagram_path: Optional[str] = None) -> List[LibraryBlock]:
    """Scan the search paths and return the available library blocks.

    Results are sorted by display name.  When the same ``id`` appears in more
    than one folder the higher-priority folder wins (see
    :func:`library_search_paths`).
    """
    found: Dict[str, LibraryBlock] = {}
    for folder in library_search_paths(diagram_path):
        if not folder or not os.path.isdir(folder):
            continue
        try:
            entries = sorted(os.listdir(folder))
        except OSError as exc:
            logger.warning("Could not list library folder %s: %s", folder, exc)
            continue
        for entry in entries:
            if not entry.lower().endswith(_LIBRARY_SUFFIXES):
                continue
            full = os.path.join(folder, entry)
            if not os.path.isfile(full):
                continue
            lib_block = read_library_file(full)
            if lib_block is None:
                continue
            found.setdefault(lib_block.block_id, lib_block)
    return sorted(found.values(), key=lambda lb: (lb.category.lower(), lb.name.lower()))


def find_library_block(block_id: str, diagram_path: Optional[str] = None) -> Optional[LibraryBlock]:
    """Look up one library block by id across the search paths."""
    if not block_id:
        return None
    for lib_block in discover_library_blocks(diagram_path):
        if lib_block.block_id == block_id:
            return lib_block
    return None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def build_library_document(
    block_data: Dict[str, Any],
    block_id: str,
    mask: Optional[Dict[str, Any]] = None,
    sim_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the ``.diablos``-compatible document for a library block."""
    payload = copy.deepcopy(block_data)
    params = payload.setdefault("params", {})
    if mask is None:
        mask = params.get(MASK_KEY)
    if mask is not None:
        mask = normalize_mask(mask)
        params[MASK_KEY] = copy.deepcopy(mask)
    # An instance's back-reference must not be baked into the library file.
    params.pop(LIBRARY_REF_KEY, None)

    name = (mask or {}).get("name") or payload.get("username") or block_id
    category = (mask or {}).get("category") or "User Library"
    description = (mask or {}).get("description", "")

    return {
        "version": "2.0",
        "sim_data": sim_data
        or {
            "wind_width": 1280,
            "wind_height": 770,
            "fps": 60,
            "sim_time": 1.0,
            "sim_dt": 0.01,
            "sim_trange": 100,
            "solver_method": "RK45",
            "rtol": 1e-9,
            "atol": 1e-12,
        },
        "blocks_data": [copy.deepcopy(payload)],
        "lines_data": [],
        LIBRARY_SECTION: {
            "format_version": LIBRARY_FORMAT_VERSION,
            "id": block_id,
            "name": name,
            "category": category,
            "description": description,
            "mask": copy.deepcopy(mask) if mask is not None else None,
            "block": payload,
        },
    }


def write_library_file(
    block_data: Dict[str, Any],
    directory: Optional[str] = None,
    block_id: Optional[str] = None,
    mask: Optional[Dict[str, Any]] = None,
    sim_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Write one library block file and return its path.

    ``directory`` defaults to the per-user library folder, created on demand.
    ``block_id`` defaults to a slug of the mask/display name.
    """
    if mask is None:
        mask = (block_data.get("params") or {}).get(MASK_KEY)
    if block_id is None:
        source = (mask or {}).get("name") or block_data.get("username") or "library_block"
        block_id = slugify(source)
    else:
        block_id = slugify(block_id)

    target_dir = directory or user_library_dir(create=True)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        raise LibraryError("Could not create library folder {}: {}".format(target_dir, exc))

    document = build_library_document(block_data, block_id, mask=mask, sim_data=sim_data)
    path = os.path.join(target_dir, block_id + ".diablos")
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(document, fp, indent=4)
    except OSError as exc:
        raise LibraryError("Could not write library file {}: {}".format(path, exc))
    logger.info("Saved library block %r to %s", block_id, path)
    return path


# ---------------------------------------------------------------------------
# Instance <-> library back-reference
# ---------------------------------------------------------------------------


def library_ref_for(lib_block: LibraryBlock) -> Dict[str, Any]:
    return lib_block.reference()


def attach_library_ref(block: Any, ref: Optional[Dict[str, Any]]) -> None:
    """Stamp (or clear) the library back-reference on a canvas block."""
    params = getattr(block, "params", None)
    if params is None:
        return
    if ref is None:
        params.pop(LIBRARY_REF_KEY, None)
    else:
        params[LIBRARY_REF_KEY] = copy.deepcopy(ref)
    init_list = getattr(block, "init_params_list", None)
    if isinstance(init_list, list):
        if ref is None:
            if LIBRARY_REF_KEY in init_list:
                init_list.remove(LIBRARY_REF_KEY)
        elif LIBRARY_REF_KEY not in init_list:
            init_list.append(LIBRARY_REF_KEY)


def get_library_ref(block: Any) -> Optional[Dict[str, Any]]:
    """Return the block's library back-reference, if it has one."""
    params = getattr(block, "params", None)
    if not isinstance(params, dict):
        return None
    ref = params.get(LIBRARY_REF_KEY)
    return ref if isinstance(ref, dict) else None
