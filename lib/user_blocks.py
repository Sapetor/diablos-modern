"""User (third-party) block modules -- discovery, isolated loading, reporting.

A *user block* is an ordinary Python module that defines one or more
:class:`~blocks.base_block.BaseBlock` subclasses, dropped into a folder DiaBloS
scans at start-up.  Nothing has to be registered and nothing in the app has to
be edited: the classes found here join the palette next to the built-in blocks
and are saved/loaded in ``.diablos`` files under their ``block_name``.

``docs/BLOCK_API.md`` is the reference for block authors; this module is the
loader behind it.

Discovery
---------
:func:`user_block_search_paths` mirrors :func:`lib.library.library_search_paths`
so the two user-extension mechanisms behave the same way.  Highest priority
first:

1. every folder listed in the ``DIABLOS_BLOCKS_PATH`` environment variable
   (``os.pathsep``-separated);
2. a ``blocks/`` folder next to the currently open diagram (project-local);
3. the per-user folder ``<user data dir>/blocks``
   (:func:`lib.app_paths.user_data_path`).

The first folder providing a given module file name wins, so a project-local
copy shadows the user's own.

In a *development* checkout ``user_data_path("blocks")`` resolves to the
repository's own ``blocks/`` package; that folder is dropped from the search
paths (it is already loaded as the built-in registry) so built-ins are never
imported twice under a second module name.  Frozen builds have a real,
separate per-user folder.

Isolation
---------
Every module is imported on its own: a module that raises is logged with its
traceback and skipped, and the rest still load.  Each class is then checked
with :func:`blocks.base_block.validate_block_class`; a class that breaks the
contract is skipped with the validator's message rather than half-registered.
A user block whose ``block_name`` collides with a built-in is skipped too --
built-ins always win, because the name is the key stored in saved diagrams.

Frozen builds
-------------
Loading goes through ``importlib`` on an explicit file path, so it works
identically in a PyInstaller bundle, where the bundled ``blocks/`` package is
*not* scanned (see ``lib/block_loader._BLOCK_MODULES``).
"""

import logging
import os
import sys
import types
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from blocks.base_block import (
    BLOCK_API_VERSION,
    BaseBlock,
    block_contract_errors,
)
from lib.app_paths import get_user_data_dir
from lib.i18n import tr

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCKS_ENV_VAR",
    "BLOCKS_DIR_NAME",
    "USER_MODULE_PREFIX",
    "UserBlockProblem",
    "UserBlockReport",
    "user_blocks_dir",
    "project_blocks_dir",
    "user_block_search_paths",
    "discover_user_block_files",
    "load_user_blocks",
    "is_user_block",
    "user_block_source",
    "missing_block_names",
    "missing_blocks_message",
]

#: Environment variable holding extra user-block folders (os.pathsep separated).
BLOCKS_ENV_VAR = "DIABLOS_BLOCKS_PATH"

#: Folder name looked for next to the open diagram and under the user data dir.
BLOCKS_DIR_NAME = "blocks"

#: Package name user modules are imported under, so they can never shadow a
#: real import (``blocks.gain`` stays the built-in) and are easy to purge on a
#: reload.
USER_MODULE_PREFIX = "diablos_user_blocks"

#: Marker attributes stamped on a successfully loaded user block class.
USER_BLOCK_FLAG = "__diablos_user_block__"
USER_BLOCK_SOURCE = "__diablos_block_source__"


def _builtin_blocks_dirs() -> Set[str]:
    """Real paths of the built-in ``blocks/`` package directory.

    ``blocks`` is a namespace package (no ``__init__.py``), so its ``__file__``
    is None; the location comes from ``base_block``'s own file, plus every
    ``__path__`` entry for good measure.
    """
    found: Set[str] = set()
    base_file = getattr(sys.modules.get("blocks.base_block"), "__file__", None)
    if base_file:
        found.add(os.path.realpath(os.path.dirname(os.path.abspath(base_file))))
    package = sys.modules.get("blocks")
    for entry in getattr(package, "__path__", []) or []:
        try:
            found.add(os.path.realpath(entry))
        except (TypeError, ValueError):  # pragma: no cover - exotic loaders
            continue
    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class UserBlockProblem:
    """One module or class that could not be loaded, and why."""

    __slots__ = ("path", "message")

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<UserBlockProblem {!r}: {}>".format(self.path, self.message)

    def describe(self) -> str:
        """One-line ``<file>: <message>`` summary for a dialog or the log."""
        name = os.path.basename(self.path) if self.path else "?"
        return "{}: {}".format(name, self.message)


class UserBlockReport:
    """Outcome of one :func:`load_user_blocks` pass."""

    __slots__ = ("classes", "problems", "search_paths")

    def __init__(
        self,
        classes: Optional[List[type]] = None,
        problems: Optional[List[UserBlockProblem]] = None,
        search_paths: Optional[List[str]] = None,
    ) -> None:
        self.classes = classes if classes is not None else []
        self.problems = problems if problems is not None else []
        self.search_paths = search_paths if search_paths is not None else []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<UserBlockReport {} block(s), {} problem(s)>".format(
            len(self.classes), len(self.problems)
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def user_blocks_dir(create: bool = False) -> str:
    """Path of the per-user blocks folder (``<user data dir>/blocks``).

    Resolves through :func:`lib.app_paths.get_user_data_dir`, so it is the
    platform data directory in a frozen build and the project root in dev.
    """
    path = os.path.join(get_user_data_dir(), BLOCKS_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _env_block_paths() -> List[str]:
    raw = os.environ.get(BLOCKS_ENV_VAR, "")
    if not raw:
        return []
    return [os.path.abspath(os.path.expanduser(p)) for p in raw.split(os.pathsep) if p.strip()]


def project_blocks_dir(diagram_path: Optional[str]) -> Optional[str]:
    """``blocks/`` folder next to ``diagram_path`` (None when not applicable)."""
    if not diagram_path:
        return None
    folder = diagram_path if os.path.isdir(diagram_path) else os.path.dirname(diagram_path)
    if not folder:
        return None
    return os.path.join(os.path.abspath(folder), BLOCKS_DIR_NAME)


def user_block_search_paths(diagram_path: Optional[str] = None) -> List[str]:
    """Folders scanned for user block modules, highest priority first.

    Order: ``DIABLOS_BLOCKS_PATH`` entries, then a ``blocks/`` folder beside the
    open diagram, then the per-user blocks folder.  Duplicates are removed
    while preserving priority; non-existent folders are kept in the list (the
    caller skips them) so the order is easy to reason about and can be shown to
    the user verbatim.  The built-in ``blocks/`` package directory is filtered
    out: in a dev checkout the per-user folder resolves to exactly that path.
    """
    builtin = _builtin_blocks_dirs()
    paths: List[str] = []

    def _add(candidate: Optional[str]) -> None:
        if not candidate:
            return
        if os.path.realpath(candidate) in builtin:
            logger.debug(
                "Skipping %s as a user block folder: it is the built-in package", candidate
            )
            return
        if candidate not in paths:
            paths.append(candidate)

    for entry in _env_block_paths():
        _add(entry)
    _add(project_blocks_dir(diagram_path))
    _add(user_blocks_dir())
    return paths


def discover_user_block_files(diagram_path: Optional[str] = None) -> List[str]:
    """Module files to load, in priority order, one per module file name.

    Only top-level ``*.py`` files are picked up; names starting with ``_`` are
    skipped (``__init__.py``, private helpers).  A helper module a user block
    imports is best kept next to it with a leading underscore, or installed as
    a normal Python package.
    """
    chosen: Dict[str, str] = {}
    ordered: List[str] = []
    for folder in user_block_search_paths(diagram_path):
        if not folder or not os.path.isdir(folder):
            continue
        try:
            entries = sorted(os.listdir(folder))
        except OSError as exc:
            logger.warning("Could not list user block folder %s: %s", folder, exc)
            continue
        for entry in entries:
            if not entry.endswith(".py") or entry.startswith("_"):
                continue
            full = os.path.join(folder, entry)
            if not os.path.isfile(full):
                continue
            stem = entry[:-3]
            if stem in chosen:
                logger.info(
                    "User block module %s in %s is shadowed by %s",
                    entry,
                    folder,
                    chosen[stem],
                )
                continue
            chosen[stem] = full
            ordered.append(full)
    return ordered


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _ensure_parent_package() -> None:
    """Register a namespace parent so dotted user module names resolve."""
    if USER_MODULE_PREFIX in sys.modules:
        return
    package = types.ModuleType(USER_MODULE_PREFIX)
    package.__path__ = []  # namespace package: no filesystem of its own
    sys.modules[USER_MODULE_PREFIX] = package


def purge_user_modules() -> None:
    """Drop every previously imported user module so a reload re-reads the file."""
    for name in [n for n in sys.modules if n.startswith(USER_MODULE_PREFIX + ".")]:
        del sys.modules[name]


def _import_module_file(path: str) -> types.ModuleType:
    """Import ``path`` as ``diablos_user_blocks.<stem>`` (raises on failure).

    The source is compiled and executed directly rather than going through
    ``SourceFileLoader``, for two reasons: no ``__pycache__`` folder appears in
    the user's blocks directory, and "Reload user blocks" always re-reads the
    file.  (A cached ``.pyc`` is validated on source mtime *in whole seconds*
    plus size, so an edit that keeps the size and lands in the same second --
    exactly what tweaking a constant does -- would otherwise reload stale
    bytecode.)
    """
    _ensure_parent_package()
    stem = os.path.splitext(os.path.basename(path))[0]
    module_name = "{}.{}".format(USER_MODULE_PREFIX, stem)

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    code = compile(source, path, "exec")

    module = types.ModuleType(module_name)
    module.__file__ = path
    module.__package__ = USER_MODULE_PREFIX
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _module_api_version(module: types.ModuleType) -> Optional[int]:
    declared = getattr(module, "BLOCK_API_VERSION", None)
    if declared is None:
        return None
    try:
        return int(declared)
    except (TypeError, ValueError):
        return None


def _classes_defined_in(module: types.ModuleType) -> List[type]:
    """Concrete ``BaseBlock`` subclasses *defined* in ``module``.

    Classes merely imported into the module (``from blocks.gain import
    GainBlock``) are ignored, so a helper import cannot re-register a built-in.
    """
    found: List[type] = []
    for name in dir(module):
        obj = getattr(module, name, None)
        if not isinstance(obj, type) or not issubclass(obj, BaseBlock) or obj is BaseBlock:
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        found.append(obj)
    return found


def load_user_blocks(
    diagram_path: Optional[str] = None,
    builtin_names: Optional[Iterable[str]] = None,
    reload: bool = False,
) -> UserBlockReport:
    """Import every discovered user block module and return the valid classes.

    Args:
        diagram_path: path of the open diagram, for the project-local folder.
        builtin_names: ``block_name`` values already taken by built-in blocks.
            A user block that collides with one is skipped -- built-ins win,
            because the name is what saved diagrams store.
        reload: drop previously imported user modules first, so edited files
            are re-read (the "Reload user blocks" action).

    Returns:
        A :class:`UserBlockReport`; problems are also logged.
    """
    if reload:
        purge_user_modules()

    taken: Set[str] = {str(n) for n in (builtin_names or [])}
    report = UserBlockReport(search_paths=user_block_search_paths(diagram_path))

    for path in discover_user_block_files(diagram_path):
        try:
            module = _import_module_file(path)
        except BaseException as exc:  # noqa: BLE001 - a user module may raise anything
            logger.warning("Skipping user block module %s: %s", path, exc, exc_info=True)
            report.problems.append(UserBlockProblem(path, "{}: {}".format(type(exc).__name__, exc)))
            continue

        declared = _module_api_version(module)
        if declared is not None and declared > BLOCK_API_VERSION:
            message = (
                "declares BLOCK_API_VERSION {} but this build implements {}; "
                "update DiaBloS or the block".format(declared, BLOCK_API_VERSION)
            )
            logger.warning("Skipping user block module %s: %s", path, message)
            report.problems.append(UserBlockProblem(path, message))
            continue

        classes = _classes_defined_in(module)
        if not classes:
            logger.info("User block module %s defines no BaseBlock subclass", path)
            continue

        for cls in classes:
            problems = block_contract_errors(cls)
            if problems:
                message = "{} is not a valid block: {}".format(cls.__name__, "; ".join(problems))
                logger.warning("Skipping user block %s from %s: %s", cls.__name__, path, message)
                report.problems.append(UserBlockProblem(path, message))
                continue
            try:
                name = str(cls().block_name)
            except Exception as exc:  # pragma: no cover - validator already ran
                report.problems.append(UserBlockProblem(path, "{}: {}".format(cls.__name__, exc)))
                continue
            if name in taken:
                message = (
                    "block_name '{}' is already taken by a built-in or an "
                    "earlier user block; rename it".format(name)
                )
                logger.warning("Skipping user block %s from %s: %s", cls.__name__, path, message)
                report.problems.append(UserBlockProblem(path, message))
                continue
            taken.add(name)
            setattr(cls, USER_BLOCK_FLAG, True)
            setattr(cls, USER_BLOCK_SOURCE, path)
            report.classes.append(cls)
            logger.info("Loaded user block %r from %s", name, path)

    return report


def is_user_block(obj: Any) -> bool:
    """True when ``obj`` (a block class or palette entry) came from a user folder."""
    cls = getattr(obj, "block_class", obj)
    return bool(getattr(cls, USER_BLOCK_FLAG, False))


def user_block_source(obj: Any) -> str:
    """Path of the file a user block was loaded from ("" for built-ins)."""
    cls = getattr(obj, "block_class", obj)
    return str(getattr(cls, USER_BLOCK_SOURCE, "") or "")


# ---------------------------------------------------------------------------
# Diagrams that reference a block type we do not have
# ---------------------------------------------------------------------------

#: Block types reconstructed directly by the loader, without a palette entry
#: (see ``lib/services/file_service._construct_block``).
_INTRINSIC_BLOCK_TYPES = frozenset({"Subsystem", "Inport", "Outport"})


def missing_block_names(diagram_data: Any, known: Iterable[str]) -> List[str]:
    """Block types a saved diagram uses that ``known`` does not provide.

    ``diagram_data`` is a parsed ``.diablos`` document (or any nested
    blocks_data/sub_blocks structure).  Nested subsystems are walked too, so a
    user block buried inside a subsystem is reported as well.  The result is
    sorted and de-duplicated.
    """
    available = {str(n) for n in known} | _INTRINSIC_BLOCK_TYPES
    missing: Set[str] = set()

    def _walk(records: Any) -> None:
        if not isinstance(records, (list, tuple)):
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            block_fn = record.get("block_fn")
            if isinstance(block_fn, str) and block_fn and block_fn not in available:
                missing.add(block_fn)
            _walk(record.get("sub_blocks"))

    if isinstance(diagram_data, dict):
        _walk(diagram_data.get("blocks_data"))
    else:
        _walk(diagram_data)
    return sorted(missing)


def missing_blocks_message(names: Sequence[str], diagram_path: Optional[str] = None) -> str:
    """Translated explanation for block types the diagram needs but we lack."""
    return tr(
        "This diagram uses block types that are not installed: {blocks}.\n\n"
        "Those blocks were left out of the diagram. User block modules are "
        "loaded from:\n  {paths}\n\n"
        "Put the block's .py file in one of those folders (or list its folder "
        "in the {env} environment variable) and open the diagram again.",
        blocks=", ".join(names),
        paths="\n  ".join(user_block_search_paths(diagram_path)),
        env=BLOCKS_ENV_VAR,
    )
