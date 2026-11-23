"""Compatibility with code written for the pre-package layout.

Until November 2026 DiaBloS ran only from a checkout, and its code lived in
top-level packages named ``lib`` (the engine), ``blocks`` (the block library)
and ``modern_ui`` (the GUI). They are now ``diablos.core``, ``diablos.blocks``
and ``diablos.gui``. Two kinds of outside code used the old names:

* **User blocks** (``from blocks.base_block import BaseBlock``, ``from
  lib.engine.compiler_kernels import kernel``). DiaBloS's block loader maps
  these automatically, for the block file only (see
  :func:`legacy_import_hook`); nothing to do.
* **Scripts** (``from lib.api import load``). Change them to ``from
  diablos.api import load``. Until then, call :func:`install_legacy_imports`
  before the old imports::

      import diablos.compat
      diablos.compat.install_legacy_imports()
      from lib.api import load        # works, with a DeprecationWarning

The old names are mapped onto the very same module objects, so a class such
as ``BaseBlock`` has one identity whichever name imported it. They are never
installed as real top-level packages (that would shadow any other ``lib`` /
``blocks`` package in the environment), and the mapping is refused for a name
that a real installed package already uses.
"""

import builtins
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import sys
import warnings

logger = logging.getLogger(__name__)

#: Old top-level package -> canonical package.
LEGACY_PACKAGES = {
    "lib": "diablos.core",
    "blocks": "diablos.blocks",
    "modern_ui": "diablos.gui",
}


def canonical_name(name):
    """``"blocks.base_block"`` -> ``"diablos.blocks.base_block"`` (None if not legacy)."""
    head, _, rest = name.partition(".")
    target = LEGACY_PACKAGES.get(head)
    if target is None:
        return None
    return target + ("." + rest if rest else "")


def _real_package_exists(head):
    """True when a genuine (non-DiaBloS) top-level ``head`` package is importable.

    A plain directory without ``__init__.py`` (a namespace package) does not
    count: a user's project folder has its own ``blocks/`` directory of
    project blocks, and the current directory is often on ``sys.path``.
    """
    module = sys.modules.get(head)
    if module is not None:
        name = getattr(module, "__name__", head)
        return not name.startswith("diablos.") and getattr(module, "__file__", None) is not None
    try:
        for finder in sys.meta_path:
            if isinstance(finder, _LegacyFinder):
                continue
            find = getattr(finder, "find_spec", None)
            spec = find(head, None) if find is not None else None
            if spec is not None and spec.origin not in (None, "namespace"):
                return True
    except (ImportError, ValueError):
        return False
    return False


def _deprecated(old, new, where=""):
    message = f"'{old}' is a pre-2026 DiaBloS module name; import '{new}' instead{where}."
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    return message


# --------------------------------------------------------------- user blocks
def legacy_import_hook(source_path):
    """An ``__import__`` for one user block file that maps the old names.

    ``diablos.core.user_blocks`` puts it in the block module's own
    ``__builtins__``, so it applies to that file's code only (including
    imports inside its methods) and never to the rest of the process.
    """
    real_import = builtins.__import__
    noted = set()

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            target = canonical_name(name)
            head = name.partition(".")[0]
            if target is not None and not _real_package_exists(head):
                if head not in noted:
                    noted.add(head)
                    logger.info(
                        "%s",
                        _deprecated(name, target, f" (in {source_path})"),
                    )
                module = real_import(target, globals, locals, fromlist, 0)
                if fromlist:
                    return module
                # ``import blocks.base_block`` binds ``blocks``: hand back the
                # canonical package standing in for it.
                return importlib.import_module(LEGACY_PACKAGES[head])
        return real_import(name, globals, locals, fromlist, level)

    return _import


# ------------------------------------------------------------------- scripts
class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        return importlib.import_module(self.canonical)

    def exec_module(self, module):
        pass  # already executed under its canonical name


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        canonical = canonical_name(name)
        if canonical is None:
            return None
        _deprecated(name, canonical)
        spec = importlib.util.find_spec(canonical)
        if spec is None:
            return None
        alias = importlib.machinery.ModuleSpec(
            name, _AliasLoader(canonical), is_package=spec.submodule_search_locations is not None
        )
        return alias


_installed = None


def install_legacy_imports():
    """Let ``import lib.*`` / ``blocks.*`` / ``modern_ui.*`` resolve to DiaBloS.

    For scripts written before the package layout; prefer changing the
    imports to ``diablos.*``. Raises ``ImportError`` if another installed
    package already owns one of the old names. Calling it twice is harmless.
    """
    global _installed
    if _installed is not None:
        return
    taken = [head for head in LEGACY_PACKAGES if _real_package_exists(head)]
    if taken:
        raise ImportError(
            "Cannot map the old DiaBloS module names: another package already provides "
            + ", ".join(repr(t) for t in taken)
            + ". Import from 'diablos' instead (e.g. 'from diablos.api import load')."
        )
    _installed = _LegacyFinder()
    # First in line, for the same reason as install_moved_module_aliases().
    sys.meta_path.insert(0, _installed)


# ------------------------------------------------- modules moved out of core
#: GUI modules that lived in ``diablos.core`` until 2026-11 (stage 3b of the
#: Qt-free core). Importing an old path returns the module object at the new
#: one (so a theme signal, a class, a monkeypatched global is the same object
#: either way), with a DeprecationWarning. Submodules follow their package.
MOVED_MODULES = {
    "diablos.core.theming.theme_manager": "diablos.gui.themes.theme_manager",
    "diablos.core.plotting.signal_plot": "diablos.gui.plotting.signal_plot",
    "diablos.core.plotting.pyqtgraph_scope_mixin": "diablos.gui.plotting.pyqtgraph_scope_mixin",
    "diablos.core.plotting.field_scope_mixin": "diablos.gui.plotting.field_scope_mixin",
    "diablos.core.dialogs": "diablos.gui.widgets.simulation_dialogs",
    "diablos.core.services.diagram_service": "diablos.gui.services.diagram_service",
    "diablos.core.services.file_dialogs": "diablos.gui.services.file_dialogs",
    "diablos.core.analysis.analyzers": "diablos.gui.analysis.analyzers",
    "diablos.core.analysis.control_system_analyzer": "diablos.gui.analysis.control_system_analyzer",
}


def moved_name(name):
    """New name for a module moved out of ``diablos.core`` (None if not moved)."""
    for old, new in MOVED_MODULES.items():
        if name == old or name.startswith(old + "."):
            return new + name[len(old) :]
    return None


class _MovedFinder(importlib.abc.MetaPathFinder):
    """Resolves the old ``diablos.core.*`` paths of moved modules only."""

    def find_spec(self, name, path, target=None):
        new = moved_name(name)
        if new is None:
            return None
        _deprecated(name, new)
        spec = importlib.util.find_spec(new)
        if spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            name, _AliasLoader(new), is_package=spec.submodule_search_locations is not None
        )


def install_moved_module_aliases():
    """Called by ``import diablos``; touches nothing outside ``diablos.core.*``."""
    # First in line: for a submodule of an aliased package (e.g. the old
    # ``...analyzers.bode``) the regular path finder would otherwise find the
    # file through the package's __path__ and load a second copy under the old
    # name. This finder answers only for the exact moved names.
    if not any(isinstance(f, _MovedFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MovedFinder())
