"""Block icons are toolkit-neutral IconPaths (src/diablos/core/icon_path.py).

Every block's draw_icon used to import PyQt6's QPainterPath, so the block
library needed Qt to describe its own icons. It now records the same calls in
an IconPath, which the renderer replays onto a QPainterPath.
"""

import ast
import os

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QPainterPath

from diablos.core.icon_path import IconPath, to_qpainterpath

pytestmark = pytest.mark.unit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One call per supported operation, with the same numbers for both paths.
CALLS = [
    ("moveTo", (0.1, 0.2)),
    ("lineTo", (0.9, 0.2)),
    ("quadTo", (0.95, 0.5, 0.9, 0.8)),
    ("cubicTo", (0.7, 0.9, 0.3, 0.9, 0.1, 0.8)),
    ("closeSubpath", ()),
    ("arcMoveTo", (0.2, 0.2, 0.6, 0.6, 30.0)),
    ("arcTo", (0.2, 0.2, 0.6, 0.6, 30.0, 120.0)),
    ("addRect", (0.3, 0.3, 0.2, 0.1)),
    ("addEllipse", (0.4, 0.4, 0.2, 0.2)),
    ("addRoundedRect", (0.1, 0.6, 0.8, 0.3, 0.05, 0.05)),
]


def _elements(path):
    return [
        (path.elementAt(i).type, path.elementAt(i).x, path.elementAt(i).y)
        for i in range(path.elementCount())
    ]


def test_replay_matches_qpainterpath_element_for_element(qapp):
    icon, direct = IconPath(), QPainterPath()
    for name, args in CALLS:
        getattr(icon, name)(*args)
        getattr(direct, name)(*args)
    assert _elements(to_qpainterpath(icon)) == _elements(direct)
    assert [op[0] for op in icon.ops] == [name for name, _ in CALLS]


def test_is_empty_agrees_with_the_replayed_qpainterpath(qapp):
    moves_only = IconPath()
    moves_only.moveTo(0.1, 0.1)
    moves_only.moveTo(0.2, 0.2)
    drawn = IconPath()
    drawn.moveTo(0.1, 0.1)
    drawn.lineTo(0.9, 0.9)
    for path in (IconPath(), moves_only, drawn):
        assert path.isEmpty() == to_qpainterpath(path).isEmpty()


def test_wrong_argument_count_is_a_clear_error():
    with pytest.raises(TypeError, match="lineTo"):
        IconPath().lineTo(0.5)  # a QPointF-style call, not supported


def test_legacy_qpainterpath_and_none_are_accepted(qapp):
    legacy = QPainterPath()
    legacy.lineTo(1, 1)
    assert to_qpainterpath(legacy) is legacy
    assert to_qpainterpath(None).isEmpty()
    with pytest.raises(TypeError, match="IconPath"):
        to_qpainterpath("not a path")


def test_user_block_icons_are_not_shared_between_instances(qapp):
    """The icon cache keys on (class, geometry); a user block's draw_icon may
    read instance state, so two instances must keep their own icons."""
    from types import SimpleNamespace

    from diablos.gui.renderers.block_renderer import BlockRenderer

    class _Stateful:
        def __init__(self, n):
            self.n = n

        def draw_icon(self, rect):
            path = IconPath()
            for i in range(self.n):
                path.moveTo(0.1, 0.1 * (i + 1))
                path.lineTo(0.9, 0.1 * (i + 1))
            return path

    _Stateful.__module__ = "diablos_user_blocks.stateful"

    def block(inst):
        return SimpleNamespace(
            block_instance=inst,
            block_fn="Stateful",
            rect=QRect(0, 0, 80, 60),
            left=0,
            top=0,
            width=80,
            height=60,
            flipped=False,
        )

    renderer = BlockRenderer()
    one = renderer._icon_source_path(block(_Stateful(1)))
    two = renderer._icon_source_path(block(_Stateful(2)))
    assert (one.elementCount(), two.elementCount()) == (2, 4)


def test_renderer_draws_a_user_block_that_still_returns_qpainterpath(qapp):
    """Third-party blocks written before IconPath keep their icons."""
    from types import SimpleNamespace

    from diablos.gui.renderers.block_renderer import BlockRenderer

    class _OldStyle:
        def draw_icon(self, rect):
            path = QPainterPath()
            path.moveTo(0.1, 0.1)
            path.lineTo(0.9, 0.9)
            return path

    block = SimpleNamespace(
        block_instance=_OldStyle(),
        block_fn="OldStyleUserBlock",
        rect=QRect(0, 0, 80, 60),
        left=0,
        top=0,
        width=80,
        height=60,
        flipped=False,
    )
    path = BlockRenderer()._icon_source_path(block)
    assert path.elementCount() == 2


def _is_pyqt_import(node):
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").split(".")[0] == "PyQt6"
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == "PyQt6" for a in node.names)
    return False


def _pyqt_imports(tree, module_level_only):
    """Line numbers of PyQt6 imports in ``tree``.

    ``module_level_only`` keeps the imports that run when the module is
    imported: everything outside function bodies, including top-level
    ``try`` / ``if`` / ``with`` blocks and class bodies.
    """
    if not module_level_only:
        return [n.lineno for n in ast.walk(tree) if _is_pyqt_import(n)]
    found = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return  # runs only when called
        if _is_pyqt_import(node):
            found.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


def test_guard_sees_imports_in_try_if_and_class_bodies():
    src = (
        "try:\n    from PyQt6.QtCore import QPoint\nexcept ImportError:\n    pass\n"
        "if True:\n    import PyQt6\n"
        "class A:\n    from PyQt6 import QtGui\n"
        "def f():\n    from PyQt6 import QtWidgets\n"
    )
    assert _pyqt_imports(ast.parse(src), module_level_only=True) == [2, 6, 8]


def test_block_library_imports_no_pyqt6():
    offenders = {}
    for root, _dirs, files in os.walk(os.path.join(REPO, "src", "diablos", "blocks")):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                with open(path, encoding="utf-8") as fh:
                    lines = _pyqt_imports(ast.parse(fh.read()), module_level_only=False)
                if lines:
                    offenders[os.path.relpath(path, REPO)] = lines
    assert offenders == {}


# The engine side (no PyQt6 / pyqtgraph / diablos.gui in diablos.core, at any
# nesting depth) is guarded by tests/unit/test_core_gui_boundary.py.
