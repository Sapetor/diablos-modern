#!/usr/bin/env python3
"""Extract translatable strings and sync them into ``locales/*.json``.

What counts as a translatable string
------------------------------------
1. The first argument of any ``tr("...")`` or ``tr_noop("...")`` call (also
   attribute calls with those names, e.g. ``i18n.tr(...)``) found anywhere
   under the scanned source roots.  Literal strings -- including implicit
   concatenation and references to a module-level string constant -- are
   collected; a ``tr(variable)`` call cannot be, so mark the literal with
   ``tr_noop`` where it is declared instead.
2. Block *display* metadata, which is translated at display time rather than at
   the source site: a block's ``category`` name, its ``doc`` blurb, and every
   ``"doc"``/``"group"`` value inside its ``params`` spec.  ``block_name``,
   parameter keys and category identifiers are **never** rewritten -- they are
   persisted in ``.diablos`` files and used as registry keys -- so only the
   human-readable text is extracted.

Usage
-----
    python scripts/extract_strings.py              # report only
    python scripts/extract_strings.py --update     # add missing keys
    python scripts/extract_strings.py --update --fill-english
    python scripts/extract_strings.py --list       # print every extracted key

``--update`` adds missing keys with an empty value (``--fill-english`` seeds
them with the English source instead), rewrites each catalog sorted by key with
``_meta`` first, and **never deletes** an existing translation.  Keys no longer
found in the source are reported as "stale" so a human can decide.

Exit status is 1 when a catalog is missing keys (and ``--update`` was not
passed), which makes the script usable as a CI gate.
"""

import argparse
import ast
import json
import os
import sys
from typing import Dict, List, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALES_DIR = os.path.join(REPO_ROOT, "locales")

#: Directories scanned for ``tr(...)`` calls.
SOURCE_ROOTS = ("modern_ui", "lib", "blocks", "diablos_modern.py")

#: Directory whose modules also contribute block display metadata.
BLOCKS_ROOT = "blocks"

SKIP_DIRS = {"__pycache__", ".git", "build", "dist", "archive", ".venv", ".venv-win"}

#: Call names whose first argument is a translatable string. ``tr_noop`` marks
#: a literal declared away from its display site (a table of menu labels, a
#: helper's argument) -- see ``lib.i18n.tr_noop``.
TR_FUNCTIONS = ("tr", "tr_noop")


# --- file discovery ---------------------------------------------------------
def iter_python_files(roots=SOURCE_ROOTS) -> List[str]:
    """Yield absolute paths of every ``.py`` file under ``roots``."""
    found: List[str] = []
    for root in roots:
        path = os.path.join(REPO_ROOT, root)
        if os.path.isfile(path):
            found.append(path)
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    found.append(os.path.join(dirpath, name))
    return sorted(found)


def _parse(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError) as exc:
        print("warning: could not parse {}: {}".format(path, exc), file=sys.stderr)
        return None


def _literal_str(node) -> str:
    """Return the value of a string literal node, else ``""``.

    Handles implicit concatenation of adjacent literals, which is how long
    messages are usually written.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_str(node.left)
        right = _literal_str(node.right)
        if left and right:
            return left + right
    return ""


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _module_string_constants(tree: ast.Module) -> Dict[str, str]:
    """Map module-level ``NAME = "literal"`` assignments to their value.

    A few long messages are kept as module constants so tests can assert on
    them (``FIRST_RUN_WELCOME_MESSAGE``, ``WINDOW_TITLE``); ``tr(CONSTANT)``
    must still be extractable.
    """
    constants: Dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = _literal_str(node.value)
        if text:
            constants[target.id] = text
    return constants


# --- extraction -------------------------------------------------------------
def extract_tr_calls(path: str) -> Set[str]:
    """Collect the first arguments of ``tr(...)`` calls in one file.

    String literals (including implicit concatenation) and references to a
    module-level string constant are resolved; anything else -- a variable, an
    f-string, an attribute -- cannot be extracted and should be avoided at
    ``tr()`` call sites.
    """
    tree = _parse(path)
    if tree is None:
        return set()
    constants = _module_string_constants(tree)
    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in TR_FUNCTIONS and node.args:
            arg = node.args[0]
            text = _literal_str(arg)
            if not text and isinstance(arg, ast.Name):
                text = constants.get(arg.id, "")
            if text:
                found.add(text)
    return found


def _property_return_strings(class_node: ast.ClassDef, name: str) -> List[ast.AST]:
    """Return the returned expressions of a property named ``name``."""
    values = []
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            for sub in ast.walk(item):
                if isinstance(sub, ast.Return) and sub.value is not None:
                    values.append(sub.value)
    return values


def extract_block_metadata(path: str) -> Tuple[Set[str], Set[str]]:
    """Collect ``category`` names and param ``doc`` strings from a block file."""
    tree = _parse(path)
    if tree is None:
        return set(), set()
    categories: Set[str] = set()
    docs: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for value in _property_return_strings(node, "category"):
            text = _literal_str(value)
            if text:
                categories.add(text)
        # The block's own `doc` blurb, rendered in the property editor's
        # documentation section (property_editor.py: tr(str(block.doc))).
        for value in _property_return_strings(node, "doc"):
            text = _literal_str(value)
            if text:
                docs.add(text)
        for value in _property_return_strings(node, "params"):
            for sub in ast.walk(value):
                if not isinstance(sub, ast.Dict):
                    continue
                for key, val in zip(sub.keys, sub.values):
                    if key is None:
                        continue
                    # "doc" is the tooltip; "group" is the collapsible section
                    # title the property editor renders above the field.
                    if _literal_str(key) in ("doc", "group"):
                        text = _literal_str(val)
                        if text:
                            docs.add(text)
    return categories, docs


def extract_all() -> Dict[str, Set[str]]:
    """Return the full extraction grouped by origin."""
    tr_strings: Set[str] = set()
    for path in iter_python_files():
        tr_strings |= extract_tr_calls(path)

    categories: Set[str] = set()
    docs: Set[str] = set()
    for path in iter_python_files((BLOCKS_ROOT,)):
        cat, doc = extract_block_metadata(path)
        categories |= cat
        docs |= doc

    return {"tr": tr_strings, "categories": categories, "docs": docs}


def extracted_keys() -> Set[str]:
    """Every string that must exist in a complete catalog."""
    groups = extract_all()
    return groups["tr"] | groups["categories"] | groups["docs"]


# --- catalogs ---------------------------------------------------------------
def catalog_paths() -> List[str]:
    try:
        names = sorted(f for f in os.listdir(LOCALES_DIR) if f.endswith(".json"))
    except OSError:
        return []
    return [os.path.join(LOCALES_DIR, n) for n in names]


def load_catalog_file(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("{} is not a JSON object".format(path))
    return data


def write_catalog_file(path: str, data: Dict[str, object]) -> None:
    """Write a catalog with ``_meta`` first and the rest sorted by key."""
    ordered = {}
    for key in sorted(k for k in data if k.startswith("_")):
        ordered[key] = data[key]
    for key in sorted(k for k in data if not k.startswith("_")):
        ordered[key] = data[key]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")


def missing_keys(catalog: Dict[str, object], keys: Set[str]) -> List[str]:
    """Keys absent or present-but-empty in ``catalog``."""
    missing = []
    for key in sorted(keys):
        value = catalog.get(key)
        if not isinstance(value, str) or not value.strip():
            missing.append(key)
    return missing


def stale_keys(catalog: Dict[str, object], keys: Set[str]) -> List[str]:
    return sorted(k for k in catalog if not k.startswith("_") and k not in keys)


# --- CLI --------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--update", action="store_true", help="add missing keys to every catalog")
    parser.add_argument(
        "--fill-english",
        action="store_true",
        help="with --update, seed new entries with the English source instead of an empty string",
    )
    parser.add_argument("--list", action="store_true", help="print every extracted key")
    parser.add_argument(
        "--locale", action="append", help="restrict to these locale codes (repeatable)"
    )
    args = parser.parse_args(argv)

    groups = extract_all()
    keys = groups["tr"] | groups["categories"] | groups["docs"]
    print(
        "Extracted {} strings: {} tr() calls, {} block categories, {} param docs".format(
            len(keys), len(groups["tr"]), len(groups["categories"]), len(groups["docs"])
        )
    )
    if args.list:
        for key in sorted(keys):
            print("  {!r}".format(key))

    paths = catalog_paths()
    if args.locale:
        wanted = set(args.locale)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
    if not paths:
        print("No catalogs found in {}".format(LOCALES_DIR))
        return 0

    incomplete = False
    for path in paths:
        code = os.path.splitext(os.path.basename(path))[0]
        try:
            catalog = load_catalog_file(path)
        except (OSError, ValueError) as exc:
            print("error: {}".format(exc), file=sys.stderr)
            incomplete = True
            continue

        missing = missing_keys(catalog, keys)
        stale = stale_keys(catalog, keys)
        print(
            "\n{}: {} translated, {} missing, {} stale".format(
                code, len(keys) - len(missing), len(missing), len(stale)
            )
        )
        for key in missing[:20]:
            print("  missing: {!r}".format(key))
        if len(missing) > 20:
            print("  ... and {} more".format(len(missing) - 20))
        for key in stale[:10]:
            print("  stale:   {!r}".format(key))
        if len(stale) > 10:
            print("  ... and {} more stale".format(len(stale) - 10))

        if args.update:
            for key in missing:
                if key not in catalog or not isinstance(catalog.get(key), str):
                    catalog[key] = key if args.fill_english else ""
            write_catalog_file(path, catalog)
            print("  updated {}".format(os.path.relpath(path, REPO_ROOT)))
        elif missing:
            incomplete = True

    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
