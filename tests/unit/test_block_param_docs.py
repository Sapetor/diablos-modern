"""Every public block param must carry a non-empty ``doc``.

Docs are shown in the parameter dialog and translated at display time via
``tr(doc)``; the param validator does not enforce them, so this gate does.
Internal params (leading underscore) are exempt.
"""

import pytest

from lib.block_loader import load_builtin_blocks

# Blocks exempt from the check. PIDBlock is owned by a separate change.
DOC_SKIP = {"PIDBlock"}

_CLASSES = sorted(load_builtin_blocks(), key=lambda c: (c.__module__, c.__name__))


def _public_params():
    for cls in _CLASSES:
        if cls.__name__ in DOC_SKIP:
            continue
        try:
            spec = cls().params
        except Exception:
            continue
        for name, meta in spec.items():
            if name.startswith("_"):
                continue
            yield pytest.param(cls, name, meta, id=f"{cls.__name__}.{name}")


@pytest.mark.unit
@pytest.mark.parametrize("cls,name,meta", list(_public_params()))
def test_public_param_has_doc(cls, name, meta):
    assert isinstance(meta, dict), f"{cls.__name__}.{name} spec is not a dict"
    doc = meta.get("doc")
    assert isinstance(doc, str) and doc.strip(), f"{cls.__name__}.{name} has no doc"
