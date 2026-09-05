"""CI gate: every extractable UI string must be translated in every catalog.

Runs the same extraction logic as ``scripts/extract_strings.py`` -- literal
``tr("...")`` arguments plus block display metadata (category names and param
``doc`` strings) -- and asserts each shipped ``locales/*.json`` has a non-empty
entry for every one of them.  A new untranslated string therefore fails CI
rather than silently shipping as English.

To fix a failure::

    python scripts/extract_strings.py --update   # add the missing keys
    # ...then translate the new empty values in locales/<code>.json
"""

import json
import os

import pytest

from scripts import extract_strings

REPO_ROOT = extract_strings.REPO_ROOT
LOCALES_DIR = extract_strings.LOCALES_DIR


def _catalog_codes():
    try:
        return sorted(
            os.path.splitext(f)[0] for f in os.listdir(LOCALES_DIR) if f.endswith(".json")
        )
    except OSError:  # pragma: no cover - locales/ always ships
        return []


@pytest.mark.unit
class TestExtractor:
    def test_extracts_block_metadata(self):
        groups = extract_strings.extract_all()
        assert "Sources" in groups["categories"]
        assert "Sinks" in groups["categories"]
        assert groups["docs"], "no param doc strings were extracted from blocks/"

    def test_extracts_tr_calls(self):
        groups = extract_strings.extract_all()
        assert len(groups["tr"]) > 100, "tr() extraction looks broken"

    def test_keys_are_free_of_fstring_leftovers(self):
        """A tr() key must never contain an un-substituted f-string artefact."""
        for key in extract_strings.extracted_keys():
            assert "{}" not in key, "positional placeholder in key: {!r}".format(key)

    def test_missing_keys_helper(self):
        keys = {"a", "b"}
        assert extract_strings.missing_keys({"a": "x"}, keys) == ["b"]
        assert extract_strings.missing_keys({"a": "x", "b": " "}, keys) == ["b"]


@pytest.mark.unit
@pytest.mark.parametrize("code", _catalog_codes())
class TestCatalogCompleteness:
    def test_catalog_is_complete(self, code):
        path = os.path.join(LOCALES_DIR, code + ".json")
        with open(path, "r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        missing = extract_strings.missing_keys(catalog, extract_strings.extracted_keys())
        assert not missing, (
            "locales/{}.json is missing {} translations, e.g. {}. "
            "Run: python scripts/extract_strings.py --update".format(
                code, len(missing), missing[:5]
            )
        )

    def test_placeholders_are_preserved(self, code):
        """A translation must keep every {named} placeholder of its key."""
        import re

        path = os.path.join(LOCALES_DIR, code + ".json")
        with open(path, "r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        pattern = re.compile(r"\{(\w+)\}")
        broken = []
        for key, value in catalog.items():
            if key.startswith("_") or not isinstance(value, str) or not value:
                continue
            if set(pattern.findall(key)) != set(pattern.findall(value)):
                broken.append(key)
        assert not broken, "placeholders differ between key and translation: {}".format(broken[:5])

    def test_catalog_has_meta(self, code):
        path = os.path.join(LOCALES_DIR, code + ".json")
        with open(path, "r", encoding="utf-8") as fh:
            catalog = json.load(fh)
        meta = catalog.get("_meta", {})
        assert meta.get("code") == code
        assert meta.get("name"), "_meta.name is the label shown in the Language menu"
