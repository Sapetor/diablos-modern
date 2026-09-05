"""Unit tests for the translation core (``lib/i18n.py``) and the extractor."""

import json
import os

import pytest

from lib import i18n


@pytest.fixture
def temp_locales(tmp_path):
    """Point :mod:`lib.i18n` at a throwaway catalog directory."""
    directory = tmp_path / "locales"
    directory.mkdir()
    (directory / "xx.json").write_text(
        json.dumps(
            {
                "_meta": {"code": "xx", "name": "Ejemplo", "english_name": "Example"},
                "_comment": "metadata keys are ignored as translations",
                "File": "Archivo",
                "Loaded {name}": "Cargado {name}",
                "Empty": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    previous = i18n.current_language()
    i18n.set_locales_dir(str(directory))
    i18n.set_language(i18n.DEFAULT_LANGUAGE)
    yield directory
    i18n.set_locales_dir(None)
    i18n.set_language(previous)


@pytest.mark.unit
class TestTranslate:
    def test_falls_back_to_english_when_untranslated(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("Not in the catalog") == "Not in the catalog"

    def test_translates_known_key(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("File") == "Archivo"

    def test_english_is_identity(self, temp_locales):
        i18n.set_language("en")
        assert i18n.tr("File") == "File"

    def test_named_placeholders_are_formatted(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("Loaded {name}", name="demo.diablos") == "Cargado demo.diablos"

    def test_placeholder_formatting_without_translation(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("Saved {name}", name="a.diablos") == "Saved a.diablos"

    def test_empty_translation_is_ignored(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("Empty") == "Empty"

    def test_metadata_keys_are_not_translations(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("_meta") == "_meta"

    def test_broken_placeholder_in_translation_falls_back(self, temp_locales, tmp_path):
        (temp_locales / "yy.json").write_text(
            json.dumps({"Loaded {name}": "Cargado {nombre}"}, ensure_ascii=False),
            encoding="utf-8",
        )
        i18n.clear_cache()
        i18n.set_language("yy")
        assert i18n.tr("Loaded {name}", name="x") == "Loaded x"

    def test_non_string_passes_through(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.tr("") == ""

    def test_has_translation(self, temp_locales):
        i18n.set_language("xx")
        assert i18n.has_translation("File")
        assert not i18n.has_translation("Nope")


@pytest.mark.unit
class TestLanguageSelection:
    def test_set_and_current_language(self, temp_locales):
        assert i18n.set_language("xx") == "xx"
        assert i18n.current_language() == "xx"

    def test_unknown_language_degrades_to_english(self, temp_locales):
        assert i18n.set_language("zz") == "en"
        assert i18n.current_language() == "en"

    def test_regional_code_is_reduced_to_language(self, temp_locales):
        assert i18n.resolve_language("xx_YY") == "xx"

    def test_system_setting_uses_system_locale(self, temp_locales, monkeypatch):
        monkeypatch.setattr(i18n, "system_language", lambda: "xx")
        assert i18n.resolve_language(i18n.SYSTEM_LANGUAGE) == "xx"
        monkeypatch.setattr(i18n, "system_language", lambda: "zz")
        assert i18n.resolve_language(i18n.SYSTEM_LANGUAGE) == "en"

    def test_explicit_setting_beats_system_locale(self, temp_locales, monkeypatch):
        monkeypatch.setattr(i18n, "system_language", lambda: "zz")
        assert i18n.resolve_language("xx") == "xx"

    def test_system_language_from_environment(self, monkeypatch):
        monkeypatch.setattr(
            i18n, "system_language", i18n.system_language
        )  # keep the real implementation
        monkeypatch.setenv("LANG", "es_ES.UTF-8")
        # QLocale wins when Qt is importable; the env fallback must still parse.
        assert isinstance(i18n.system_language(), str)

    def test_listeners_fire_on_language_change(self, temp_locales):
        seen = []
        i18n.add_language_listener(seen.append)
        try:
            i18n.set_language("xx")
            assert seen == ["xx"]
        finally:
            i18n.remove_language_listener(seen.append)

    def test_broken_listener_does_not_break_switching(self, temp_locales):
        def boom(_code):
            raise RuntimeError("listener bug")

        i18n.add_language_listener(boom)
        try:
            assert i18n.set_language("xx") == "xx"
        finally:
            i18n.remove_language_listener(boom)


@pytest.mark.unit
class TestCatalogs:
    def test_available_languages_lists_english_first(self, temp_locales):
        languages = i18n.available_languages()
        assert languages[0]["code"] == "en"
        codes = [entry["code"] for entry in languages]
        assert "xx" in codes

    def test_available_languages_uses_meta_name(self, temp_locales):
        entry = next(e for e in i18n.available_languages() if e["code"] == "xx")
        assert entry["name"] == "Ejemplo"
        assert entry["english_name"] == "Example"

    def test_catalog_meta(self, temp_locales):
        assert i18n.catalog_meta("xx")["name"] == "Ejemplo"
        assert i18n.catalog_meta("en")["code"] == "en"
        assert i18n.catalog_meta("zz") == {}

    def test_missing_catalog_loads_empty(self, temp_locales):
        assert i18n.load_catalog("zz") == {}

    def test_malformed_catalog_is_tolerated(self, temp_locales):
        (temp_locales / "bad.json").write_text("{ not json", encoding="utf-8")
        i18n.clear_cache()
        assert i18n.load_catalog("bad") == {}

    def test_non_object_catalog_is_tolerated(self, temp_locales):
        (temp_locales / "arr.json").write_text("[1, 2]", encoding="utf-8")
        i18n.clear_cache()
        assert i18n.load_catalog("arr") == {}


@pytest.mark.unit
class TestShippedSpanishCatalog:
    """Guards on the real ``locales/es.json`` that ships with the app."""

    def test_spanish_catalog_exists_and_is_valid(self):
        path = os.path.join(i18n.locales_dir(), "es.json")
        assert os.path.exists(path), "locales/es.json is missing"
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert data["_meta"]["code"] == "es"
        assert data["_meta"]["name"]

    def test_spanish_is_offered(self):
        codes = [entry["code"] for entry in i18n.available_languages()]
        assert "es" in codes
