# Language & Translations

DiaBloS Modern ships in **English** and **Spanish**, and adding another language
takes one JSON file and no code changes.

## Switching language

**View ▸ Language** lists:

- **System default** — follow the operating system's locale. This is the
  packaged default.
- **English**
- **Español**

The choice takes effect immediately: the menu bar, toolbar, block palette, dock
titles and status bar retranslate in place. It is remembered across sessions.

!!! note "Windows already open keep their old language"
    A toast confirms the switch with *"Language changed. Open windows keep the
    previous language."* Result windows, dialogs and scope windows that were
    already on screen were built in the previous language; close and reopen them
    to pick up the new one.

### How the language is chosen

In order of priority:

1. **Your explicit choice** in View ▸ Language, stored under the `ui/language`
   setting.
2. **The system locale**, when the setting is *System default*.
3. **English**, if neither yields a catalog that exists.

Asking for a language that has no catalog silently falls back to English rather
than failing.

## What is and is not translated

Translations are keyed by the **English source string**, so an untranslated
entry renders as the original English rather than as a blank or a key name. A
partially translated language is therefore usable rather than broken.

Translated: menus, toolbars, dialogs, the block palette, property panel labels,
block **category** names and parameter **doc** strings, status messages and
error dialogs.

Not translated, by design:

- **Block names** (`Step`, `TranFn`, `Scope`) — they are registry keys and are
  written into saved `.diablos` files, so translating them would break diagram
  portability.
- **Parameter keys** (`gain`, `numerator`) — same reason.
- Log messages and developer-facing exception text.

A few strings currently escape translation as an oversight rather than a
decision — notably the analyzer plot titles (`Bode Magnitude Plot: …`,
`Nyquist Plot: …`, `Root Locus: …`, `LQR Result: …`) and the UI-scale
percentages. They stay English in every language.

## Contributing a translation

Catalogs are plain UTF-8 JSON in `locales/`, one file per language, named by its
language code. There is no `.ts`/`.qm` compile step and no Qt Linguist involved.

### 1. Create the catalog

Copy the metadata header into `locales/<code>.json` — for French:

```json
{
  "_meta": {"code": "fr", "name": "Français", "english_name": "French"}
}
```

`name` is what appears in the **View ▸ Language** menu, so write it in the
language itself.

### 2. Fill it with every key

```bash
python scripts/extract_strings.py --update
```

This scans the source for `tr("…")` arguments plus block category names and
parameter `doc` strings, adds any missing key with an empty value, keeps each
file sorted (with `_meta` first), and **never deletes an existing translation**.
Keys that no longer exist in the source are reported as *stale* rather than
removed.

`--fill-english` seeds new entries with the English text instead of an empty
string, which is handy if you want to translate in place. `--list` prints every
extracted key.

### 3. Translate

Fill in the empty values. Two rules:

- **Keep every `{named}` placeholder exactly as it appears in the key.**
  `"Loaded {name}"` must stay `"Cargado {name}"`, never `"Cargado {nombre}"`. A
  test enforces this.
- Leave the keyboard-shortcut suffix alone where you see one — shortcuts are
  appended outside the translated text.

### 4. Verify

```bash
python scripts/extract_strings.py     # exits non-zero while keys are missing
pytest tests/unit/test_i18n.py tests/unit/test_locale_catalog_complete.py -q
```

`tests/unit/test_locale_catalog_complete.py` runs the extractor over every
catalog in `locales/`, so a new untranslated string fails CI rather than
silently shipping as English. That also means a partial translation cannot be
merged as-is — fill every key, using the English text where you are unsure.

### 5. Done

The language appears in **View ▸ Language** automatically; the menu is built
from `locales/*.json` and labelled from each catalog's `_meta.name`. Nothing
else needs registering, and `locales/` is bundled into the frozen builds, so
your language ships with the next release.

## For developers

The implementation notes — how `tr()` works, what must never be wrapped, how
live retranslation is wired, and how to add strings safely — are in the
[Developer Guide](../DEVELOPER_GUIDE.md) under *Localization*.
