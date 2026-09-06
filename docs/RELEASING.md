# Releasing

Cutting a release is two commands. Everything else is automated by
`.github/workflows/release.yml`, which builds three platform artifacts and
publishes them to GitHub Releases.

```bash
git tag v1.2.0
git push --tags
```

That is the whole trigger — but do the checklist below first, because the tag is
what gets built and there is no way to amend a published release cleanly.

---

## The version is single-sourced

`[project] version` in **`pyproject.toml`** is the only place a version number
is written. Everything else reads it back:

| Consumer | How it reads the version |
|---|---|
| `modern_ui/__init__.py` | Parses `pyproject.toml` in a checkout; reads the bundled `_version.txt` in a frozen build |
| `diablos.spec` | Parses it for `CFBundleShortVersionString` |
| `tools/build.sh` | Parses it for the DMG filename |
| `.github/workflows/release.yml` | Parses it for the Windows zip and Linux tarball filenames |

So bumping one line in `pyproject.toml` renames the artifacts, changes the macOS
bundle version, and updates the string in the window title, all at once.

The **tag** is not derived from that line. Keep them in step by hand: tag
`v1.2.0` after setting `version = "1.2.0"`.

---

## Release checklist

### 1. Make sure main is green

The release workflow calls `.github/workflows/ci.yml` through `workflow_call`
and every build job depends on it, so a tag can never publish an untested tree.
Save yourself the round trip and check locally first:

```bash
ruff check .
ruff format --check .
python tools/sync_block_registry.py --check
QT_QPA_PLATFORM=offscreen pytest tests/
```

`sync_block_registry.py --check` matters: a new block file that nobody synced
works in a checkout and is *silently missing from the packaged app*. CI gates on
it, but catching it here is cheaper.

### 2. Bump the version

Edit `[project] version` in `pyproject.toml`. Use semantic versioning:
breaking changes to the `.diablos` file format or the CLI are a major bump.

### 3. Update the CHANGELOG

Move everything under `## [Unreleased]` into a new
`## [1.2.0] - YYYY-MM-DD` section, and leave a fresh empty `[Unreleased]`
heading above it. The release notes on GitHub are auto-generated from commits,
but `CHANGELOG.md` is what a human reads, and the release body points at it.

### 4. Check the documentation

The docs site is built from the same commit. If the release adds a block,
a menu item or a CLI flag, the pages describing them should already be updated:

```bash
python scripts/audit_wiki_docs.py    # blocks vs. docs/wiki/
mkdocs build --strict                # links, nav, API reference
```

### 5. Commit, tag, push

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "release: 1.2.0"
git push
git tag v1.2.0
git push --tags
```

The workflow triggers on `push` of a tag matching `v*`.

---

## What the workflow produces

`release.yml` runs four jobs after the CI gate:

| Job | Runner | Output |
|---|---|---|
| `macos` | `macos-latest` (arm64) | `dist/DiaBloS-<version>-arm64.dmg`, via `tools/build.sh` |
| `windows` | `windows-latest` | `dist/DiaBloS-<version>-windows-x64.zip` — the whole one-folder `dist/DiaBloS/` layout, zipped |
| `linux` | `ubuntu-latest` | `dist/DiaBloS-<version>-linux-x86_64.tar.gz` — same layout, tarred so the executable bit survives |
| `release` | `ubuntu-latest` | The GitHub release, with all three attached |

All three builds use Python 3.12 and PyInstaller with `diablos.spec`. The macOS
job runs `tools/build.sh`, which also syncs the block registry and builds the
DMG with `hdiutil`; the Windows and Linux jobs run the portable equivalent
(`sync_block_registry.py` then `pyinstaller --noconfirm diablos.spec`) because
`build.sh` is macOS-only.

The `release` job is guarded by `if: startsWith(github.ref, 'refs/tags/v')`, so a
manual `workflow_dispatch` run builds and uploads artifacts for inspection
without publishing anything.

`fail_on_unmatched_files: true` means a missing artifact fails the release
rather than publishing a partial one.

### What the release notes say

The body is a fixed template plus GitHub's auto-generated commit list. It tells
users how to get past the unsigned-binary warnings:

- **macOS** — `xattr -rd com.apple.quarantine /Applications/DiaBloS-arm64.app`
- **Windows** — SmartScreen ▸ More info ▸ Run anyway
- **Linux** — `tar -xzf`, then `./DiaBloS/DiaBloS`

---

## Verifying the artifacts

Do this before announcing the release.

1. **All three assets are attached** and their filenames carry the new version.
2. **macOS**: download the DMG on an Apple Silicon Mac, drag to Applications,
   run the `xattr` command from the release notes, and launch it. The window
   title reads `DiaBloS Modern <version> - Block Diagram Simulator`; check that
   the version is the one you tagged.
3. **Windows**: extract the *whole* `DiaBloS` folder from the zip and run
   `DiaBloS.exe`. Running it from inside the zip viewer will fail — that is
   expected, not a bug.
4. **Linux**: `tar -xzf` and run `./DiaBloS/DiaBloS`.
5. **Blocks are all there.** Open the palette and confirm nothing is missing —
   this is what the registry sync protects, and a packaged build is where a
   missed sync shows up.
6. **Open an example** from **File ▸ Examples** and run it. A packaged build
   resolves resource paths differently from a checkout
   (`lib/app_paths.py`), so this catches path bugs a source run never sees.

There is no automated Intel-macOS or ARM-Linux artifact. Build those from source
if you need them — see [Building & Packaging](building.md).

---

## How the documentation deploys

`.github/workflows/docs.yml` builds the MkDocs site with `--strict` on every
push and pull request, and on a push to `main` it also runs
`mkdocs gh-deploy --force`, which commits the rendered site to the `gh-pages`
branch. GitHub Pages serves it from there.

Two consequences:

- **A broken link or a page missing from the nav fails the branch**, before it
  can reach the site.
- **The site follows `main`, not the tags.** It is documentation of the current
  code, not of the last release. Merging a documentation fix publishes it
  without a release.

`readthedocs.yaml` builds the same `mkdocs.yml` on Read the Docs if that project
is configured. Neither depends on the other, so the site survives either one
being down.
