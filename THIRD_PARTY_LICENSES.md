# Third-party licences and redistribution notice

DiaBloS Modern's own source code is under the **MIT** licence (see `LICENSE`).
Nothing in this file changes that: the source tree stays MIT, and you may copy
it, vendor a block, build a course around it, or ship a closed-source tool on
top of it.

This file is about the **prebuilt bundles** on the releases page, which are a
different artifact from the source. They embed the Qt 6 libraries and the PyQt6
bindings, and PyQt6 is GPL-3.0-only. It is shipped inside every bundle, next to
the executable, together with the full licence texts in `licenses/`.

**Using the application imposes nothing on you.** The obligations below fall on
whoever *redistributes* a bundle.

## What is in a bundle

| Component | Licence | Notes |
|---|---|---|
| DiaBloS Modern | MIT | this project; `LICENSE` |
| PyQt6 bindings | **GPL-3.0-only** (Riverbank Computing; a commercial licence is also sold) | <https://www.riverbankcomputing.com/software/pyqt/intro> |
| Qt 6 (`Qt6Core`, `Qt6Gui`, `Qt6Widgets`, `Qt6Svg`, platform plugins), shipped as `PyQt6-Qt6` | LGPL v3 | <https://doc.qt.io/qt-6/lgpl.html> |
| `PyQt6-sip` | BSD-2-Clause | <https://pypi.org/project/PyQt6-sip/> |
| NumPy | BSD-3-Clause (with 0BSD / MIT / Zlib / CC0 components) | <https://numpy.org/doc/stable/license.html> |
| SciPy | BSD-3-Clause | <https://github.com/scipy/scipy/blob/main/LICENSE.txt> |
| Matplotlib | Matplotlib licence (PSF-derived, BSD-compatible) | <https://matplotlib.org/stable/project/license.html> |
| pyqtgraph | MIT | <https://github.com/pyqtgraph/pyqtgraph/blob/master/LICENSE.txt> |
| Pillow | MIT-CMU | <https://github.com/python-pillow/Pillow/blob/main/LICENSE> |
| tqdm | MPL-2.0 AND MIT | <https://github.com/tqdm/tqdm/blob/master/LICENCE> |
| SymPy (optional, only if the build environment had it) | BSD-3-Clause | <https://github.com/sympy/sympy/blob/master/LICENSE> |
| CPython runtime | PSF licence v2 | <https://docs.python.org/3/license.html> |

Every licence above is compatible with the GPL. Most are permissive and need
only this notice; tqdm's MPL-2.0 is file-level copyleft, so it reaches tqdm's
own files and nothing else.

## The bundle as a whole is conveyed under GPL v3

DiaBloS Modern imports PyQt6 directly, and PyQt6 is GPL-3.0-only. A bundle that
contains both is therefore a combined work that may only be redistributed under
the terms of the GNU General Public License, version 3 (`licenses/GPL-3.0.txt`).

Two things this does **not** mean:

- It does not relicense DiaBloS Modern's source. The GPL governs the terms on
  which the *combination* may be conveyed; the MIT grant on this project's own
  code is unaffected, and an MIT licence is GPL-compatible precisely so that
  such a combination is permitted.
- It does not affect anyone who installs from source. `pip install -r
  requirements.txt` makes the combination on your own machine for your own use,
  which is not distribution, and nothing is conveyed.

GPL-3.0-**only** on PyQt6 also means a bundle can be conveyed under v3 and not
under a later version of the GPL.

## Corresponding source (GPL v3 section 6)

The complete corresponding source for a bundle is the tagged commit it was
built from, at <https://github.com/Sapetor/diablos-modern>, together with the
exact dependency versions pinned in `requirements.txt` at that tag, all of
which are published on PyPI. Upstream sources: Qt 6 at
<https://download.qt.io/official_releases/qt/> and PyQt6 at
<https://pypi.org/project/PyQt6/#files>. No component is patched. On request,
the maintainers will provide the corresponding source on a physical medium for
no more than the cost of distribution.

## LGPL v3 notice for Qt 6

Qt itself is used under LGPL v3 (`licenses/LGPL-3.0.txt`), not GPL. Qt ships as
ordinary shared libraries, not statically linked and not compressed into a
single-file executable: on macOS inside `DiaBloS.app/Contents/Frameworks/`, on
Windows and Linux beside the executable in the one-folder layout emitted by
`diablos.spec`. A recipient can replace any of those `.dylib`/`.dll`/`.so`
files with a modified build of the same Qt version and re-run the application,
which is the "suitable shared library mechanism" of LGPL v3 section 4(d)(0).
