"""Copy the repository-root data folders into the wheel as ``diablos/resources``.

Everything else is configured in pyproject.toml. The folders stay at the root
of the checkout (one canonical copy, used directly when running from source);
``diablos.core.app_paths.resource_path`` resolves to ``diablos/resources`` in
an installed package. ``MANIFEST.in`` puts the same files into the sdist, so a
wheel built from the sdist is identical.
"""

import fnmatch
import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))

#: (source, destination under diablos/resources) pairs; directories are copied whole.
RESOURCES = [
    ("config/default_config.json", "config/default_config.json"),
    ("config/logging.json", "config/logging.json"),
    ("locales", "locales"),
    ("examples", "examples"),
    ("licenses", "licenses"),
    ("LICENSE", "LICENSE"),
    ("THIRD_PARTY_LICENSES.md", "THIRD_PARTY_LICENSES.md"),
    ("docs/USER_MANUAL.md", "docs/USER_MANUAL.md"),
]

#: Never shipped: caches and per-user state that can appear in a checkout.
EXCLUDE = ["__pycache__", "*.pyc", ".DS_Store", "recent_files.json", ".autosave*"]


def _ignored(_dir, names):
    return [n for n in names if any(fnmatch.fnmatch(n, pat) for pat in EXCLUDE)]


class BuildWithResources(build_py):
    def run(self):
        super().run()
        target = os.path.join(self.build_lib, "diablos", "resources")
        for src, dst in RESOURCES:
            src_path = os.path.join(HERE, src)
            dst_path = os.path.join(target, dst)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"resource missing from the source tree: {src}")
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, ignore=_ignored, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)


setup(cmdclass={"build_py": BuildWithResources})
