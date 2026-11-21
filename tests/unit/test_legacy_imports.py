"""Code written for the pre-package layout keeps working (diablos/compat.py).

Until November 2026 the engine, block library and GUI were the top-level
packages ``lib``, ``blocks`` and ``modern_ui``; they are now ``diablos.core``,
``diablos.blocks`` and ``diablos.gui``. User blocks in the wild import
``blocks.base_block`` and ``lib.engine.compiler_kernels``; scripts import
``lib.api``. Neither name may come back as a real top-level package.
"""

import importlib
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import diablos.core.user_blocks as ub
from diablos.blocks.base_block import BaseBlock

pytestmark = pytest.mark.unit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEGACY_BLOCK = textwrap.dedent(
    """
    import numpy as np
    import blocks.base_block
    from blocks.base_block import BaseBlock
    from lib.engine.compiler_kernels import kernel


    class LegacyTriplerBlock(BaseBlock):
        @property
        def block_name(self):
            return "LegacyTripler"

        @property
        def category(self):
            return "Math"

        @property
        def params(self):
            return {}

        @property
        def inputs(self):
            return [{"name": "in", "type": "any"}]

        @property
        def outputs(self):
            return [{"name": "out", "type": "any"}]

        def execute(self, time, inputs, params, **kwargs):
            # A method-level import of an old name works too.
            from lib.icon_path import IconPath  # noqa: F401
            return {0: np.atleast_1d(inputs.get(0, 0.0)) * 3.0}

        def draw_icon(self, block_rect):
            from lib.icon_path import IconPath
            path = IconPath()
            path.moveTo(0.1, 0.5)
            path.lineTo(0.9, 0.5)
            return path


    ALIAS_OK = blocks.base_block.BaseBlock is BaseBlock


    @kernel("Legacytripler")
    def build_legacy_tripler(ctx):
        b_name, src = ctx.b_name, ctx.input_sources[0]

        def exec_legacy_tripler(t, y, dy_vec, signals):
            signals[b_name] = 3.0 * signals.get(src, 0.0)

        return exec_legacy_tripler
    """
)


@pytest.fixture
def legacy_block(tmp_path, monkeypatch):
    folder = tmp_path / "blocks"
    folder.mkdir()
    (folder / "legacy_tripler.py").write_text(LEGACY_BLOCK)
    monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(folder))
    home = tmp_path / "userdata"
    home.mkdir()
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(home))
    ub.purge_user_modules()
    yield folder
    ub.purge_user_modules()


def test_user_block_with_old_imports_loads(legacy_block, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="diablos.compat"):
        report = ub.load_user_blocks()
    assert [c().block_name for c in report.classes] == ["LegacyTripler"]
    cls = report.classes[0]
    # One BaseBlock, whichever name imported it.
    assert issubclass(cls, BaseBlock)
    assert sys.modules[cls.__module__].ALIAS_OK
    out = cls().execute(0.0, {0: np.array([2.0])}, {})
    assert out[0][0] == 6.0
    assert "import 'diablos.blocks.base_block' instead" in caplog.text


def test_old_names_are_not_installed_globally(legacy_block):
    ub.load_user_blocks()
    assert "lib" not in sys.modules and "blocks" not in sys.modules
    with pytest.raises(ImportError):
        importlib.import_module("lib.api")


def _script(code):
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return out


def test_script_compat_maps_old_names_onto_the_same_modules():
    out = _script(
        """
        import warnings
        warnings.simplefilter("error", DeprecationWarning)
        try:
            from lib.api import load
        except ImportError:
            pass
        else:
            raise SystemExit("lib.api imported without install_legacy_imports()")

        import diablos.compat
        diablos.compat.install_legacy_imports()
        warnings.simplefilter("ignore", DeprecationWarning)
        from lib.api import load
        import lib.api, diablos.core.api, blocks.base_block, diablos.blocks.base_block
        assert lib.api is diablos.core.api
        assert blocks.base_block.BaseBlock is diablos.blocks.base_block.BaseBlock
        res = load("examples/first_order_step_response.diablos").run(t=2.0)
        print(round(float(res["output"][-1]), 4))
        """
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-1] == "0.3935"


def test_script_compat_refuses_to_shadow_a_real_lib_package(tmp_path):
    pkg = tmp_path / "lib"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("OTHER = True\n")
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import diablos.compat\n"
            "try:\n"
            "    diablos.compat.install_legacy_imports()\n"
            "except ImportError as e:\n"
            "    print('refused', 'lib' in str(e))\n"
            "import lib; print(lib.OTHER)\n",
        ],
        cwd=str(tmp_path),
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ["PYTHONPATH"]])),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split() == ["refused", "True", "True"]
