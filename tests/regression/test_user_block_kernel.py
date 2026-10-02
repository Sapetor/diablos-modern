"""A user block that registers a kernel runs on the compiled fast path.

``SystemCompiler.COMPILABLE_BLOCKS`` is a hard-coded allowlist, so before this
fix a diagram containing *any* user block fell back to the interpreter even
when the block's module registered ``@kernel(...)`` -- the documented way to
contribute a compiled kernel (``docs/examples/custom_kernel_template.py``).

Now a user block (``lib.user_blocks``) with a registered kernel is compilable,
while the allowlist still gates built-ins (Impulse and Noise have kernels but
are deliberately interpreted), and a user module can no longer replace a
built-in kernel by registering under its name.

The diagram is ``Step(2) -> SoftClip(limit=1) -> 1/(s+1) -> Scope``: the clip
runs inside the ODE right-hand side, so the TranFn state settles at 1.0, not 2.0.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.regression

ROOT = Path(__file__).parent.parent.parent
TEMPLATE = ROOT / "docs" / "examples" / "custom_kernel_template.py"
BASE_DIAGRAM = ROOT / "examples" / "optimization_data_fit_demo.diablos"


@pytest.fixture
def user_block_diagram(tmp_path, monkeypatch):
    """A user-blocks folder holding the kernel template, plus a diagram using it."""
    from lib import user_blocks as ub

    blocks_dir = tmp_path / "user_blocks"
    blocks_dir.mkdir()
    shutil.copy(TEMPLATE, blocks_dir / "soft_clip.py")
    monkeypatch.setenv(ub.BLOCKS_ENV_VAR, str(blocks_dir))
    monkeypatch.setattr(ub, "get_user_data_dir", lambda: str(tmp_path / "no_user_dir"))

    data = json.loads(BASE_DIAGRAM.read_text())
    for block in data["blocks_data"]:
        if block["block_fn"] == "Step":
            block["params"]["value"] = 2.0
        elif block["block_fn"] == "Gain":
            block["block_fn"] = "SoftClip"
            block["fn_name"] = "softclip"
            block["params"] = {"limit": 1.0}
            old_name, new_name = "gain{}".format(block["sid"]), "softclip{}".format(block["sid"])
    for line in data["lines_data"]:
        for end in ("srcblock", "dstblock"):
            if line[end] == old_name:
                line[end] = new_name
    diagram = tmp_path / "soft_clip_demo.diablos"
    diagram.write_text(json.dumps(data))
    yield diagram
    ub.purge_user_modules()


def _run(diagram, use_fast_solver):
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    previous = WorkspaceManager._instance
    WorkspaceManager._instance = None
    try:
        dsim = DSim()
        data = dsim.file_service.load(filepath=str(diagram))
        assert data is not None
        dsim.file_service.apply_loaded_data(data)
        assert any(b.block_fn == "SoftClip" for b in dsim.model.blocks_list), (
            "the user block did not load into the diagram"
        )
        dsim.use_fast_solver = use_fast_solver
        ok, message = dsim.run_tuning_simulation(10.0, 0.01)
        assert ok, message
        scope = next(b for b in dsim.engine.active_blocks_list if b.block_fn == "Scope")
        trace = np.asarray(getattr(scope, "exec_params", scope.params)["vector"], dtype=float)
        return dsim, trace.ravel()
    finally:
        WorkspaceManager._instance = previous


@pytest.mark.qt
def test_user_kernel_diagram_compiles(qapp, user_block_diagram):
    from lib.engine.system_compiler import SystemCompiler

    calls = []
    original = SystemCompiler.compile_system

    def spy(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    SystemCompiler.compile_system = spy
    try:
        dsim, trace = _run(user_block_diagram, use_fast_solver=True)
    finally:
        SystemCompiler.compile_system = original

    assert dsim.engine.get_compile_fallback_reason() is None
    assert calls, "the fast solver fell back to the interpreter"
    assert np.isclose(trace[-1], 1.0 - np.exp(-10.0), rtol=1e-3)


@pytest.mark.qt
def test_both_engines_agree(qapp, user_block_diagram):
    _, compiled = _run(user_block_diagram, use_fast_solver=True)
    _, interpreted = _run(user_block_diagram, use_fast_solver=False)
    n = min(compiled.size, interpreted.size)
    assert n > 100
    assert np.allclose(compiled[:n], interpreted[:n], atol=2e-2)


@pytest.mark.unit
class TestKernelGate:
    def _block(self, block_fn, cls):
        return type("FakeDBlock", (), {"block_fn": block_fn, "block_instance": cls()})()

    def test_builtin_with_a_kernel_but_off_the_allowlist_stays_interpreted(self):
        from blocks.noise import NoiseBlock
        from lib.engine.compiler_kernels import get_kernel_builder
        from lib.engine.system_compiler import SystemCompiler

        assert get_kernel_builder("Noise") is not None
        assert not SystemCompiler._has_user_kernel(self._block("Noise", NoiseBlock))

    def test_user_block_without_a_kernel_is_not_compilable(self, monkeypatch):
        from blocks.base_block import BaseBlock
        from lib import user_blocks as ub
        from lib.engine.system_compiler import SystemCompiler

        class NoKernelBlock(BaseBlock):
            block_name = "NoKernelXyz"
            params = {}
            inputs = []
            outputs = []

            def execute(self, time, inputs, params, **kwargs):
                return {}

        setattr(NoKernelBlock, ub.USER_BLOCK_FLAG, True)
        assert not SystemCompiler._has_user_kernel(self._block("NoKernelXyz", NoKernelBlock))

    def test_a_user_module_cannot_replace_a_builtin_kernel(self):
        from lib.engine.compiler_kernels import KERNEL_BUILDERS, kernel

        builtin = KERNEL_BUILDERS["Gain"]

        def hijack(ctx):  # pragma: no cover - must never be registered
            raise AssertionError

        hijack.__module__ = "diablos_user_blocks.evil"
        kernel("Gain")(hijack)
        assert KERNEL_BUILDERS["Gain"] is builtin
