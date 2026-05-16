"""
Regression test for audit priority #4: interpreter memory-block propagation lag.

Hypothesis: in the interpreter execution path, memory blocks (Integrator,
TranFn, StateSpace, etc.) execute twice per step — output_only first (emit
y[k]), then a state-updating full execute (advance x[k+1] = f(x[k], u[k])).
The audit suspected scope captures y[k] AFTER state already advanced to
x[k+1], producing a one-step lag.

Investigation verdict: FALSE POSITIVE.

The dual execution is by design.  At the start of step k the memory block
emits y[k] = C @ x[k] via output_only and propagate_outputs writes y[k]
into the scope's input_queue.  Then the outer hierarchy loop advances state
x[k] → x[k+1] but explicitly does NOT re-propagate memory-block outputs
(lib.py:965 `if block.name not in self.memory_blocks`).  So the scope
reads y[k], not y[k+1], at time t = k*dt.

This test guards the property empirically by running an example with a
strictly-proper TranFn (a memory block) in both compiled and interpreter
modes and asserting the two scope traces agree sample-by-sample within
the discretization tolerance of forward-Euler.  A one-step lag would
show up as interpreter[k] ≈ compiled[k-1] (or vice versa).
"""

from pathlib import Path

import numpy as np
import pytest


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def _load_example(filename: str):
    """Load an example .diablos file and return the configured DSim."""
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    path = EXAMPLES_DIR / filename
    data = dsim.file_service.load(filepath=str(path))
    assert data is not None, f"Failed to load {path}"
    dsim.file_service.apply_loaded_data(data)
    return dsim


def _scope_vector(dsim, scope_username: str = None):
    """Return the (n_samples, vec_dim) ndarray captured by the named scope."""
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        if scope_username and b.params.get("username") != scope_username:
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        vec_dim = params.get("vec_dim", 1)
        if vec is None:
            continue
        return np.asarray(vec).reshape(-1, vec_dim)
    raise AssertionError(
        f"No Scope with captured vector found "
        f"(username={scope_username!r})"
    )


@pytest.mark.regression
class TestInterpreterCompiledEquivalence:
    """
    Run the same example in compiled and interpreter modes and verify
    the resulting scope traces agree.  If the interpreter had a one-step
    lag, the traces would differ by one sample on memory-block outputs.
    """

    def _run(self, filename: str, use_fast: bool):
        dsim = _load_example(filename)
        dsim.use_fast_solver = use_fast
        success, err = dsim.run_tuning_simulation(
            dsim.sim_time, dsim.sim_dt
        )
        assert success, f"Simulation ({use_fast=}) failed: {err}"
        return dsim

    def test_tank_feedback_compiled_matches_interpreter(self, qapp):
        """
        c01_tank_feedback: Step → Sum → Gain → TranFn (strictly proper)
        → Scope, with a unity feedback loop.  The TranFn is the memory
        block exercising the dual-execute path.

        Both solver modes must produce the same closed-loop response;
        a one-step lag in the interpreter would show up as
        interp[k] ≈ compiled[k-1].
        """
        compiled = self._run("c01_tank_feedback.diablos", use_fast=True)
        interp = self._run("c01_tank_feedback.diablos", use_fast=False)

        c_scope = _scope_vector(compiled)
        i_scope = _scope_vector(interp)

        # Equal sample counts (both ran the same time × dt).
        assert c_scope.shape == i_scope.shape, (
            f"Shape mismatch: compiled={c_scope.shape} "
            f"interpreter={i_scope.shape}"
        )

        # Per-sample agreement — same dt forward-Euler should match exactly
        # up to floating-point tolerance.  A one-step lag would shift the
        # entire trajectory by one sample.
        max_diff = float(np.max(np.abs(c_scope - i_scope)))
        assert max_diff < 1e-6, (
            f"Compiled vs interpreter scope traces disagree by max "
            f"{max_diff:.3e} > 1e-6 — possible memory-block timing lag.  "
            f"c_scope[:5,0]={c_scope[:5,0]}, i_scope[:5,0]={i_scope[:5,0]}"
        )

        # Bug-signature check: if there were a one-step lag the SHIFTED
        # traces would agree better than the aligned ones.
        if c_scope.shape[0] >= 3:
            aligned_err = float(np.max(np.abs(c_scope - i_scope)))
            lag1_err = float(np.max(np.abs(c_scope[1:] - i_scope[:-1])))
            assert aligned_err <= lag1_err, (
                f"Aligned error ({aligned_err:.3e}) exceeded one-step-lag "
                f"error ({lag1_err:.3e}) — interpreter output looks "
                f"shifted by one step relative to compiled."
            )

    def test_initial_sample_matches(self, qapp):
        """
        Verify scope[0] is the same in both modes — a lag bug would make
        interpreter scope[0] be the IC and compiled scope[0] be the
        IC + one Euler step (or vice versa).
        """
        compiled = self._run("c01_tank_feedback.diablos", use_fast=True)
        interp = self._run("c01_tank_feedback.diablos", use_fast=False)

        c = _scope_vector(compiled)[0]
        i = _scope_vector(interp)[0]
        assert np.allclose(c, i, atol=1e-9), (
            f"scope[0] mismatch: compiled={c} interpreter={i} — "
            f"would indicate the initial sample is captured at different "
            f"points in the state-update cycle."
        )
