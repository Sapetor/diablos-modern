"""Sampled-data kernels for the compiled (hybrid) path.

A discrete block (ZeroOrderHold, DiscreteTranFn, DiscreteStateSpace), or a
continuous one given a rate (TranFn, StateSpace, PID, Integrator, Derivative
with ``sampling_time > 0``, which the interpreter runs only at its instants,
discretized over ``dtime = Ts``), holds its output between sample instants, so
inside the ODE right-hand side it is a pure
read of a per-compile *holder*: a source, as far as the RHS is concerned.  The
holder changes only at sample instants, where the runner calls
:func:`sample_update` (see ``lib.engine.zero_crossing.solve_with_events``,
``breakpoints``).

The update runs the block's own ``execute()`` on a run-local copy of its
params, so the discrete math has one source of truth and the compiled samples
equal the interpreter's.  The copy starts with ``_init_start_`` set, because
the engine's initialization has already advanced the live ``exec_params`` once
at t = 0; :func:`reset_sampled` restores it at the start of every run, which
matters on a compile-cache hit.

Holders join the compiler's mode registry, so ``solve_with_events`` records
their outputs in ``mode_history`` at every sample instant and the post-solve
replay (``ModeHistoryReplayer``) shows the value that held at each output
sample.
"""

import copy

import numpy as np

from lib.engine.block_names import canonical_fn
from lib.engine.compiler_kernels import kernel

#: Canonical fn-names of the blocks that are always sampled.
DISCRETE_FNS = frozenset({"Zeroorderhold", "Discretetranfn", "Discretestatespace"})

#: Continuous blocks that become sampled when given a rate (``sampling_time >
#: 0``, or inheriting one). FirstOrderHold and RateTransition stay out: they run
#: every solver step on their own internal schedule, and FOH ramps between
#: samples instead of holding.
RATED_FNS = frozenset({"TransferFcn", "StateSpace", "PID", "Pid", "Integrator", "Deriv"})


def sample_rate(block) -> float:
    """The block's resolved rate (> 0) or its declared one; <= 0 means none."""
    rate = getattr(block, "effective_sample_time", -1.0)
    if isinstance(rate, (int, float)) and rate > 0:
        return float(rate)
    try:
        return float(block.resolve_sample_time())
    except (AttributeError, TypeError, ValueError):
        return -1.0


def is_sampled_block(block) -> bool:
    """True for a block the hybrid path runs as a sampled holder."""
    fn = canonical_fn(getattr(block, "block_fn", ""))
    if fn in DISCRETE_FNS:
        return True
    return fn in RATED_FNS and sample_rate(block) > 0


def _signal_key(b_name, port):
    return b_name if port == 0 else f"{b_name}_out{port}"


@kernel(*sorted(DISCRETE_FNS))
def build_sampled(ctx):
    b_name = ctx.b_name
    block = ctx.block
    rate = sample_rate(block)
    if not rate > 0:
        raise ValueError(f"{b_name}: sampled block without a resolved sample time")
    template = copy.deepcopy(dict(ctx.params))
    holder = ctx.latch(mode=None, init=None)
    holder.update(
        kind="sampled",
        event_driven=True,
        ts=rate,
        block=block,
        input_sources=list(ctx.input_sources),
        template=template,
        params=None,
        reset=reset_sampled,
    )
    reset_sampled(holder)

    def exec_sampled(t, y, dy_vec, signals, _holder=holder):
        outs = _holder["mode"]
        if outs is None:
            signals[b_name] = 0.0
            return
        for port, value in outs.items():
            signals[_signal_key(b_name, port)] = value

    return exec_sampled


def reset_sampled(holder):
    """Fresh run-local params and no output yet (seeded at t0 by the runner)."""
    params = copy.deepcopy(holder["template"])
    params["_init_start_"] = True
    holder["params"] = params
    holder["mode"] = None


def is_sampled(holder):
    return holder.get("kind") == "sampled"


def sample_update(holder, t, signals):
    """Take one sample at ``t``: run the block on ``signals``, latch its outputs.

    Raises RuntimeError when the block reports an error, so the run stops
    instead of silently holding a stale value.
    """
    block = holder["block"]
    inputs = {
        port: signals.get(key, 0.0)
        for port, key in enumerate(holder["input_sources"])
        if key is not None
    }
    params = holder["params"]
    # The runner owns the schedule: every call here is a sample instant.
    if not params.get("_init_start_", True) and "_next_sample_time_" in params:
        params["_next_sample_time_"] = min(float(params["_next_sample_time_"]), float(t))
    instance = block.block_instance
    held = None
    if getattr(instance, "output_is_post_update", False):
        # An Integrator returns its advanced state; the interpreter holds the
        # output_only value taken before the advance (stamp_held_outputs).
        held = _checked(
            block,
            t,
            instance.execute(
                time=t,
                inputs=inputs,
                params=params,
                output_only=True,
                next_add_in_memory=False,
                dtime=holder["ts"],
            ),
        )
    out = _checked(block, t, instance.execute(time=t, inputs=inputs, params=params))
    if "mem" in params and canonical_fn(block.block_fn) == "Integrator":
        # SimulationEngine.sync_integrator_output: the next output_only call
        # reports a copy of the advanced state.
        params["output"] = np.array(params["mem"], copy=True)
    holder["mode"] = {
        port: np.array(value, dtype=float, copy=True)
        for port, value in (held or out).items()
        if isinstance(port, int)
    }


def _checked(block, t, out):
    """``out`` if the block reported no error, else RuntimeError, so the run
    stops instead of silently holding a stale value."""
    if not isinstance(out, dict) or out.get("E"):
        msg = out.get("error", "block error") if isinstance(out, dict) else repr(out)
        raise RuntimeError(f"{block.name} failed at its sample t={t:.6g}: {msg}")
    return out
