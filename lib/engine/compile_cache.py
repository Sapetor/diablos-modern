"""Compiled-system cache fingerprinting.

Pure helpers extracted from :class:`~lib.engine.simulation_engine.SimulationEngine`
that turn block parameters and diagram structure into stable, hashable cache
keys. They carry no engine state — the engine keeps the small stateful cache
(``clear_compile_cache`` / ``_compile_system_cached``) and calls these to build
its keys, so the (subtle) normalization/fingerprint logic can be unit-tested in
isolation.
"""

import hashlib

import numpy as np

from lib.engine.block_names import canonical_fn

# Runtime/history keys that must not invalidate the compiled-system cache.
# The compiler only needs user-facing/resolved parameters and port metadata;
# these values are written during interpreted initialization or post-solve
# replay and would otherwise make every repeat run look structurally new.
_COMPILE_CACHE_ALLOWED_INTERNAL_KEYS = frozenset({"_inputs_", "_outputs_"})
_COMPILE_CACHE_IGNORED_PARAM_KEYS = frozenset(
    {
        "_source_params_fingerprint",
        "_init_start_",
        "_rng",
        "_prev",
        "_x_",
        "_held_output_",
        "_held_value_",
        "_last_value_",
        "_last_time_",
        "_next_sample_time_",
        "_current_value_",
        "_sample_index_",
        "_time_buffer_",
        "_value_buffer_",
        "_vector_buf",
        "_vector_len",
        "_field_history_",
        "_field_history_2d_",
        "_time_history_",
        "_replay_state_",
        "_replay_pending_",
        "_replay_hyst_state_",
        "vector",
        "vec_dim",
        "vec_labels",
        "mem",
        "mem_list",
        "mem_len",
        "output",
        "aux",
    }
)


def normalize_cache_value(value):
    """Convert parameter values to stable, hashable cache-key fragments."""
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.dtype == object:
            return (
                "ndarray-object",
                tuple(arr.shape),
                tuple(normalize_cache_value(x) for x in arr.ravel().tolist()),
            )
        contiguous = np.ascontiguousarray(arr)
        digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
        return ("ndarray", tuple(contiguous.shape), str(contiguous.dtype), digest)
    if isinstance(value, np.generic):
        return normalize_cache_value(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return ("float", "nan")
        if np.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return value
    if isinstance(value, dict):
        return tuple(
            (str(k), normalize_cache_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(normalize_cache_value(v) for v in value)
    if isinstance(value, set):
        return tuple(
            sorted(
                (normalize_cache_value(v) for v in value),
                key=repr,
            )
        )
    try:
        hash(value)
        return value
    except TypeError:
        return (value.__class__.__name__, repr(value))


def compile_param_items(params):
    """Return cache-relevant parameter items, excluding runtime history."""
    items = []
    for key, value in sorted((params or {}).items(), key=lambda item: str(item[0])):
        key_s = str(key)
        if key_s in _COMPILE_CACHE_IGNORED_PARAM_KEYS:
            continue
        if key_s.startswith("_") and key_s not in _COMPILE_CACHE_ALLOWED_INTERNAL_KEYS:
            continue
        items.append((key_s, normalize_cache_value(value)))
    return tuple(items)


def source_params_fingerprint(params):
    """Fingerprint raw block params for exec_params stale-cache detection."""
    return compile_param_items(params)


def compiled_system_fingerprint(blocks, sorted_blocks, lines, dt):
    """Build a conservative cache key for compile_system outputs."""
    block_fp = []
    for block in blocks:
        block_fp.append(
            (
                getattr(block, "name", ""),
                getattr(block, "block_fn", ""),
                canonical_fn(getattr(block, "block_fn", "")),
                int(getattr(block, "in_ports", 0) or 0),
                int(getattr(block, "out_ports", 0) or 0),
                int(getattr(block, "b_type", 0) or 0),
                int(getattr(block, "hierarchy", -1)),
                float(getattr(block, "effective_sample_time", -1.0)),
                compile_param_items(getattr(block, "params", {})),
                compile_param_items(getattr(block, "exec_params", {}) or {}),
            )
        )

    line_fp = tuple(
        (
            getattr(line, "srcblock", None),
            int(getattr(line, "srcport", 0) or 0),
            getattr(line, "dstblock", None),
            int(getattr(line, "dstport", 0) or 0),
            bool(getattr(line, "hidden", False)),
        )
        for line in lines
    )
    order_fp = tuple(getattr(block, "name", "") for block in sorted_blocks)
    return (float(dt), tuple(block_fp), order_fp, line_fp)
