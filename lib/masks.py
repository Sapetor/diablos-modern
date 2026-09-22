"""Subsystem masks -- the model behind user library blocks.

A *mask* turns a Subsystem into a reusable component with its own small
parameter surface.  The user names the block, gives it an icon and a
category, and declares a handful of parameters; the blocks *inside* the
subsystem then reference those parameter names in their own values
(an inner Gain with ``gain = "K"``, a TranFn with ``denominator = "[m, b]"``).

Storage
-------
Everything lives in the Subsystem block's ``params`` dict, so it round-trips
through ``FileService`` for free:

``params["_mask"]``
    The mask *definition*: display name, description, icon, category and the
    ordered parameter specs.  The leading underscore keeps it out of the
    property editor's generic parameter loop.

``params[<mask param name>]``
    The instance *value* of each mask parameter -- an ordinary params entry,
    so the existing property-editor / undo / clipboard plumbing edits it with
    no special cases.  Values may be literals or expression strings.

Resolution
----------
:func:`resolve_mask_scope` evaluates the mask parameter values (through
``lib.safe_eval``, with the diagram's workspace variables underneath) into a
plain ``{name: value}`` scope.  :func:`resolve_params_in_scope` then rewrites
an inner block's parameter strings against that scope.  Both are pure: the
caller decides where the resolved dict goes.  The flattener applies them to
its *clones*, and ``DSim._resolve_block_params`` applies them to
``exec_params`` -- the stored ``params`` of the user's blocks are never
touched, so ``gain = "K"`` survives save/load and every re-run.

Nested masks resolve outer-to-inner: an inner masked subsystem's parameter
*values* are evaluated in the outer mask's scope, and its own resolved scope
is what its children see.
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from lib.i18n import tr, tr_noop
from lib.safe_eval import SafeEvalError, safe_expr

logger = logging.getLogger(__name__)

__all__ = [
    "MASK_FORMAT_VERSION",
    "MASK_KEY",
    "MASK_PARAM_TYPES",
    "MASK_SHAPES",
    "MaskError",
    "default_mask",
    "normalize_mask",
    "get_mask",
    "set_mask",
    "clear_mask",
    "is_masked",
    "is_subsystem",
    "mask_parameters",
    "mask_display_name",
    "mask_icon_text",
    "mask_shape",
    "mask_values",
    "coerce_mask_value",
    "resolve_mask_scope",
    "resolve_params_in_scope",
    "child_scope_for",
    "refresh_saveable_params",
    "apply_mask_appearance",
]

#: Bumped when the on-disk shape of a mask dict changes incompatibly.
MASK_FORMAT_VERSION = 1

#: Key under which the mask definition is stored in a Subsystem's ``params``.
MASK_KEY = "_mask"

#: Parameter types a mask may declare (mirrors ``BaseBlock.params`` types).
MASK_PARAM_TYPES = ("float", "int", "string", "choice", "list", "bool")

#: Outline shapes a masked block may draw itself with.
MASK_SHAPES = ("rect", "triangle", "circle", "tag")

# Structural params written by DBlock/the UI; a mask parameter may not shadow one.
_RESERVED_PARAM_NAMES = frozenset({"_name_", "_inputs_", "_outputs_", MASK_KEY})


class MaskError(ValueError):
    """Raised when a mask is malformed or one of its parameters cannot resolve."""


# ---------------------------------------------------------------------------
# Definition helpers
# ---------------------------------------------------------------------------


def default_mask(name: str = tr_noop("Masked Subsystem")) -> Dict[str, Any]:
    """Return a new, empty but valid mask definition.

    ``name`` is a default-parameter literal, evaluated once at import time
    before a language is necessarily active -- ``tr_noop`` marks it for
    extraction without translating it here. A caller that wants the display
    string in the active language passes ``tr("Masked Subsystem")`` itself
    (see ``modern_ui/widgets/mask_editor_dialog.py``).
    """
    return {
        "format_version": MASK_FORMAT_VERSION,
        "name": name,
        "description": "",
        "icon": "",
        "shape": "rect",
        "category": tr_noop("User Library"),
        "parameters": [],
    }


def _normalize_param(spec: Any, index: int, seen: set) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise MaskError(
            tr(
                "Mask parameter #{index} must be a dict, got {type_name}",
                index=index + 1,
                type_name=type(spec).__name__,
            )
        )

    name = str(spec.get("name", "") or "").strip()
    if not name:
        raise MaskError(tr("Mask parameter #{index} has no name", index=index + 1))
    if not name.isidentifier():
        raise MaskError(
            tr(
                "Mask parameter '{name}' is not a valid identifier "
                "(letters, digits and underscores; must not start with a digit)",
                name=name,
            )
        )
    if name in _RESERVED_PARAM_NAMES:
        raise MaskError(
            tr("Mask parameter '{name}' collides with a reserved block parameter", name=name)
        )
    if name in seen:
        raise MaskError(tr("Duplicate mask parameter '{name}'", name=name))
    seen.add(name)

    ptype = str(spec.get("type", "float") or "float").strip().lower()
    if ptype not in MASK_PARAM_TYPES:
        raise MaskError(
            tr(
                "Mask parameter '{name}' has unknown type '{ptype}' (expected one of: {types})",
                name=name,
                ptype=ptype,
                types=", ".join(MASK_PARAM_TYPES),
            )
        )

    options = spec.get("options") or []
    if not isinstance(options, (list, tuple)):
        raise MaskError(tr("Mask parameter '{name}': 'options' must be a list", name=name))
    options = [str(o) for o in options]
    if ptype == "choice" and not options:
        raise MaskError(
            tr("Mask parameter '{name}' is a choice but declares no options", name=name)
        )

    out = {
        "name": name,
        "type": ptype,
        "default": spec.get("default", _blank_default(ptype, options)),
        "doc": str(spec.get("doc", "") or ""),
        "options": options,
    }
    return out


def _blank_default(ptype: str, options: List[str]) -> Any:
    if ptype == "float":
        return 0.0
    if ptype == "int":
        return 0
    if ptype == "bool":
        return False
    if ptype == "list":
        return []
    if ptype == "choice":
        return options[0] if options else ""
    return ""


def normalize_mask(mask: Any) -> Dict[str, Any]:
    """Validate ``mask`` and return a normalized copy.

    Raises :class:`MaskError` when the definition cannot be repaired.
    """
    if not isinstance(mask, dict):
        raise MaskError(tr("Mask must be a dict, got {type_name}", type_name=type(mask).__name__))

    name = str(mask.get("name", "") or "").strip()
    if not name:
        raise MaskError(tr("Mask needs a display name"))

    shape = str(mask.get("shape", "rect") or "rect").strip().lower()
    if shape not in MASK_SHAPES:
        raise MaskError(
            tr(
                "Unknown mask shape '{shape}' (expected one of: {shapes})",
                shape=shape,
                shapes=", ".join(MASK_SHAPES),
            )
        )

    raw_params = mask.get("parameters", []) or []
    if not isinstance(raw_params, (list, tuple)):
        raise MaskError(tr("Mask 'parameters' must be an ordered list"))

    seen: set = set()
    parameters = [_normalize_param(spec, i, seen) for i, spec in enumerate(raw_params)]

    category = str(mask.get("category", "") or "").strip() or "User Library"

    out = {
        "format_version": int(mask.get("format_version", MASK_FORMAT_VERSION) or 1),
        "name": name,
        "description": str(mask.get("description", "") or ""),
        "icon": str(mask.get("icon", "") or ""),
        "shape": shape,
        "category": category,
        "parameters": parameters,
    }
    # Carry a library back-reference through unchanged when present.
    if mask.get("library_ref"):
        out["library_ref"] = copy.deepcopy(mask["library_ref"])
    return out


# ---------------------------------------------------------------------------
# Block accessors
# ---------------------------------------------------------------------------


def is_subsystem(block: Any) -> bool:
    """True when ``block`` is a Subsystem container (by fn, type or class)."""
    if block is None:
        return False
    if getattr(block, "block_fn", "") == "Subsystem":
        return True
    if getattr(block, "block_type", "") == "Subsystem":
        return True
    return block.__class__.__name__ == "Subsystem"


def get_mask(block: Any) -> Optional[Dict[str, Any]]:
    """Return the block's mask definition, or None when it is not masked.

    A stored mask that no longer validates is reported and treated as absent
    rather than raising into a paint or save path.
    """
    params = getattr(block, "params", None)
    if not isinstance(params, dict):
        return None
    raw = params.get(MASK_KEY)
    if not raw:
        return None
    try:
        return normalize_mask(raw)
    except MaskError as exc:
        logger.warning("Ignoring malformed mask on block %r: %s", getattr(block, "name", "?"), exc)
        return None


def is_masked(block: Any) -> bool:
    """True when ``block`` carries a valid mask definition."""
    return get_mask(block) is not None


def mask_parameters(mask: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ordered parameter specs of ``mask`` (empty list when there is no mask)."""
    if not mask:
        return []
    return list(mask.get("parameters", []) or [])


def mask_display_name(block: Any) -> Optional[str]:
    mask = get_mask(block)
    return mask["name"] if mask else None


def mask_icon_text(block: Any) -> str:
    mask = get_mask(block)
    return (mask or {}).get("icon", "") or ""


def mask_shape(block: Any) -> str:
    mask = get_mask(block)
    return (mask or {}).get("shape", "rect") or "rect"


def coerce_mask_value(spec: Dict[str, Any], raw: Any) -> Any:
    """Best-effort coercion of ``raw`` to the spec's declared type.

    Strings that do not parse are kept verbatim: they are expressions to be
    resolved later against the mask scope / workspace variables.
    """
    ptype = spec.get("type", "float")
    if raw is None:
        return spec.get("default")
    if ptype == "string" or ptype == "choice":
        return raw if isinstance(raw, str) else str(raw)
    if isinstance(raw, str):
        text = raw.strip()
        try:
            if ptype == "int":
                return int(text)
            if ptype == "float":
                return float(text)
            if ptype == "bool":
                return text.lower() in ("1", "true", "yes", "on")
        except ValueError:
            return raw
        return raw  # list / unknown: keep the expression string
    if ptype == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if ptype == "float":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if ptype == "bool":
        return bool(raw)
    return raw


def mask_values(block: Any) -> Dict[str, Any]:
    """Return ``{mask param name: stored raw value}`` for ``block``.

    Falls back to each spec's default when the block has no stored value
    (a mask parameter added after the instance was created).
    """
    mask = get_mask(block)
    if not mask:
        return {}
    params = getattr(block, "params", {}) or {}
    out = {}
    for spec in mask_parameters(mask):
        name = spec["name"]
        out[name] = params[name] if name in params else spec.get("default")
    return out


def refresh_saveable_params(block: Any) -> None:
    """Make sure mask keys survive ``DBlock.saving_params``.

    ``init_params_list`` is frozen at construction time, so keys added later
    (the mask definition and its parameter values) would be dropped on save.
    """
    params = getattr(block, "params", None)
    if not isinstance(params, dict):
        return
    if not hasattr(block, "init_params_list") or block.init_params_list is None:
        block.init_params_list = []
    wanted = [MASK_KEY] + [spec["name"] for spec in mask_parameters(get_mask(block))]
    for key in wanted:
        if key in params and key not in block.init_params_list:
            block.init_params_list.append(key)


def _has_default_username(block: Any) -> bool:
    """True if ``username`` is empty or a label the app generated for it.

    Generated labels differ by creation path (``name``, ``block_fn``,
    ``block_fn + sid``), so all are accepted, case-insensitively.
    """
    current = (getattr(block, "username", "") or "").lower()
    if not current:
        return True
    block_fn = getattr(block, "block_fn", "") or ""
    generated = {
        getattr(block, "name", "") or "",
        block_fn,
        "%s%s" % (block_fn, getattr(block, "sid", "")),
    }
    return current in {label.lower() for label in generated if label}


def apply_mask_appearance(block: Any, force: bool = False) -> None:
    """Adopt the mask's display name as the block's ``username``.

    Only replaces a default/auto-generated username unless ``force`` is set,
    so a user who renamed an instance keeps their label.
    """
    mask = get_mask(block)
    if not mask:
        return
    if force or _has_default_username(block):
        block.username = mask["name"]


def set_mask(block: Any, mask: Optional[Dict[str, Any]], seed_values: bool = True) -> None:
    """Attach ``mask`` to ``block`` (or remove it when ``mask`` is None).

    Seeds any missing parameter values from the specs' defaults, drops values
    for parameters that no longer exist, and keeps ``init_params_list`` in
    sync so everything persists.
    """
    if mask is None:
        clear_mask(block)
        return

    normalized = normalize_mask(mask)
    params = getattr(block, "params", None)
    if params is None:
        block.params = params = {}

    previous = params.get(MASK_KEY) or {}
    previous_names = {
        spec.get("name") for spec in (previous.get("parameters") or []) if isinstance(spec, dict)
    }
    new_names = {spec["name"] for spec in normalized["parameters"]}

    # Retire values whose parameter was removed from the mask.
    for stale in previous_names - new_names:
        params.pop(stale, None)
        if hasattr(block, "init_params_list") and stale in (block.init_params_list or []):
            block.init_params_list.remove(stale)

    params[MASK_KEY] = normalized

    if seed_values:
        for spec in normalized["parameters"]:
            name = spec["name"]
            if name not in params:
                params[name] = copy.deepcopy(spec.get("default"))

    refresh_saveable_params(block)
    # A label still showing the previous mask's name came from that mask, so it follows a rename.
    apply_mask_appearance(
        block, force=bool(previous) and getattr(block, "username", None) == previous.get("name")
    )


def clear_mask(block: Any) -> None:
    """Remove the mask (and its parameter values) from ``block``."""
    params = getattr(block, "params", None)
    if not isinstance(params, dict):
        return
    mask = get_mask(block)
    for spec in mask_parameters(mask):
        params.pop(spec["name"], None)
        if hasattr(block, "init_params_list") and spec["name"] in (block.init_params_list or []):
            block.init_params_list.remove(spec["name"])
    params.pop(MASK_KEY, None)
    if hasattr(block, "init_params_list") and MASK_KEY in (block.init_params_list or []):
        block.init_params_list.remove(MASK_KEY)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _workspace_variables() -> Dict[str, Any]:
    try:
        from lib.workspace import WorkspaceManager

        return dict(WorkspaceManager().get_all_variables() or {})
    except Exception:  # pragma: no cover - workspace is optional
        logger.debug("Workspace variables unavailable for mask resolution", exc_info=True)
        return {}


def resolve_mask_scope(
    mask: Optional[Dict[str, Any]],
    values: Optional[Dict[str, Any]] = None,
    outer_scope: Optional[Dict[str, Any]] = None,
    workspace: Optional[Dict[str, Any]] = None,
    block_label: str = "masked subsystem",
) -> Dict[str, Any]:
    """Evaluate a mask's parameter values into a flat ``{name: value}`` scope.

    Expression strings are evaluated with :func:`lib.safe_eval.safe_expr` over
    ``workspace`` (the diagram's variables, read from the ``WorkspaceManager``
    when not supplied), overlaid with ``outer_scope`` (the enclosing mask's
    scope, so nested masks resolve outer-to-inner), overlaid with the mask
    parameters resolved so far -- a later parameter may therefore reference an
    earlier one.

    Raises :class:`MaskError` naming the block, the parameter and the missing
    variable when an expression cannot be evaluated.
    """
    if not mask:
        return {}
    if workspace is None:
        workspace = _workspace_variables()
    if values is None:
        values = {}

    env: Dict[str, Any] = dict(workspace)
    if outer_scope:
        env.update(outer_scope)

    scope: Dict[str, Any] = {}
    for spec in mask_parameters(mask):
        name = spec["name"]
        raw = values[name] if name in values else spec.get("default")
        ptype = spec.get("type", "float")

        if ptype in ("string", "choice") or not isinstance(raw, str):
            resolved = raw
        else:
            text = raw.strip()
            if not text:
                resolved = raw
            else:
                try:
                    resolved = safe_expr(text, variables=env, allow_numpy=True)
                except SafeEvalError as exc:
                    available = sorted(set(env) | set(scope))
                    raise MaskError(
                        tr(
                            "Mask parameter '{param}' of '{label}' could not be resolved: "
                            "{expr!r} -- {reason}. Available names: {names}",
                            param=name,
                            label=block_label,
                            expr=text,
                            reason=exc,
                            names=", ".join(available) if available else "(none)",
                        )
                    ) from exc
        scope[name] = resolved
        env[name] = resolved

    return scope


def resolve_params_in_scope(
    params: Dict[str, Any], scope: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Return a copy of ``params`` with expression strings resolved in ``scope``.

    Only public keys are touched, and only strings: an unresolvable string
    (a Scope label, a solver name, a workspace variable resolved later) is
    kept verbatim, exactly like ``WorkspaceManager.resolve_params``.  With an
    empty scope the input dict is returned unchanged, so diagrams without
    masks take a strictly identical path.
    """
    if not scope or not isinstance(params, dict):
        return params
    resolved = dict(params)
    for key, value in params.items():
        if key.startswith("_") or not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        try:
            resolved[key] = safe_expr(text, variables=scope, allow_numpy=True)
        except (SafeEvalError, ValueError, SyntaxError, TypeError):
            continue
    return resolved


def child_scope_for(
    block: Any,
    outer_scope: Optional[Dict[str, Any]] = None,
    workspace: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Scope that the children of subsystem ``block`` should resolve against.

    A masked subsystem opens a fresh scope built from its own mask parameters
    (evaluated in ``outer_scope``); an unmasked one simply passes the enclosing
    scope through.
    """
    mask = get_mask(block)
    if not mask:
        return outer_scope
    return resolve_mask_scope(
        mask,
        mask_values(block),
        outer_scope=outer_scope,
        workspace=workspace,
        block_label=getattr(block, "username", None) or getattr(block, "name", "?"),
    )
