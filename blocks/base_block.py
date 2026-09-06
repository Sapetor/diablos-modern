"""
Base block module providing the abstract base class for all simulation blocks.

This module is the public, stable entry point for third-party block authors:
``BaseBlock`` is the contract every block implements, ``BLOCK_API_VERSION``
identifies the revision of that contract, and :func:`validate_block_class`
checks a class against it.  See ``docs/BLOCK_API.md`` for the full reference.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np

try:
    from numpy.typing import NDArray
except ImportError:
    # numpy < 1.20 compatibility
    NDArray = np.ndarray

# Type aliases for block interfaces
BlockParams = Dict[str, Any]
BlockInputs = Dict[int, Union[float, int, NDArray]]
BlockOutput = Dict[int, Union[float, int, NDArray]]
BlockResult = Union[BlockOutput, Dict[str, Union[bool, str]]]
PortDefinition = Dict[str, str]

#: Revision of the block contract implemented by this module.
#:
#: Bumped only when a change would break an existing third-party block (a new
#: required member, a changed ``execute()`` signature, a changed return shape).
#: Additive, backwards-compatible changes -- a new *optional* hook, a new port
#: ``type`` token -- leave it alone.  A user block module may declare its own
#: module-level ``BLOCK_API_VERSION``; the loader refuses to import a module
#: that asks for a newer API than this build provides, instead of failing
#: obscurely somewhere inside ``execute()``.
BLOCK_API_VERSION = 1

#: Outline tokens ``resolve_block_shape`` understands (see ``shape`` below).
VALID_BLOCK_SHAPES = ("rect", "triangle", "circle", "tag")

#: Values ``io_editable`` may take.
VALID_IO_EDITABLE = (None, "input", "output", "both")

#: Members every concrete block must provide.
REQUIRED_BLOCK_MEMBERS = ("block_name", "params", "inputs", "outputs", "execute")


class BaseBlock(ABC):
    """
    Abstract base class for all simulation blocks.

    All custom blocks must inherit from this class and implement
    the required abstract properties and methods.

    Attributes:
        optional_inputs: Set of port indices that are optional (not required to be connected).
        optional_outputs: Set of port indices that are optional outputs.
    """

    # Optional ports (override in subclasses)
    optional_inputs: set = set()
    optional_outputs: set = set()

    @property
    @abstractmethod
    def block_name(self) -> str:
        """The user-facing name of the block."""
        pass

    @property
    @abstractmethod
    def params(self) -> BlockParams:
        """A dictionary defining the block's parameters, their types, and default values."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> List[PortDefinition]:
        """A list of input port definitions."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> List[PortDefinition]:
        """A list of output port definitions."""
        pass

    @abstractmethod
    def execute(
        self, time: float, inputs: BlockInputs, params: BlockParams, **kwargs: Any
    ) -> BlockResult:
        """
        The core simulation function for the block.

        Args:
            time: The current simulation time.
            inputs: A dictionary of input values, keyed by port index.
            params: A dictionary of the block's current parameter values.
            **kwargs: Additional keyword arguments (e.g., output_only, dtime).

        Returns:
            A dictionary of output values keyed by port index,
            or an error dict {'E': True, 'error': 'message'} on failure.
        """
        pass

    @property
    def shape(self) -> str:
        """
        Outline shape the canvas renderer draws for this block.

        Supported values:
            "rect"     - rounded rectangle (default)
            "triangle" - amplifier triangle pointing toward the output (Gain)
            "circle"   - ellipse inscribed in the block rect (Sum, Product);
                         the renderer falls back to "rect" when the block has
                         too many ports for a circle to look right
            "tag"      - pentagon pointing along the signal flow (Goto/From)

        Returns:
            Shape token; unknown tokens are drawn as "rect".
        """
        return "rect"

    @property
    def use_port_grid_snap(self) -> bool:
        """
        Whether port positions should snap to grid.

        Some blocks (like triangular gain blocks) need precise port alignment
        without grid snapping for proper visual geometry.

        Returns:
            True to snap ports to grid, False otherwise.
        """
        return True  # Default: use grid snapping

    @property
    def requires_inputs(self) -> bool:
        """
        Whether this block requires all inputs to be connected.

        Source blocks typically don't require inputs, while most other blocks do.
        Override this in subclasses for custom behavior.

        Returns:
            True if inputs must be connected, False otherwise.
        """
        # Default: blocks require inputs unless they're Sources
        return getattr(self, "category", "Other") not in ["Sources"]

    @property
    def requires_outputs(self) -> bool:
        """
        Whether this block requires outputs to be connected.

        Sink blocks and utility blocks typically don't require outputs to be connected.
        Override this in subclasses for custom behavior.

        Returns:
            True if outputs must be connected, False otherwise.
        """
        # Default: blocks require outputs unless they're Sinks or Other
        return getattr(self, "category", "Other") not in ["Sinks", "Other"]

    @property
    def output_is_post_update(self) -> bool:
        """
        Whether execute() returns the state AFTER this step's state update.

        Most stateful blocks compute their output from the pre-update state
        (y = Cx + Du, then x <- Ax + Bu), so the value returned by execute()
        is the output belonging to the current instant.  A block that instead
        returns the freshly advanced state (the Integrator returns x[k+1])
        must not have that value held between discrete samples: when the
        block is gated to a sample time, holding x[k+1] across
        [kTs, (k+1)Ts) makes the trace lead the true sampled response by a
        full sample period.  Override to True and the engine will hold the
        output_only value (the pre-update state) instead.

        Returns:
            True if execute() returns the post-update state, False otherwise.
        """
        return False

    @property
    def requires_sample_time(self) -> bool:
        """
        Whether this block is meaningless without a sample period.

        A block whose whole behaviour is a recursion in the sample index k
        (DiscreteTransferFunction, DiscreteStateSpace) has no continuous-time
        interpretation: if no rate can be resolved for it, it advances one k
        per solver step, so its physical response silently changes when the
        user changes sim_dt.  The engine warns once per such block at
        initialization when its rate resolves to continuous, instead of
        quietly producing a solver-step-dependent answer.

        Returns:
            True if the block needs a resolved sample time to be well defined.
        """
        return False

    @property
    def io_editable(self) -> Optional[str]:
        """
        Whether the block supports user-editable port counts.

        Override in subclasses that allow variable port numbers.

        Returns:
            'input' - user can change input port count
            'output' - user can change output port count
            'both' - user can change both
            None - port count is fixed (default)
        """
        return None

    def draw_icon(self, block_rect: Any) -> Optional[Any]:
        """
        Return a QPainterPath for the block's icon in normalized coordinates.

        The path should use coordinates from 0.0 to 1.0 where (0,0) is top-left
        and (1,1) is bottom-right. The path will be automatically scaled and
        positioned within the block's drawing area.

        Override this method in subclasses to provide custom block icons.
        If not overridden, the fallback switch-based drawing in DBlock is used.

        Args:
            block_rect: QRect of the block (for context, not for positioning).

        Returns:
            QPainterPath in 0-1 normalized coordinates, or None to use fallback.
        """
        return None  # Default: use fallback switch-based drawing

    def symbolic_execute(
        self, inputs: Dict[int, Any], params: BlockParams
    ) -> Optional[Dict[int, Any]]:
        """
        Symbolic execution for equation extraction and linearization.

        This method returns symbolic expressions (using SymPy) instead of
        numeric values. Used by the SymbolicEngine for:
        - Automatic equation extraction
        - Transfer function computation
        - Linearization at operating points
        - LaTeX/MathML export

        Override this method in subclasses to provide symbolic behavior.
        If not overridden, returns None indicating no symbolic support.

        Args:
            inputs: Dict of symbolic input expressions (port_idx -> sympy expr).
            params: Dict of block parameters (may contain symbolic params).

        Returns:
            Dict of symbolic output expressions {port_idx: sympy_expr}, or
            None if block doesn't support symbolic execution.

        Example for Gain block::

            def symbolic_execute(self, inputs, params):
                from sympy import Symbol
                K = params.get('gain', Symbol('K'))
                u = inputs.get(0, Symbol('u'))
                return {0: K * u}
        """
        return None  # Default: no symbolic support

    def get_symbolic_params(self, params: BlockParams) -> Dict[str, Any]:
        """
        Convert numeric parameters to symbolic for equation extraction.

        Override to specify which parameters should be symbolic
        and how they should be named.

        Args:
            params: Dict of block parameters.

        Returns:
            Dict of symbolic parameters (name -> sympy Symbol).
        """
        return {}  # Default: no symbolic parameters


# ---------------------------------------------------------------------------
# Contract validation
#
# The block contract used to be enforced only by whatever blew up first at
# runtime: a block with a misspelled port key or a params spec of the wrong
# shape imported fine, appeared in the palette and then failed deep inside the
# engine.  ``validate_block_class`` turns those into one actionable message at
# load time.  It is advisory for built-ins (dev mode logs the problems) and
# mandatory for user blocks, which are skipped when they fail.
#
# Messages here are developer-facing and stay English (see CLAUDE.md).
# ---------------------------------------------------------------------------


class BlockContractError(TypeError):
    """Raised by :func:`validate_block_class` when a class breaks the contract."""


def _describe(cls: Any) -> str:
    module = getattr(cls, "__module__", "?")
    name = getattr(cls, "__name__", repr(cls))
    return "{}.{}".format(module, name)


def _check_params_spec(spec: Any, problems: List[str]) -> None:
    """Validate the nested ``params`` spec dict."""
    if not isinstance(spec, dict):
        problems.append(
            "params must return a dict mapping parameter name -> spec, got {}".format(
                type(spec).__name__
            )
        )
        return
    for key, meta in spec.items():
        if not isinstance(key, str) or not key:
            problems.append(
                "params has a non-string key {!r}; parameter names must be str".format(key)
            )
            continue
        if not isinstance(meta, dict):
            # A bare value is accepted as shorthand for {"default": value}.
            continue
        if "default" not in meta:
            problems.append(
                "params['{}'] is a spec dict without a 'default' key; add "
                "{{'type': ..., 'default': ..., 'doc': ...}}".format(key)
            )
        if "type" in meta and not isinstance(meta["type"], str):
            problems.append(
                "params['{}']['type'] must be a string token (e.g. 'float', "
                "'int', 'bool', 'string', 'vector', 'choice'), got {}".format(
                    key, type(meta["type"]).__name__
                )
            )
        if "doc" in meta and not isinstance(meta["doc"], str):
            problems.append(
                "params['{}']['doc'] must be a string (it is shown as a tooltip "
                "and translated at display time)".format(key)
            )
        for choice_key in ("choices", "options"):
            if choice_key in meta and not isinstance(meta[choice_key], (list, tuple)):
                problems.append(
                    "params['{}']['{}'] must be a list of allowed values".format(key, choice_key)
                )
        if "range" in meta:
            rng = meta["range"]
            ok = isinstance(rng, (list, tuple)) and len(rng) == 2
            if ok:
                ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in rng)
            if not ok:
                problems.append(
                    "params['{}']['range'] must be a (min, max) pair of numbers".format(key)
                )


def _check_ports(ports: Any, kind: str, problems: List[str]) -> int:
    """Validate an ``inputs``/``outputs`` port list; returns the port count."""
    if not isinstance(ports, (list, tuple)):
        problems.append(
            "{} must return a list of port dicts like "
            "[{{'name': 'in', 'type': 'any'}}], got {}".format(kind, type(ports).__name__)
        )
        return 0
    for index, port in enumerate(ports):
        where = "{}[{}]".format(kind, index)
        if not isinstance(port, dict):
            problems.append(
                "{} must be a dict like {{'name': 'in', 'type': 'any'}}, got {}".format(
                    where, type(port).__name__
                )
            )
            continue
        name = port.get("name")
        if not isinstance(name, str) or not name.strip():
            problems.append("{} needs a non-empty string 'name'".format(where))
        if "type" in port and not isinstance(port["type"], str):
            problems.append(
                "{}['type'] must be a string token ('any', 'float', 'vector', "
                "'matrix'), got {}".format(where, type(port["type"]).__name__)
            )
    return len(ports)


def _check_optional_ports(instance: Any, attr: str, count: int, problems: List[str]) -> None:
    value = getattr(instance, attr, set())
    if isinstance(value, (set, frozenset, list, tuple)):
        for index in value:
            if not isinstance(index, int) or isinstance(index, bool):
                problems.append("{} must hold integer port indices, found {!r}".format(attr, index))
            elif not 0 <= index < count:
                problems.append(
                    "{} lists port index {} but the block declares {} port(s)".format(
                        attr, index, count
                    )
                )
    else:
        problems.append(
            "{} must be a set of port indices (e.g. {{1}}), got {}".format(
                attr, type(value).__name__
            )
        )


def _check_execute_signature(cls: Any, problems: List[str]) -> None:
    execute = getattr(cls, "execute", None)
    if not callable(execute):
        problems.append("execute must be a method")
        return
    try:
        signature = inspect.signature(execute)
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return
    names = [
        p.name
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and p.name != "self"
    ]
    for required in ("time", "inputs", "params"):
        if required not in names:
            problems.append(
                "execute() must accept a '{}' argument -- the engine calls it as "
                "execute(time=..., inputs=..., params=..., **kwargs)".format(required)
            )
    if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        problems.append(
            "execute() must end with **kwargs -- the engine passes extra keywords "
            "(dtime, output_only, ...) to some blocks and a block without **kwargs "
            "raises TypeError at run time"
        )


def block_contract_errors(cls: Any) -> List[str]:
    """Return a list of contract violations for ``cls`` (empty when valid).

    Every message names the member at fault and says what to write instead, so
    it can be shown verbatim to the author of a custom block.
    """
    problems: List[str] = []

    if not inspect.isclass(cls):
        return ["{!r} is not a class; a block must be a class deriving from BaseBlock".format(cls)]
    if not issubclass(cls, BaseBlock):
        return ["{} does not inherit from blocks.base_block.BaseBlock".format(_describe(cls))]
    if cls is BaseBlock:
        return ["BaseBlock itself is abstract and cannot be registered as a block"]

    if inspect.isabstract(cls):
        missing = sorted(getattr(cls, "__abstractmethods__", ()) or ())
        return [
            "{} is abstract: it does not implement {}. Every block must define "
            "block_name, params, inputs, outputs (as @property) and execute().".format(
                _describe(cls), ", ".join(missing) or "the required members"
            )
        ]

    try:
        instance = cls()
    except Exception as exc:
        return [
            "{} could not be instantiated as cls(): {}: {}. A block class must be "
            "constructible with no arguments.".format(_describe(cls), type(exc).__name__, exc)
        ]

    def _read(member: str) -> Any:
        try:
            return getattr(instance, member)
        except Exception as exc:
            problems.append("reading '{}' raised {}: {}".format(member, type(exc).__name__, exc))
            return None

    name = _read("block_name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        problems.append(
            "block_name must return a non-empty string (it is the palette name and "
            "the key stored in saved diagrams), got {!r}".format(name)
        )

    params_spec = _read("params")
    if params_spec is not None:
        _check_params_spec(params_spec, problems)

    inputs = _read("inputs")
    n_in = _check_ports(inputs, "inputs", problems) if inputs is not None else 0
    outputs = _read("outputs")
    n_out = _check_ports(outputs, "outputs", problems) if outputs is not None else 0

    _check_optional_ports(instance, "optional_inputs", n_in, problems)
    _check_optional_ports(instance, "optional_outputs", n_out, problems)

    _check_execute_signature(cls, problems)

    category = getattr(instance, "category", "Other")
    if not isinstance(category, str) or not category.strip():
        problems.append(
            "category must be a non-empty string (it groups the block in the "
            "palette), got {!r}".format(category)
        )

    shape = _read("shape")
    if shape is not None and shape not in VALID_BLOCK_SHAPES:
        problems.append(
            "shape must be one of {}, got {!r}".format(", ".join(VALID_BLOCK_SHAPES), shape)
        )

    io_editable = _read("io_editable")
    if io_editable not in VALID_IO_EDITABLE:
        problems.append(
            "io_editable must be one of None, 'input', 'output', 'both'; got {!r}".format(
                io_editable
            )
        )

    for hook in ("draw_icon", "symbolic_execute"):
        member = getattr(cls, hook, None)
        if member is not None and not callable(member):
            problems.append("{} must be a method when defined".format(hook))

    return problems


def validate_block_class(cls: Any) -> None:
    """Check ``cls`` against the block contract, raising on any violation.

    Args:
        cls: The candidate block class.

    Raises:
        BlockContractError: with one line per problem found.
    """
    problems = block_contract_errors(cls)
    if problems:
        raise BlockContractError(
            "{} does not satisfy the DiaBloS block contract (BLOCK_API_VERSION {}):\n  - {}".format(
                _describe(cls), BLOCK_API_VERSION, "\n  - ".join(problems)
            )
        )
