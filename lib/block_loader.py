"""Block registry: the built-in blocks plus any user blocks the app can find.

Built-ins load two ways -- a live filesystem scan of ``blocks/`` in a dev
checkout, and the static ``_BLOCK_MODULES`` list in a frozen PyInstaller build
(kept in sync by ``tools/sync_block_registry.py``).  User blocks (see
``lib/user_blocks.py`` and ``docs/BLOCK_API.md``) load the same way in both
modes, by file path, and are appended after the built-ins so a name collision
always resolves in favour of the built-in.
"""

import os
import sys
import logging
import importlib
import inspect
from typing import List, Optional

from blocks.base_block import BaseBlock, block_contract_errors

logger = logging.getLogger(__name__)

# Static registry of all block modules — used when running as a frozen
# PyInstaller bundle (where filesystem scanning doesn't work).
_BLOCK_MODULES = [
    "blocks.abs_block",
    "blocks.agent_scope",
    "blocks.assert_block",
    "blocks.bodemag",
    "blocks.bodephase",
    "blocks.chirp",
    "blocks.compare_to_constant",
    "blocks.constant",
    "blocks.deadband",
    "blocks.delay",
    "blocks.demux",
    "blocks.derivative",
    "blocks.discrete_statespace",
    "blocks.discrete_transfer_function",
    "blocks.display",
    "blocks.exponential",
    "blocks.export",
    "blocks.external",
    "blocks.fft",
    "blocks.first_order_hold",
    "blocks.from_block",
    "blocks.from_file",
    "blocks.function",
    "blocks.gain",
    "blocks.goto",
    "blocks.hysteresis",
    "blocks.impulse",
    "blocks.inport",
    "blocks.integrator",
    "blocks.logical_operator",
    "blocks.lookup_table",
    "blocks.lqr",
    "blocks.math_function",
    "blocks.matrix_gain",
    "blocks.mux",
    "blocks.network_channel",
    "blocks.noise",
    "blocks.nyquist",
    "blocks.outport",
    "blocks.packet_loss",
    "blocks.pid",
    "blocks.prbs",
    "blocks.product",
    "blocks.ramp",
    "blocks.random_source",
    "blocks.rate_limiter",
    "blocks.rate_transition",
    "blocks.relational_operator",
    "blocks.rootlocus",
    "blocks.saturation",
    "blocks.scope",
    "blocks.selector",
    "blocks.sigproduct",
    "blocks.sine",
    "blocks.statespace",
    "blocks.step",
    "blocks.subsystem",
    "blocks.sum",
    "blocks.switch",
    "blocks.terminator",
    "blocks.transfer_function",
    "blocks.transport_delay",
    "blocks.variable_transport_delay",
    "blocks.wave_generator",
    "blocks.xygraph",
    "blocks.zero_order_hold",
    "blocks.optimization.constraint",
    "blocks.optimization.cost_function",
    "blocks.optimization.data_fit",
    "blocks.optimization.optimizer",
    "blocks.optimization.parameter",
    "blocks.optimization_primitives.adam",
    "blocks.optimization_primitives.linear_system_solver",
    "blocks.optimization_primitives.momentum",
    "blocks.optimization_primitives.numerical_gradient",
    "blocks.optimization_primitives.objective_function",
    "blocks.optimization_primitives.residual_norm",
    "blocks.optimization_primitives.root_finder",
    "blocks.optimization_primitives.state_variable",
    "blocks.optimization_primitives.vector_gain",
    "blocks.optimization_primitives.vector_perturb",
    "blocks.optimization_primitives.vector_sum",
    "blocks.pde.advection_equation_1d",
    "blocks.pde.advection_equation_2d",
    "blocks.pde.diffusion_reaction_1d",
    "blocks.pde.field_processing",
    "blocks.pde.field_processing_2d",
    "blocks.pde.heat_equation_1d",
    "blocks.pde.heat_equation_2d",
    "blocks.pde.wave_equation_1d",
    "blocks.pde.wave_equation_2d",
]


def _collect_block_classes(module_names):
    """Import modules by name and collect BaseBlock subclasses."""
    block_classes = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseBlock)
                    and obj is not BaseBlock
                    and not inspect.isabstract(obj)
                ):
                    block_classes.append(obj)
        except Exception as e:
            logger.error(f"Error loading block {module_name}: {e}", exc_info=True)
    return block_classes


def _validate_builtin_blocks(block_classes):
    """Log contract violations in built-in blocks (development/test only).

    Built-ins are still registered -- refusing to load one would break an
    existing diagram over a cosmetic spec problem -- but the violation is
    reported at load time instead of surfacing as an obscure failure deep in
    the engine.  ``tests/unit/test_block_api.py`` turns the same check into a
    hard CI gate; a frozen build skips it (the tree is fixed at build time).
    """
    for cls in block_classes:
        try:
            problems = block_contract_errors(cls)
        except Exception:  # pragma: no cover - the validator must never break loading
            logger.debug("Could not validate %s", cls, exc_info=True)
            continue
        if problems:
            logger.error(
                "Built-in block %s.%s breaks the block contract:\n  - %s",
                cls.__module__,
                cls.__name__,
                "\n  - ".join(problems),
            )


def load_user_block_classes(diagram_path=None, builtin_classes=None, reload=False):
    """Discovered user block classes, ready to register (never raises).

    ``builtin_classes`` supplies the ``block_name`` values already taken; a
    user block colliding with one is skipped by the loader.
    """
    from lib.user_blocks import load_user_blocks

    taken = []
    for cls in builtin_classes or []:
        try:
            taken.append(cls().block_name)
        except Exception:  # pragma: no cover - a broken built-in is reported elsewhere
            logger.debug("Could not read block_name from %s", cls, exc_info=True)

    try:
        report = load_user_blocks(diagram_path, builtin_names=taken, reload=reload)
    except Exception:
        logger.exception("User block discovery failed; only built-in blocks are available")
        return []
    if report.problems:
        logger.warning("%d user block problem(s); see the log above", len(report.problems))
    return report.classes


def load_builtin_blocks():
    """Import the built-in block modules and return their classes.

    In frozen (PyInstaller) mode, uses a static registry since filesystem
    scanning is not available. In development mode, scans the blocks directory.
    """
    if getattr(sys, "frozen", False):
        return _collect_block_classes(_BLOCK_MODULES)

    # Development mode: scan the filesystem
    block_modules = []
    blocks_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "blocks"))

    logger.info(f"Scanning blocks from: {blocks_dir} (cwd={os.getcwd()})")

    # Load from top-level blocks directory
    for filename in os.listdir(blocks_dir):
        if (
            filename.endswith(".py")
            and not filename.startswith("__")
            and filename != "base_block.py"
        ):
            block_modules.append(f"blocks.{filename[:-3]}")

    # Load from subdirectories (pde, optimization, etc.)
    for subdir in os.listdir(blocks_dir):
        subdir_path = os.path.join(blocks_dir, subdir)
        if os.path.isdir(subdir_path) and not subdir.startswith("__"):
            init_file = os.path.join(subdir_path, "__init__.py")
            if os.path.exists(init_file):
                for filename in os.listdir(subdir_path):
                    if filename.endswith(".py") and not filename.startswith("__"):
                        block_modules.append(f"blocks.{subdir}.{filename[:-3]}")

    classes = _collect_block_classes(block_modules)
    logger.info(f"Loaded {len(classes)} block classes from {len(block_modules)} modules")
    _validate_builtin_blocks(classes)
    return classes


def load_blocks(
    diagram_path: Optional[str] = None,
    include_user: bool = True,
    reload_user: bool = False,
) -> List[type]:
    """Every block class available to the app: built-ins first, then user blocks.

    Args:
        diagram_path: path of the open diagram, so a ``blocks/`` folder next to
            it is searched too (see ``lib.user_blocks``).
        include_user: set False to get the built-ins only (used by tests and by
            tooling that must not execute third-party code).
        reload_user: re-read user modules from disk instead of reusing the
            already-imported ones (the "Reload user blocks" action).

    Returns:
        A list of block classes. User blocks are appended, so any lookup that
        takes the first match keeps preferring the built-in.
    """
    classes = load_builtin_blocks()
    if not include_user:
        return classes
    user_classes = load_user_block_classes(
        diagram_path, builtin_classes=classes, reload=reload_user
    )
    return classes + user_classes
