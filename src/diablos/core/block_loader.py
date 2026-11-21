"""Block registry: the built-in blocks plus any user blocks the app can find.

Built-ins load two ways -- a live filesystem scan of ``blocks/`` in a dev
checkout, and the static ``_BLOCK_MODULES`` list in a frozen PyInstaller build
(kept in sync by ``tools/sync_block_registry.py``).  User blocks (see
``src/diablos/core/user_blocks.py`` and ``docs/BLOCK_API.md``) load the same way in both
modes, by file path, and are appended after the built-ins so a name collision
always resolves in favour of the built-in.
"""

import os
import sys
import logging
import importlib
import inspect
from typing import List, Optional

from diablos.blocks.base_block import BaseBlock, block_contract_errors

logger = logging.getLogger(__name__)

# Static registry of all block modules — used when running as a frozen
# PyInstaller bundle (where filesystem scanning doesn't work).
_BLOCK_MODULES = [
    "diablos.blocks.abs_block",
    "diablos.blocks.agent_scope",
    "diablos.blocks.assert_block",
    "diablos.blocks.bodemag",
    "diablos.blocks.bodephase",
    "diablos.blocks.chirp",
    "diablos.blocks.compare_to_constant",
    "diablos.blocks.constant",
    "diablos.blocks.deadband",
    "diablos.blocks.delay",
    "diablos.blocks.demux",
    "diablos.blocks.derivative",
    "diablos.blocks.discrete_statespace",
    "diablos.blocks.discrete_transfer_function",
    "diablos.blocks.display",
    "diablos.blocks.exponential",
    "diablos.blocks.export",
    "diablos.blocks.external",
    "diablos.blocks.fft",
    "diablos.blocks.first_order_hold",
    "diablos.blocks.from_block",
    "diablos.blocks.from_file",
    "diablos.blocks.function",
    "diablos.blocks.gain",
    "diablos.blocks.goto",
    "diablos.blocks.hysteresis",
    "diablos.blocks.impulse",
    "diablos.blocks.inport",
    "diablos.blocks.integrator",
    "diablos.blocks.logical_operator",
    "diablos.blocks.lookup_table",
    "diablos.blocks.lqr",
    "diablos.blocks.math_function",
    "diablos.blocks.matrix_gain",
    "diablos.blocks.mux",
    "diablos.blocks.network_channel",
    "diablos.blocks.noise",
    "diablos.blocks.nyquist",
    "diablos.blocks.outport",
    "diablos.blocks.packet_loss",
    "diablos.blocks.pid",
    "diablos.blocks.prbs",
    "diablos.blocks.product",
    "diablos.blocks.ramp",
    "diablos.blocks.random_source",
    "diablos.blocks.rate_limiter",
    "diablos.blocks.rate_transition",
    "diablos.blocks.relational_operator",
    "diablos.blocks.rootlocus",
    "diablos.blocks.saturation",
    "diablos.blocks.scope",
    "diablos.blocks.selector",
    "diablos.blocks.serial_read",
    "diablos.blocks.serial_write",
    "diablos.blocks.sigproduct",
    "diablos.blocks.sine",
    "diablos.blocks.statespace",
    "diablos.blocks.step",
    "diablos.blocks.subsystem",
    "diablos.blocks.sum",
    "diablos.blocks.switch",
    "diablos.blocks.system_identification",
    "diablos.blocks.terminator",
    "diablos.blocks.transfer_function",
    "diablos.blocks.transport_delay",
    "diablos.blocks.udp_receive",
    "diablos.blocks.udp_send",
    "diablos.blocks.variable_transport_delay",
    "diablos.blocks.wave_generator",
    "diablos.blocks.xygraph",
    "diablos.blocks.zero_order_hold",
    "diablos.blocks.optimization.constraint",
    "diablos.blocks.optimization.cost_function",
    "diablos.blocks.optimization.data_fit",
    "diablos.blocks.optimization.optimizer",
    "diablos.blocks.optimization.parameter",
    "diablos.blocks.optimization_primitives.adam",
    "diablos.blocks.optimization_primitives.linear_system_solver",
    "diablos.blocks.optimization_primitives.momentum",
    "diablos.blocks.optimization_primitives.numerical_gradient",
    "diablos.blocks.optimization_primitives.objective_function",
    "diablos.blocks.optimization_primitives.residual_norm",
    "diablos.blocks.optimization_primitives.root_finder",
    "diablos.blocks.optimization_primitives.state_variable",
    "diablos.blocks.optimization_primitives.vector_gain",
    "diablos.blocks.optimization_primitives.vector_perturb",
    "diablos.blocks.optimization_primitives.vector_sum",
    "diablos.blocks.pde.advection_equation_1d",
    "diablos.blocks.pde.advection_equation_2d",
    "diablos.blocks.pde.diffusion_reaction_1d",
    "diablos.blocks.pde.field_processing",
    "diablos.blocks.pde.field_processing_2d",
    "diablos.blocks.pde.heat_equation_1d",
    "diablos.blocks.pde.heat_equation_2d",
    "diablos.blocks.pde.wave_equation_1d",
    "diablos.blocks.pde.wave_equation_2d",
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
    from diablos.core.user_blocks import load_user_blocks

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
            it is searched too (see ``diablos.core.user_blocks``).
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
