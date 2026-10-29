"""Workspace variables: named values that block parameters may reference.

A :class:`Workspace` belongs to one diagram. Every ``DSim`` owns its own
instance (``dsim.workspace``, shared with its ``SimulationEngine``) and threads
it to every consumer -- parameter resolution, mask scopes in the flattener, the
linearizer, the LQR analyzer and the Python exporter -- so two diagrams in one
process never see each other's variables.  Analysis clones
(``DSim.clone_for_analysis``) get an independent :meth:`Workspace.copy`.

:class:`WorkspaceManager` is kept for backward compatibility as the
process-level *default* workspace: ``WorkspaceManager()`` still returns one
shared instance.  The GUI's single ``DSim`` is built on it, so third-party or
user-block code calling ``WorkspaceManager()`` keeps seeing the variables loaded
in the window.  Nothing under ``lib/`` reads it: a headless ``DSim`` (CLI,
importable API, analysis clones) owns a fresh ``Workspace`` the default never
sees.
"""

import ast
import copy
import logging
import os
from typing import Any, Dict, Optional

from lib.safe_eval import safe_expr

logger = logging.getLogger(__name__)


class Workspace:
    """A diagram's workspace variables, optionally loaded from a text file."""

    def __init__(
        self, variables: Optional[Dict[str, Any]] = None, workspace_file: Optional[str] = None
    ):
        self.variables: Dict[str, Any] = dict(variables or {})
        self.workspace_file: Optional[str] = workspace_file

    def copy(self) -> "Workspace":
        """An independent plain :class:`Workspace` with deep-copied variables."""
        return Workspace.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        """Picklable snapshot ``{"variables": {...}, "workspace_file": str | None}``.

        The variables are deep-copied, so later changes to either side never
        leak across.
        """
        return {
            "variables": copy.deepcopy(dict(self.variables)),
            "workspace_file": self.workspace_file,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Workspace":
        """Rebuild a plain :class:`Workspace` from :meth:`to_dict` output."""
        data = data or {}
        return Workspace(
            copy.deepcopy(dict(data.get("variables") or {})), data.get("workspace_file")
        )

    def load_from_file(self, filepath):
        """
        Load variables from a text file.
        Supported syntax: variable = value (Python syntax)
        """
        if not os.path.exists(filepath):
            logger.error(f"Workspace file not found: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Parse the file content safely
            tree = ast.parse(content)
            new_variables = {}

            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Evaluate the value
                            try:
                                value = ast.literal_eval(node.value)
                                new_variables[target.id] = value
                            except ValueError:
                                # Handle simple math expressions if needed, or just skip complex ones
                                logger.warning(f"Could not evaluate value for {target.id}")

            self.variables = new_variables
            self.workspace_file = filepath
            logger.info(f"Loaded {len(self.variables)} variables from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading workspace file: {e}")
            return False

    def get_value(self, name):
        """Get the value of a variable."""
        return self.variables.get(name)

    def resolve_params(self, params):
        """
        Resolve parameters in a dictionary.
        Returns a new dictionary with strings resolved to values where possible.
        """
        resolved = params.copy()
        for key, value in params.items():
            if isinstance(value, str):
                # Check if it's a direct variable match
                if value in self.variables:
                    resolved[key] = self.variables[value]
                else:
                    # Try to evaluate expression using workspace variables
                    try:
                        # Use safe_expr with the variables dictionary as locals
                        # This allows expressions like "[K, K]" or "2*K" to be resolved
                        val = safe_expr(value, variables=self.variables, allow_numpy=False)
                        resolved[key] = val
                    except (ValueError, SyntaxError, NameError, TypeError):
                        # Keep as string if evaluation fails
                        pass
            elif isinstance(value, list):
                # Recursively resolve lists (though usually lists are numbers)
                # For now, we assume lists are already resolved or contain literals
                pass

        return resolved

    def set_variable(self, name, value):
        """Set or update a variable."""
        self.variables[name] = value

    def delete_variable(self, name):
        """Delete a variable."""
        if name in self.variables:
            del self.variables[name]

    def get_all_variables(self):
        """Get all variables as a dictionary."""
        return self.variables

    def save_to_file(self, filepath=None):
        """Save variables to the current or specified file."""
        path = filepath or self.workspace_file
        if not path:
            logger.warning("No filepath specified for saving workspace.")
            return False

        try:
            with open(path, "w") as f:
                for name, value in self.variables.items():
                    # We use repr() to ensure the value is written as a valid Python literal (e.g. string with quotes)
                    f.write(f"{name} = {repr(value)}\n")
            logger.info(f"Saved workspace to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving workspace file: {e}")
            return False


class WorkspaceManager(Workspace):
    """The process-level default workspace (backward-compatible singleton).

    ``WorkspaceManager()`` always returns the same instance, and the GUI's
    ``DSim`` is built on it.  Library code never calls it -- it uses the owning
    diagram's ``dsim.workspace`` -- so it only serves third-party code written
    against the old singleton API.  Setting ``WorkspaceManager._instance =
    None`` starts a fresh default; a ``DSim`` already built on the old one keeps
    it.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            Workspace.__init__(instance)
            cls._instance = instance
        return cls._instance

    def __init__(self):
        # State is set once in __new__: re-running Workspace.__init__ on every
        # ``WorkspaceManager()`` call would wipe the shared variables.
        pass
