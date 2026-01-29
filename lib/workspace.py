import logging
import ast
import os

logger = logging.getLogger(__name__)

class WorkspaceManager:
    """
    Manages workspace variables loaded from text files.
    Singleton pattern to ensure consistent state across the application.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorkspaceManager, cls).__new__(cls)
            cls._instance.variables = {}
            cls._instance.workspace_file = None
        return cls._instance

    def load_from_file(self, filepath):
        """
        Load variables from a text file.
        Supported syntax: variable = value (Python syntax)
        """
        if not os.path.exists(filepath):
            logger.error(f"Workspace file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
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
                        # Use eval with the variables dictionary as locals
                        # This allows expressions like "[K, K]" or "2*K" to be resolved
                        # We use a restricted globals dict for basic safety, but allow math
                        import math
                        safe_globals = {"__builtins__": {}, "math": math, "list": list, "int": int, "float": float}
                        val = eval(value, safe_globals, self.variables)
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
            with open(path, 'w') as f:
                for name, value in self.variables.items():
                    # We use repr() to ensure the value is written as a valid Python literal (e.g. string with quotes)
                    f.write(f"{name} = {repr(value)}\n")
            logger.info(f"Saved workspace to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving workspace file: {e}")
            return False

