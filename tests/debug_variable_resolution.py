#!/usr/bin/env python3
"""
Quick test to verify variable resolution is working
"""
import sys
sys.path.insert(0, '/Users/apeters/diablos-modern')

from lib.workspace import WorkspaceManager

# Test 1: Store and retrieve variable
print("Test 1: Store and retrieve variable")
wm = WorkspaceManager()
wm.set_variable('K', 10)
wm.set_variable('num', [1, 2])
wm.set_variable('den', [1, 3, 2])

print(f"  Variables: {wm.variables}")
print(f"  K = {wm.get_variable('K')}")
print(f"  num = {wm.get_variable('num')}")
print(f"  den = {wm.get_variable('den')}")

# Test 2: Resolve parameters
print("\nTest 2: Resolve parameters")
params = {
    'gain': 'K',
    'num': 'num',
    'den': 'den',
    'constant_value': 5  # Not a variable
}
print(f"  Original params: {params}")
resolved = wm.resolve_params(params)
print(f"  Resolved params: {resolved}")

# Test 3: Test from string (like Variable Editor does)
print("\nTest 3: Parse from string")
code = """
K = 10
num = [1, 2]
den = [1, 3, 2]
"""
wm2 = WorkspaceManager()  # Fresh instance
import ast
tree = ast.parse(code)
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                try:
                    var_value = ast.literal_eval(node.value)
                    wm2.set_variable(var_name, var_value)
                    print(f"  Set {var_name} = {var_value}")
                except (ValueError, SyntaxError) as e:
                    print(f"  Error parsing {var_name}: {e}")

print(f"\nFinal workspace: {wm2.variables}")
