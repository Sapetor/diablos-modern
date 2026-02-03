"""
Integration tests for example diagram files.

These tests verify that example diagrams can be loaded and simulated
without errors.
"""

import pytest
import json
import os
import numpy as np
from pathlib import Path


# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def get_example_files():
    """Get list of example diagram files."""
    if not EXAMPLES_DIR.exists():
        return []
    return list(EXAMPLES_DIR.glob("*.diablos"))


@pytest.mark.integration
class TestExampleDiagramsLoad:
    """Test that example diagrams can be loaded as valid JSON."""

    @pytest.mark.parametrize("example_file", get_example_files(),
                             ids=lambda f: f.name)
    def test_example_loads_as_json(self, example_file):
        """Test that example file is valid JSON."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        assert 'blocks_data' in data, "Missing blocks_data"
        assert 'lines_data' in data, "Missing lines_data"
        assert 'sim_data' in data, "Missing sim_data"

    @pytest.mark.parametrize("example_file", get_example_files(),
                             ids=lambda f: f.name)
    def test_example_has_blocks(self, example_file):
        """Test that example has at least one block."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        assert len(data['blocks_data']) > 0, "Example has no blocks"

    @pytest.mark.parametrize("example_file", get_example_files(),
                             ids=lambda f: f.name)
    def test_example_sim_params_valid(self, example_file):
        """Test that simulation parameters are valid."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        sim_data = data['sim_data']
        assert sim_data.get('sim_time', 0) > 0, "sim_time must be positive"
        assert sim_data.get('sim_dt', 0) > 0, "sim_dt must be positive"


@pytest.mark.integration
class TestVerificationExamples:
    """Test that verification examples have proper structure."""

    def get_verification_files(self):
        """Get list of verification example files."""
        return [f for f in get_example_files() if 'verification' in f.name]

    @pytest.mark.parametrize("example_file",
                             [f for f in get_example_files() if 'verification' in f.name],
                             ids=lambda f: f.name)
    def test_verification_has_notes(self, example_file):
        """Test that verification examples have verification notes."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        assert '_verification_notes' in data, \
            f"Verification example {example_file.name} missing _verification_notes"

    @pytest.mark.parametrize("example_file",
                             [f for f in get_example_files() if 'verification' in f.name],
                             ids=lambda f: f.name)
    def test_verification_notes_complete(self, example_file):
        """Test that verification notes have required fields."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        notes = data.get('_verification_notes', {})
        assert 'problem' in notes, "Missing 'problem' in verification notes"
        assert 'analytical_solution' in notes or 'analytical_solutions' in notes or 'equation' in notes, \
            "Missing analytical solution or equation in verification notes"


def get_full_optimization_examples():
    """Get optimization examples that use the full optimization workflow.

    Excludes optimization_data_fit_demo.diablos which is a simplified
    first-order system response demo, not a full optimization example.
    """
    return [f for f in get_example_files()
            if 'optimization' in f.name
            and 'data_fit' not in f.name]


@pytest.mark.integration
class TestOptimizationExamples:
    """Test that optimization examples have proper structure."""

    @pytest.mark.parametrize("example_file",
                             get_full_optimization_examples(),
                             ids=lambda f: f.name)
    def test_optimization_has_parameter_blocks(self, example_file):
        """Test that optimization examples have Parameter blocks."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        block_types = [b.get('block_fn', '') for b in data['blocks_data']]
        assert 'Parameter' in block_types, \
            f"Optimization example {example_file.name} missing Parameter block"

    @pytest.mark.parametrize("example_file",
                             get_full_optimization_examples(),
                             ids=lambda f: f.name)
    def test_optimization_has_cost_function(self, example_file):
        """Test that optimization examples have CostFunction blocks."""
        with open(example_file, 'r') as f:
            data = json.load(f)

        block_types = [b.get('block_fn', '') for b in data['blocks_data']]
        has_cost = 'CostFunction' in block_types or 'DataFit' in block_types
        assert has_cost, \
            f"Optimization example {example_file.name} missing cost function block"
