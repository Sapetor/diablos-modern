# DiaBloS Testing Suite

This directory contains the test suite for DiaBloS Modern.

## Setup

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

### Run all tests

```bash
pytest
```

### Run with verbose output

```bash
pytest -v
```

### Run specific test file

```bash
pytest tests/unit/test_simulation_model.py
```

### Run specific test class

```bash
pytest tests/unit/test_simulation_model.py::TestAddBlock
```

### Run specific test

```bash
pytest tests/unit/test_simulation_model.py::TestAddBlock::test_add_block_creates_block_instance
```

### Run only unit tests

```bash
pytest -m unit
```

### Run only integration tests

```bash
pytest -m integration
```

### Run with coverage report

```bash
pytest --cov=lib --cov=modern_ui --cov-report=html --cov-report=term
```

View coverage report:
```bash
# Open htmlcov/index.html in browser
```

## Test Organization

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for complete workflows
- `tests/fixtures/` - Test data and fixtures
- `conftest.py` - Shared pytest fixtures and configuration

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.qt` - Tests that require PyQt
- `@pytest.mark.file_io` - Tests involving file operations

## Writing Tests

### Test file naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example test

```python
import pytest

@pytest.mark.unit
@pytest.mark.qt
def test_add_block(simulation_model):
    \"\"\"Test that blocks can be added to the model.\"\"\"
    initial_count = len(simulation_model.blocks_list)

    # Add block logic here

    assert len(simulation_model.blocks_list) == initial_count + 1
```

### Using fixtures

Common fixtures are available in `conftest.py`:

- `qapp` - QApplication instance
- `simulation_model` - SimulationModel instance
- `simulation_engine` - SimulationEngine instance
- `file_service` - FileService instance
- `sample_block` - Sample DBlock for testing
- `sample_line` - Sample DLine for testing
- `sample_colors` - Color palette dictionary

## Test Coverage Goals

- Core MVC components: 80%+
- Block execution: 70%+
- File I/O: 80%+
- Overall: 70%+

## Continuous Integration

Tests are run automatically on:

- Pull requests
- Commits to main branch
- Pre-commit hooks (if configured)

## Troubleshooting

### Qt platform plugin errors

If you see errors about Qt platform plugins on headless systems:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

### Import errors

Make sure you're running pytest from the project root:

```bash
cd /path/to/diablos-modern
pytest
```

### Python path issues

The conftest.py file adds the project root to Python path automatically.
