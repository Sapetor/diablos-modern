"""
Pytest configuration and shared fixtures for DiaBloS tests.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import diablos modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor

# Need a QApplication instance for PyQt tests
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for tests that need Qt."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def sample_colors():
    """Provide sample color palette for tests."""
    return {
        'black': QColor(0, 0, 0),
        'red': QColor(255, 0, 0),
        'blue': QColor(0, 0, 255),
        'green': QColor(0, 255, 0),
        'magenta': QColor(255, 0, 255),
    }


@pytest.fixture
def sample_block_rect():
    """Provide a sample QRect for block positioning."""
    return QRect(100, 100, 100, 80)


@pytest.fixture
def sample_position():
    """Provide a sample QPoint for positioning."""
    return QPoint(200, 200)


@pytest.fixture
def simulation_model(qapp, sample_colors):
    """Create a SimulationModel instance for testing."""
    from lib.models.simulation_model import SimulationModel
    model = SimulationModel()
    return model


@pytest.fixture
def simulation_engine(qapp, simulation_model):
    """Create a SimulationEngine instance for testing."""
    from lib.engine.simulation_engine import SimulationEngine
    engine = SimulationEngine(simulation_model)
    return engine


@pytest.fixture
def file_service(qapp, simulation_model):
    """Create a FileService instance for testing."""
    from lib.services.file_service import FileService
    service = FileService(simulation_model)
    return service


@pytest.fixture
def sample_block(qapp, sample_block_rect, sample_colors):
    """Create a sample DBlock for testing."""
    from lib.simulation.block import DBlock
    block = DBlock(
        block_fn='TestBlock',
        sid=0,
        coords=sample_block_rect,
        color='red',
        in_ports=1,
        out_ports=1,
        b_type=2,
        io_edit='none',
        fn_name='testblock',
        params={'gain': 1.0},
        external=False,
        colors=sample_colors
    )
    return block


@pytest.fixture
def sample_line(qapp):
    """Create a sample DLine for testing."""
    from lib.simulation.connection import DLine
    from PyQt5.QtCore import QPoint

    line = DLine(
        sid=0,
        srcblock='block1',
        srcport=0,
        dstblock='block2',
        dstport=0,
        points=[QPoint(100, 100), QPoint(200, 200)]
    )
    return line


@pytest.fixture
def temp_diagram_file(tmp_path):
    """Provide a temporary file path for diagram testing."""
    return tmp_path / "test_diagram.dat"
